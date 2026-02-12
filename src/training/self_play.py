import os
import time
import json
import queue
import threading
import numpy as np
import chess
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from src.model.network import ChessNet, encode_board, POLICY_SIZE
from src.model.mcts import MCTS
from src.training.config import TrainingConfig
from src.data.openings import get_random_opening, apply_opening


# shared state for live game broadcasting
_live_games_path = None
_live_lock = threading.Lock()
_active_games = {}
_completed_count = 0

# global stop flag for clean shutdown
_stop_requested = False


def request_stop():
    global _stop_requested
    _stop_requested = True


def reset_stop():
    global _stop_requested
    _stop_requested = False


def set_live_broadcast_path(path):
    global _live_games_path
    _live_games_path = path


def _broadcast_game_state(game_id, board, move_num, result=None):
    """Write current game state to the live broadcast file."""
    if _live_games_path is None:
        return
    global _completed_count
    with _live_lock:
        if result is not None:
            _active_games.pop(str(game_id), None)
            _completed_count += 1
        else:
            _active_games[str(game_id)] = {
                'fen': board.fen(),
                'move_num': move_num,
                'last_move': board.peek().uci() if board.move_stack else None,
                'turn': 'white' if board.turn == chess.WHITE else 'black',
            }
        snapshot = {
            'timestamp': time.time(),
            'active_games': dict(_active_games),
            'completed': _completed_count,
        }
        try:
            with open(_live_games_path, 'w') as f:
                json.dump(snapshot, f)
        except:
            pass


class GameRecord:
    """Stores the data from one self-play game."""
    def __init__(self):
        self.boards = []
        self.policies = []
        self.turns = []  # which side was to move
        self.moves = []
        self.result = None  # 1.0, -1.0, or 0.0 from white's perspective

    def add_position(self, board_tensor, policy_target, turn):
        self.boards.append(board_tensor)
        self.policies.append(policy_target)
        self.turns.append(turn)

    def set_result(self, result_str, draw_penalty=0.0):
        if result_str == '1-0':
            self.result = 1.0
        elif result_str == '0-1':
            self.result = -1.0
        else:
            # penalize draws so the model doesn't learn to play safe
            # a small negative value for both sides incentivizes winning
            self.result = -abs(draw_penalty)

    def get_training_samples(self):
        """Return (boards, policies, values) arrays with values from each player's perspective."""
        n = len(self.boards)
        boards = np.array(self.boards)
        policies = np.array(self.policies)
        values = np.zeros(n, dtype=np.float32)

        for i in range(n):
            # value from the perspective of the player who was to move
            if self.turns[i] == chess.WHITE:
                values[i] = self.result
            else:
                values[i] = -self.result

        return boards, policies, values


def make_eval_fn(model, device):
    """Create an evaluation function that runs the model on a single board position."""
    def eval_fn(board):
        board_tensor = encode_board(board)
        inp = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            policy_logits, value = model(inp)
        return policy_logits.cpu().numpy()[0], value.item()
    return eval_fn


def play_single_game(model, device, config: TrainingConfig, game_id=0) -> GameRecord:
    """Play one self-play game and return the game record."""
    eval_fn = make_eval_fn(model, device)
    mcts = MCTS(
        eval_fn=eval_fn,
        num_sims=config.num_sims,
        cpuct=config.cpuct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_weight=config.dirichlet_weight,
    )

    board = chess.Board()
    record = GameRecord()
    resign_counter = 0
    move_num = 0

    # apply a random opening from the book to diversify positions
    if config.use_opening_book:
        opening_name, opening_moves = get_random_opening()
        applied = apply_opening(board, opening_moves)
        # record the opening moves in the game record (but not as training data)
        for m in board.move_stack:
            record.moves.append(m.uci())
        move_num = applied

    while not board.is_game_over(claim_draw=True):
        if _stop_requested:
            record.set_result('1/2-1/2', draw_penalty=config.draw_penalty)
            return record

        # temperature schedule
        if move_num < config.temperature_moves:
            temp = 1.0
        else:
            temp = config.temperature_final

        move, visit_counts = mcts.pick_move(board, temperature=temp)

        # store training data
        board_tensor = encode_board(board)
        # normalize visit counts to get policy target
        total_visits = visit_counts.sum()
        if total_visits > 0:
            policy_target = visit_counts / total_visits
        else:
            policy_target = np.zeros(POLICY_SIZE, dtype=np.float32)

        record.add_position(board_tensor, policy_target, board.turn)
        record.moves.append(move.uci())

        # check resign condition
        eval_fn_result = eval_fn(board)
        value = eval_fn_result[1]
        if value < config.resign_threshold:
            resign_counter += 1
        else:
            resign_counter = 0

        if resign_counter >= config.resign_count and move_num > 20:
            # resign — treat as loss for current player
            if board.turn == chess.WHITE:
                record.set_result('0-1')
            else:
                record.set_result('1-0')
            board.push(move)
            _broadcast_game_state(game_id, board, move_num, result=record.result)
            break

        board.push(move)
        move_num += 1
        _broadcast_game_state(game_id, board, move_num)

        # safety cap at 512 moves
        if move_num >= 512:
            break

    if record.result is None:
        result = board.result(claim_draw=True)
        record.set_result(result, draw_penalty=config.draw_penalty)

    _broadcast_game_state(game_id, board, move_num, result=record.result)
    return record


class BatchEvaluator:
    """Collects board positions from multiple games and evaluates them in GPU batches."""

    def __init__(self, model, device, batch_size=16):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.request_queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running:
            batch = []
            events = []

            try:
                item = self.request_queue.get(timeout=0.01)
                batch.append(item)
                events.append(item[1])
            except queue.Empty:
                continue

            # collect more items up to batch_size
            while len(batch) < self.batch_size:
                try:
                    item = self.request_queue.get_nowait()
                    batch.append(item)
                    events.append(item[1])
                except queue.Empty:
                    break

            # run batch inference
            boards = np.stack([b[0] for b in batch])
            inp = torch.tensor(boards, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                policy_logits, values = self.model(inp)

            policy_np = policy_logits.cpu().numpy()
            values_np = values.cpu().numpy()

            for i, (_, event, result_container) in enumerate(batch):
                result_container['policy'] = policy_np[i]
                result_container['value'] = float(values_np[i])
                event.set()

    def evaluate(self, board_tensor):
        """Submit a board for evaluation, block until result is ready."""
        event = threading.Event()
        result = {}
        self.request_queue.put((board_tensor, event, result))
        event.wait()
        return result['policy'], result['value']

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)


def _game_worker(game_id, model, device, config, results_list, lock, pbar=None):
    """Worker that plays a single game (for threading-based parallelism)."""
    record = play_single_game(model, device, config, game_id=game_id)
    with lock:
        results_list.append(record)
        if pbar is not None:
            pbar.update(1)


def run_self_play(model, device, config: TrainingConfig, num_games=None):
    """Run self-play games. Uses threading for parallel games on GPU."""
    global _completed_count, _active_games
    _completed_count = 0
    _active_games = {}

    if num_games is None:
        num_games = config.games_per_iter

    model.eval()
    records = []

    pbar = tqdm(total=num_games, desc='  Self-play', unit='game',
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    if device == "cpu" or config.parallel_games <= 1:
        for i in range(num_games):
            if _stop_requested:
                break
            record = play_single_game(model, device, config, game_id=i)
            records.append(record)
            pbar.update(1)
    else:
        lock = threading.Lock()
        remaining = num_games
        game_counter = 0
        while remaining > 0:
            if _stop_requested:
                break
            batch = min(remaining, config.parallel_games)
            threads = []
            for g in range(batch):
                t = threading.Thread(
                    target=_game_worker,
                    args=(game_counter + g, model, device, config, records, lock, pbar)
                )
                threads.append(t)
                t.start()

            for t in threads:
                while t.is_alive():
                    t.join(timeout=0.5)
                    if _stop_requested:
                        break

            remaining -= batch
            game_counter += batch

    pbar.close()

    # compute some stats
    total_moves = sum(len(r.moves) for r in records)
    wins = sum(1 for r in records if r.result == 1.0)
    losses = sum(1 for r in records if r.result == -1.0)
    draws = sum(1 for r in records if r.result != 1.0 and r.result != -1.0)
    avg_length = total_moves / max(1, len(records))
    game_lengths = [len(r.moves) for r in records]

    game_details = []
    for r in records:
        if r.result == 1.0:
            outcome = 'white_win'
        elif r.result == -1.0:
            outcome = 'black_win'
        else:
            outcome = 'draw'  # includes draw penalty values like -0.1
        game_details.append({
            'result': outcome,
            'length': len(r.moves),
            'positions': len(r.boards),
        })

    stats = {
        'games': len(records),
        'white_wins': wins,
        'black_wins': losses,
        'draws': draws,
        'avg_game_length': avg_length,
        'min_game_length': min(game_lengths) if game_lengths else 0,
        'max_game_length': max(game_lengths) if game_lengths else 0,
        'total_positions': sum(len(r.boards) for r in records),
        'game_details': game_details,
    }
    print(f"  Results: W={wins} B={losses} D={draws}, avg length={avg_length:.0f}")

    return records, stats
