import time
import queue
import threading
import numpy as np
import chess
import torch
import torch.multiprocessing as mp

from src.model.network import ChessNet, encode_board, POLICY_SIZE
from src.model.mcts import MCTS
from src.training.config import TrainingConfig


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

    def set_result(self, result_str):
        if result_str == '1-0':
            self.result = 1.0
        elif result_str == '0-1':
            self.result = -1.0
        else:
            self.result = 0.0

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


def play_single_game(model, device, config: TrainingConfig) -> GameRecord:
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

    while not board.is_game_over(claim_draw=True):
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
            break

        board.push(move)
        move_num += 1

        # safety cap at 512 moves
        if move_num >= 512:
            break

    if record.result is None:
        result = board.result(claim_draw=True)
        record.set_result(result)

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


def _game_worker(game_id, model, device, config, results_list, lock):
    """Worker that plays a single game (for threading-based parallelism)."""
    record = play_single_game(model, device, config)
    with lock:
        results_list.append(record)


def run_self_play(model, device, config: TrainingConfig, num_games=None):
    """Run self-play games. Uses threading for parallel games on GPU."""
    if num_games is None:
        num_games = config.games_per_iter

    model.eval()
    records = []

    # for GPU: use threading (shared model, GIL released during torch ops)
    # for CPU: just run sequentially
    if device == "cpu" or config.parallel_games <= 1:
        for i in range(num_games):
            record = play_single_game(model, device, config)
            records.append(record)
            if (i + 1) % 10 == 0:
                print(f"  Self-play: {i + 1}/{num_games} games done")
    else:
        # threaded parallel self-play
        lock = threading.Lock()
        remaining = num_games
        while remaining > 0:
            batch = min(remaining, config.parallel_games)
            threads = []
            for g in range(batch):
                t = threading.Thread(
                    target=_game_worker,
                    args=(g, model, device, config, records, lock)
                )
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            remaining -= batch
            done = num_games - remaining
            print(f"  Self-play: {done}/{num_games} games done")

    # compute some stats
    total_moves = sum(len(r.moves) for r in records)
    wins = sum(1 for r in records if r.result == 1.0)
    losses = sum(1 for r in records if r.result == -1.0)
    draws = sum(1 for r in records if r.result == 0.0)
    avg_length = total_moves / max(1, len(records))
    game_lengths = [len(r.moves) for r in records]

    game_details = []
    for r in records:
        if r.result == 1.0:
            outcome = 'white_win'
        elif r.result == -1.0:
            outcome = 'black_win'
        else:
            outcome = 'draw'
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
