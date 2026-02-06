import os
import json
import math
import time
import chess
import torch
import numpy as np
from datetime import datetime

from src.model.network import ChessNet, encode_board
from src.model.mcts import MCTS
from src.training.config import TrainingConfig


def play_vs_stockfish(model, device, config: TrainingConfig, sf_depth=5,
                      num_games=50, num_sims=200, model_color=None):
    """Play games against Stockfish at the given depth. Returns results dict."""
    try:
        from stockfish import Stockfish
        sf = Stockfish(config.stockfish_path, depth=sf_depth)
    except Exception as e:
        print(f"Could not start Stockfish: {e}")
        print("Make sure Stockfish is installed and STOCKFISH_PATH is set.")
        return None

    model.eval()

    def eval_fn(board):
        board_tensor = encode_board(board)
        inp = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            policy_logits, value = model(inp)
        return policy_logits.cpu().numpy()[0], value.item()

    mcts = MCTS(eval_fn=eval_fn, num_sims=num_sims, cpuct=config.cpuct)

    results = {'wins': 0, 'losses': 0, 'draws': 0, 'games': []}
    depth_start = time.time()

    for game_idx in range(num_games):
        # alternate colors each game unless specified
        if model_color is not None:
            model_is_white = model_color == chess.WHITE
        else:
            model_is_white = (game_idx % 2 == 0)

        board = chess.Board()
        game_moves = []

        while not board.is_game_over(claim_draw=True):
            is_model_turn = (board.turn == chess.WHITE) == model_is_white

            if is_model_turn:
                move, _ = mcts.pick_move(board, temperature=0.1)
            else:
                sf.set_fen_position(board.fen())
                sf_move = sf.get_best_move()
                if sf_move is None:
                    break
                move = chess.Move.from_uci(sf_move)

            game_moves.append(move.uci())
            board.push(move)

            if len(game_moves) > 512:
                break

        result = board.result(claim_draw=True)
        game_record = {
            'model_white': model_is_white,
            'result': result,
            'moves': len(game_moves),
            'game_moves': game_moves,
        }

        if result == '1-0':
            if model_is_white:
                results['wins'] += 1
            else:
                results['losses'] += 1
        elif result == '0-1':
            if model_is_white:
                results['losses'] += 1
            else:
                results['wins'] += 1
        else:
            results['draws'] += 1

        results['games'].append(game_record)

        w, l, d = results['wins'], results['losses'], results['draws']
        print(f"  Game {game_idx + 1}/{num_games}: {result} "
              f"(model={'W' if model_is_white else 'B'}) | "
              f"Running: W={w} L={l} D={d}")

    # estimate win rate
    total = results['wins'] + results['losses'] + results['draws']
    win_rate = (results['wins'] + 0.5 * results['draws']) / max(1, total)
    results['win_rate'] = win_rate
    results['time_s'] = round(time.time() - depth_start, 1)

    return results


def estimate_elo(win_rate, opponent_elo):
    """Rough ELO estimate from win rate against a known-strength opponent.
    Uses the inverse of the expected score formula."""
    if win_rate <= 0.001:
        return opponent_elo - 400
    if win_rate >= 0.999:
        return opponent_elo + 400
    return opponent_elo - 400 * math.log10(1.0 / win_rate - 1)


# approximate ELO for Stockfish at various depths (rough calibration)
SF_DEPTH_ELO = {
    1: 800,
    2: 1000,
    3: 1200,
    4: 1400,
    5: 1600,
    6: 1800,
    7: 2000,
    8: 2200,
    10: 2500,
}


def run_benchmark(model, device, config: TrainingConfig, generation: int,
                  sf_depths=None, num_games=None, num_sims=None,
                  checkpoint_path=None):
    """Benchmark model against Stockfish at multiple depths. Saves results."""
    if sf_depths is None:
        sf_depths = [1, 3, 5]
    if num_games is None:
        num_games = config.benchmark_games
    if num_sims is None:
        num_sims = config.num_sims

    print(f"\nBenchmarking generation {generation}")
    print("-" * 40)

    bench_start = time.time()

    all_results = {
        'generation': generation,
        'timestamp': datetime.now().isoformat(),
        'model': {
            'num_blocks': config.num_blocks,
            'channels': config.channels,
            'checkpoint': checkpoint_path or 'unknown',
        },
        'settings': {
            'num_sims': num_sims,
            'num_games_per_depth': num_games,
            'cpuct': config.cpuct,
            'sf_depths': sf_depths,
            'device': str(device) if not isinstance(device, str) else device,
        },
        'depths': {},
    }

    for depth in sf_depths:
        print(f"\nStockfish depth {depth}:")
        result = play_vs_stockfish(
            model, device, config,
            sf_depth=depth,
            num_games=num_games,
            num_sims=num_sims,
        )
        if result is None:
            continue

        opponent_elo = SF_DEPTH_ELO.get(depth, 1000 + depth * 200)
        estimated = estimate_elo(result['win_rate'], opponent_elo)
        result['estimated_elo'] = estimated

        print(f"  Win rate: {result['win_rate']:.1%}, estimated ELO: {estimated:.0f}")

        per_game = []
        for g in result['games']:
            per_game.append({
                'model_white': g['model_white'],
                'result': g['result'],
                'moves': g['moves'],
            })

        all_results['depths'][str(depth)] = {
            'sf_depth': depth,
            'sf_estimated_elo': opponent_elo,
            'wins': result['wins'],
            'losses': result['losses'],
            'draws': result['draws'],
            'win_rate': result['win_rate'],
            'estimated_elo': estimated,
            'time_s': result.get('time_s', 0),
            'games': per_game,
        }

    total_time = round(time.time() - bench_start, 1)
    all_results['total_time_s'] = total_time

    # compute overall summary across all depths
    total_w = sum(d['wins'] for d in all_results['depths'].values())
    total_l = sum(d['losses'] for d in all_results['depths'].values())
    total_d = sum(d['draws'] for d in all_results['depths'].values())
    total_g = total_w + total_l + total_d
    elo_estimates = [d['estimated_elo'] for d in all_results['depths'].values()]
    all_results['summary'] = {
        'total_wins': total_w,
        'total_losses': total_l,
        'total_draws': total_d,
        'total_games': total_g,
        'overall_win_rate': (total_w + 0.5 * total_d) / max(1, total_g),
        'avg_estimated_elo': sum(elo_estimates) / max(1, len(elo_estimates)),
        'min_estimated_elo': min(elo_estimates) if elo_estimates else 0,
        'max_estimated_elo': max(elo_estimates) if elo_estimates else 0,
    }

    # save results
    os.makedirs(config.analytics_dir, exist_ok=True)
    results_file = os.path.join(config.analytics_dir, 'benchmark_results.json')

    # append to existing results
    existing = []
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            existing = json.load(f)

    existing.append(all_results)
    with open(results_file, 'w') as f:
        json.dump(existing, f, indent=2)

    return all_results
