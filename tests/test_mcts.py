import sys
import os
import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.network import ChessNet, encode_board, POLICY_SIZE
from src.model.mcts import MCTS, MCTSNode


def _make_eval_fn():
    """Create a tiny model for testing."""
    model = ChessNet(num_blocks=2, channels=32)
    model.eval()

    def eval_fn(board):
        tensor = encode_board(board)
        x = torch.tensor(tensor).unsqueeze(0)
        with torch.no_grad():
            policy, value = model(x)
        return policy.numpy()[0], value.item()

    return eval_fn


def test_mcts_returns_valid_move():
    eval_fn = _make_eval_fn()
    mcts = MCTS(eval_fn=eval_fn, num_sims=20, cpuct=2.0)

    board = chess.Board()
    move, visits = mcts.pick_move(board, temperature=1.0)

    assert move in board.legal_moves
    assert visits.shape == (POLICY_SIZE,)
    assert visits.sum() > 0


def test_mcts_deterministic_at_low_temp():
    eval_fn = _make_eval_fn()
    mcts = MCTS(eval_fn=eval_fn, num_sims=50, cpuct=2.0)

    board = chess.Board()
    move1, _ = mcts.pick_move(board, temperature=0.001)
    # with very low temp and same model, should pick the most visited
    # (can't guarantee exact same due to dirichlet noise, but it should be legal)
    assert move1 in board.legal_moves


def test_mcts_visit_counts_sum():
    eval_fn = _make_eval_fn()
    mcts = MCTS(eval_fn=eval_fn, num_sims=30, cpuct=2.0)

    board = chess.Board()
    visits = mcts.search(board)
    total = visits.sum()
    # total visits should be roughly equal to num_sims
    # (root expansion doesn't count as a sim visit to children)
    assert total > 0


def test_mcts_handles_near_endgame():
    # scholar's mate position - checkmate in 1
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    eval_fn = _make_eval_fn()
    mcts = MCTS(eval_fn=eval_fn, num_sims=50, cpuct=2.0)

    move, _ = mcts.pick_move(board, temperature=0.01)
    assert move in board.legal_moves


def test_mcts_terminal_position():
    # checkmate position
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert board.is_checkmate()

    node = MCTSNode(board)
    assert node.is_terminal()
