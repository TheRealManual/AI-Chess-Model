import sys
import os
import numpy as np
import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.network import (
    encode_board, move_to_index, index_to_move, get_legal_move_mask,
    ChessNet, NUM_PLANES, POLICY_SIZE,
)


def test_encode_starting_position():
    board = chess.Board()
    tensor = encode_board(board)
    assert tensor.shape == (NUM_PLANES, 8, 8)
    assert tensor.dtype == np.float32

    # white pawns should be on rank 1 (from white's perspective)
    pawn_plane = tensor[0]  # own pawns
    assert pawn_plane.sum() == 8  # 8 pawns

    # side to move plane should be all 1s (last plane)
    assert tensor[NUM_PLANES - 1].sum() == 64


def test_encode_flips_for_black():
    board = chess.Board()
    board.push_san("e4")  # now it's black to move

    tensor = encode_board(board)
    # "own" pawns (black) should still look like starting position pawns from black's view
    own_pawns = tensor[0]
    assert own_pawns.sum() == 8


def test_move_index_roundtrip():
    board = chess.Board()
    for move in board.legal_moves:
        idx = move_to_index(move, board)
        assert 0 <= idx < POLICY_SIZE
        recovered = index_to_move(idx, board)
        assert recovered == move, f"Roundtrip failed: {move} -> {idx} -> {recovered}"


def test_move_index_roundtrip_black():
    board = chess.Board()
    board.push_san("e4")
    for move in board.legal_moves:
        idx = move_to_index(move, board)
        assert 0 <= idx < POLICY_SIZE
        recovered = index_to_move(idx, board)
        assert recovered == move


def test_legal_move_mask():
    board = chess.Board()
    mask = get_legal_move_mask(board)
    assert mask.shape == (POLICY_SIZE,)
    assert mask.sum() == len(list(board.legal_moves))
    assert mask.dtype == np.float32


def test_promotion_encoding():
    # position where a pawn can promote
    board = chess.Board("8/P7/8/8/8/8/8/4K2k w - - 0 1")
    legal = list(board.legal_moves)
    promo_moves = [m for m in legal if m.promotion is not None]
    assert len(promo_moves) >= 1  # should have queen promo at minimum

    for move in promo_moves:
        idx = move_to_index(move, board)
        recovered = index_to_move(idx, board)
        assert recovered == move, f"Promo roundtrip: {move} -> {idx} -> {recovered}"


def test_network_forward_pass():
    import torch
    model = ChessNet(num_blocks=2, channels=32)  # small for testing
    model.eval()

    x = torch.randn(4, NUM_PLANES, 8, 8)
    with torch.no_grad():
        policy, value = model(x)

    assert policy.shape == (4, POLICY_SIZE)
    assert value.shape == (4,)
    # value should be in [-1, 1] due to tanh
    assert value.min() >= -1.0
    assert value.max() <= 1.0


def test_network_single_position():
    import torch
    model = ChessNet(num_blocks=2, channels=32)
    model.eval()

    board = chess.Board()
    tensor = encode_board(board)
    x = torch.tensor(tensor).unsqueeze(0)

    with torch.no_grad():
        policy, value = model(x)

    assert policy.shape == (1, POLICY_SIZE)
    assert value.shape == (1,)
