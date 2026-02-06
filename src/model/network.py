import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np

# --- Board Encoding ---
# 18 input planes: 12 piece types (6 per color), 4 castling, 1 en passant, 1 side to move
# Board is always flipped so the current player sees from their own perspective.

NUM_PLANES = 18
POLICY_SIZE = 4672  # 8x8 source squares * 73 move types

PIECE_INDICES = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
}

# move encoding: 73 planes per source square
# 0-55: queen-style moves (8 directions * 7 distances)
# 56-63: knight moves (8 directions)
# 64-72: underpromotions (3 directions * 3 piece types: N, B, R)

# 8 directions for queen-style: N, NE, E, SE, S, SW, W, NW
QUEEN_DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
# 8 knight move offsets
KNIGHT_MOVES = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]

_move_to_idx = {}
_idx_to_move = {}


def _build_move_tables():
    """Build lookup tables between UCI moves and policy indices once at import time."""
    if _move_to_idx:
        return

    for from_sq in range(64):
        from_rank, from_file = divmod(from_sq, 8)

        # queen-style moves (covers rook, bishop, queen moves and pawn pushes/captures)
        for dir_idx, (dr, df) in enumerate(QUEEN_DIRS):
            for dist in range(1, 8):
                to_rank = from_rank + dr * dist
                to_file = from_file + df * dist
                if 0 <= to_rank < 8 and 0 <= to_file < 8:
                    to_sq = to_rank * 8 + to_file
                    plane = dir_idx * 7 + (dist - 1)
                    idx = from_sq * 73 + plane
                    _move_to_idx[(from_sq, to_sq, None)] = idx
                    _idx_to_move[idx] = (from_sq, to_sq, None)

        # knight moves
        for k_idx, (dr, df) in enumerate(KNIGHT_MOVES):
            to_rank = from_rank + dr
            to_file = from_file + df
            if 0 <= to_rank < 8 and 0 <= to_file < 8:
                to_sq = to_rank * 8 + to_file
                plane = 56 + k_idx
                idx = from_sq * 73 + plane
                _move_to_idx[(from_sq, to_sq, None)] = idx
                _idx_to_move[idx] = (from_sq, to_sq, None)

        # underpromotions (only from ranks where pawns can promote)
        # 3 directions: straight, capture-left, capture-right (from current player's view)
        # pawns move "up" (increasing rank) toward promotion rank 7
        # 3 pieces: knight, bishop, rook (queen promo is already a queen-style move)
        promo_dirs = [(1, 0), (1, -1), (1, 1)]
        promo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

        for d_idx, (dr, df) in enumerate(promo_dirs):
            for p_idx, piece in enumerate(promo_pieces):
                to_rank = from_rank + dr
                to_file = from_file + df
                if 0 <= to_rank < 8 and 0 <= to_file < 8:
                    to_sq = to_rank * 8 + to_file
                    plane = 64 + d_idx * 3 + p_idx
                    idx = from_sq * 73 + plane
                    _move_to_idx[(from_sq, to_sq, piece)] = idx
                    _idx_to_move[idx] = (from_sq, to_sq, piece)


_build_move_tables()


def encode_board(board: chess.Board) -> np.ndarray:
    """Convert a board position to an 18x8x8 float array from the current player's perspective."""
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    flip = board.turn == chess.BLACK

    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            continue

        rank, file = divmod(sq, 8)
        if flip:
            rank = 7 - rank

        if piece.color == board.turn:
            plane_idx = PIECE_INDICES[piece.piece_type]
        else:
            plane_idx = PIECE_INDICES[piece.piece_type] + 6

        planes[plane_idx, rank, file] = 1.0

    # castling rights
    if board.turn == chess.WHITE:
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[12] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[13] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[14] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[15] = 1.0
    else:
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[12] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[13] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[14] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[15] = 1.0

    # en passant
    if board.ep_square is not None:
        ep_rank, ep_file = divmod(board.ep_square, 8)
        if flip:
            ep_rank = 7 - ep_rank
        planes[16, ep_rank, ep_file] = 1.0

    # side to move (always 1 since we flip the board)
    planes[17] = 1.0

    return planes


def _flip_square(sq):
    """Flip a square index vertically (for black's perspective)."""
    rank, file = divmod(sq, 8)
    return (7 - rank) * 8 + file


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Convert a chess.Move to a policy index for the current player's perspective."""
    from_sq = move.from_square
    to_sq = move.to_square
    promo = move.promotion

    if board.turn == chess.BLACK:
        from_sq = _flip_square(from_sq)
        to_sq = _flip_square(to_sq)

    # queen promotion uses the regular queen-style move encoding
    if promo == chess.QUEEN:
        promo = None

    key = (from_sq, to_sq, promo)
    return _move_to_idx[key]


def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    """Convert a policy index back to a chess.Move."""
    from_sq, to_sq, promo = _idx_to_move[idx]

    if board.turn == chess.BLACK:
        from_sq = _flip_square(from_sq)
        to_sq = _flip_square(to_sq)

    # if a pawn reaches the last rank without an underpromotion, it's a queen promo
    if promo is None:
        from_rank = chess.square_rank(from_sq)
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(to_sq)
            if (board.turn == chess.WHITE and to_rank == 7) or \
               (board.turn == chess.BLACK and to_rank == 0):
                promo = chess.QUEEN

    return chess.Move(from_sq, to_sq, promotion=promo)


def get_legal_move_mask(board: chess.Board) -> np.ndarray:
    """Return a binary mask over the 4672-dim policy vector for legal moves."""
    mask = np.zeros(POLICY_SIZE, dtype=np.float32)
    for move in board.legal_moves:
        idx = move_to_index(move, board)
        mask[idx] = 1.0
    return mask


# --- Neural Network ---

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class ChessNet(nn.Module):
    """Small residual network for chess. Outputs policy logits and a scalar value."""

    def __init__(self, num_blocks=6, channels=128, input_planes=NUM_PLANES):
        super().__init__()
        self.input_conv = nn.Conv2d(input_planes, channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)

        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])

        # policy head
        self.policy_conv = nn.Conv2d(channels, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, POLICY_SIZE)

        # value head
        self.value_conv = nn.Conv2d(channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks:
            x = block(x)

        # policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v.squeeze(-1)
