from pydantic import BaseModel, field_validator
from typing import Optional
import chess


def _check_fen(v):
    try:
        chess.Board(v)
    except ValueError:
        raise ValueError(f"Invalid FEN: {v}")
    return v


class MoveRequest(BaseModel):
    fen: str
    difficulty: str = "medium"

    @field_validator('fen')
    @classmethod
    def validate_fen(cls, v):
        return _check_fen(v)

    @field_validator('difficulty')
    @classmethod
    def validate_difficulty(cls, v):
        if v not in ('easy', 'medium', 'hard', 'max'):
            raise ValueError(f"Difficulty must be one of: easy, medium, hard, max")
        return v


class MoveResponse(BaseModel):
    move: str
    value: float
    confidence: float
    think_time_ms: float


class EvalRequest(BaseModel):
    fen: str

    @field_validator('fen')
    @classmethod
    def validate_fen(cls, v):
        return _check_fen(v)


class MoveScore(BaseModel):
    move: str
    visits: int
    score: float


class EvalResponse(BaseModel):
    value: float
    top_moves: list[MoveScore]


class AnalyzeRequest(BaseModel):
    fen: str
    num_sims: int = 400

    @field_validator('fen')
    @classmethod
    def validate_fen(cls, v):
        return _check_fen(v)


class AnalyzeResponse(BaseModel):
    value: float
    moves: list[MoveScore]
    total_simulations: int


# ---------------------------------------------------------------------------
# Training mode schemas
# ---------------------------------------------------------------------------


class TrainingAnalyzeMoveRequest(BaseModel):
    fen: str
    player_move: str

    @field_validator('fen')
    @classmethod
    def validate_fen(cls, v):
        return _check_fen(v)

    @field_validator('player_move')
    @classmethod
    def validate_move(cls, v):
        try:
            chess.Move.from_uci(v)
        except (ValueError, chess.InvalidMoveError):
            raise ValueError(f"Invalid UCI move: {v}")
        return v


class TrainingAnalyzeMoveResponse(BaseModel):
    player_move: str
    player_move_rank: int
    player_move_score: float
    best_move: str
    best_move_score: float
    rating: str
    explanation: str
    suggestion: str
    value_before: float
    value_after: float
    top_moves: list[MoveScore]


class TrainingSuggestRequest(BaseModel):
    fen: str

    @field_validator('fen')
    @classmethod
    def validate_fen(cls, v):
        return _check_fen(v)


class TrainingSuggestResponse(BaseModel):
    suggested_move: str
    explanation: str
    confidence: float
    value: float
    alternatives: list[MoveScore]


class PieceInfoRequest(BaseModel):
    fen: str
    square: str

    @field_validator('fen')
    @classmethod
    def validate_fen(cls, v):
        return _check_fen(v)

    @field_validator('square')
    @classmethod
    def validate_square(cls, v):
        v = v.lower()
        if len(v) != 2 or v[0] not in 'abcdefgh' or v[1] not in '12345678':
            raise ValueError(f"Invalid square: {v}")
        return v


class LegalDestination(BaseModel):
    square: str
    is_capture: bool


class PieceInfoResponse(BaseModel):
    piece_name: str
    piece_color: str
    square: str
    movement_rules: str
    legal_destinations: list[LegalDestination]
