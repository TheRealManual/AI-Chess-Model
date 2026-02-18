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
