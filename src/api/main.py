import os
import time
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.engine import ChessEngine
from src.api.schemas import (
    MoveRequest, MoveResponse,
    EvalRequest, EvalResponse,
    AnalyzeRequest, AnalyzeResponse,
    MoveScore,
)

engine: ChessEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    model_path = os.environ.get("MODEL_PATH", "chess_model.onnx")
    print(f"Loading model from {model_path}")
    engine = ChessEngine(model_path)
    print("Engine ready")
    yield
    print("Shutting down")


app = FastAPI(title="Chess AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": engine is not None}


@app.post("/api/move", response_model=MoveResponse)
async def get_move(req: MoveRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    loop = asyncio.get_event_loop()
    t0 = time.time()
    move, value, confidence = await loop.run_in_executor(
        None, engine.get_move, req.fen, req.difficulty
    )
    elapsed = (time.time() - t0) * 1000

    return MoveResponse(
        move=move,
        value=round(value, 4),
        confidence=round(confidence, 4),
        think_time_ms=round(elapsed, 1),
    )


@app.post("/api/evaluate", response_model=EvalResponse)
async def evaluate(req: EvalRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    loop = asyncio.get_event_loop()
    value, top_moves = await loop.run_in_executor(
        None, engine.evaluate, req.fen
    )

    return EvalResponse(
        value=round(value, 4),
        top_moves=[
            MoveScore(move=m['move'], visits=m['visits'], score=round(m['score'], 4))
            for m in top_moves
        ],
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    loop = asyncio.get_event_loop()
    value, all_moves, total_sims = await loop.run_in_executor(
        None, engine.analyze, req.fen, req.num_sims
    )

    return AnalyzeResponse(
        value=round(value, 4),
        moves=[
            MoveScore(move=m['move'], visits=m['visits'], score=round(m['score'], 4))
            for m in all_moves
        ],
        total_simulations=total_sims,
    )
