# AI Chess Model

A chess engine trained through self-play using Monte Carlo Tree Search and a small residual neural network. The trained model exports to ONNX format and runs as a FastAPI microservice, designed for integration with other projects (e.g. a Node.js backend).

## Project Structure

```
src/
  model/       - neural network and MCTS implementation
  training/    - self-play, trainer, replay buffer, config
  benchmark/   - play against Stockfish and estimate ELO
  analytics/   - training plots and dashboard
  api/         - FastAPI server and ONNX inference engine
  export.py    - PyTorch to ONNX conversion
scripts/       - CLI entry points (train, benchmark, export, serve)
tests/         - unit and integration tests
```

## Setup

```bash
pip install -e ".[dev]"
```

For benchmarking, download Stockfish from https://stockfishchess.org/download/ and either put it on your PATH or set `STOCKFISH_PATH`:

```bash
set STOCKFISH_PATH=path/to/stockfish.exe
```

## Training

```bash
python scripts/train.py --iterations 100 --games-per-iter 500 --sims 200
```

Key flags:
- `--parallel-games 12` — concurrent self-play games (defaults to CPU count)
- `--no-amp` — disable mixed precision (on by default, uses GPU if available)
- `--training-steps 1000` — gradient steps per iteration
- `--lr 0.01` — learning rate

Training is resumable — it picks up from the last checkpoint automatically. Checkpoints save to `checkpoints/` and training history (losses, self-play stats, timings, config) saves to `analytics_output/training_history.json`.

## Benchmarking

```bash
python scripts/benchmark.py --games 50 --depths 1 3 5 --sims 200
```

Plays the model against Stockfish at each depth, estimates ELO, and saves full results (per-game records, win/loss/draw, settings) to `analytics_output/benchmark_results.json`. Each run appends to the file so you keep a running history.

## ONNX Export

```bash
python scripts/export.py
```

Creates `chess_model.onnx` from the latest checkpoint. BatchNorm layers are folded for faster inference.

## Running the API

```bash
python scripts/serve.py
```

Or with Docker:

```bash
docker build -t chess-api .
docker run -p 8000:8000 chess-api
```

### Endpoints

- `POST /api/move` — get the best move for a position
- `POST /api/evaluate` — position evaluation with top moves
- `POST /api/analyze` — detailed analysis with visit counts
- `GET /api/health` — health check

### Difficulty Levels

Pass `"difficulty"` in the request body. Each level controls how many MCTS simulations run:

| Level | Simulations | Strength |
|-------|-------------|----------|
| easy | 50 | Fast, weak play |
| medium | 200 | Reasonable |
| hard | 400 | Strong |
| max | 800 | Full strength |

Example:

```bash
curl -X POST http://localhost:8000/api/move \
  -H "Content-Type: application/json" \
  -d '{"fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "difficulty": "medium"}'
```

## Tests

```bash
pytest tests/ -v
```
