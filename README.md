# AI Chess Model

A chess engine trained entirely through self-play using Monte Carlo Tree Search (MCTS) and a residual neural network, inspired by DeepMind's AlphaZero. The trained model exports to ONNX format and serves as a lightweight FastAPI microservice, designed for integration with web backends (e.g. a Node.js chess app on AWS App Runner).

## Overview

- **Self-play training** — the model learns chess from scratch by playing against itself
- **Opening book** — 59 common openings ensure diverse training positions and prevent draw collapse
- **MCTS + neural network** — combines deep search with learned position evaluation
- **ONNX inference** — fast, lightweight inference without PyTorch dependency
- **REST API** — FastAPI server with configurable difficulty levels for concurrent games
- **Stockfish benchmarking** — automated ELO estimation against Stockfish at multiple depths
- **Export snapshots** — full backups of models, checkpoints, and analytics for version tracking

---

## Project Structure

```
AI-Chess-Model/
├── src/
│   ├── model/
│   │   ├── network.py       # ResNet (6 blocks x 128ch, ~2.4M params), board encoding, move indexing
│   │   └── mcts.py          # Monte Carlo Tree Search with PUCT selection
│   ├── training/
│   │   ├── config.py         # All hyperparameters in one dataclass
│   │   ├── self_play.py      # Self-play game generation with parallel threading
│   │   ├── trainer.py        # Training loop, checkpoints, history tracking
│   │   └── replay_buffer.py  # Circular buffer for training positions
│   ├── data/
│   │   └── openings.py       # 60-opening book (Italian, Sicilian, French, QGD, KID, etc.)
│   ├── benchmark/
│   │   └── stockfish_bench.py # Play model vs Stockfish, estimate ELO
│   ├── analytics/
│   │   └── plots.py          # Loss curves, self-play stats, dashboard generation
│   ├── api/
│   │   ├── main.py           # FastAPI app with CORS, lifespan management
│   │   ├── engine.py         # ONNX Runtime inference + MCTS wrapper
│   │   └── schemas.py        # Pydantic request/response models
│   └── export.py             # PyTorch → ONNX conversion with BatchNorm folding
├── scripts/
│   ├── train.py              # Main training entry point
│   ├── benchmark.py          # Standalone Stockfish benchmarking
│   ├── export.py             # Export model + create backup snapshot
│   ├── serve.py              # Start the API server
│   ├── watch.py              # Live ASCII board viewer for self-play
│   └── status.py             # Check training progress from terminal
├── tests/
│   ├── test_model.py         # Network, encoding, move indexing tests (8 tests)
│   ├── test_mcts.py          # MCTS search and move selection tests (5 tests)
│   ├── test_training.py      # Replay buffer tests (4 tests)
│   └── test_api.py           # API endpoint tests (5 tests)
├── exported_models/           # Versioned model snapshots (committed to git)
├── Dockerfile                 # Production container for the API
├── docker-compose.yml         # Docker Compose for local deployment
├── pyproject.toml             # Dependencies and project metadata
├── AI_USE_STATEMENT.md        # AI assistance disclosure
└── README.md
```

---

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU recommended (trains on CPU but very slowly)

### Install

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

### PyTorch with CUDA (recommended)

If you have an NVIDIA GPU, install PyTorch with CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Stockfish (for benchmarking)

Download Stockfish from https://stockfishchess.org/download/ and set the path:

```bash
# Windows PowerShell:
$env:STOCKFISH_PATH = "stockfish\stockfish-windows-x86-64-avx2.exe"

# Linux/Mac:
export STOCKFISH_PATH=/path/to/stockfish
```

---

## Architecture

### Neural Network

| Component | Details |
|-----------|---------|
| Type | Residual CNN |
| Blocks | 6 residual blocks |
| Channels | 128 per layer |
| Parameters | ~2,400,000 |
| Input | 18 planes × 8×8 (12 piece + 4 castling + 1 en passant + 1 side-to-move) |
| Policy head | 4672-dim output (AlphaZero move encoding: 8×8×73) |
| Value head | Single scalar (-1 to +1) |
| ONNX size | ~3–9 MB |

### MCTS

- **PUCT selection** with cpuct=2.0
- **Dirichlet noise** (α=0.3, ε=0.25) at root for exploration
- **Temperature scheduling** — τ=1.0 for first 30 moves, then τ=0.1
- Configurable simulation count per difficulty level

### Opening Book

59 named openings covering all major families:
- King pawn: Italian, Ruy Lopez, Scotch, King's Gambit, Vienna, Petrov, Philidor
- Sicilian: Open, Najdorf, Dragon, Scheveningen, Classical, Alapin, Closed
- French: Advance, Winawer, Tarrasch
- Caro-Kann, Scandinavian, Pirc, Modern, Alekhine
- Queen's Gambit: Declined, Accepted, Slav, Semi-Slav
- Indian: King's Indian, Nimzo-Indian, Queen's Indian, Grunfeld, Catalan, Benoni, Dutch
- Flank: English, Reti, Bird, Larsen
- Systems: London, Trompowsky, Torre, Colle

Each self-play game starts from a randomly chosen opening, preventing the model from converging to repetitive drawing lines.

---

## Scripts

### `scripts/train.py` — Train the Model

Runs the full self-play training loop: generate games → train network → save checkpoint → repeat.

```bash
python scripts/train.py --iterations 30 --games-per-iter 30 --sims 50 --training-steps 500 --parallel-games 8 --watch
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--iterations` | 100 | Number of training generations |
| `--games-per-iter` | 500 | Self-play games per generation |
| `--sims` | 200 | MCTS simulations per move during self-play |
| `--training-steps` | 1000 | Gradient descent steps per generation |
| `--parallel-games` | auto | Concurrent self-play threads (auto-detects CPU cores on GPU, 1 on CPU) |
| `--batch-size` | 256 | Training batch size |
| `--lr` | 0.01 | Learning rate (cosine annealing with warmup) |
| `--buffer-size` | 200000 | Replay buffer capacity (positions) |
| `--benchmark-every` | 10 | Run Stockfish benchmark every N generations (0 to disable) |
| `--benchmark-games` | 50 | Games per benchmark run |
| `--no-amp` | — | Disable mixed precision training |
| `--no-opening-book` | — | Disable random opening book (start all games from initial position) |
| `--watch` | — | Open a live ASCII board viewer in a new terminal window |
| `--checkpoint-dir` | checkpoints | Directory for saving checkpoints |
| `--analytics-dir` | analytics_output | Directory for plots and history |

**Features:**
- Auto-resumes from the latest checkpoint
- Graceful shutdown: first Ctrl+C finishes current games, second force-quits
- Generates loss curve plots and dashboard after each generation
- Saves rich training history (per-generation losses, game stats, timings)

---

### `scripts/benchmark.py` — Benchmark Against Stockfish

Plays the model against Stockfish at specified depths and estimates ELO.

```bash
$env:STOCKFISH_PATH = "stockfish\stockfish-windows-x86-64-avx2.exe"
python scripts/benchmark.py --games 20 --depths 1 3 5 --sims 200
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--games` | 50 | Number of games per Stockfish depth |
| `--depths` | 1 3 5 | Stockfish search depths to test against |
| `--sims` | 200 | MCTS simulations per move for the model |
| `--checkpoint` | latest | Path to a specific checkpoint (auto-finds latest) |
| `--checkpoint-dir` | checkpoints | Where to look for checkpoints |

Results are saved to `analytics_output/benchmark_results.json` with per-game records, win/loss/draw counts, and estimated ELO per depth.

---

### `scripts/export.py` — Export Model & Create Snapshot

Exports the model to ONNX and creates a full backup of the current training state.

```bash
python scripts/export.py --name "v1_opening_book"
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--name` | export | A label for this export (used in folder name) |
| `--checkpoint` | latest | Path to a specific checkpoint |
| `--checkpoint-dir` | checkpoints | Where to find checkpoints |
| `--analytics-dir` | analytics_output | Where to find analytics |
| `--no-verify` | — | Skip ONNX verification step |

Creates a timestamped folder in `exported_models/`:

```
exported_models/
  gen0030_v1_opening_book_20260207_150000/
    chess_model.onnx        # ONNX model ready for deployment
    export_info.json        # Metadata (generation, losses, benchmark results)
    checkpoints/            # All .pt checkpoint files + replay buffer
    analytics_output/       # Plots, dashboard, training history, benchmark data
```

The `exported_models/` directory is tracked by git, so each export is a permanent snapshot you can compare or roll back to.

---

### `scripts/serve.py` — Start the API Server

Launches the FastAPI inference server using an exported ONNX model.

```bash
python scripts/serve.py --model chess_model.onnx --port 8000
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | chess_model.onnx | Path to the ONNX model file |
| `--host` | 0.0.0.0 | Bind address |
| `--port` | 8000 | Port number |
| `--workers` | 1 | Number of Uvicorn workers |

Or with Docker:

```bash
docker build -t chess-api .
docker run -p 8000:8000 chess-api
```

---

### `scripts/watch.py` — Live Game Viewer

Opens an ASCII chess board display showing self-play games in real-time during training.

```bash
python scripts/watch.py --file analytics_output/live_games.json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--file` | analytics_output/live_games.json | Live game data file written by the trainer |
| `--refresh` | 0.5 | Refresh interval in seconds |
| `--max-boards` | 4 | Maximum number of boards to display simultaneously |

Launched automatically when you use `--watch` with the training script. Shows Unicode chess pieces, highlights the last move, and displays game count.

---

### `scripts/status.py` — Check Training Progress

Prints a formatted table of training progress from the history file.

```bash
python scripts/status.py
```

No parameters — reads directly from `analytics_output/training_history.json`. Shows:
- Per-generation policy/value loss
- Self-play results (W/L/D) and average game length
- Time per generation and ETA for completion

---

## API Endpoints

### `POST /api/move` — Get Best Move

Request:
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
  "difficulty": "medium"
}
```

Response:
```json
{
  "move": "e7e5",
  "value": 0.12,
  "confidence": 0.45,
  "think_time_ms": 82.3
}
```

### `POST /api/evaluate` — Position Evaluation

Request:
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
}
```

Response:
```json
{
  "value": 0.12,
  "top_moves": [
    {"move": "e7e5", "visits": 89, "score": 0.445},
    {"move": "c7c5", "visits": 42, "score": 0.21}
  ]
}
```

### `POST /api/analyze` — Full Analysis

Request:
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
  "num_sims": 400
}
```

Response:
```json
{
  "value": 0.12,
  "moves": [
    {"move": "e7e5", "visits": 156, "score": 0.39},
    {"move": "c7c5", "visits": 87, "score": 0.2175}
  ],
  "total_simulations": 400
}
```

### `GET /api/health` — Health Check

Response:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Difficulty Levels

| Level | MCTS Simulations | Temperature | Strength |
|-------|:---:|:---:|----------|
| easy | 50 | 1.0 | Fast, exploratory play |
| medium | 200 | 0.5 | Balanced |
| hard | 400 | 0.1 | Strong, focused |
| max | 800 | 0.05 | Full strength |

---

## Analytics

Training generates several outputs in `analytics_output/`:

| File | Contents |
|------|----------|
| `training_history.json` | Full training log — config snapshot, per-generation losses, self-play stats, timings |
| `benchmark_results.json` | Stockfish benchmark results — per-game records, W/L/D, estimated ELO |
| `loss_curves.png` | Policy and value loss over generations |
| `policy_entropy.png` | Policy entropy over time |
| `self_play_stats.png` | Game length and outcome distributions |
| `dashboard.html` | Interactive Plotly dashboard |
| `live_games.json` | Real-time game state for the live viewer |

---

## Testing

```bash
# Run all 22 tests
pytest tests/ -v

# Run specific test files
pytest tests/test_model.py -v     # 8 tests — network, encoding, move indexing
pytest tests/test_mcts.py -v      # 5 tests — search, move selection, endgames
pytest tests/test_training.py -v  # 4 tests — replay buffer operations
pytest tests/test_api.py -v       # 5 tests — API endpoints and engine
```

---

## Deployment

### Docker

```bash
docker build -t chess-api .
docker run -p 8000:8000 chess-api
```

### Docker Compose

```bash
docker compose up
```

### AWS App Runner

1. Export the model: `python scripts/export.py --name "production"`
2. Copy the ONNX file to the project root
3. Push to a GitHub repo connected to App Runner
4. App Runner auto-builds from the Dockerfile and exposes port 8000

The API is stateless and lightweight — a single instance handles multiple concurrent games since ONNX inference is fast (~1-10ms per position evaluation).

---

## Quick Start (Full Workflow)

```bash
# 1. Install
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cu124  # GPU

# 2. Train (moderate run, ~3.5 hours on RTX 3080)
python scripts/train.py --iterations 30 --games-per-iter 30 --sims 50 --training-steps 500 --parallel-games 8 --watch

# 3. Check progress
python scripts/status.py

# 4. Benchmark against Stockfish
$env:STOCKFISH_PATH = "stockfish\stockfish-windows-x86-64-avx2.exe"
python scripts/benchmark.py --games 20 --depths 1 3 5 --sims 200

# 5. Export model + backup
python scripts/export.py --name "v1_trained"

# 6. Start API server
python scripts/serve.py --model exported_models/<your_export>/chess_model.onnx

# 7. Test it
curl -X POST http://localhost:8000/api/move -H "Content-Type: application/json" -d '{"fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "difficulty": "medium"}'

# 8. Run tests
pytest tests/ -v
```
