# AI Chess Model

A chess engine built with a residual neural network and Monte Carlo Tree Search (MCTS), inspired by DeepMind's AlphaZero. The model uses supervised pre-training on strong human games from Lichess, followed by self-play reinforcement learning for fine-tuning. It exports to ONNX format and runs as a lightweight FastAPI microservice, designed for integration with web backends (e.g. a Node.js chess app on AWS App Runner).

## Overview

- **Two-stage training pipeline** — supervised pre-training on Lichess games, then self-play fine-tuning
- **History-aware input** — 102-plane board encoding with 8 timesteps of position history
- **Batched GPU evaluation** — parallel self-play games share GPU batches for faster training
- **Opening book** — 59 common openings ensure diverse training positions and prevent draw collapse
- **MCTS + neural network** — combines deep search with learned position evaluation
- **ONNX inference** — fast, lightweight inference without PyTorch dependency
- **REST API** — FastAPI server with configurable difficulty levels for concurrent games
- **Shareable packages** — bundle any exported model into a standalone folder anyone can run and play
- **Stockfish benchmarking** — automated ELO estimation against Stockfish at multiple depths
- **Export snapshots** — full backups of models, checkpoints, and analytics for version tracking

---

## Project Structure

```
AI-Chess-Model/
├── src/
│   ├── model/
│   │   ├── network.py        # ResNet (6 blocks x 128ch, ~2.4M params), 102-plane board encoding
│   │   └── mcts.py           # Monte Carlo Tree Search with PUCT selection
│   ├── training/
│   │   ├── config.py          # All hyperparameters in one dataclass
│   │   ├── self_play.py       # Self-play game generation with batched GPU eval
│   │   ├── trainer.py         # Training loop, checkpoints, history tracking
│   │   └── replay_buffer.py   # Circular buffer for training positions
│   ├── data/
│   │   ├── openings.py        # 59-opening book (Italian, Sicilian, French, QGD, KID, etc.)
│   │   └── lichess_dataset.py # PGN streaming parser for supervised pre-training
│   ├── benchmark/
│   │   └── stockfish_bench.py # Play model vs Stockfish, estimate ELO
│   ├── analytics/
│   │   └── plots.py           # Loss curves, self-play stats chart generation
│   ├── api/
│   │   ├── main.py            # FastAPI app with CORS
│   │   ├── engine.py          # ONNX Runtime inference + MCTS wrapper
│   │   └── schemas.py         # Pydantic request/response models
│   └── export.py              # PyTorch → ONNX conversion with BatchNorm folding
├── scripts/
│   ├── pretrain.py            # Supervised pre-training on Lichess PGN databases
│   ├── train.py               # Self-play training entry point
│   ├── benchmark.py           # Standalone Stockfish benchmarking
│   ├── export.py              # Export model + create backup snapshot
│   ├── serve.py               # Start the API server
│   ├── package.py             # Bundle exported model into shareable package
│   ├── analyze.py             # Deep training analytics and diagnostics
│   ├── watch.py               # Live ASCII board viewer for self-play
│   └── status.py              # Check training progress from terminal
├── tests/
│   ├── test_model.py          # Network, encoding, move indexing tests (8 tests)
│   ├── test_mcts.py           # MCTS search and move selection tests (5 tests)
│   ├── test_training.py       # Replay buffer tests (4 tests)
│   └── test_api.py            # API endpoint tests (5 tests)
├── exported_models/            # Versioned model snapshots (committed to git)
├── Dockerfile                  # Production container for the API
├── docker-compose.yml          # Docker Compose for local deployment
├── pyproject.toml              # Dependencies and project metadata
├── AI_USE_STATEMENT.md         # AI assistance disclosure
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

# For pre-training on .pgn.zst files (Lichess databases)
pip install zstandard
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
| Input | 102 planes × 8×8 (see Board Encoding below) |
| Policy head | 4672-dim output (AlphaZero move encoding: 8×8×73) |
| Value head | Single scalar (-1 to +1) |
| ONNX size | ~3–9 MB |

### Board Encoding (102 Input Planes)

The network sees 8 timesteps of board history, giving it awareness of piece trajectories, repetitions, and tempo:

| Planes | Content |
|--------|---------|
| 0–11 | Current position: 6 piece types × 2 colors |
| 12–23 | Position 1 move ago |
| 24–35 | Position 2 moves ago |
| ... | ... |
| 84–95 | Position 7 moves ago |
| 96 | White kingside castling rights |
| 97 | White queenside castling rights |
| 98 | Black kingside castling rights |
| 99 | Black queenside castling rights |
| 100 | En passant square |
| 101 | Side to move (all 1s = white, all 0s = black) |

The board is always flipped so the current player sees from their own perspective (row 0 = their back rank).

### MCTS

- **PUCT selection** with cpuct=2.0
- **Dirichlet noise** (α=0.3, ε=0.25) at root for exploration
- **Temperature scheduling** — τ=1.0 for the first 60 moves, then τ=0.3
- **Batched GPU evaluation** — parallel self-play games pool positions into GPU batches (batch size 16) for efficient inference
- Configurable simulation count per difficulty level

### Training Stabilization

Several parameters prevent common training pathologies:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `value_loss_weight` | 4.0 | Upweights value head to prevent collapse |
| `draw_penalty` | 0.1 | Draws score as -0.1 for both sides, discouraging draw-seeking |
| `temperature_moves` | 60 | Keeps exploration high for more of each game |
| `temperature_final` | 0.3 | Maintains some exploration even in endgames |
| `resign_threshold` | -0.80 | Resigns lost positions to avoid wasting data |

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

## Training Pipeline

The training process has two stages:

### Stage 1: Supervised Pre-Training

Bootstrap the neural network with chess knowledge from strong human games. This is dramatically more effective than training from random weights.

1. Download a Lichess database from https://database.lichess.org/ (`.pgn.zst` format)
2. Run the pre-training script:

```bash
pip install zstandard
python scripts/pretrain.py --pgn lichess_db_standard_rated_2024-01.pgn.zst --min-elo 1800
```

This streams positions from the PGN file, filters by ELO, and trains the network to predict both human moves (policy) and game outcomes (value). It saves a `gen_0000.pt` checkpoint compatible with the self-play trainer.

### Stage 2: Self-Play Fine-Tuning

Once pre-trained, the model improves further through self-play:

```bash
python scripts/train.py --checkpoint-dir checkpoints_pretrained --iterations 50 --games-per-iter 50 --sims 200
```

---

## Scripts

### `scripts/pretrain.py` — Supervised Pre-Training

Pre-trains the neural network on human games from a Lichess PGN database. Streams positions directly from compressed `.pgn.zst` files without loading the entire dataset into memory.

```bash
python scripts/pretrain.py --pgn lichess_db_standard_rated_2024-01.pgn.zst --min-elo 1800
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pgn` | (required) | Path to PGN file (`.pgn` or `.pgn.zst`) |
| `--min-elo` | 1800 | Minimum ELO for both players (lower-rated games are skipped) |
| `--batch-size` | 512 | Training batch size |
| `--lr` | 0.001 | Peak learning rate (Adam optimizer) |
| `--target-positions` | 5000000 | Target positions for cosine LR schedule |
| `--max-games` | all | Max number of games to process |
| `--skip-draws` | — | Skip drawn games |
| `--value-weight` | 4.0 | Weight for value loss component |
| `--num-blocks` | 6 | Residual blocks in network |
| `--channels` | 128 | Channels per layer |
| `--checkpoint-dir` | checkpoints_pretrained | Where to save checkpoints |
| `--resume` | — | Path to pretrain checkpoint to resume from |
| `--save-every` | 200 | Save checkpoint every N batches |
| `--no-amp` | — | Disable mixed precision training |

**Features:**
- Adam optimizer with cosine LR schedule and linear warmup (500 steps)
- Gradient clipping (max norm 1.0) for stability
- Mixed precision (AMP) for fast GPU training
- Resume support — pick up where you left off with `--resume`
- Graceful Ctrl+C — saves checkpoint before exiting
- Saves `gen_0000.pt` compatible with the self-play trainer

---

### `scripts/train.py` — Self-Play Training

Runs the full self-play training loop: generate games → train network → save checkpoint → repeat.

```bash
python scripts/train.py --iterations 50 --games-per-iter 50 --sims 200 --training-steps 500 --watch
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
| `--buffer-size` | 100000 | Replay buffer capacity (positions) |
| `--benchmark-every` | 10 | Run Stockfish benchmark every N generations (0 to disable) |
| `--benchmark-games` | 50 | Games per benchmark run |
| `--no-amp` | — | Disable mixed precision training |
| `--no-opening-book` | — | Disable random opening book (start all games from initial position) |
| `--watch` | — | Open a live ASCII board viewer in a new terminal window |
| `--checkpoint-dir` | checkpoints | Directory for saving checkpoints |
| `--analytics-dir` | analytics_output | Directory for plots and history |

**Features:**
- Auto-resumes from the latest checkpoint (handles architecture mismatches gracefully)
- Batched GPU evaluation across parallel self-play games
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
python scripts/export.py --name "v1_pretrained"
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
  gen0030_v1_pretrained_20260207_150000/
    chess_model.onnx        # ONNX model ready for deployment
    export_info.json        # Metadata (generation, losses, benchmark results)
    checkpoints/            # All .pt checkpoint files + replay buffer
    analytics_output/       # Plots, dashboard, training history, benchmark data
```

The `exported_models/` directory is tracked by git, so each export is a permanent snapshot you can compare or roll back to.

---

### `scripts/serve.py` — Start the API Server

Launches the FastAPI inference server using an exported ONNX model (API only, no UI).

```bash
python scripts/serve.py --model chess_model.onnx --port 8000
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | chess_model.onnx | Path to the ONNX model file |
| `--host` | 0.0.0.0 | Bind address |
| `--port` | 8000 | Port number |
| `--workers` | 1 | Number of Uvicorn workers |

---

### `scripts/package.py` — Bundle a Shareable Package

Creates a standalone folder with everything needed to play against the model: ONNX model, inference server, browser UI, and a one-command run script.

```bash
python scripts/package.py exported_models/gen0048_v3_pretrained_gen48
```

The output lands in `packages/<export-name>_<timestamp>/` and contains:

| File | Purpose |
|------|---------|
| `run.py` | Starts the server and opens the browser |
| `chess_model.onnx` | The exported model |
| `index.html` | Browser-based chess UI |
| `src/` | Inference code (API + MCTS + encoding) |
| `requirements.txt` | Python dependencies |
| `README.txt` | Quick-start instructions |

**Sharing:** zip the package folder and send it to anyone. They just need Python 3.10+ installed:

```bash
pip install -r requirements.txt
python run.py
```

Or with Docker:

```bash
docker build -t chess-api .
docker run -p 8000:8000 chess-api
```

---

### `scripts/analyze.py` — Deep Training Analytics

Reads `analytics_output/training_history.json` and prints detailed per-generation diagnostics: policy/value losses, win/loss/draw rates, decisive game percentages, game lengths, and buffer utilization.

```bash
python scripts/analyze.py
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

## Play Against the AI

Bundle any exported model into a shareable package and play it in your browser:

```bash
python scripts/package.py exported_models/gen0048_v3_pretrained_gen48
python packages/<generated-folder>/run.py
```

This opens a browser-based chess UI with:
- Play as white or black
- Four difficulty levels (easy / medium / hard / max)
- Real-time evaluation bar showing the model's assessment
- Move history panel
- Undo support
- New game button

The package is fully standalone — zip it and share with anyone who has Python 3.10+.

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
| `loss_curves.png` | Policy and value loss over generations (with interpretation guide) |
| `self_play_stats.png` | Game outcome ratios and average game length (with interpretation guide) |
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
pip install zstandard  # for .pgn.zst support

# 2. Download a Lichess database for pre-training
#    Browse https://database.lichess.org/ and pick a monthly database
#    Example (January 2024, ~25 GB compressed):
#    wget https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst

# 3. Pre-train on human games
python scripts/pretrain.py --pgn lichess_db_standard_rated_2024-01.pgn.zst --min-elo 1800

# 4. Fine-tune with self-play
python scripts/train.py --checkpoint-dir checkpoints_pretrained --iterations 50 --games-per-iter 50 --sims 200 --watch

# 5. Check progress
python scripts/status.py

# 6. Benchmark against Stockfish
$env:STOCKFISH_PATH = "stockfish\stockfish-windows-x86-64-avx2.exe"
python scripts/benchmark.py --games 20 --depths 1 3 5 --sims 200

# 7. Export model + backup
python scripts/export.py --name "v1_pretrained" --checkpoint-dir checkpoints_pretrained

# 8. Package and play against the model
python scripts/package.py exported_models/<your_export>
python packages/<generated-folder>/run.py

# 9. Run tests
pytest tests/ -v
```
