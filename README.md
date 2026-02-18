# AI Chess Model

Chess engine that learns to play using a neural network and Monte Carlo Tree Search, based on the AlphaZero approach. Trained in two stages: first on millions of Lichess games to learn basic chess, then through self-play reinforcement learning to get stronger on its own. Runs as a lightweight API with an ONNX model so you don't need a GPU to play against it.

## What It Does

- Learns chess from human games (supervised pre-training on Lichess databases)
- Improves through self-play (reinforcement learning loop)
- Uses 8 positions of history so it can see piece trajectories and repetitions
- 59 opening book entries to keep training games diverse
- Exports to ONNX for fast CPU inference
- REST API with four difficulty levels
- Can be packaged into a standalone folder anyone can run and play in their browser
- Benchmarks against Stockfish to estimate ELO

---

## Project Structure

```
AI-Chess-Model/
├── src/
│   ├── model/
│   │   ├── network.py         # ResNet architecture + board encoding
│   │   └── mcts.py            # Monte Carlo Tree Search
│   ├── training/
│   │   ├── config.py           # Hyperparameters
│   │   ├── self_play.py        # Self-play game generation
│   │   ├── trainer.py          # Training loop + checkpoints
│   │   └── replay_buffer.py    # Circular position buffer
│   ├── data/
│   │   ├── openings.py         # 59-opening book
│   │   └── lichess_dataset.py  # PGN parser for pre-training
│   ├── benchmark/
│   │   └── stockfish_bench.py  # Stockfish matchups + ELO estimation
│   ├── analytics/
│   │   └── plots.py            # Training charts
│   ├── api/
│   │   ├── main.py             # FastAPI server
│   │   ├── engine.py           # ONNX inference wrapper
│   │   └── schemas.py          # Request/response models
│   └── export.py               # PyTorch → ONNX with BatchNorm folding
├── scripts/
│   ├── pretrain.py             # Supervised pre-training
│   ├── train.py                # Self-play training
│   ├── benchmark.py            # Run Stockfish benchmarks
│   ├── export.py               # Export + backup snapshot
│   ├── serve.py                # Start API server
│   └── package.py              # Bundle into shareable package
├── tests/                      # 22 tests (model, MCTS, training, API)
├── exported_models/            # Versioned model snapshots (tracked in git)
├── pyproject.toml
├── AI_USE_STATEMENT.md
└── README.md
```

---

## Setup

**Requirements:** Python 3.10+, CUDA GPU recommended (CPU works but training is slow)

```bash
python -m venv .venv

# Windows:
.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

pip install -e ".[dev]"

# for .pgn.zst files (Lichess databases):
pip install zstandard
```

If you have an NVIDIA GPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Stockfish (optional, for benchmarking)

Stockfish isn't included in the repo (it's ~30 MB). Download it from https://stockfishchess.org/download/ and set the path:

```bash
# Windows:
$env:STOCKFISH_PATH = "stockfish\stockfish-windows-x86-64-avx2.exe"
# Linux/Mac:
export STOCKFISH_PATH=/path/to/stockfish
```

---

## Architecture

### Network

Residual CNN with 6 blocks, 128 channels (~2.4M parameters). Takes in 102 input planes (8 timesteps of 12 piece planes + 6 meta planes for castling/en passant/turn) and outputs a 4672-dim policy vector + scalar value in [-1, 1].

### Board Encoding

| Planes | What |
|--------|------|
| 0–11 | Current position (6 piece types × 2 colors) |
| 12–23 | 1 move ago |
| 24–35 | 2 moves ago |
| ... | ... |
| 84–95 | 7 moves ago |
| 96–99 | Castling rights |
| 100 | En passant square |
| 101 | Side to move |

Board is always flipped so the current player sees from their own perspective.

### MCTS

PUCT selection (cpuct=2.0), Dirichlet noise at root for exploration (α=0.3, ε=0.25), temperature scheduling (τ=1.0 for first 60 moves, then τ=0.3). During self-play, parallel games batch their positions together for GPU efficiency.

### Training Tricks

| Param | Value | Why |
|-------|-------|-----|
| `value_loss_weight` | 4.0 | Prevents value head from collapsing |
| `draw_penalty` | 0.1 | Draws score -0.1 for both sides to discourage draw-seeking |
| `temperature_moves` | 60 | Keeps exploration high longer |
| `resign_threshold` | -0.80 | Cuts off lost games early to save data |

---

## Training

### Stage 1: Pre-Training on Lichess

Download a database from https://database.lichess.org/ — these are multi-GB compressed archives of rated games, not included in this repo.

```bash
pip install zstandard
python scripts/pretrain.py --pgn lichess_db_standard_rated_2024-01.pgn.zst --min-elo 1800
```

This streams positions from the PGN, trains the network to predict human moves and game outcomes, and saves a `gen_0000.pt` checkpoint.

**Key flags:**

| Flag | Default | What |
|------|---------|------|
| `--pgn` | required | Path to .pgn or .pgn.zst |
| `--min-elo` | 1800 | Skip games below this ELO |
| `--batch-size` | 512 | Training batch size |
| `--lr` | 0.001 | Learning rate (Adam) |
| `--target-positions` | 5M | For cosine LR schedule |
| `--max-games` | all | Cap number of games |
| `--checkpoint-dir` | checkpoints_pretrained | Output dir |
| `--resume` | — | Resume from checkpoint |

### Stage 2: Self-Play

```bash
python scripts/train.py --checkpoint-dir checkpoints_pretrained --iterations 50 --games-per-iter 50 --sims 200
```

Each iteration: play self-play games → add positions to buffer → train on buffer → save checkpoint.

**Key flags:**

| Flag | Default | What |
|------|---------|------|
| `--iterations` | 100 | Number of generations |
| `--games-per-iter` | 500 | Games per generation |
| `--sims` | 200 | MCTS sims per move |
| `--training-steps` | 1000 | Gradient steps per gen |
| `--batch-size` | 256 | Training batch |
| `--lr` | 0.01 | Learning rate (SGD) |
| `--benchmark-every` | 10 | Stockfish benchmark interval |
| `--no-opening-book` | — | Start from initial position |

Auto-resumes from latest checkpoint. Ctrl+C finishes current games then saves.

---

## Other Scripts

### Benchmark
```bash
python scripts/benchmark.py --games 20 --depths 1 3 5 --sims 200
```
Plays the model vs Stockfish at different depths and estimates ELO.

### Export
```bash
python scripts/export.py --name "v1_pretrained"
```
Exports to ONNX and creates a versioned snapshot in `exported_models/` with the model, checkpoint, analytics, and metadata.

### Serve
```bash
python scripts/serve.py --model chess_model.onnx --port 8000
```
Starts the FastAPI inference server.

### Package
```bash
python scripts/package.py exported_models/gen0048_v3_pretrained_gen48
```
Bundles the model into a standalone folder with all dependencies, a run script, and a browser UI. Zip it and send to anyone with Python 3.10+ — they just run `python run.py`.

---

## Playing Against It

```bash
python scripts/package.py exported_models/<your_export>
python packages/<generated-folder>/run.py
```

Opens a browser UI where you can:
- Play as white or black
- Pick difficulty (easy / medium / hard / max)
- See the model's evaluation bar
- Undo moves

The package is self-contained — all dependencies bundled in `lib/`.

---

## API

### `POST /api/move`
```json
// request
{"fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "difficulty": "medium"}
// response
{"move": "e7e5", "value": 0.12, "confidence": 0.45, "think_time_ms": 82.3}
```

### `POST /api/evaluate`
```json
// request
{"fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"}
// response
{"value": 0.12, "top_moves": [{"move": "e7e5", "visits": 89, "score": 0.445}]}
```

### `POST /api/analyze`
```json
// request
{"fen": "...", "num_sims": 400}
// response
{"value": 0.12, "moves": [...], "total_simulations": 400}
```

### `GET /api/health`
```json
{"status": "ok", "model_loaded": true}
```

### Difficulty Levels

| Level | Sims | Temp | |
|-------|:----:|:----:|--|
| easy | 50 | 1.0 | Fast, random-ish |
| medium | 200 | 0.5 | Balanced |
| hard | 400 | 0.1 | Strong |
| max | 800 | 0.05 | Full strength |

---

## Testing

```bash
pytest tests/ -v
```

22 tests covering the network, encoding, MCTS, replay buffer, and API endpoints.

---

## Quick Start

```bash
# install
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install zstandard

# pre-train (download a Lichess DB first from database.lichess.org)
python scripts/pretrain.py --pgn lichess_db_standard_rated_2024-01.pgn.zst --min-elo 1800

# self-play
python scripts/train.py --checkpoint-dir checkpoints_pretrained --iterations 50 --games-per-iter 50 --sims 200

# benchmark
python scripts/benchmark.py --games 20 --depths 1 3 5 --sims 200

# export
python scripts/export.py --name "v1" --checkpoint-dir checkpoints_pretrained

# package and play
python scripts/package.py exported_models/<your_export>
python packages/<generated-folder>/run.py

# tests
pytest tests/ -v
```
