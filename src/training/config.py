import os
from dataclasses import dataclass, field
import torch


@dataclass
class TrainingConfig:
    # network
    num_blocks: int = 6
    channels: int = 128

    # MCTS
    num_sims: int = 200
    cpuct: float = 2.0
    dirichlet_alpha: float = 0.3
    dirichlet_weight: float = 0.25

    # self-play
    games_per_iter: int = 500
    parallel_games: int = 0  # 0 = auto-detect from cpu count
    temperature_moves: int = 30  # use temp=1.0 for this many moves, then drop
    temperature_final: float = 0.1
    resign_threshold: float = -0.95
    resign_count: int = 5  # resign if value stays below threshold for this many moves
    gpu_batch_size: int = 16  # batched NN eval during self-play
    use_opening_book: bool = True  # start games from random book openings for diversity

    # training
    training_steps_per_iter: int = 1000
    batch_size: int = 256
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_schedule: str = "cosine"
    warmup_steps: int = 500
    use_amp: bool = True

    # replay buffer
    buffer_capacity: int = 200_000

    # checkpoints
    checkpoint_dir: str = "checkpoints"
    keep_checkpoints: int = 20

    # stockfish benchmarking
    benchmark_games: int = 50
    benchmark_every: int = 10  # every N iterations
    stockfish_path: str = ""

    # analytics
    analytics_dir: str = "analytics_output"

    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.parallel_games <= 0:
            if self.device == "cuda":
                self.parallel_games = max(1, os.cpu_count() - 1)
            else:
                self.parallel_games = 1

        if not self.stockfish_path:
            self.stockfish_path = os.environ.get("STOCKFISH_PATH", "stockfish")

    def print_summary(self):
        print(f"Device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"VRAM: {mem:.1f} GB")
        print(f"CPU cores: {os.cpu_count()}")
        print(f"Parallel self-play games: {self.parallel_games}")
        print(f"Network: {self.num_blocks} blocks x {self.channels} channels")
        print(f"MCTS simulations: {self.num_sims}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Replay buffer: {self.buffer_capacity} positions")
        print(f"Training batch size: {self.batch_size}")
        print()
