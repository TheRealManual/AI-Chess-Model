import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.network import ChessNet
from src.training.config import TrainingConfig
from src.benchmark.stockfish_bench import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Benchmark model against Stockfish")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint. Uses latest if not provided.")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    config = TrainingConfig(
        benchmark_games=args.games,
        checkpoint_dir=args.checkpoint_dir,
    )
    device = config.device

    # find checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpts = sorted([
            f for f in os.listdir(config.checkpoint_dir)
            if f.startswith('gen_') and f.endswith('.pt')
        ])
        if not ckpts:
            print("No checkpoints found. Train the model first.")
            sys.exit(1)
        ckpt_path = os.path.join(config.checkpoint_dir, ckpts[-1])

    print(f"Loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    saved_config = ckpt.get('config')
    if saved_config:
        model = ChessNet(num_blocks=saved_config.num_blocks, channels=saved_config.channels)
    else:
        model = ChessNet()

    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    generation = ckpt.get('generation', 0)
    config.num_sims = args.sims

    run_benchmark(model, device, config, generation=generation,
                  sf_depths=args.depths, num_games=args.games,
                  num_sims=args.sims, checkpoint_path=ckpt_path)


if __name__ == "__main__":
    main()
