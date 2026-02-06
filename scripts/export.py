import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.export import export_to_onnx


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="chess_model.onnx")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpts = sorted([
            f for f in os.listdir(args.checkpoint_dir)
            if f.startswith('gen_') and f.endswith('.pt')
        ])
        if not ckpts:
            print("No checkpoints found. Train the model first.")
            sys.exit(1)
        ckpt_path = os.path.join(args.checkpoint_dir, ckpts[-1])

    export_to_onnx(ckpt_path, args.output, verify=not args.no_verify)


if __name__ == "__main__":
    main()
