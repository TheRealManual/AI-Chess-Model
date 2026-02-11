import sys
import os
import shutil
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.export import export_to_onnx

EXPORTED_DIR = "exported_models"


def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent gen_XXXX.pt checkpoint."""
    ckpts = sorted([
        f for f in os.listdir(checkpoint_dir)
        if f.startswith('gen_') and f.endswith('.pt')
    ])
    if not ckpts:
        return None, 0
    latest = ckpts[-1]
    # extract generation number from filename like gen_0016.pt
    gen = int(latest.replace('gen_', '').replace('.pt', ''))
    return os.path.join(checkpoint_dir, latest), gen


def create_export_folder(name, generation):
    """Create a uniquely named folder inside exported_models/.

    Naming scheme: gen{N}_{name}_{timestamp}
    e.g.  gen16_opening_book_20260207_143022
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = name.replace(' ', '_').replace('/', '_')
    folder_name = f"gen{generation:04d}_{safe_name}_{timestamp}"
    folder_path = os.path.join(EXPORTED_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def copy_directory(src, dst_parent, dir_name):
    """Copy a directory into dst_parent/dir_name. Skips if src doesn't exist."""
    if not os.path.isdir(src):
        print(f"  Skipping {src} (not found)")
        return
    dst = os.path.join(dst_parent, dir_name)
    shutil.copytree(src, dst)
    count = sum(len(files) for _, _, files in os.walk(dst))
    print(f"  Copied {src}/ -> {dir_name}/ ({count} files)")


def copy_file(src, dst_dir, filename=None):
    """Copy a single file into dst_dir. Skips if src doesn't exist."""
    if not os.path.isfile(src):
        return
    fname = filename or os.path.basename(src)
    shutil.copy2(src, os.path.join(dst_dir, fname))


def main():
    parser = argparse.ArgumentParser(
        description="Export model to ONNX and create a full backup snapshot"
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint. Uses latest if not provided.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--analytics-dir", type=str, default="analytics_output")
    parser.add_argument("--name", type=str, default="export",
                        help="A short label for this export (e.g. 'opening_book', 'baseline')")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    # ── find checkpoint ──────────────────────────────────────────────
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path, gen = find_latest_checkpoint(args.checkpoint_dir)
        if ckpt_path is None:
            print("No checkpoints found. Train the model first.")
            sys.exit(1)
    else:
        # try to parse generation from filename
        base = os.path.basename(ckpt_path)
        try:
            gen = int(base.replace('gen_', '').replace('.pt', ''))
        except ValueError:
            gen = 0

    print(f"Checkpoint: {ckpt_path} (generation {gen})")

    # ── create export folder ─────────────────────────────────────────
    export_folder = create_export_folder(args.name, gen)
    print(f"Export folder: {export_folder}/\n")

    # ── export ONNX model ────────────────────────────────────────────
    onnx_path = os.path.join(export_folder, "chess_model.onnx")
    export_to_onnx(ckpt_path, onnx_path, verify=not args.no_verify)
    print()

    # ── copy checkpoints ─────────────────────────────────────────────
    print("Backing up checkpoints...")
    copy_directory(args.checkpoint_dir, export_folder, "checkpoints")

    # ── copy analytics ───────────────────────────────────────────────
    print("Backing up analytics...")
    copy_directory(args.analytics_dir, export_folder, "analytics_output")

    # ── copy training history from root if it exists ─────────────────
    copy_file("training_history.json", export_folder)

    # ── write export metadata ────────────────────────────────────────
    metadata = {
        "export_time": datetime.now().isoformat(),
        "name": args.name,
        "generation": gen,
        "checkpoint": ckpt_path,
        "onnx_file": "chess_model.onnx",
    }

    # pull some stats from training history if available
    history_path = os.path.join(args.analytics_dir, "training_history.json")
    if os.path.isfile(history_path):
        try:
            with open(history_path) as f:
                history = json.load(f)
            iters = history.get("iterations", [])
            if iters:
                last = iters[-1]
                metadata["final_policy_loss"] = last.get("policy_loss")
                metadata["final_value_loss"] = last.get("value_loss")
                metadata["total_iterations"] = len(iters)
                metadata["total_positions"] = last.get("buffer_size")
        except Exception:
            pass

    # pull benchmark results if available
    bench_path = os.path.join(args.analytics_dir, "benchmark_results.json")
    if os.path.isfile(bench_path):
        try:
            with open(bench_path) as f:
                metadata["benchmark"] = json.load(f)
        except Exception:
            pass

    meta_path = os.path.join(export_folder, "export_info.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata written to export_info.json")

    # ── summary ──────────────────────────────────────────────────────
    total_size = 0
    for root, dirs, files in os.walk(export_folder):
        for fname in files:
            total_size += os.path.getsize(os.path.join(root, fname))
    size_mb = total_size / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"Export complete: {export_folder}/")
    print(f"Total size: {size_mb:.1f} MB")
    print(f"Contents:")
    for item in sorted(os.listdir(export_folder)):
        item_path = os.path.join(export_folder, item)
        if os.path.isdir(item_path):
            count = sum(len(files) for _, _, files in os.walk(item_path))
            print(f"  {item}/ ({count} files)")
        else:
            fsize = os.path.getsize(item_path) / (1024 * 1024)
            print(f"  {item} ({fsize:.1f} MB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
