"""Supervised pre-training on Lichess PGN games.

Usage:
    python scripts/pretrain.py --pgn path/to/games.pgn.zst
    python scripts/train.py --checkpoint-dir checkpoints_pretrained
"""

import sys
import os
import signal
import argparse
import time
import math
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.network import ChessNet, NUM_PLANES, POLICY_SIZE
from src.training.config import TrainingConfig
from src.data.lichess_dataset import stream_training_positions, batch_positions

_shutting_down = False


def handle_interrupt(sig, frame):
    global _shutting_down
    if _shutting_down:
        print('\nForce quit.')
        sys.exit(1)
    _shutting_down = True
    print('\n\nCtrl+C detected — saving checkpoint and exiting...')


def save_pretrain_checkpoint(model, optimizer, scaler, positions, batches, args,
                              policy_loss, value_loss, checkpoint_dir):
    """Save pre-training progress."""
    path = os.path.join(checkpoint_dir, 'pretrain_latest.pt')
    data = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'positions_trained': positions,
        'batches_trained': batches,
        'avg_policy_loss': policy_loss,
        'avg_value_loss': value_loss,
        'timestamp': datetime.now().isoformat(),
        'args': {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool))},
    }
    if scaler is not None:
        data['scaler_state'] = scaler.state_dict()
    torch.save(data, path)
    print(f"  Saved pretrain checkpoint: {path} ({positions:,} positions)")


def save_training_checkpoint(model, positions, args, checkpoint_dir):
    """Save a checkpoint compatible with the self-play trainer.

    This creates gen_0000.pt so 'python scripts/train.py' continues from here.
    """
    path = os.path.join(checkpoint_dir, 'gen_0000.pt')

    config = TrainingConfig(
        num_blocks=args.num_blocks,
        channels=args.channels,
    )

    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': {},
        'generation': 0,
        'total_training_steps': 0,
        'history': {
            'generations': [],
            'policy_losses': [],
            'value_losses': [],
            'total_losses': [],
            'self_play_stats': [],
            'benchmark_results': [],
            'policy_entropy': [],
            'iterations': [],
        },
        'config': config,
        'pretrained_from': {
            'pgn': os.path.basename(args.pgn),
            'positions': positions,
            'min_elo': args.min_elo,
            'timestamp': datetime.now().isoformat(),
        },
    }, path)
    print(f"  Saved training-compatible checkpoint: {path}")
    print(f"  Continue with: python scripts/train.py --checkpoint-dir {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train the chess model on human games from a Lichess PGN database"
    )
    parser.add_argument("--pgn", type=str, required=True,
                        help="Path to PGN file (.pgn or .pgn.zst)")
    parser.add_argument("--num-blocks", type=int, default=6,
                        help="Number of residual blocks (default: 6)")
    parser.add_argument("--channels", type=int, default=128,
                        help="Number of channels (default: 128)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Peak learning rate (default: 0.001)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Training batch size (default: 512)")
    parser.add_argument("--min-elo", type=int, default=1800,
                        help="Minimum ELO for both players (default: 1800)")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Max number of games to process (default: all)")
    parser.add_argument("--target-positions", type=int, default=5_000_000,
                        help="Target positions for LR schedule (default: 5M)")
    parser.add_argument("--skip-draws", action="store_true",
                        help="Skip drawn games")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_pretrained",
                        help="Where to save checkpoints (default: checkpoints_pretrained)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to pretrain checkpoint to resume from")
    parser.add_argument("--save-every", type=int, default=200,
                        help="Save checkpoint every N batches (default: 200)")
    parser.add_argument("--value-weight", type=float, default=4.0,
                        help="Weight for value loss (default: 4.0)")
    parser.add_argument("--use-amp", action="store_true", default=True,
                        help="Use mixed precision training (default: True)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision")
    args = parser.parse_args()

    if not os.path.exists(args.pgn):
        print(f"Error: PGN file not found: {args.pgn}")
        print()
        print("Download a Lichess database from https://database.lichess.org/")
        print("Example:")
        print("  wget https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst")
        sys.exit(1)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    use_amp = args.use_amp and not args.no_amp
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create model
    model = ChessNet(
        num_blocks=args.num_blocks,
        channels=args.channels,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())

    # Adam for supervised pre-training (converges faster than SGD for this task)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scaler = torch.amp.GradScaler('cuda') if use_amp and device == 'cuda' else None

    # resume from checkpoint
    start_positions = 0
    start_batches = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_positions = ckpt.get('positions_trained', 0)
        start_batches = ckpt.get('batches_trained', 0)
        if scaler and 'scaler_state' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state'])
        print(f"  Resumed at {start_positions:,} positions")

    signal.signal(signal.SIGINT, handle_interrupt)

    # print config
    print(f"\n{'='*60}")
    print(f"Supervised Pre-Training")
    print(f"{'='*60}")
    print(f"Model: {args.num_blocks} blocks x {args.channels} channels ({param_count:,} params)")
    print(f"Input planes: {NUM_PLANES}")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PGN: {args.pgn}")
    print(f"Min ELO: {args.min_elo}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Value loss weight: {args.value_weight}")
    print(f"Mixed precision: {use_amp}")
    print(f"Target positions: {args.target_positions:,}")
    if args.max_games:
        print(f"Max games: {args.max_games:,}")
    print(f"Checkpoint dir: {args.checkpoint_dir}/")
    print(f"{'='*60}\n")

    # training loop
    model.train()
    total_positions = start_positions
    running_policy_loss = 0.0
    running_value_loss = 0.0
    batch_count = start_batches
    t0 = time.time()

    position_stream = stream_training_positions(
        args.pgn,
        min_elo=args.min_elo,
        max_games=args.max_games,
        skip_draws=args.skip_draws,
    )

    for boards, policies, values in batch_positions(position_stream, args.batch_size):
        if _shutting_down:
            break

        boards_t = torch.tensor(boards).to(device)
        policies_t = torch.tensor(policies).to(device)
        values_t = torch.tensor(values).to(device)

        # cosine LR decay with warmup
        warmup_steps = 500
        if batch_count < warmup_steps:
            lr = args.lr * (batch_count + 1) / warmup_steps
        else:
            progress = min(total_positions / max(args.target_positions, 1), 1.0)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                policy_logits, value_pred = model(boards_t)
                policy_loss = -(policies_t * torch.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
                value_loss = nn.functional.mse_loss(value_pred, values_t)
                loss = policy_loss + args.value_weight * value_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            policy_logits, value_pred = model(boards_t)
            policy_loss = -(policies_t * torch.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
            value_loss = nn.functional.mse_loss(value_pred, values_t)
            loss = policy_loss + args.value_weight * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        running_policy_loss += policy_loss.item()
        running_value_loss += value_loss.item()
        batch_count += 1
        total_positions += len(boards)

        # log progress
        if batch_count % 50 == 0:
            elapsed = time.time() - t0
            avg_p = running_policy_loss / batch_count
            avg_v = running_value_loss / batch_count
            pos_sec = total_positions / max(elapsed, 1)
            print(f"  [{total_positions:>10,} pos] "
                  f"policy={avg_p:.4f}  value={avg_v:.4f}  "
                  f"lr={lr:.6f}  speed={pos_sec:.0f} pos/s")

        # save checkpoint periodically
        if batch_count % args.save_every == 0:
            avg_p = running_policy_loss / batch_count
            avg_v = running_value_loss / batch_count
            save_pretrain_checkpoint(
                model, optimizer, scaler, total_positions, batch_count, args,
                avg_p, avg_v, args.checkpoint_dir,
            )

    # final save
    avg_p = running_policy_loss / max(batch_count, 1)
    avg_v = running_value_loss / max(batch_count, 1)
    save_pretrain_checkpoint(
        model, optimizer, scaler, total_positions, batch_count, args,
        avg_p, avg_v, args.checkpoint_dir,
    )

    # save as training-compatible checkpoint for self-play continuation
    save_training_checkpoint(model, total_positions, args, args.checkpoint_dir)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Pre-training complete!")
    print(f"{'='*60}")
    print(f"Trained on {total_positions:,} positions in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Final avg policy loss: {avg_p:.4f}")
    print(f"Final avg value loss: {avg_v:.4f}")
    print(f"\nNext steps:")
    print(f"  1. Run self-play training to fine-tune:")
    print(f"     python scripts/train.py --checkpoint-dir {args.checkpoint_dir}")
    print(f"  2. Or export directly for testing:")
    print(f"     python scripts/export.py --checkpoint-dir {args.checkpoint_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
