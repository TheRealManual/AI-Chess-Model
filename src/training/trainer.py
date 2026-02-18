import os
import time
import json
import math
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from tqdm import tqdm

from src.model.network import ChessNet
from src.training.config import TrainingConfig
from src.training.replay_buffer import ReplayBuffer
from src.training.self_play import run_self_play, set_live_broadcast_path


class Trainer:
    """Handles the training loop, loss computation, and checkpoint management."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device

        self.model = ChessNet(
            num_blocks=config.num_blocks,
            channels=config.channels,
        ).to(self.device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

        self.scaler = torch.amp.GradScaler('cuda') if config.use_amp and self.device == 'cuda' else None
        self.buffer = ReplayBuffer(capacity=config.buffer_capacity)

        self.generation = 0
        self.total_training_steps = 0
        self.history = {
            'generations': [],
            'policy_losses': [],
            'value_losses': [],
            'total_losses': [],
            'self_play_stats': [],
            'benchmark_results': [],
            'policy_entropy': [],
            'iterations': [],
        }
        self.session_start = datetime.now().isoformat()

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.analytics_dir, exist_ok=True)

        # enable live game broadcasting
        live_path = os.path.join(config.analytics_dir, 'live_games.json')
        set_live_broadcast_path(live_path)

    def save_checkpoint(self):
        path = os.path.join(self.config.checkpoint_dir, f'gen_{self.generation:04d}.pt')
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'generation': self.generation,
            'total_training_steps': self.total_training_steps,
            'history': self.history,
            'config': self.config,
        }, path)

        # save buffer separately (it's large)
        buf_dir = os.path.join(self.config.checkpoint_dir, 'buffer')
        self.buffer.save_data(buf_dir)

        # clean up old checkpoints
        self._cleanup_checkpoints()

        print(f"  Checkpoint saved: gen {self.generation}")

    def _cleanup_checkpoints(self):
        ckpts = sorted([
            f for f in os.listdir(self.config.checkpoint_dir)
            if f.startswith('gen_') and f.endswith('.pt')
        ])
        while len(ckpts) > self.config.keep_checkpoints:
            old = os.path.join(self.config.checkpoint_dir, ckpts.pop(0))
            os.remove(old)

    def load_latest_checkpoint(self):
        """Find and load the most recent checkpoint. Returns True if found."""
        if not os.path.exists(self.config.checkpoint_dir):
            return False

        ckpts = sorted([
            f for f in os.listdir(self.config.checkpoint_dir)
            if f.startswith('gen_') and f.endswith('.pt')
        ])
        if not ckpts:
            return False

        path = os.path.join(self.config.checkpoint_dir, ckpts[-1])
        print(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        try:
            self.model.load_state_dict(ckpt['model_state'])
        except RuntimeError as e:
            if 'size mismatch' in str(e):
                print(f"  Architecture mismatch — checkpoint uses different input size.")
                print(f"  Starting with fresh weights. Pre-train first for best results.")
                return False
            raise

        # Load optimizer state if available (may be empty for pre-trained checkpoints)
        opt_state = ckpt.get('optimizer_state', {})
        if opt_state and 'param_groups' in opt_state:
            self.optimizer.load_state_dict(opt_state)
        else:
            print(f"  No optimizer state found — using fresh optimizer (normal for pre-trained models)")
        self.generation = ckpt['generation']
        self.total_training_steps = ckpt['total_training_steps']
        self.history = ckpt['history']

        # load replay buffer
        buf_dir = os.path.join(self.config.checkpoint_dir, 'buffer')
        self.buffer.load_data(buf_dir)

        print(f"  Resumed at generation {self.generation}, "
              f"buffer has {len(self.buffer)} positions, "
              f"{self.total_training_steps} total training steps")
        return True

    def train_on_buffer(self):
        """Run training steps on randomly sampled positions from the replay buffer."""
        if len(self.buffer) < self.config.batch_size:
            print("  Buffer too small to train, skipping")
            return {}

        self.model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_steps = self.config.training_steps_per_iter

        pbar = tqdm(range(num_steps), desc='  Training', unit='step',
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] loss={postfix}')

        for step in pbar:
            boards, policies, values = self.buffer.sample(self.config.batch_size)

            boards_t = torch.tensor(boards).to(self.device)
            policies_t = torch.tensor(policies).to(self.device)
            values_t = torch.tensor(values).to(self.device)

            # learning rate warmup
            lr = self._get_lr()
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr

            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    policy_logits, value_pred = self.model(boards_t)
                    policy_loss = self._policy_loss(policy_logits, policies_t)
                    value_loss = nn.functional.mse_loss(value_pred, values_t)
                    loss = policy_loss + self.config.value_loss_weight * value_loss

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                policy_logits, value_pred = self.model(boards_t)
                policy_loss = self._policy_loss(policy_logits, policies_t)
                value_loss = nn.functional.mse_loss(value_pred, values_t)
                loss = policy_loss + self.config.value_loss_weight * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            self.total_training_steps += 1

            if step % 10 == 0:
                pbar.set_postfix_str(f'{(total_policy_loss + total_value_loss) / (step + 1):.4f}')

        pbar.close()

        avg_policy = total_policy_loss / num_steps
        avg_value = total_value_loss / num_steps
        avg_total = avg_policy + avg_value

        # estimate policy entropy from last batch
        with torch.no_grad():
            probs = torch.softmax(policy_logits, dim=-1)
            log_probs = torch.log(probs + 1e-8)
            entropy = -(probs * log_probs).sum(dim=-1).mean().item()
            if math.isnan(entropy) or math.isinf(entropy):
                entropy = 0.0

        stats = {
            'policy_loss': avg_policy,
            'value_loss': avg_value,
            'total_loss': avg_total,
            'policy_entropy': entropy,
            'learning_rate': lr,
        }
        print(f"  Training: policy_loss={avg_policy:.4f}, value_loss={avg_value:.4f}, "
              f"entropy={entropy:.2f}, lr={lr:.6f}")
        return stats

    def _policy_loss(self, logits, targets):
        """Cross-entropy between MCTS visit distribution and predicted policy."""
        log_probs = torch.log_softmax(logits, dim=-1)
        return -(targets * log_probs).sum(dim=-1).mean()

    def _get_lr(self):
        """Learning rate with linear warmup then cosine decay."""
        if self.total_training_steps < self.config.warmup_steps:
            return self.config.lr * (self.total_training_steps + 1) / self.config.warmup_steps

        if self.config.lr_schedule == "cosine":
            # cosine decay over expected total steps
            t = self.total_training_steps - self.config.warmup_steps
            total = max(1, 100 * self.config.training_steps_per_iter)  # rough estimate
            return self.config.lr * 0.5 * (1 + math.cos(math.pi * min(t, total) / total))

        return self.config.lr

    def _config_snapshot(self):
        c = self.config
        return {
            'num_blocks': c.num_blocks,
            'channels': c.channels,
            'history_length': c.history_length,
            'num_sims': c.num_sims,
            'cpuct': c.cpuct,
            'dirichlet_alpha': c.dirichlet_alpha,
            'dirichlet_weight': c.dirichlet_weight,
            'games_per_iter': c.games_per_iter,
            'parallel_games': c.parallel_games,
            'temperature_moves': c.temperature_moves,
            'temperature_final': c.temperature_final,
            'resign_threshold': c.resign_threshold,
            'draw_penalty': c.draw_penalty,
            'training_steps_per_iter': c.training_steps_per_iter,
            'batch_size': c.batch_size,
            'lr': c.lr,
            'momentum': c.momentum,
            'weight_decay': c.weight_decay,
            'value_loss_weight': c.value_loss_weight,
            'lr_schedule': c.lr_schedule,
            'warmup_steps': c.warmup_steps,
            'use_amp': c.use_amp,
            'buffer_capacity': c.buffer_capacity,
            'device': c.device,
        }

    def run_iteration(self, iteration):
        """One full cycle: self-play -> add to buffer -> train."""
        print(f"\n{'='*60}")
        print(f"Generation {self.generation + 1} (iteration {iteration})")
        print(f"{'='*60}")

        # self-play
        t0 = time.time()
        records, sp_stats = run_self_play(
            self.model, self.device, self.config
        )
        sp_time = time.time() - t0
        print(f"  Self-play took {sp_time:.1f}s")

        # add games to replay buffer
        positions_added = 0
        for record in records:
            boards, policies, values = record.get_training_samples()
            self.buffer.add_batch(boards, policies, values)
            positions_added += len(boards)
        print(f"  Added {positions_added} positions to buffer (total: {len(self.buffer)})")

        # train
        t0 = time.time()
        train_stats = self.train_on_buffer()
        train_time = time.time() - t0
        print(f"  Training took {train_time:.1f}s")

        # update history
        self.generation += 1
        self.history['generations'].append(self.generation)
        self.history['self_play_stats'].append(sp_stats)

        if train_stats:
            self.history['policy_losses'].append(train_stats['policy_loss'])
            self.history['value_losses'].append(train_stats['value_loss'])
            self.history['total_losses'].append(train_stats['total_loss'])
            self.history['policy_entropy'].append(train_stats['policy_entropy'])

        iteration_record = {
            'generation': self.generation,
            'timestamp': datetime.now().isoformat(),
            'self_play_time_s': round(sp_time, 1),
            'training_time_s': round(train_time, 1),
            'total_time_s': round(sp_time + train_time, 1),
            'buffer_size': len(self.buffer),
            'positions_added': positions_added,
            'total_training_steps': self.total_training_steps,
            'learning_rate': train_stats.get('learning_rate', 0),
            'policy_loss': train_stats.get('policy_loss', 0),
            'value_loss': train_stats.get('value_loss', 0),
            'total_loss': train_stats.get('total_loss', 0),
            'policy_entropy': train_stats.get('policy_entropy', 0),
            'self_play': sp_stats,
        }
        self.history['iterations'].append(iteration_record)

        # save checkpoint
        self.save_checkpoint()

        # save training history as JSON too (for analytics)
        history_path = os.path.join(self.config.analytics_dir, 'training_history.json')
        os.makedirs(self.config.analytics_dir, exist_ok=True)

        full_history = {
            'session_start': self.session_start,
            'last_updated': datetime.now().isoformat(),
            'config': self._config_snapshot(),
            **self.history,
        }
        with open(history_path, 'w') as f:
            json.dump(full_history, f, indent=2)

        return {**sp_stats, **train_stats}
