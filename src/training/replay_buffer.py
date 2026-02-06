import numpy as np
import json
import os
from pathlib import Path

from src.model.network import POLICY_SIZE, NUM_PLANES


class ReplayBuffer:
    """Fixed-capacity FIFO buffer of training positions."""

    def __init__(self, capacity=200_000):
        self.capacity = capacity
        self.boards = np.zeros((capacity, NUM_PLANES, 8, 8), dtype=np.float32)
        self.policies = np.zeros((capacity, POLICY_SIZE), dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.index = 0
        self.total_added = 0

    def add(self, board_tensor, policy_target, value_target):
        """Add a single training sample."""
        self.boards[self.index] = board_tensor
        self.policies[self.index] = policy_target
        self.values[self.index] = value_target
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_added += 1

    def add_batch(self, board_tensors, policy_targets, value_targets):
        """Add multiple samples at once."""
        n = len(board_tensors)
        for i in range(n):
            self.add(board_tensors[i], policy_targets[i], value_targets[i])

    def sample(self, batch_size):
        """Sample a random batch of positions."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.boards[indices],
            self.policies[indices],
            self.values[indices],
        )

    def save_metadata(self, path):
        """Save buffer state metadata (not the full data — that's too big)."""
        meta = {
            'size': self.size,
            'index': self.index,
            'total_added': self.total_added,
            'capacity': self.capacity,
        }
        with open(path, 'w') as f:
            json.dump(meta, f)

    def save_data(self, dir_path):
        """Save the actual buffer arrays to disk."""
        os.makedirs(dir_path, exist_ok=True)
        np.save(os.path.join(dir_path, 'boards.npy'), self.boards[:self.size])
        np.save(os.path.join(dir_path, 'policies.npy'), self.policies[:self.size])
        np.save(os.path.join(dir_path, 'values.npy'), self.values[:self.size])
        self.save_metadata(os.path.join(dir_path, 'buffer_meta.json'))

    def load_data(self, dir_path):
        """Load buffer data from disk."""
        meta_path = os.path.join(dir_path, 'buffer_meta.json')
        if not os.path.exists(meta_path):
            return False

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        boards = np.load(os.path.join(dir_path, 'boards.npy'))
        policies = np.load(os.path.join(dir_path, 'policies.npy'))
        values = np.load(os.path.join(dir_path, 'values.npy'))

        n = len(boards)
        self.boards[:n] = boards
        self.policies[:n] = policies
        self.values[:n] = values
        self.size = meta['size']
        self.index = meta['index']
        self.total_added = meta['total_added']

        return True

    def __len__(self):
        return self.size
