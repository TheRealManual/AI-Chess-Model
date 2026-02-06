import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.network import NUM_PLANES, POLICY_SIZE
from src.training.replay_buffer import ReplayBuffer


def test_buffer_add_and_sample():
    buf = ReplayBuffer(capacity=100)

    for i in range(50):
        board = np.random.randn(NUM_PLANES, 8, 8).astype(np.float32)
        policy = np.random.randn(POLICY_SIZE).astype(np.float32)
        value = np.random.uniform(-1, 1)
        buf.add(board, policy, value)

    assert len(buf) == 50

    boards, policies, values = buf.sample(16)
    assert boards.shape == (16, NUM_PLANES, 8, 8)
    assert policies.shape == (16, POLICY_SIZE)
    assert values.shape == (16,)


def test_buffer_wraps_around():
    buf = ReplayBuffer(capacity=10)

    for i in range(25):
        board = np.ones((NUM_PLANES, 8, 8), dtype=np.float32) * i
        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        buf.add(board, policy, float(i))

    assert len(buf) == 10
    assert buf.total_added == 25
    # newest entries should be 15-24
    assert buf.values[buf.index - 1] == 24.0


def test_buffer_save_load(tmp_path):
    buf = ReplayBuffer(capacity=50)

    for i in range(30):
        board = np.random.randn(NUM_PLANES, 8, 8).astype(np.float32)
        policy = np.random.randn(POLICY_SIZE).astype(np.float32)
        buf.add(board, policy, float(i))

    save_dir = str(tmp_path / "buffer")
    buf.save_data(save_dir)

    buf2 = ReplayBuffer(capacity=50)
    loaded = buf2.load_data(save_dir)

    assert loaded is True
    assert len(buf2) == len(buf)
    assert buf2.total_added == buf.total_added
    np.testing.assert_array_equal(buf2.boards[:30], buf.boards[:30])


def test_buffer_batch_add():
    buf = ReplayBuffer(capacity=100)

    boards = np.random.randn(20, NUM_PLANES, 8, 8).astype(np.float32)
    policies = np.random.randn(20, POLICY_SIZE).astype(np.float32)
    values = np.random.randn(20).astype(np.float32)

    buf.add_batch(boards, policies, values)
    assert len(buf) == 20
