import os
import numpy as np
import chess
import onnxruntime as ort

from src.model.network import encode_board, move_to_index, index_to_move, get_legal_move_mask, POLICY_SIZE
from src.model.mcts import MCTS


# difficulty -> (num_sims, temperature)
DIFFICULTY_PRESETS = {
    'easy': (50, 1.0),
    'medium': (200, 0.5),
    'hard': (400, 0.1),
    'max': (800, 0.05),
}


class ChessEngine:
    """ONNX-based chess engine wrapping the model and MCTS for inference."""

    def __init__(self, model_path="chess_model.onnx"):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_cpu_mem_arena = True

        # reduce idle CPU usage on constrained environments
        try:
            sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        except Exception:
            pass

        self.session = ort.InferenceSession(
            model_path, sess_options,
            providers=["CPUExecutionProvider"]
        )

    def _eval_fn(self, board):
        """Evaluate a board position using the ONNX model."""
        board_tensor = encode_board(board)
        inp = board_tensor[np.newaxis, ...]
        policy_logits, value = self.session.run(None, {"board": inp})
        return policy_logits[0], float(value[0])

    def get_move(self, fen: str, difficulty: str = "medium"):
        """Get the best move for a position at the given difficulty level."""
        board = chess.Board(fen)
        num_sims, temp = DIFFICULTY_PRESETS.get(difficulty, DIFFICULTY_PRESETS['medium'])

        mcts = MCTS(eval_fn=self._eval_fn, num_sims=num_sims, cpuct=2.0)
        move, visit_counts = mcts.pick_move(board, temperature=temp)

        # get the value of the position
        _, value = self._eval_fn(board)

        # confidence = visit share of the chosen move
        total = visit_counts.sum()
        move_idx = move_to_index(move, board)
        confidence = visit_counts[move_idx] / max(total, 1)

        return move.uci(), value, confidence

    def evaluate(self, fen: str, num_sims=200):
        """Evaluate a position and return top moves."""
        board = chess.Board(fen)

        mcts = MCTS(eval_fn=self._eval_fn, num_sims=num_sims, cpuct=2.0)
        visit_counts = mcts.search(board)

        _, value = self._eval_fn(board)

        # extract top moves
        moves_with_visits = []
        for move in board.legal_moves:
            idx = move_to_index(move, board)
            visits = int(visit_counts[idx])
            if visits > 0:
                moves_with_visits.append({
                    'move': move.uci(),
                    'visits': visits,
                    'score': visits / max(visit_counts.sum(), 1),
                })

        moves_with_visits.sort(key=lambda x: x['visits'], reverse=True)
        return value, moves_with_visits[:10]

    def analyze(self, fen: str, num_sims=400):
        """Full analysis — return all legal moves ranked by MCTS visits."""
        board = chess.Board(fen)

        mcts = MCTS(eval_fn=self._eval_fn, num_sims=num_sims, cpuct=2.0)
        visit_counts = mcts.search(board)

        _, value = self._eval_fn(board)

        all_moves = []
        for move in board.legal_moves:
            idx = move_to_index(move, board)
            visits = int(visit_counts[idx])
            all_moves.append({
                'move': move.uci(),
                'visits': visits,
                'score': visits / max(visit_counts.sum(), 1),
            })

        all_moves.sort(key=lambda x: x['visits'], reverse=True)
        total_sims = int(visit_counts.sum())
        return value, all_moves, total_sims
