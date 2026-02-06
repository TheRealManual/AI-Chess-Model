import math
import numpy as np
import chess

from src.model.network import encode_board, move_to_index, index_to_move, get_legal_move_mask, POLICY_SIZE


class MCTSNode:
    """A single node in the search tree."""

    __slots__ = ['parent', 'move', 'children', 'visit_count', 'total_value',
                 'prior', 'board']

    def __init__(self, board, parent=None, move=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = []
        self.visit_count = 0
        self.total_value = 0.0

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def is_leaf(self):
        return len(self.children) == 0

    def is_terminal(self):
        return self.board.is_game_over()

    def select_child(self, cpuct):
        """Pick the child with the highest PUCT score."""
        total_visits = sum(c.visit_count for c in self.children)
        sqrt_total = math.sqrt(total_visits + 1)

        best_score = -float('inf')
        best_child = None

        for child in self.children:
            exploit = child.q_value
            explore = cpuct * child.prior * sqrt_total / (1 + child.visit_count)
            score = exploit + explore

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self, policy_logits):
        """Create child nodes for all legal moves using the policy network output."""
        mask = get_legal_move_mask(self.board)

        # mask illegal moves and compute probabilities
        masked = np.where(mask > 0, policy_logits, -1e9)
        # softmax
        exp = np.exp(masked - masked.max())
        exp = exp * mask
        total = exp.sum()
        if total > 0:
            probs = exp / total
        else:
            # fallback to uniform if something goes wrong
            probs = mask / mask.sum()

        for move in self.board.legal_moves:
            idx = move_to_index(move, self.board)
            child_board = self.board.copy()
            child_board.push(move)
            child = MCTSNode(child_board, parent=self, move=move, prior=probs[idx])
            self.children.append(child)

    def backpropagate(self, value):
        """Walk back up the tree, updating visit counts and values.
        Value is negated at each level since players alternate."""
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value
            node = node.parent


class MCTS:
    """Monte Carlo Tree Search using a neural network for evaluation."""

    def __init__(self, eval_fn, num_sims=200, cpuct=2.0,
                 dirichlet_alpha=0.3, dirichlet_weight=0.25):
        self.eval_fn = eval_fn
        self.num_sims = num_sims
        self.cpuct = cpuct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight

    def search(self, board: chess.Board) -> np.ndarray:
        """Run MCTS from the given position. Returns visit count distribution over all moves."""
        root = MCTSNode(board.copy())

        # evaluate and expand root
        policy, value = self.eval_fn(root.board)
        root.expand(policy)

        # add dirichlet noise to root priors for exploration
        if root.children:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.children))
            for child, n in zip(root.children, noise):
                child.prior = (1 - self.dirichlet_weight) * child.prior + self.dirichlet_weight * n

        for _ in range(self.num_sims):
            node = root

            # selection: walk down the tree
            while not node.is_leaf() and not node.is_terminal():
                node = node.select_child(self.cpuct)

            # if terminal, use the game result directly
            if node.is_terminal():
                result = node.board.result()
                if result == '1-0':
                    # white won — value from perspective of side that just moved
                    value = 1.0 if node.board.turn == chess.BLACK else -1.0
                elif result == '0-1':
                    value = 1.0 if node.board.turn == chess.WHITE else -1.0
                else:
                    value = 0.0
            else:
                # expansion + evaluation
                policy, value = self.eval_fn(node.board)
                node.expand(policy)
                # value is from current player's perspective, negate for backprop
                value = -value

            node.backpropagate(value)

        # build visit count distribution
        visit_counts = np.zeros(POLICY_SIZE, dtype=np.float32)
        for child in root.children:
            idx = move_to_index(child.move, board)
            visit_counts[idx] = child.visit_count

        return visit_counts

    def pick_move(self, board: chess.Board, temperature=1.0):
        """Run search and pick a move based on visit counts and temperature."""
        visit_counts = self.search(board)

        if temperature < 0.05:
            # pick the most visited move
            best_idx = np.argmax(visit_counts)
            move = index_to_move(best_idx, board)
            return move, visit_counts
        else:
            # sample proportional to visit_count^(1/temp)
            nonzero = visit_counts > 0
            # use log-space to avoid overflow: exp(log(v) / temp)
            adjusted = np.zeros_like(visit_counts)
            log_counts = np.log(visit_counts[nonzero] + 1e-8)
            scaled = log_counts / temperature
            scaled -= scaled.max()  # numerical stability
            adjusted[nonzero] = np.exp(scaled)
            total = adjusted.sum()
            if total > 0:
                probs = adjusted / total
            else:
                mask = get_legal_move_mask(board)
                probs = mask / mask.sum()
            idx = np.random.choice(len(probs), p=probs)
            move = index_to_move(idx, board)
            return move, visit_counts
