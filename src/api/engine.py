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

# Piece movement rule descriptions
PIECE_MOVEMENT_RULES = {
    chess.PAWN: (
        "Pawn",
        "Pawns move forward one square, or two squares from their starting rank. "
        "They capture diagonally one square forward. Pawns can also capture en passant "
        "and promote to any piece (queen, rook, bishop, or knight) upon reaching the last rank."
    ),
    chess.KNIGHT: (
        "Knight",
        "Knights move in an L-shape: two squares in one direction and one square perpendicular. "
        "Knights are the only piece that can jump over other pieces."
    ),
    chess.BISHOP: (
        "Bishop",
        "Bishops move diagonally any number of squares. They cannot jump over other pieces. "
        "Each bishop stays on its original square color for the entire game."
    ),
    chess.ROOK: (
        "Rook",
        "Rooks move horizontally or vertically any number of squares. They cannot jump over "
        "other pieces. Rooks are involved in castling with the king."
    ),
    chess.QUEEN: (
        "Queen",
        "The queen combines the power of a rook and bishop, moving any number of squares "
        "horizontally, vertically, or diagonally. She cannot jump over other pieces."
    ),
    chess.KING: (
        "King",
        "The king moves one square in any direction. The king can also castle with a rook "
        "if neither piece has moved, the squares between them are empty, and the king does "
        "not pass through or land on a square attacked by an enemy piece."
    ),
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

    # ------------------------------------------------------------------
    # Training mode methods
    # ------------------------------------------------------------------

    def analyze_player_move(self, fen: str, player_move_uci: str, num_sims=400):
        """Analyze a player's move: compare to AI's best, rate it, explain why."""
        board = chess.Board(fen)
        player_move = chess.Move.from_uci(player_move_uci)

        if player_move not in board.legal_moves:
            raise ValueError(f"Illegal move {player_move_uci} in position {fen}")

        # Evaluate position before the move
        _, value_before = self._eval_fn(board)

        # Run MCTS to get AI's move ranking
        mcts = MCTS(eval_fn=self._eval_fn, num_sims=num_sims, cpuct=2.0)
        visit_counts = mcts.search(board)

        # Rank all legal moves
        ranked_moves = []
        for move in board.legal_moves:
            idx = move_to_index(move, board)
            visits = int(visit_counts[idx])
            total = max(visit_counts.sum(), 1)
            ranked_moves.append({
                'move': move.uci(),
                'visits': visits,
                'score': visits / total,
            })
        ranked_moves.sort(key=lambda x: x['visits'], reverse=True)

        # Find where the player's move ranks
        player_rank = 1
        player_score = 0.0
        for i, m in enumerate(ranked_moves):
            if m['move'] == player_move_uci:
                player_rank = i + 1
                player_score = m['score']
                break

        best_move_info = ranked_moves[0]

        # Evaluate position after the player's move
        board_after = board.copy()
        board_after.push(player_move)
        _, value_after_raw = self._eval_fn(board_after)
        # Negate because _eval_fn returns value from current player's perspective
        # and the current player changed after pushing the move
        value_after = -value_after_raw

        # Rate the move
        rating, explanation, suggestion = self._rate_move(
            board, player_move, player_rank, len(ranked_moves),
            player_score, best_move_info, value_before, value_after,
        )

        return {
            'player_move': player_move_uci,
            'player_move_rank': player_rank,
            'player_move_score': player_score,
            'best_move': best_move_info['move'],
            'best_move_score': best_move_info['score'],
            'rating': rating,
            'explanation': explanation,
            'suggestion': suggestion,
            'value_before': value_before,
            'value_after': value_after,
            'top_moves': ranked_moves[:5],
        }

    def suggest_move(self, fen: str, num_sims=400):
        """Suggest a move for the player with an explanation of why."""
        board = chess.Board(fen)

        mcts = MCTS(eval_fn=self._eval_fn, num_sims=num_sims, cpuct=2.0)
        visit_counts = mcts.search(board)

        _, value = self._eval_fn(board)

        ranked_moves = []
        for move in board.legal_moves:
            idx = move_to_index(move, board)
            visits = int(visit_counts[idx])
            total = max(visit_counts.sum(), 1)
            ranked_moves.append({
                'move': move.uci(),
                'visits': visits,
                'score': visits / total,
            })
        ranked_moves.sort(key=lambda x: x['visits'], reverse=True)

        best = ranked_moves[0]
        best_move = chess.Move.from_uci(best['move'])
        confidence = best['score']

        explanation = self._explain_suggestion(board, best_move, ranked_moves, value)

        return {
            'suggested_move': best['move'],
            'explanation': explanation,
            'confidence': confidence,
            'value': value,
            'alternatives': ranked_moves[1:4],
        }

    @staticmethod
    def get_piece_info(fen: str, square_name: str):
        """Return piece info and legal destination squares for a selected piece."""
        board = chess.Board(fen)
        sq = chess.parse_square(square_name)
        piece = board.piece_at(sq)

        if piece is None:
            raise ValueError(f"No piece on square {square_name}")

        piece_name, movement_rules = PIECE_MOVEMENT_RULES.get(
            piece.piece_type, ("Unknown", "Unknown piece type.")
        )
        color = "white" if piece.color == chess.WHITE else "black"

        legal_destinations = []
        for move in board.legal_moves:
            if move.from_square == sq:
                dest_name = chess.square_name(move.to_square)
                is_capture = board.is_capture(move)
                legal_destinations.append({
                    'square': dest_name,
                    'is_capture': is_capture,
                })

        return {
            'piece_name': piece_name,
            'piece_color': color,
            'square': square_name,
            'movement_rules': movement_rules,
            'legal_destinations': legal_destinations,
        }

    def _rate_move(self, board, player_move, rank, total_moves,
                   player_score, best_move_info, value_before, value_after):
        """Rate a player's move and generate explanation + suggestion."""
        value_delta = value_after - value_before
        best_move = chess.Move.from_uci(best_move_info['move'])
        is_best = (rank == 1)
        player_san = board.san(player_move)
        best_san = board.san(best_move)

        # Determine rating
        if is_best:
            rating = "excellent"
        elif rank <= 3 and player_score >= 0.15:
            rating = "good"
        elif rank <= 5 and value_delta > -0.15:
            rating = "okay"
        elif value_delta > -0.3:
            rating = "inaccuracy"
        elif value_delta > -0.6:
            rating = "mistake"
        else:
            rating = "blunder"

        player_traits = self._analyze_move_traits(board, player_move)
        best_traits = self._analyze_move_traits(board, best_move)

        # Build explanation
        if is_best:
            good_parts = self._summarize_good_traits(player_traits, player_san)
            if good_parts:
                explanation = (
                    f"{player_san} is the top choice! {good_parts}"
                )
            else:
                explanation = (
                    f"{player_san} is the top choice! The AI's analysis agrees this is the "
                    f"strongest move in this position."
                )
        elif rating == "good":
            good_parts = self._summarize_good_traits(player_traits, player_san)
            explanation = (
                f"{player_san} is a solid move (ranked #{rank}). "
            )
            if good_parts:
                explanation += good_parts + " "
            explanation += (
                f"The AI slightly prefers {best_san}."
            )
        elif rating == "okay":
            explanation = (
                f"{player_san} is a reasonable move (ranked #{rank}). "
            )
            problems = self._explain_move_problems(board, player_move, player_traits)
            if problems:
                explanation += problems + " "
            explanation += f"{best_san} is stronger."
        else:
            severity = {
                "inaccuracy": "a small inaccuracy",
                "mistake": "a mistake",
                "blunder": "a serious blunder",
            }[rating]
            explanation = f"{player_san} is {severity}. "
            problems = self._explain_move_problems(board, player_move, player_traits)
            if problems:
                explanation += problems + " "
            else:
                explanation += (
                    f"The position evaluation dropped by {abs(value_delta):.2f}. "
                )
            why_better = self._explain_why_better(board, best_move, best_san, best_traits, player_traits)
            explanation += why_better

        # Build suggestion
        if is_best:
            suggestion = "Great move! Keep playing like this."
        else:
            weakness = self._explain_weakness_vs_best(board, player_move, player_san, best_move, best_san, player_traits, best_traits)
            suggestion = weakness

        return rating, explanation, suggestion

    @staticmethod
    def _analyze_move_traits(board, move):
        """Inspect a move for tactical and positional features."""
        traits = {}
        piece = board.piece_at(move.from_square)
        if piece is None:
            return traits
        traits['piece_type'] = piece.piece_type
        traits['piece_name'] = PIECE_MOVEMENT_RULES.get(piece.piece_type, ("piece",))[0].lower()

        to_rank = chess.square_rank(move.to_square)
        to_file = chess.square_file(move.to_square)
        from_rank = chess.square_rank(move.from_square)
        from_file = chess.square_file(move.from_square)
        is_white = piece.color == chess.WHITE

        # Capture info
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured:
                cap_name = PIECE_MOVEMENT_RULES.get(captured.piece_type, ("piece",))[0].lower()
                traits['captures'] = cap_name
                traits['captures_type'] = captured.piece_type

        # Check / checkmate
        board_after = board.copy()
        board_after.push(move)
        if board_after.is_checkmate():
            traits['checkmate'] = True
        elif board_after.is_check():
            traits['check'] = True

        # Castling
        if board.is_castling(move):
            traits['castles'] = True

        # Center control (d4-d5-e4-e5)
        center_squares = {chess.D4, chess.D5, chess.E4, chess.E5}
        extended_center = {chess.C3, chess.C4, chess.C5, chess.C6,
                          chess.D3, chess.D6, chess.E3, chess.E6,
                          chess.F3, chess.F4, chess.F5, chess.F6}
        if move.to_square in center_squares:
            traits['center_occupation'] = True
        elif move.to_square in extended_center:
            traits['extended_center'] = True

        # Development (moving knight/bishop off back rank)
        back_rank = 0 if is_white else 7
        if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            if from_rank == back_rank and to_rank != back_rank:
                traits['develops'] = True

        # Knight on the rim
        if piece.piece_type == chess.KNIGHT and to_file in (0, 7):
            traits['knight_rim'] = True

        # Pawn structure observations
        if piece.piece_type == chess.PAWN:
            # Early flank pawn push (a or h pawn moving forward early)
            fullmove = board.fullmove_number
            if to_file in (0, 7) and fullmove <= 10:
                traits['edge_pawn_push'] = True
            # Pawn moving far forward early without support
            advance_rank = to_rank if is_white else (7 - to_rank)
            if advance_rank >= 4 and fullmove <= 8:
                traits['early_pawn_overextend'] = True
            # Blocks own bishop or pieces
            if is_white and to_rank == 2 and to_file in (3, 4):
                # check if own bishop is behind
                behind_sq = chess.square(to_file, 1)
                behind_piece = board.piece_at(behind_sq)
                if behind_piece and behind_piece.color == piece.color and behind_piece.piece_type == chess.BISHOP:
                    traits['blocks_bishop'] = True
            if not is_white and to_rank == 5 and to_file in (3, 4):
                behind_sq = chess.square(to_file, 6)
                behind_piece = board.piece_at(behind_sq)
                if behind_piece and behind_piece.color == piece.color and behind_piece.piece_type == chess.BISHOP:
                    traits['blocks_bishop'] = True

        # King safety: moving king without castling in the opening
        if piece.piece_type == chess.KING and not board.is_castling(move) and board.fullmove_number <= 15:
            traits['king_walk'] = True

        # Leaves back rank undeveloped
        undeveloped = ChessEngine._count_undeveloped(board, piece.color)
        if undeveloped >= 3 and piece.piece_type not in (chess.KNIGHT, chess.BISHOP) and board.fullmove_number <= 12:
            traits['many_undeveloped'] = undeveloped

        # Creates or resolves threats
        board_after2 = board.copy()
        board_after2.push(move)
        # Count attacks on opponent pieces after this move
        opp_color = not piece.color
        attacks_high = 0
        for sq in chess.SQUARES:
            opp_piece = board_after2.piece_at(sq)
            if opp_piece and opp_piece.color == opp_color and opp_piece.piece_type in (chess.QUEEN, chess.ROOK):
                if board_after2.is_attacked_by(piece.color, sq):
                    attacks_high += 1
        if attacks_high > 0:
            traits['threatens_high_value'] = attacks_high

        return traits

    @staticmethod
    def _summarize_good_traits(traits, move_san):
        """Summarize the positive aspects of a move in natural language."""
        parts = []
        if traits.get('checkmate'):
            return f"{move_san} delivers checkmate!"
        if traits.get('check'):
            parts.append("gives check")
        if traits.get('captures'):
            parts.append(f"captures the {traits['captures']}")
        if traits.get('castles'):
            parts.append("castles to improve king safety and connect the rooks")
        if traits.get('develops'):
            parts.append(f"develops the {traits.get('piece_name', 'piece')} toward active play")
        if traits.get('center_occupation'):
            parts.append("takes control of the center")
        elif traits.get('extended_center'):
            parts.append("strengthens your influence in the center")
        if traits.get('threatens_high_value'):
            parts.append("creates threats against high-value pieces")
        if not parts:
            return ""
        return "It " + ", ".join(parts) + "."

    def _explain_move_problems(self, board, move, traits):
        """Explain what is wrong with a move based on its traits."""
        problems = []
        piece = board.piece_at(move.from_square)
        if piece is None:
            return ""

        if traits.get('knight_rim'):
            problems.append(
                "Placing the knight on the edge of the board limits it to fewer squares — "
                "\"a knight on the rim is dim.\""
            )
        if traits.get('edge_pawn_push'):
            problems.append(
                "Pushing a flank pawn this early doesn't help develop your pieces or control the center."
            )
        if traits.get('early_pawn_overextend'):
            problems.append(
                "Advancing this pawn so far forward early on can leave it weak and hard to defend."
            )
        if traits.get('blocks_bishop'):
            problems.append(
                "This pawn blocks your own bishop, limiting its activity."
            )
        if traits.get('king_walk'):
            problems.append(
                "Moving the king without castling this early exposes it to potential attacks."
            )
        if traits.get('many_undeveloped'):
            n = traits['many_undeveloped']
            problems.append(
                f"You still have {n} undeveloped minor pieces — "
                f"prioritizing development would give you more active play."
            )

        # If no specific problem found, try a generic positional note
        if not problems:
            is_white = piece.color == chess.WHITE
            if not board.has_castling_rights(piece.color) and board.fullmove_number <= 15:
                pass  # already castled or lost rights, not relevant
            elif board.has_castling_rights(piece.color) and piece.piece_type == chess.PAWN:
                undeveloped = self._count_undeveloped(board, piece.color)
                if undeveloped >= 2:
                    problems.append(
                        f"With {undeveloped} pieces still on the back rank, "
                        f"developing them first would activate your position faster."
                    )

        return " ".join(problems)

    @staticmethod
    def _count_undeveloped(board, color):
        """Count knights and bishops still on their starting rank."""
        back_rank = 0 if color == chess.WHITE else 7
        count = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == color and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                if chess.square_rank(sq) == back_rank:
                    count += 1
        return count

    def _explain_why_better(self, board, best_move, best_san, best_traits, player_traits):
        """Explain why the best move is better than what was played."""
        parts = []
        if best_traits.get('checkmate'):
            return f"{best_san} delivers checkmate!"
        if best_traits.get('captures'):
            cap = best_traits['captures']
            parts.append(f"{best_san} wins material by capturing the {cap}")
        if best_traits.get('check'):
            parts.append(f"{best_san} gives check, seizing the initiative")
        if best_traits.get('castles'):
            parts.append(f"{best_san} castles, securing your king and activating your rook")
        if best_traits.get('develops') and not player_traits.get('develops'):
            parts.append(
                f"{best_san} develops the {best_traits.get('piece_name', 'piece')}, "
                f"getting your pieces into the game faster"
            )
        if best_traits.get('center_occupation') and not player_traits.get('center_occupation'):
            parts.append(f"{best_san} seizes control of a key central square")
        if best_traits.get('threatens_high_value') and not player_traits.get('threatens_high_value'):
            parts.append(f"{best_san} creates pressure against high-value enemy pieces")

        if parts:
            return " ".join(parts) + "."
        # Fallback: generic but at least mentions the move
        return f"{best_san} leads to a more active position with better piece coordination."

    def _explain_weakness_vs_best(self, board, player_move, player_san,
                                   best_move, best_san, player_traits, best_traits):
        """Build the suggestion string contrasting the two moves."""
        problems = []

        # What the player's move lacks
        if best_traits.get('develops') and not player_traits.get('develops'):
            problems.append(
                f"Your move doesn't develop a piece, while {best_san} brings the "
                f"{best_traits.get('piece_name', 'piece')} into play."
            )
        if best_traits.get('center_occupation') and not player_traits.get('center_occupation'):
            problems.append(
                f"{best_san} fights for central control, which your move doesn't address."
            )
        if player_traits.get('knight_rim'):
            problems.append(
                f"Placing your knight on the rim limits its mobility. "
                f"{best_san} would keep your pieces more active."
            )
        if player_traits.get('edge_pawn_push'):
            problems.append(
                f"Pushing a flank pawn doesn't help your development. "
                f"{best_san} is more constructive."
            )
        if player_traits.get('blocks_bishop'):
            problems.append(
                f"Your pawn blocks your own bishop. "
                f"{best_san} avoids that problem."
            )
        if player_traits.get('king_walk'):
            problems.append(
                f"Moving the king early is risky. {best_san} would be safer."
            )
        if best_traits.get('captures') and not player_traits.get('captures'):
            cap = best_traits['captures']
            problems.append(f"{best_san} wins a {cap}, giving you a material advantage.")
        if best_traits.get('castles') and not player_traits.get('castles'):
            problems.append(f"{best_san} castles, improving king safety and rook activity.")

        if problems:
            return f"Consider {best_san} instead. " + " ".join(problems)
        return (
            f"Consider {best_san} instead — it leads to superior piece activity "
            f"and a stronger overall position."
        )

    def _explain_suggestion(self, board, move, ranked_moves, position_value):
        """Generate a natural-language explanation for a suggested move."""
        move_san = board.san(move)
        traits = self._analyze_move_traits(board, move)
        parts = [f"The AI suggests {move_san}."]

        # Use trait-based explanation
        good = self._summarize_good_traits(traits, move_san)
        if good:
            parts.append(good)

        # Castling extra detail
        if traits.get('castles'):
            parts.append("Castling early improves king safety and connects the rooks.")

        # Development reminder
        if traits.get('many_undeveloped'):
            n = traits['many_undeveloped']
            parts.append(f"You still have {n} undeveloped minor pieces to bring out.")

        # Positional reasoning for pieces going to center
        if traits.get('center_occupation'):
            parts.append("Occupying the center gives your pieces more scope and activity.")
        elif traits.get('extended_center'):
            parts.append("This supports your central presence.")

        # Confidence context
        confidence = ranked_moves[0]['score']
        if confidence > 0.6:
            parts.append("The AI is very confident in this move.")
        elif confidence < 0.2 and len(ranked_moves) > 1:
            parts.append(
                "The position is complex with several reasonable options."
            )

        # Position assessment
        if position_value > 0.5:
            parts.append("You have a strong advantage — keep pressing.")
        elif position_value < -0.5:
            parts.append("You're in a difficult position — accurate play is critical.")

        return " ".join(parts)
