"""Parse PGN files and stream training data for supervised pre-training."""

import io
import contextlib
import chess
import chess.pgn
import numpy as np

from src.model.network import encode_board, move_to_index, POLICY_SIZE


@contextlib.contextmanager
def open_pgn(path):
    """Open a PGN file, supporting both plain .pgn and zstandard .pgn.zst."""
    if path.endswith('.zst'):
        try:
            import zstandard as zstd
        except ImportError:
            raise ImportError(
                "zstandard package required for .pgn.zst files. "
                "Install with: pip install zstandard"
            )
        dctx = zstd.ZstdDecompressor()
        fh = open(path, 'rb')
        try:
            reader = dctx.stream_reader(fh)
            wrapper = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
            yield wrapper
        finally:
            fh.close()
    else:
        fh = open(path, 'r', encoding='utf-8', errors='replace')
        try:
            yield fh
        finally:
            fh.close()


def stream_training_positions(pgn_path, min_elo=1800, max_games=None,
                               skip_draws=False, min_moves=10):
    """Yield (board_tensor, policy_target, value_target) from PGN games."""
    games_processed = 0
    games_skipped = 0
    positions_yielded = 0

    with open_pgn(pgn_path) as f:
        while True:
            if max_games and games_processed >= max_games:
                break

            try:
                game = chess.pgn.read_game(f)
            except Exception:
                continue
            if game is None:
                break

            # filter by ELO
            try:
                white_elo = int(game.headers.get("WhiteElo", "0"))
                black_elo = int(game.headers.get("BlackElo", "0"))
            except (ValueError, TypeError):
                games_skipped += 1
                continue

            if white_elo < min_elo or black_elo < min_elo:
                games_skipped += 1
                continue

            # filter by result
            result_str = game.headers.get("Result", "*")
            if result_str == "1-0":
                result_white = 1.0
            elif result_str == "0-1":
                result_white = -1.0
            elif result_str == "1/2-1/2":
                if skip_draws:
                    games_skipped += 1
                    continue
                result_white = 0.0
            else:
                games_skipped += 1
                continue

            # replay the game and extract training positions
            moves = list(game.mainline_moves())
            if len(moves) < min_moves:
                games_skipped += 1
                continue

            board = game.board()
            for move in moves:
                # encode current position (with full move history from the game)
                board_tensor = encode_board(board)

                # policy target: one-hot on the move actually played
                policy = np.zeros(POLICY_SIZE, dtype=np.float32)
                try:
                    idx = move_to_index(move, board)
                    policy[idx] = 1.0
                except (KeyError, IndexError):
                    board.push(move)
                    continue  # skip positions with encoding issues

                # value target from current player's perspective
                if board.turn == chess.WHITE:
                    value = result_white
                else:
                    value = -result_white

                yield board_tensor, policy, float(value)
                positions_yielded += 1

                board.push(move)

            games_processed += 1
            if games_processed % 5000 == 0:
                print(f"  Processed {games_processed:,} games "
                      f"({games_skipped:,} skipped), "
                      f"{positions_yielded:,} positions extracted")

    print(f"  Final: {games_processed:,} games processed, "
          f"{games_skipped:,} skipped, {positions_yielded:,} positions")


def batch_positions(position_generator, batch_size=512):
    """Collect positions from the streaming generator into numpy batches.

    Yields:
        Tuple of (boards_batch, policies_batch, values_batch) numpy arrays.
    """
    boards, policies, values = [], [], []

    for board_tensor, policy, value in position_generator:
        boards.append(board_tensor)
        policies.append(policy)
        values.append(value)

        if len(boards) >= batch_size:
            yield (
                np.array(boards, dtype=np.float32),
                np.array(policies, dtype=np.float32),
                np.array(values, dtype=np.float32),
            )
            boards, policies, values = [], [], []

    # yield remaining partial batch
    if boards:
        yield (
            np.array(boards, dtype=np.float32),
            np.array(policies, dtype=np.float32),
            np.array(values, dtype=np.float32),
        )
