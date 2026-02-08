import sys
import os
import json
import time
import argparse
import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PIECE_SYMBOLS = {
    'R': '\u2656', 'N': '\u2658', 'B': '\u2657', 'Q': '\u2655', 'K': '\u2654', 'P': '\u2659',
    'r': '\u265c', 'n': '\u265e', 'b': '\u265d', 'q': '\u265b', 'k': '\u265a', 'p': '\u265f',
}


def render_board(board, last_move=None):
    lines = []
    lines.append('    a   b   c   d   e   f   g   h')
    lines.append('  +---+---+---+---+---+---+---+---+')

    for rank in range(7, -1, -1):
        row = f'{rank + 1} |'
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)

            highlight = False
            if last_move:
                try:
                    mv = chess.Move.from_uci(last_move)
                    if sq == mv.to_square:
                        highlight = True
                except:
                    pass

            if piece:
                symbol = PIECE_SYMBOLS.get(piece.symbol(), piece.symbol())
            else:
                symbol = ' '

            if highlight:
                row += f'>{symbol}< |'
            else:
                row += f' {symbol} |'

        lines.append(row)
        lines.append('  +---+---+---+---+---+---+---+---+')

    return '\n'.join(lines)


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def main():
    parser = argparse.ArgumentParser(description="Watch live self-play games")
    parser.add_argument("--file", type=str, default="analytics_output/live_games.json")
    parser.add_argument("--refresh", type=float, default=0.5, help="Refresh interval in seconds")
    parser.add_argument("--max-boards", type=int, default=4, help="Max boards to display at once")
    args = parser.parse_args()

    print("Waiting for training to start...")
    print(f"Watching: {args.file}")
    print("Press Ctrl+C to stop\n")

    last_data = None

    while True:
        try:
            if not os.path.exists(args.file):
                time.sleep(1)
                continue

            try:
                with open(args.file, 'r') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                time.sleep(args.refresh)
                continue

            if data == last_data:
                time.sleep(args.refresh)
                continue
            last_data = data

            clear_screen()

            active = data.get('active_games', {})
            completed = data.get('completed', 0)
            total_active = len(active)

            print('=' * 60)
            print(f'  LIVE SELF-PLAY  |  Active: {total_active}  |  Completed: {completed}')
            print('=' * 60)

            if not active:
                print('\n  No active games — waiting for next batch...')
                time.sleep(args.refresh)
                continue

            # show up to max_boards games
            game_ids = sorted(active.keys(), key=lambda x: int(x))[:args.max_boards]

            for gid in game_ids:
                game = active[gid]
                fen = game.get('fen', chess.STARTING_FEN)
                move_num = game.get('move_num', 0)
                last_move = game.get('last_move')
                turn = game.get('turn', '?')

                board = chess.Board(fen)

                print(f'\n  Game #{gid}  |  Move {move_num}  |  {turn} to play')
                if last_move:
                    print(f'  Last move: {last_move}')
                print()
                print(render_board(board, last_move))

            if total_active > args.max_boards:
                print(f'\n  ... and {total_active - args.max_boards} more games running')

            time.sleep(args.refresh)

        except KeyboardInterrupt:
            print("\nStopped watching.")
            break


if __name__ == "__main__":
    main()
