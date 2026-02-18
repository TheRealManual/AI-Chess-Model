"""Opening book for self-play diversity."""

import random
import chess

OPENING_BOOK = [
    # === KING PAWN (1.e4) ===
    # Italian Game
    {"name": "Italian Game",
     "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"]},
    # Italian — Giuoco Piano
    {"name": "Giuoco Piano",
     "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3"]},
    # Italian — Evans Gambit
    {"name": "Evans Gambit",
     "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "b2b4"]},
    # Ruy Lopez
    {"name": "Ruy Lopez",
     "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]},
    # Ruy Lopez — Berlin Defense
    {"name": "Ruy Lopez Berlin",
     "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6"]},
    # Ruy Lopez — Morphy Defense
    {"name": "Ruy Lopez Morphy",
     "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "f1a4"]},
    # Scotch Game
    {"name": "Scotch Game",
     "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f3d4"]},
    # King's Gambit
    {"name": "King's Gambit",
     "moves": ["e2e4", "e7e5", "f2f4"]},
    # King's Gambit Accepted
    {"name": "King's Gambit Accepted",
     "moves": ["e2e4", "e7e5", "f2f4", "e5f4"]},
    # Vienna Game
    {"name": "Vienna Game",
     "moves": ["e2e4", "e7e5", "b1c3"]},
    # Four Knights
    {"name": "Four Knights",
     "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "b1c3", "g8f6"]},
    # Petrov Defense
    {"name": "Petrov Defense",
     "moves": ["e2e4", "e7e5", "g1f3", "g8f6"]},
    # Philidor Defense
    {"name": "Philidor Defense",
     "moves": ["e2e4", "e7e5", "g1f3", "d7d6"]},
    # Center Game
    {"name": "Center Game",
     "moves": ["e2e4", "e7e5", "d2d4", "e5d4", "d1d4"]},
    # Bishop's Opening
    {"name": "Bishop's Opening",
     "moves": ["e2e4", "e7e5", "f1c4"]},

    # === SICILIAN (1.e4 c5) ===
    # Open Sicilian
    {"name": "Sicilian Open",
     "moves": ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"]},
    # Sicilian Najdorf
    {"name": "Sicilian Najdorf",
     "moves": ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"]},
    # Sicilian Dragon
    {"name": "Sicilian Dragon",
     "moves": ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "g7g6"]},
    # Sicilian Scheveningen
    {"name": "Sicilian Scheveningen",
     "moves": ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "e7e6"]},
    # Sicilian Classical
    {"name": "Sicilian Classical",
     "moves": ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "b8c6"]},
    # Sicilian Alapin
    {"name": "Sicilian Alapin",
     "moves": ["e2e4", "c7c5", "c2c3"]},
    # Sicilian Closed
    {"name": "Sicilian Closed",
     "moves": ["e2e4", "c7c5", "b1c3"]},

    # === FRENCH (1.e4 e6) ===
    # French Defense
    {"name": "French Defense",
     "moves": ["e2e4", "e7e6", "d2d4", "d7d5"]},
    # French Advance
    {"name": "French Advance",
     "moves": ["e2e4", "e7e6", "d2d4", "d7d5", "e4e5"]},
    # French Winawer
    {"name": "French Winawer",
     "moves": ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4"]},
    # French Tarrasch
    {"name": "French Tarrasch",
     "moves": ["e2e4", "e7e6", "d2d4", "d7d5", "b1d2"]},

    # === CARO-KANN (1.e4 c6) ===
    {"name": "Caro-Kann",
     "moves": ["e2e4", "c7c6", "d2d4", "d7d5"]},
    {"name": "Caro-Kann Advance",
     "moves": ["e2e4", "c7c6", "d2d4", "d7d5", "e4e5"]},
    {"name": "Caro-Kann Classical",
     "moves": ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "c8f5"]},

    # === SCANDINAVIAN ===
    {"name": "Scandinavian",
     "moves": ["e2e4", "d7d5"]},
    {"name": "Scandinavian Modern",
     "moves": ["e2e4", "d7d5", "e4d5", "g8f6"]},

    # === PIRC / MODERN ===
    {"name": "Pirc Defense",
     "moves": ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6"]},
    {"name": "Modern Defense",
     "moves": ["e2e4", "g7g6"]},

    # === ALEKHINE ===
    {"name": "Alekhine Defense",
     "moves": ["e2e4", "g8f6"]},

    # === QUEEN PAWN (1.d4) ===
    # Queen's Gambit
    {"name": "Queen's Gambit",
     "moves": ["d2d4", "d7d5", "c2c4"]},
    # Queen's Gambit Declined
    {"name": "QGD",
     "moves": ["d2d4", "d7d5", "c2c4", "e7e6"]},
    # Queen's Gambit Accepted
    {"name": "QGA",
     "moves": ["d2d4", "d7d5", "c2c4", "d5c4"]},
    # Slav Defense
    {"name": "Slav Defense",
     "moves": ["d2d4", "d7d5", "c2c4", "c7c6"]},
    # Semi-Slav
    {"name": "Semi-Slav",
     "moves": ["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3", "e7e6"]},

    # === INDIAN DEFENSES (1.d4 Nf6) ===
    # King's Indian
    {"name": "King's Indian",
     "moves": ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"]},
    # King's Indian — Classical
    {"name": "KID Classical",
     "moves": ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6"]},
    # Nimzo-Indian
    {"name": "Nimzo-Indian",
     "moves": ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"]},
    # Queen's Indian
    {"name": "Queen's Indian",
     "moves": ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6"]},
    # Bogo-Indian
    {"name": "Bogo-Indian",
     "moves": ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "f8b4"]},
    # Grunfeld Defense
    {"name": "Grunfeld",
     "moves": ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5"]},
    # Catalan
    {"name": "Catalan",
     "moves": ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3"]},
    # Benoni
    {"name": "Benoni",
     "moves": ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5"]},
    # Dutch Defense
    {"name": "Dutch Defense",
     "moves": ["d2d4", "f7f5"]},

    # === FLANK OPENINGS ===
    # English Opening
    {"name": "English Opening",
     "moves": ["c2c4"]},
    # English — Symmetrical
    {"name": "English Symmetrical",
     "moves": ["c2c4", "c7c5"]},
    # English — Reversed Sicilian
    {"name": "English Reversed Sicilian",
     "moves": ["c2c4", "e7e5"]},
    # Reti Opening
    {"name": "Reti Opening",
     "moves": ["g1f3", "d7d5", "c2c4"]},
    # King's Indian Attack
    {"name": "King's Indian Attack",
     "moves": ["g1f3", "d7d5", "g2g3"]},
    # Bird Opening
    {"name": "Bird Opening",
     "moves": ["f2f4"]},
    # Larsen Opening
    {"name": "Larsen Opening",
     "moves": ["b2b3"]},

    # === UNCOMMON / SURPRISE WEAPONS ===
    # London System
    {"name": "London System",
     "moves": ["d2d4", "d7d5", "g1f3", "g8f6", "c1f4"]},
    # Trompowsky
    {"name": "Trompowsky",
     "moves": ["d2d4", "g8f6", "c1g5"]},
    # Torre Attack
    {"name": "Torre Attack",
     "moves": ["d2d4", "g8f6", "g1f3", "e7e6", "c1g5"]},
    # Colle System
    {"name": "Colle System",
     "moves": ["d2d4", "d7d5", "g1f3", "g8f6", "e2e3"]},
]


def get_random_opening(rng=None):
    """Pick a random opening from the book."""
    if rng is None:
        entry = random.choice(OPENING_BOOK)
    else:
        entry = rng.choice(OPENING_BOOK)
    return entry["name"], list(entry["moves"])


def apply_opening(board, moves):
    """Apply a list of UCI moves to a board. Returns number applied."""
    applied = 0
    for uci in moves:
        try:
            move = chess.Move.from_uci(uci)
            if move in board.legal_moves:
                board.push(move)
                applied += 1
            else:
                break
        except ValueError:
            break
    return applied



