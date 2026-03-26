import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# only run API tests if onnxruntime is available and a model exists
# these are integration tests, skip if no ONNX model

pytest_plugins = ('pytest_asyncio',)


def _model_exists():
    """Check if there's an exported ONNX model or we can create a dummy one."""
    return os.path.exists("chess_model.onnx")


@pytest.fixture
def dummy_model(tmp_path):
    """Create a tiny ONNX model for testing."""
    import torch
    from src.model.network import ChessNet, NUM_PLANES
    from src.export import export_to_onnx

    model = ChessNet(num_blocks=2, channels=32)
    ckpt_path = str(tmp_path / "test_ckpt.pt")
    torch.save({'model_state': model.state_dict()}, ckpt_path)

    onnx_path = str(tmp_path / "test_model.onnx")
    export_to_onnx(ckpt_path, onnx_path, verify=False)
    return onnx_path


def test_engine_get_move(dummy_model):
    from src.api.engine import ChessEngine
    engine = ChessEngine(dummy_model)

    move, value, conf = engine.get_move(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        difficulty="easy",
    )

    assert len(move) >= 4  # UCI format like "e2e4"
    assert -1.0 <= value <= 1.0
    assert 0.0 <= conf <= 1.0


def test_engine_evaluate(dummy_model):
    from src.api.engine import ChessEngine
    engine = ChessEngine(dummy_model)

    value, top_moves = engine.evaluate(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        num_sims=30,
    )

    assert -1.0 <= value <= 1.0
    assert len(top_moves) > 0
    assert 'move' in top_moves[0]
    assert 'visits' in top_moves[0]


def test_engine_analyze(dummy_model):
    from src.api.engine import ChessEngine
    engine = ChessEngine(dummy_model)

    value, moves, total_sims = engine.analyze(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        num_sims=30,
    )

    assert -1.0 <= value <= 1.0
    assert len(moves) > 0
    assert total_sims > 0


@pytest.mark.asyncio
async def test_api_health(dummy_model):
    os.environ["MODEL_PATH"] = dummy_model

    from httpx import AsyncClient, ASGITransport
    from src.api import main as api_main
    from src.api.engine import ChessEngine

    api_main.engine = ChessEngine(dummy_model)

    transport = ASGITransport(app=api_main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_api_move(dummy_model):
    os.environ["MODEL_PATH"] = dummy_model

    from httpx import AsyncClient, ASGITransport
    from src.api import main as api_main
    from src.api.engine import ChessEngine

    api_main.engine = ChessEngine(dummy_model)

    transport = ASGITransport(app=api_main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/move", json={
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "difficulty": "easy",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "move" in data
        assert "value" in data
        assert "think_time_ms" in data


# ---------------------------------------------------------------------------
# Training mode tests
# ---------------------------------------------------------------------------

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_engine_analyze_player_move(dummy_model):
    from src.api.engine import ChessEngine
    engine = ChessEngine(dummy_model)

    # e2e4 is always legal from the starting position
    result = engine.analyze_player_move(STARTING_FEN, "e2e4", num_sims=30)

    assert result['player_move'] == "e2e4"
    assert 1 <= result['player_move_rank'] <= 20
    assert 0.0 <= result['player_move_score'] <= 1.0
    assert len(result['best_move']) >= 4
    assert result['rating'] in ("excellent", "good", "okay", "inaccuracy", "mistake", "blunder")
    assert len(result['explanation']) > 0
    assert len(result['suggestion']) > 0
    assert -1.0 <= result['value_before'] <= 1.0
    assert -1.0 <= result['value_after'] <= 1.0
    assert len(result['top_moves']) > 0


def test_engine_analyze_player_move_illegal(dummy_model):
    from src.api.engine import ChessEngine
    engine = ChessEngine(dummy_model)

    with pytest.raises(ValueError, match="Illegal move"):
        engine.analyze_player_move(STARTING_FEN, "e1e5", num_sims=30)


def test_engine_suggest_move(dummy_model):
    from src.api.engine import ChessEngine
    engine = ChessEngine(dummy_model)

    result = engine.suggest_move(STARTING_FEN, num_sims=30)

    assert len(result['suggested_move']) >= 4
    assert len(result['explanation']) > 0
    assert 0.0 <= result['confidence'] <= 1.0
    assert -1.0 <= result['value'] <= 1.0
    assert isinstance(result['alternatives'], list)


def test_engine_get_piece_info():
    from src.api.engine import ChessEngine

    result = ChessEngine.get_piece_info(STARTING_FEN, "e2")

    assert result['piece_name'] == "Pawn"
    assert result['piece_color'] == "white"
    assert result['square'] == "e2"
    assert "Pawn" in result['movement_rules']
    # e2 pawn should be able to go to e3 and e4
    dest_squares = [d['square'] for d in result['legal_destinations']]
    assert "e3" in dest_squares
    assert "e4" in dest_squares


def test_engine_get_piece_info_no_piece():
    from src.api.engine import ChessEngine

    with pytest.raises(ValueError, match="No piece on square"):
        ChessEngine.get_piece_info(STARTING_FEN, "e4")


def test_engine_get_piece_info_knight():
    from src.api.engine import ChessEngine

    result = ChessEngine.get_piece_info(STARTING_FEN, "b1")

    assert result['piece_name'] == "Knight"
    assert result['piece_color'] == "white"
    dest_squares = [d['square'] for d in result['legal_destinations']]
    assert "a3" in dest_squares
    assert "c3" in dest_squares


@pytest.mark.asyncio
async def test_api_training_analyze_move(dummy_model):
    os.environ["MODEL_PATH"] = dummy_model

    from httpx import AsyncClient, ASGITransport
    from src.api import main as api_main
    from src.api.engine import ChessEngine

    api_main.engine = ChessEngine(dummy_model)

    transport = ASGITransport(app=api_main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/training/analyze-move", json={
            "fen": STARTING_FEN,
            "player_move": "e2e4",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["player_move"] == "e2e4"
        assert "rating" in data
        assert "explanation" in data
        assert "suggestion" in data
        assert "best_move" in data
        assert "top_moves" in data


@pytest.mark.asyncio
async def test_api_training_suggest(dummy_model):
    os.environ["MODEL_PATH"] = dummy_model

    from httpx import AsyncClient, ASGITransport
    from src.api import main as api_main
    from src.api.engine import ChessEngine

    api_main.engine = ChessEngine(dummy_model)

    transport = ASGITransport(app=api_main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/training/suggest", json={
            "fen": STARTING_FEN,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "suggested_move" in data
        assert "explanation" in data
        assert "confidence" in data
        assert "alternatives" in data


@pytest.mark.asyncio
async def test_api_training_piece_info(dummy_model):
    os.environ["MODEL_PATH"] = dummy_model

    from httpx import AsyncClient, ASGITransport
    from src.api import main as api_main
    from src.api.engine import ChessEngine

    api_main.engine = ChessEngine(dummy_model)

    transport = ASGITransport(app=api_main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/training/piece-info", json={
            "fen": STARTING_FEN,
            "square": "e2",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["piece_name"] == "Pawn"
        assert data["piece_color"] == "white"
        assert len(data["legal_destinations"]) >= 2
        assert data["movement_rules"] != ""


@pytest.mark.asyncio
async def test_api_training_piece_info_empty_square(dummy_model):
    os.environ["MODEL_PATH"] = dummy_model

    from httpx import AsyncClient, ASGITransport
    from src.api import main as api_main
    from src.api.engine import ChessEngine

    api_main.engine = ChessEngine(dummy_model)

    transport = ASGITransport(app=api_main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/training/piece-info", json={
            "fen": STARTING_FEN,
            "square": "e4",
        })
        assert resp.status_code == 400
