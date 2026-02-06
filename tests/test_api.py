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
