import os
import time
import asyncio
import platform
import threading
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

from src.api.engine import ChessEngine
from src.api.schemas import (
    MoveRequest, MoveResponse,
    EvalRequest, EvalResponse,
    AnalyzeRequest, AnalyzeResponse,
    MoveScore,
    TrainingAnalyzeMoveRequest, TrainingAnalyzeMoveResponse,
    TrainingSuggestRequest, TrainingSuggestResponse,
    PieceInfoRequest, PieceInfoResponse, LegalDestination,
)

# ---------------------------------------------------------------------------
# Metrics tracker
# ---------------------------------------------------------------------------


class MetricsTracker:
    """Thread-safe request metrics collector."""

    def __init__(self):
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.total_requests = 0
        self.active_requests = 0
        self.peak_active = 0
        self.endpoint_hits: dict[str, int] = defaultdict(int)
        self.endpoint_errors: dict[str, int] = defaultdict(int)
        self.endpoint_total_ms: dict[str, float] = defaultdict(float)
        self.status_codes: dict[int, int] = defaultdict(int)
        self.move_requests = 0  # dedicated counter for chess-move requests
        self.move_difficulties: dict[str, int] = defaultdict(int)

    # --- mutators (called from middleware) --------------------------------
    def request_started(self, path: str):
        with self._lock:
            self.total_requests += 1
            self.active_requests += 1
            self.peak_active = max(self.peak_active, self.active_requests)
            self.endpoint_hits[path] += 1

    def request_finished(self, path: str, status: int, elapsed_ms: float):
        with self._lock:
            self.active_requests = max(self.active_requests - 1, 0)
            self.status_codes[status] += 1
            self.endpoint_total_ms[path] += elapsed_ms
            if status >= 400:
                self.endpoint_errors[path] += 1

    def record_move(self, difficulty: str = "medium"):
        with self._lock:
            self.move_requests += 1
            self.move_difficulties[difficulty] += 1

    # --- readers ----------------------------------------------------------
    def snapshot(self) -> dict:
        now = time.time()
        uptime_s = now - self.start_time

        proc = psutil.Process()
        mem = proc.memory_info()

        with self._lock:
            endpoints = []
            for path in sorted(self.endpoint_hits):
                hits = self.endpoint_hits[path]
                avg_ms = (self.endpoint_total_ms[path] / hits) if hits else 0
                endpoints.append({
                    "path": path,
                    "requests": hits,
                    "errors": self.endpoint_errors.get(path, 0),
                    "avg_response_ms": round(avg_ms, 1),
                })

            data = {
                "uptime_seconds": round(uptime_s, 1),
                "uptime_human": _fmt_duration(uptime_s),
                "started_at": datetime.fromtimestamp(
                    self.start_time, tz=timezone.utc
                ).isoformat(),
                "system": {
                    "cpu_percent": psutil.cpu_percent(interval=0),
                    "cpu_count": psutil.cpu_count(),
                    "memory_total_mb": round(
                        psutil.virtual_memory().total / 1048576
                    ),
                    "memory_used_mb": round(
                        psutil.virtual_memory().used / 1048576
                    ),
                    "memory_percent": psutil.virtual_memory().percent,
                    "process_memory_mb": round(mem.rss / 1048576, 1),
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                },
                "requests": {
                    "total": self.total_requests,
                    "active": self.active_requests,
                    "peak_active": self.peak_active,
                    "chess_moves_served": self.move_requests,
                    "move_difficulties": dict(self.move_difficulties),
                    "status_codes": dict(self.status_codes),
                },
                "endpoints": endpoints,
            }
        return data


def _fmt_duration(seconds: float) -> str:
    d = int(seconds // 86400)
    h = int((seconds % 86400) // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts = []
    if d:
        parts.append(f"{d}d")
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

engine: ChessEngine = None
metrics = MetricsTracker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    model_path = os.environ.get("MODEL_PATH", "chess_model.onnx")
    print(f"Loading model from {model_path}")
    engine = ChessEngine(model_path)
    print("Engine ready")
    yield
    print("Shutting down")


app = FastAPI(title="Chess AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Metrics middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    path = request.url.path
    metrics.request_started(path)
    t0 = time.time()
    try:
        response = await call_next(request)
    except Exception:
        metrics.request_finished(path, 500, (time.time() - t0) * 1000)
        raise
    elapsed_ms = (time.time() - t0) * 1000
    metrics.request_finished(path, response.status_code, elapsed_ms)
    return response


# ---------------------------------------------------------------------------
# Dashboard & metrics endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Live metrics dashboard - auto-refreshes every 5 seconds."""
    return _DASHBOARD_HTML


@app.get("/play", response_class=HTMLResponse)
async def play():
    """Serve the chess UI for playing against the AI."""
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _chess_ui = os.path.join(_project_root, "scripts", "chess_ui.html")
    if os.path.isfile(_chess_ui):
        return FileResponse(_chess_ui, media_type="text/html")
    raise HTTPException(status_code=404, detail="Chess UI not found")


@app.get("/api/metrics")
async def get_metrics():
    """Raw JSON metrics for programmatic access."""
    data = metrics.snapshot()
    data["model_loaded"] = engine is not None
    return JSONResponse(data)


@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": engine is not None}


@app.post("/api/move", response_model=MoveResponse)
async def get_move(req: MoveRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    metrics.record_move(req.difficulty)

    loop = asyncio.get_event_loop()
    t0 = time.time()
    move, value, confidence = await loop.run_in_executor(
        None, engine.get_move, req.fen, req.difficulty
    )
    elapsed = (time.time() - t0) * 1000

    return MoveResponse(
        move=move,
        value=round(value, 4),
        confidence=round(confidence, 4),
        think_time_ms=round(elapsed, 1),
    )


@app.post("/api/evaluate", response_model=EvalResponse)
async def evaluate(req: EvalRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    loop = asyncio.get_event_loop()
    value, top_moves = await loop.run_in_executor(
        None, engine.evaluate, req.fen
    )

    return EvalResponse(
        value=round(value, 4),
        top_moves=[
            MoveScore(
                move=m['move'], visits=m['visits'],
                score=round(m['score'], 4),
            )
            for m in top_moves
        ],
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    loop = asyncio.get_event_loop()
    value, all_moves, total_sims = await loop.run_in_executor(
        None, engine.analyze, req.fen, req.num_sims
    )

    return AnalyzeResponse(
        value=round(value, 4),
        moves=[
            MoveScore(
                move=m['move'], visits=m['visits'],
                score=round(m['score'], 4),
            )
            for m in all_moves
        ],
        total_simulations=total_sims,
    )


# ---------------------------------------------------------------------------
# Training mode endpoints
# ---------------------------------------------------------------------------


@app.post("/api/training/analyze-move", response_model=TrainingAnalyzeMoveResponse)
async def training_analyze_move(req: TrainingAnalyzeMoveRequest):
    """Analyze a player's move: rate it, explain why, and suggest a better one."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, engine.analyze_player_move, req.fen, req.player_move
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return TrainingAnalyzeMoveResponse(
        player_move=result['player_move'],
        player_move_rank=result['player_move_rank'],
        player_move_score=round(result['player_move_score'], 4),
        best_move=result['best_move'],
        best_move_score=round(result['best_move_score'], 4),
        rating=result['rating'],
        explanation=result['explanation'],
        suggestion=result['suggestion'],
        value_before=round(result['value_before'], 4),
        value_after=round(result['value_after'], 4),
        top_moves=[
            MoveScore(
                move=m['move'], visits=m['visits'],
                score=round(m['score'], 4),
            )
            for m in result['top_moves']
        ],
    )


@app.post("/api/training/suggest", response_model=TrainingSuggestResponse)
async def training_suggest(req: TrainingSuggestRequest):
    """Suggest the best move for the player with an explanation."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, engine.suggest_move, req.fen
    )

    return TrainingSuggestResponse(
        suggested_move=result['suggested_move'],
        explanation=result['explanation'],
        confidence=round(result['confidence'], 4),
        value=round(result['value'], 4),
        alternatives=[
            MoveScore(
                move=m['move'], visits=m['visits'],
                score=round(m['score'], 4),
            )
            for m in result['alternatives']
        ],
    )


@app.post("/api/training/piece-info", response_model=PieceInfoResponse)
async def training_piece_info(req: PieceInfoRequest):
    """Get piece info and legal destination squares for a selected piece."""
    try:
        result = ChessEngine.get_piece_info(req.fen, req.square)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PieceInfoResponse(
        piece_name=result['piece_name'],
        piece_color=result['piece_color'],
        square=result['square'],
        movement_rules=result['movement_rules'],
        legal_destinations=[
            LegalDestination(square=d['square'], is_capture=d['is_capture'])
            for d in result['legal_destinations']
        ],
    )


# ---------------------------------------------------------------------------
# HTML dashboard template
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chess AI &ndash; Metrics Dashboard</title>
<style>
  :root {
    --bg: #0f1117; --card: #1a1d27; --accent: #58a6ff;
    --green: #3fb950; --yellow: #d29922; --red: #f85149;
    --text: #e6edf3; --muted: #8b949e; --border: #30363d;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
    background: var(--bg); color: var(--text);
    padding: 24px; min-height: 100vh;
  }
  h1 { font-size: 1.5rem; margin-bottom: 4px; }
  .subtitle { color: var(--muted); font-size: .85rem; margin-bottom: 20px; }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 16px; margin-bottom: 24px;
  }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 18px 20px;
  }
  .card-label {
    font-size: .75rem; text-transform: uppercase;
    letter-spacing: .08em; color: var(--muted); margin-bottom: 6px;
  }
  .card-value { font-size: 1.6rem; font-weight: 700; }
  .card-detail { font-size: .8rem; color: var(--muted); margin-top: 4px; }
  .bar-bg {
    width: 100%; height: 8px; background: var(--border);
    border-radius: 4px; margin-top: 10px; overflow: hidden;
  }
  .bar-fill {
    height: 100%; border-radius: 4px; transition: width .6s ease;
  }
  .bar-cpu  { background: var(--accent); }
  .bar-mem  { background: var(--green); }
  table {
    width: 100%; border-collapse: collapse; font-size: .85rem;
  }
  th {
    text-align: left; color: var(--muted); font-weight: 600;
    padding: 8px 12px; border-bottom: 1px solid var(--border);
  }
  td { padding: 8px 12px; border-bottom: 1px solid var(--border); }
  tr:last-child td { border-bottom: none; }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: .75rem; font-weight: 600;
  }
  .badge-ok  { background: #0d3321; color: var(--green); }
  .badge-err { background: #3d1418; color: var(--red); }
  .badge-warn { background: #3b2607; color: var(--yellow); }
  .refresh {
    font-size: .75rem; color: var(--muted);
    text-align: center; margin-top: 16px;
  }
  #status-dot {
    display: inline-block; width: 10px; height: 10px;
    border-radius: 50%; margin-right: 6px; vertical-align: middle;
  }
  .dot-ok  { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .dot-err { background: var(--red);   box-shadow: 0 0 6px var(--red); }
  .section-title {
    font-size: 1rem; margin: 20px 0 10px; color: var(--accent);
  }
  .diff-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 8px;
  }
  .diff-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 14px; text-align: center;
  }
  .diff-card .num { font-size: 1.3rem; font-weight: 700; }
  .diff-card .lbl {
    font-size: .7rem; color: var(--muted); text-transform: uppercase;
  }
</style>
</head>
<body>
<h1>
  <span id="status-dot" class="dot-ok"></span>
  Chess AI &ndash; Metrics Dashboard
</h1>
<p class="subtitle">
  AWS App Runner &middot; Live metrics &middot; auto-refresh 5 s
</p>

<div class="grid" id="cards"></div>

<div class="section-title">Chess Move Requests</div>
<div id="moves-section" class="diff-grid" style="margin-bottom:20px"></div>

<div class="section-title">Endpoint Breakdown</div>
<div class="card" style="overflow-x:auto">
  <table>
    <thead>
      <tr><th>Path</th><th>Requests</th><th>Errors</th><th>Avg (ms)</th></tr>
    </thead>
    <tbody id="ep-body"></tbody>
  </table>
</div>

<div class="section-title">Status Codes</div>
<div id="codes-section" class="diff-grid" style="margin-bottom:10px"></div>

<p class="refresh" id="refresh-line">Updating&hellip;</p>

<script>
async function fetchMetrics() {
  try {
    const r = await fetch('/api/metrics');
    const d = await r.json();
    render(d);
  } catch(e) {
    document.getElementById('status-dot').className = 'dot-err';
  }
}

function render(d) {
  const dot = document.getElementById('status-dot');
  dot.className = d.model_loaded ? 'dot-ok' : 'dot-err';

  const cards = [
    {label:'Uptime',         value:d.uptime_human,
     detail:'Since '+new Date(d.started_at).toLocaleString()},
    {label:'Model',          value:d.model_loaded?'Loaded':'Not loaded',
     detail:d.system.platform},
    {label:'Total Requests', value:d.requests.total.toLocaleString(),
     detail:'Peak concurrent: '+d.requests.peak_active},
    {label:'Active Now',     value:d.requests.active,
     detail:'Concurrent in-flight requests'},
    {label:'CPU Usage',      value:d.system.cpu_percent+'%',
     detail:d.system.cpu_count+' cores',
     bar:d.system.cpu_percent, barClass:'bar-cpu'},
    {label:'System Memory',  value:d.system.memory_percent+'%',
     detail:d.system.memory_used_mb+' / '+d.system.memory_total_mb+' MB',
     bar:d.system.memory_percent, barClass:'bar-mem'},
    {label:'Process Memory', value:d.system.process_memory_mb+' MB',
     detail:'Resident set size'},
    {label:'Python',         value:d.system.python, detail:''},
  ];
  let html = '';
  for (const c of cards) {
    html += '<div class="card">'
      +'<div class="card-label">'+c.label+'</div>'
      +'<div class="card-value">'+c.value+'</div>'
      +'<div class="card-detail">'+c.detail+'</div>'
      +(c.bar!==undefined
        ? '<div class="bar-bg"><div class="bar-fill '+c.barClass
          +'" style="width:'+c.bar+'%"></div></div>'
        : '')
      +'</div>';
  }
  document.getElementById('cards').innerHTML = html;

  // Chess moves
  let mhtml = '<div class="diff-card"><div class="num">'
    +d.requests.chess_moves_served
    +'</div><div class="lbl">Total Moves</div></div>';
  for (const [diff, cnt] of Object.entries(d.requests.move_difficulties||{})) {
    mhtml += '<div class="diff-card"><div class="num">'+cnt
      +'</div><div class="lbl">'+diff+'</div></div>';
  }
  document.getElementById('moves-section').innerHTML = mhtml;

  // Endpoints table
  let rows = '';
  for (const ep of d.endpoints) {
    const errBadge = ep.errors > 0
      ? '<span class="badge badge-err">'+ep.errors+'</span>'
      : '<span class="badge badge-ok">0</span>';
    rows += '<tr><td>'+ep.path+'</td><td>'+ep.requests
      +'</td><td>'+errBadge+'</td><td>'+ep.avg_response_ms+'</td></tr>';
  }
  document.getElementById('ep-body').innerHTML = rows
    || '<tr><td colspan="4" style="color:var(--muted)">No requests yet</td></tr>';

  // Status codes
  let chtml = '';
  for (const [code, cnt] of Object.entries(d.requests.status_codes||{})) {
    const cls = code<400 ? 'badge-ok' : code<500 ? 'badge-warn' : 'badge-err';
    chtml += '<div class="diff-card"><div class="num">'
      +'<span class="badge '+cls+'">'+code+'</span> '+cnt
      +'</div><div class="lbl">responses</div></div>';
  }
  document.getElementById('codes-section').innerHTML = chtml
    || '<p style="color:var(--muted);font-size:.85rem">No responses yet</p>';

  document.getElementById('refresh-line').textContent =
    'Last updated '+new Date().toLocaleTimeString()+' -- refreshes every 5 s';
}

fetchMetrics();
setInterval(fetchMetrics, 5000);
</script>
</body>
</html>
"""
