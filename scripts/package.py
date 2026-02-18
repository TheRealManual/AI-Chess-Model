"""Package an exported model into a standalone shareable folder.

Creates a self-contained directory that anyone can run with Python to
play against the chess AI in their browser. No GPU or training code needed.

Usage:
    python scripts/package.py exported_models/gen0048_v3_pretrained_gen48_20260217_223312
    python scripts/package.py exported_models/gen0048_... --output ~/Desktop/chess-ai
"""

import sys
import os
import shutil
import json
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Package an exported model into a standalone shareable app"
    )
    parser.add_argument("export_dir", type=str,
                        help="Path to an exported model directory (contains chess_model.onnx)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory. Defaults to packages/<export_name>")
    parser.add_argument("--name", type=str, default=None,
                        help="Custom name for the package folder")
    args = parser.parse_args()

    export_dir = os.path.normpath(args.export_dir)

    # validate the export directory
    onnx_path = os.path.join(export_dir, "chess_model.onnx")
    if not os.path.isfile(onnx_path):
        print(f"Error: No chess_model.onnx found in {export_dir}")
        print("Make sure you point to a valid exported model directory.")
        sys.exit(1)

    # read export info for metadata
    info_path = os.path.join(export_dir, "export_info.json")
    export_info = {}
    if os.path.isfile(info_path):
        with open(info_path) as f:
            export_info = json.load(f)

    export_name = os.path.basename(export_dir)
    package_name = args.name or f"chess-ai-{export_name}"

    if args.output:
        out_dir = os.path.normpath(args.output)
    else:
        out_dir = os.path.join(PROJECT_ROOT, "packages", package_name)

    if os.path.exists(out_dir):
        print(f"Output directory already exists: {out_dir}")
        print("Remove it first, or use --output to specify a different location.")
        sys.exit(1)

    print(f"Packaging: {export_dir}")
    print(f"Output:    {out_dir}")
    print()

    os.makedirs(out_dir, exist_ok=True)

    # --- copy the ONNX model ---
    shutil.copy2(onnx_path, os.path.join(out_dir, "chess_model.onnx"))
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  Copied chess_model.onnx ({onnx_size:.1f} MB)")

    # --- copy source modules needed for inference ---
    src_dst = os.path.join(out_dir, "src")
    os.makedirs(src_dst, exist_ok=True)
    _write_empty_init(src_dst)

    # src/model/ — network.py (encode_board, move mapping) and mcts.py
    model_dst = os.path.join(src_dst, "model")
    os.makedirs(model_dst, exist_ok=True)
    _write_empty_init(model_dst)
    shutil.copy2(os.path.join(PROJECT_ROOT, "src", "model", "network.py"), model_dst)
    shutil.copy2(os.path.join(PROJECT_ROOT, "src", "model", "mcts.py"), model_dst)
    print("  Copied src/model/ (network.py, mcts.py)")

    # src/api/ — main.py, engine.py, schemas.py
    api_dst = os.path.join(src_dst, "api")
    os.makedirs(api_dst, exist_ok=True)
    _write_empty_init(api_dst)
    shutil.copy2(os.path.join(PROJECT_ROOT, "src", "api", "engine.py"), api_dst)
    shutil.copy2(os.path.join(PROJECT_ROOT, "src", "api", "schemas.py"), api_dst)
    # write a modified main.py that serves the bundled index.html
    _write_package_main(api_dst)
    print("  Copied src/api/ (main.py, engine.py, schemas.py)")

    # --- write the HTML playground ---
    _write_index_html(out_dir)
    print("  Created index.html")

    # --- write the run script ---
    _write_run_script(out_dir)
    print("  Created run.py")

    # --- write requirements.txt ---
    _write_requirements(out_dir)
    print("  Created requirements.txt")

    # --- bundle dependencies into lib/ ---
    _install_deps(out_dir)
    print("  Bundled dependencies into lib/")

    # --- write README ---
    gen = export_info.get("generation", "?")
    _write_readme(out_dir, package_name, gen, export_info)
    print("  Created README.txt")

    # --- summary ---
    total_size = 0
    file_count = 0
    for root, dirs, files in os.walk(out_dir):
        for fname in files:
            total_size += os.path.getsize(os.path.join(root, fname))
            file_count += 1
    size_mb = total_size / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"Package ready: {out_dir}")
    print(f"Files: {file_count}  |  Size: {size_mb:.1f} MB")
    print(f"\nTo share: zip the folder and send it.")
    print(f"To run:   cd {os.path.basename(out_dir)} && python run.py")
    print(f"{'='*60}")


def _write_empty_init(directory):
    with open(os.path.join(directory, "__init__.py"), "w") as f:
        f.write("")


def _write_package_main(api_dir):
    """Write a simplified main.py for the package that serves index.html from the root."""
    src = os.path.join(PROJECT_ROOT, "src", "api", "main.py")
    with open(src) as f:
        content = f.read()

    # replace the playground mounting with root-level index.html serving
    old = '''# serve the playground web app if the folder exists
playground_dir = os.path.join(os.getcwd(), "playground")
if os.path.isdir(playground_dir):
    app.mount("/playground", StaticFiles(directory=playground_dir, html=True), name="playground")'''

    new = '''# serve the bundled web UI from the package root
_package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_index_html = os.path.join(_package_root, "index.html")

from fastapi.responses import FileResponse

@app.get("/")
async def serve_ui():
    return FileResponse(_index_html, media_type="text/html")'''

    if old in content:
        content = content.replace(old, new)
    else:
        # fallback: just append the route
        content += '''

# serve the bundled web UI
_package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_index_html = os.path.join(_package_root, "index.html")

from fastapi.responses import FileResponse

@app.get("/")
async def serve_ui():
    return FileResponse(_index_html, media_type="text/html")
'''

    with open(os.path.join(api_dir, "main.py"), "w") as f:
        f.write(content)


def _write_index_html(out_dir):
    """Copy the chess UI HTML template into the package."""
    src = os.path.join(PROJECT_ROOT, "scripts", "chess_ui.html")
    with open(src, encoding="utf-8") as f:
        content = f.read()

    # the existing HTML already handles this with the API_BASE logic,
    # but let's make sure it points to the root origin (not /playground)
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(content)


def _install_deps(out_dir):
    """Install all runtime dependencies into a lib/ folder inside the package."""
    lib_dir = os.path.join(out_dir, "lib")
    req_file = os.path.join(out_dir, "requirements.txt")
    print("  Installing dependencies (this may take a minute)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "--target", lib_dir,
         "-r", req_file,
         "--quiet", "--disable-pip-version-check"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  WARNING: pip install failed:\n{result.stderr}")
        print("  The package will still work if the user runs: pip install -r requirements.txt")
    else:
        # clean up __pycache__ and .dist-info to save space
        for root, dirs, files in os.walk(lib_dir):
            for d in dirs:
                if d == "__pycache__" or d.endswith(".dist-info"):
                    shutil.rmtree(os.path.join(root, d))


def _write_run_script(out_dir):
    content = '''"""Run the Chess AI server and open the browser to play."""

import os
import sys
import time
import webbrowser
import threading

def main():
    host = "127.0.0.1"
    port = 8000

    # find the ONNX model in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "chess_model.onnx")

    if not os.path.isfile(model_path):
        print("Error: chess_model.onnx not found next to run.py")
        sys.exit(1)

    os.environ["MODEL_PATH"] = model_path

    # add bundled dependencies and package root to Python path
    lib_dir = os.path.join(script_dir, "lib")
    if os.path.isdir(lib_dir):
        sys.path.insert(0, lib_dir)
    sys.path.insert(0, script_dir)

    print(f"Starting Chess AI server on http://{host}:{port}")
    print(f"Model: {model_path}")
    print()

    # open browser after a short delay
    def open_browser():
        time.sleep(1.5)
        url = f"http://{host}:{port}"
        print(f"Opening browser: {url}")
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
'''
    with open(os.path.join(out_dir, "run.py"), "w") as f:
        f.write(content)


def _write_requirements(out_dir):
    content = """# Chess AI runtime dependencies
fastapi>=0.104.0
uvicorn>=0.24.0
onnxruntime>=1.16.0
python-chess>=1.999
numpy>=1.24.0
pydantic>=2.0.0
"""
    with open(os.path.join(out_dir, "requirements.txt"), "w") as f:
        f.write(content)


def _write_readme(out_dir, name, generation, info):
    policy_loss = info.get("final_policy_loss", "N/A")
    value_loss = info.get("final_value_loss", "N/A")

    if isinstance(policy_loss, float):
        policy_loss = f"{policy_loss:.4f}"
    if isinstance(value_loss, float):
        value_loss = f"{value_loss:.6f}"

    content = f"""Chess AI - Play Against the Neural Network
==========================================

Model: Generation {generation}
Policy Loss: {policy_loss}
Value Loss:  {value_loss}

How to Run
----------
1. Install Python 3.10+ if you don't have it
2. Run the server:
       python run.py
3. Your browser will open automatically to play!

All dependencies are bundled in the lib/ folder — no pip install needed.

The AI server runs on http://127.0.0.1:8000.
Press Ctrl+C to stop the server.

Controls
--------
- Click and drag pieces to make your move
- Choose difficulty: Easy / Medium / Hard / Max
- Play as White or Black
- The AI uses Monte Carlo Tree Search with a neural network

About
-----
This chess AI was trained using an AlphaZero-inspired approach:
- Supervised pre-training on millions of Lichess games
- Self-play reinforcement learning for {generation} generations
- Neural network: ResNet with 6 blocks x 128 channels
- Search: MCTS with PUCT exploration

No GPU required to play - inference runs on CPU via ONNX Runtime.
"""
    with open(os.path.join(out_dir, "README.txt"), "w") as f:
        f.write(content)


if __name__ == "__main__":
    main()
