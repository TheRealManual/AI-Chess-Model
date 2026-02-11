import sys
import os
import signal
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.config import TrainingConfig
from src.training.trainer import Trainer
from src.training.self_play import request_stop, reset_stop
from src.benchmark.stockfish_bench import run_benchmark
from src.analytics.plots import generate_all_plots

_shutting_down = False


def handle_interrupt(sig, frame):
    global _shutting_down
    if _shutting_down:
        print('\nForce quit.')
        sys.exit(1)
    _shutting_down = True
    print('\n\nCtrl+C detected — stopping after current games finish...')
    print('Press Ctrl+C again to force quit immediately.')
    request_stop()


def launch_watcher(analytics_dir, python_exe):
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'watch.py')
    live_file = os.path.join(analytics_dir, 'live_games.json')

    if os.name == 'nt':
        cmd = f'start "Chess Live Viewer" "{python_exe}" "{script}" --file "{live_file}"'
        subprocess.Popen(cmd, shell=True)
    else:
        subprocess.Popen(
            ['x-terminal-emulator', '-e', python_exe, script, '--file', live_file],
            start_new_session=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Train the chess model via self-play")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--games-per-iter", type=int, default=500)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--parallel-games", type=int, default=0)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--training-steps", type=int, default=1000)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--benchmark-every", type=int, default=10)
    parser.add_argument("--benchmark-games", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--analytics-dir", type=str, default="analytics_output")
    parser.add_argument("--no-opening-book", action="store_true", help="Disable random opening book")
    parser.add_argument("--watch", action="store_true", help="Open a live game viewer in a new terminal")
    args = parser.parse_args()

    config = TrainingConfig(
        num_sims=args.sims,
        games_per_iter=args.games_per_iter,
        parallel_games=args.parallel_games,
        batch_size=args.batch_size,
        lr=args.lr,
        training_steps_per_iter=args.training_steps,
        buffer_capacity=args.buffer_size,
        use_amp=args.use_amp and not args.no_amp,
        use_opening_book=not args.no_opening_book,
        benchmark_every=args.benchmark_every,
        benchmark_games=args.benchmark_games,
        checkpoint_dir=args.checkpoint_dir,
        analytics_dir=args.analytics_dir,
    )

    config.print_summary()

    signal.signal(signal.SIGINT, handle_interrupt)

    if args.watch:
        launch_watcher(config.analytics_dir, sys.executable)
        print("Launched live game viewer in a new terminal.\n")

    trainer = Trainer(config)
    trainer.load_latest_checkpoint()

    start_iter = trainer.generation + 1

    for i in range(start_iter, start_iter + args.iterations):
        if _shutting_down:
            print(f'\nStopped at generation {trainer.generation}.')
            break

        reset_stop()
        trainer.run_iteration(i)

        if _shutting_down:
            print(f'\nStopped at generation {trainer.generation}.')
            break

        # periodic benchmarking
        if config.benchmark_every > 0 and i % config.benchmark_every == 0:
            try:
                ckpt_file = os.path.join(
                    config.checkpoint_dir,
                    f'gen_{trainer.generation:04d}.pt',
                )
                run_benchmark(
                    trainer.model, trainer.device, config,
                    generation=trainer.generation,
                    checkpoint_path=ckpt_file,
                )
                trainer.history['benchmark_results'].append(trainer.generation)
            except Exception as e:
                print(f"  Benchmark failed: {e}")

        # regenerate plots
        try:
            generate_all_plots(config.analytics_dir)
        except Exception as e:
            print(f"  Plot generation failed: {e}")

    print(f"\nTraining complete. Final generation: {trainer.generation}")
    print(f"Checkpoints in: {config.checkpoint_dir}/")
    print(f"Analytics in: {config.analytics_dir}/")


if __name__ == "__main__":
    main()
