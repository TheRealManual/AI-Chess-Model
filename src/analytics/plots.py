import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_history(analytics_dir):
    path = os.path.join(analytics_dir, 'training_history.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def load_benchmarks(analytics_dir):
    path = os.path.join(analytics_dir, 'benchmark_results.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def plot_training_losses(history, out_dir):
    """Plot policy loss, value loss, and total loss over generations."""
    gens = history['generations']
    if not gens or not history.get('policy_losses'):
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens[:len(history['policy_losses'])], history['policy_losses'], label='Policy Loss')
    ax.plot(gens[:len(history['value_losses'])], history['value_losses'], label='Value Loss')
    ax.plot(gens[:len(history['total_losses'])], history['total_losses'], label='Total Loss', linestyle='--')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss_curves.png'), dpi=150)
    plt.close()


def plot_self_play_stats(history, out_dir):
    """Plot self-play win rates and game lengths over generations."""
    stats = history.get('self_play_stats', [])
    gens = history['generations']
    if not stats:
        return

    n = min(len(gens), len(stats))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # win/draw/loss rates
    white_wins = [s.get('white_wins', 0) for s in stats[:n]]
    black_wins = [s.get('black_wins', 0) for s in stats[:n]]
    draws = [s.get('draws', 0) for s in stats[:n]]
    totals = [w + b + d for w, b, d in zip(white_wins, black_wins, draws)]

    w_rate = [w / max(t, 1) for w, t in zip(white_wins, totals)]
    b_rate = [b / max(t, 1) for b, t in zip(black_wins, totals)]
    d_rate = [d / max(t, 1) for d, t in zip(draws, totals)]

    ax1.stackplot(gens[:n], w_rate, d_rate, b_rate,
                  labels=['White wins', 'Draws', 'Black wins'],
                  colors=['#4CAF50', '#9E9E9E', '#F44336'], alpha=0.7)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Rate')
    ax1.set_title('Self-Play Results')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # average game length
    avg_lengths = [s.get('avg_game_length', 0) for s in stats[:n]]
    ax2.plot(gens[:n], avg_lengths, color='#2196F3')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Moves')
    ax2.set_title('Average Game Length')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'self_play_stats.png'), dpi=150)
    plt.close()


def plot_policy_entropy(history, out_dir):
    """Plot policy entropy over generations (higher = more exploration)."""
    gens = history['generations']
    entropy = history.get('policy_entropy', [])
    if not entropy:
        return

    n = min(len(gens), len(entropy))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens[:n], entropy[:n], color='#FF9800')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Entropy (nats)')
    ax.set_title('Policy Entropy')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'policy_entropy.png'), dpi=150)
    plt.close()


def plot_elo_progression(benchmarks, out_dir):
    """Plot estimated ELO over generations from Stockfish benchmarks."""
    if not benchmarks:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # group by depth
    depth_data = {}
    for bench in benchmarks:
        gen = bench['generation']
        for depth_str, data in bench.get('depths', {}).items():
            depth = int(depth_str)
            if depth not in depth_data:
                depth_data[depth] = {'gens': [], 'elos': [], 'win_rates': []}
            depth_data[depth]['gens'].append(gen)
            depth_data[depth]['elos'].append(data['estimated_elo'])
            depth_data[depth]['win_rates'].append(data['win_rate'])

    for depth in sorted(depth_data.keys()):
        d = depth_data[depth]
        ax.plot(d['gens'], d['elos'], marker='o', label=f'vs SF depth {depth}')

    ax.set_xlabel('Generation')
    ax.set_ylabel('Estimated ELO')
    ax.set_title('ELO Progression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'elo_progression.png'), dpi=150)
    plt.close()


def plot_interactive_dashboard(history, benchmarks, out_dir):
    """Create an interactive Plotly HTML dashboard with all metrics."""
    rows = 3 if benchmarks else 2
    titles = ['Training Loss', 'Self-Play Results', 'Policy Entropy']
    if benchmarks:
        titles.append('ELO vs Stockfish')

    fig = make_subplots(rows=rows, cols=2, subplot_titles=titles[:rows*2])

    gens = history.get('generations', [])

    # loss curves
    if history.get('policy_losses'):
        n = min(len(gens), len(history['policy_losses']))
        fig.add_trace(go.Scatter(x=gens[:n], y=history['policy_losses'][:n], name='Policy Loss'), row=1, col=1)
        fig.add_trace(go.Scatter(x=gens[:n], y=history['value_losses'][:n], name='Value Loss'), row=1, col=1)
        fig.add_trace(go.Scatter(x=gens[:n], y=history['total_losses'][:n], name='Total', line=dict(dash='dash')), row=1, col=1)

    # self-play stats
    stats = history.get('self_play_stats', [])
    if stats:
        n = min(len(gens), len(stats))
        avg_lens = [s.get('avg_game_length', 0) for s in stats[:n]]
        fig.add_trace(go.Scatter(x=gens[:n], y=avg_lens, name='Avg Game Length'), row=1, col=2)

    # entropy
    entropy = history.get('policy_entropy', [])
    if entropy:
        n = min(len(gens), len(entropy))
        fig.add_trace(go.Scatter(x=gens[:n], y=entropy[:n], name='Policy Entropy'), row=2, col=1)

    # self-play win rates
    if stats:
        n = min(len(gens), len(stats))
        draws = [s.get('draws', 0) / max(s.get('games', 1), 1) for s in stats[:n]]
        fig.add_trace(go.Scatter(x=gens[:n], y=draws, name='Draw Rate'), row=2, col=2)

    # ELO
    if benchmarks and rows == 3:
        depth_data = {}
        for bench in benchmarks:
            gen = bench['generation']
            for d_str, data in bench.get('depths', {}).items():
                d = int(d_str)
                if d not in depth_data:
                    depth_data[d] = {'gens': [], 'elos': []}
                depth_data[d]['gens'].append(gen)
                depth_data[d]['elos'].append(data['estimated_elo'])

        for d in sorted(depth_data.keys()):
            dd = depth_data[d]
            fig.add_trace(go.Scatter(x=dd['gens'], y=dd['elos'], name=f'vs SF d{d}', mode='lines+markers'), row=3, col=1)

    fig.update_layout(height=300*rows, title_text='Training Dashboard', showlegend=True)
    fig.write_html(os.path.join(out_dir, 'dashboard.html'))


def generate_all_plots(analytics_dir):
    """Generate all static plots and the interactive dashboard."""
    os.makedirs(analytics_dir, exist_ok=True)

    history = load_history(analytics_dir)
    benchmarks = load_benchmarks(analytics_dir)

    if history:
        plot_training_losses(history, analytics_dir)
        plot_self_play_stats(history, analytics_dir)
        plot_policy_entropy(history, analytics_dir)

    if benchmarks:
        plot_elo_progression(benchmarks, analytics_dir)

    if history:
        plot_interactive_dashboard(history, benchmarks, analytics_dir)

    print(f"Plots saved to {analytics_dir}/")
