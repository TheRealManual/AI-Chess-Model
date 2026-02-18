import os
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_history(analytics_dir):
    path = os.path.join(analytics_dir, 'training_history.json')
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
    ax.plot(gens[:len(history['policy_losses'])], history['policy_losses'], label='Policy Loss', linewidth=2)
    ax.plot(gens[:len(history['value_losses'])], history['value_losses'], label='Value Loss', linewidth=2)
    ax.plot(gens[:len(history['total_losses'])], history['total_losses'], label='Total Loss', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_self_play_stats(history, out_dir):
    """Plot self-play win rates and game lengths over generations."""
    stats = history.get('self_play_stats', [])
    gens = history['generations']
    if not stats:
        return

    n = min(len(gens), len(stats))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

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
    ax1.set_title('Self-Play Game Outcomes')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # average game length
    avg_lengths = [s.get('avg_game_length', 0) for s in stats[:n]]
    ax2.plot(gens[:n], avg_lengths, color='#2196F3', linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Moves')
    ax2.set_title('Average Game Length')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'self_play_stats.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_policy_entropy(history, out_dir):
    """Plot policy entropy over generations (higher = more exploration)."""
    gens = history['generations']
    entropy = history.get('policy_entropy', [])
    if not entropy:
        return

    # filter out NaN values
    valid = [(g, e) for g, e in zip(gens, entropy)
             if isinstance(e, (int, float)) and not math.isnan(e)]
    if not valid:
        return

    valid_gens, valid_entropy = zip(*valid)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(valid_gens, valid_entropy, color='#FF9800', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Entropy (nats)')
    ax.set_title('Policy Entropy')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'policy_entropy.png'), dpi=150, bbox_inches='tight')
    plt.close()


def generate_all_plots(analytics_dir):
    """Generate all static plots from training history."""
    os.makedirs(analytics_dir, exist_ok=True)

    history = load_history(analytics_dir)

    if history:
        plot_training_losses(history, analytics_dir)
        plot_self_play_stats(history, analytics_dir)
        plot_policy_entropy(history, analytics_dir)

    print(f"Plots saved to {analytics_dir}/")
