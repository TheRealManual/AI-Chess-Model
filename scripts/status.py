import json

with open('analytics_output/training_history.json') as f:
    h = json.load(f)

gens = h['generations']
pl = h['policy_losses']
vl = h['value_losses']
iters = h.get('iterations', [])

print(f"Generations completed: {len(gens)}")
print(f"Started: {h['session_start']}")
print(f"Updated: {h['last_updated']}")
print()
print(f"{'Gen':>4} {'Policy':>8} {'Value':>8} {'Total':>8} {'Games':>6} {'AvgLen':>8} {'W':>3} {'B':>3} {'D':>3} {'SPTime':>8}")
print('-' * 72)
for it in iters:
    g = it['generation']
    sp = it.get('self_play', {})
    w = sp.get('white_wins', 0)
    b = sp.get('black_wins', 0)
    d = sp.get('draws', 0)
    avg_len = sp.get('avg_game_length', 0)
    games = sp.get('games', 0)
    print(f"{g:4d} {it['policy_loss']:8.4f} {it['value_loss']:8.4f} {it['total_loss']:8.4f} {games:6d} {avg_len:8.1f} {w:3d} {b:3d} {d:3d} {it['self_play_time_s']:7.0f}s")

print()
total_time = sum(it['total_time_s'] for it in iters)
print(f"Total time so far: {total_time/3600:.1f} hours")
avg_per_gen = total_time / len(iters) if iters else 0
remaining = 30 - len(gens)
print(f"Avg per gen: {avg_per_gen/60:.1f} min")
print(f"Remaining: {remaining} gens, ETA ~{remaining * avg_per_gen / 3600:.1f} hours")
