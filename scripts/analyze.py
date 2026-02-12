"""Deep analysis of training history to diagnose issues and suggest improvements."""

import json
import sys

def main():
    with open("analytics_output/training_history.json") as f:
        data = json.load(f)
    
    iters = data["iterations"]
    
    print("=" * 85)
    print("TRAINING ANALYTICS - DEEP ANALYSIS")
    print("=" * 85)
    
    # Per-generation summary
    print("\n--- PER-GENERATION SUMMARY ---")
    header = f"{'Gen':>4} {'PLoss':>7} {'VLoss':>8} {'LR':>10} {'W':>3} {'B':>3} {'D':>3} {'Dec%':>5} {'AvgLen':>6} {'Buffer':>6} {'NewPos':>6}"
    print(header)
    print("-" * 85)
    
    for rec in iters:
        g = rec["generation"]
        sp = rec["self_play"]
        dec = sp["white_wins"] + sp["black_wins"]
        dec_pct = dec / sp["games"] * 100
        print(
            f"{g:4d} {rec['policy_loss']:7.4f} {rec['value_loss']:8.6f} "
            f"{rec['learning_rate']:10.7f} {sp['white_wins']:3d} {sp['black_wins']:3d} "
            f"{sp['draws']:3d} {dec_pct:5.1f} {sp['avg_game_length']:6.1f} "
            f"{rec['buffer_size']:6d} {rec['positions_added']:6d}"
        )
    
    # Game length distribution
    print("\n--- GAME LENGTH ANALYSIS ---")
    for rec in iters:
        g = rec["generation"]
        details = rec["self_play"]["game_details"]
        
        short = sum(1 for d in details if d["length"] < 50)
        medium = sum(1 for d in details if 50 <= d["length"] < 100)
        long_g = sum(1 for d in details if d["length"] >= 100)
        
        win_lengths = [d["length"] for d in details if d["result"] != "draw"]
        draw_lengths = [d["length"] for d in details if d["result"] == "draw"]
        
        avg_win = sum(win_lengths) / len(win_lengths) if win_lengths else 0
        avg_draw = sum(draw_lengths) / len(draw_lengths) if draw_lengths else 0
        
        print(
            f"Gen {g:2d}: Short(<50)={short:2d}  Med(50-100)={medium:2d}  "
            f"Long(100+)={long_g:2d}  | AvgDecisive={avg_win:5.1f}  AvgDraw={avg_draw:5.1f}"
        )
    
    # Value head diagnosis
    print("\n--- VALUE HEAD DIAGNOSIS ---")
    for rec in iters:
        g = rec["generation"]
        sp = rec["self_play"]
        dec = sp["white_wins"] + sp["black_wins"]
        vloss = rec["value_loss"]
        if vloss < 0.01:
            status = "CRITICAL - predicting all draws (near-zero loss = zero gradient)"
        elif vloss < 0.05:
            status = "LOW - weak win/loss signal"
        elif vloss < 0.2:
            status = "HEALTHY - learning from wins/losses"
        else:
            status = "HIGH - still learning basic patterns"
        print(f"Gen {g:2d}: value_loss={vloss:.6f}  decisive={dec:2d}/30 -> {status}")
    
    # Policy loss curve
    print("\n--- POLICY LEARNING CURVE ---")
    losses = [(r["generation"], r["policy_loss"]) for r in iters]
    for i in range(1, len(losses)):
        delta = losses[i][1] - losses[i - 1][1]
        arrow = "IMPROVING" if delta < -0.05 else "STALLED" if abs(delta) < 0.05 else "REGRESSING"
        print(
            f"Gen {losses[i-1][0]:2d} -> Gen {losses[i][0]:2d}: "
            f"{losses[i-1][1]:.4f} -> {losses[i][1]:.4f} ({delta:+.4f}) [{arrow}]"
        )
    
    # Draw pattern analysis
    print("\n--- DRAW PATTERN ANALYSIS ---")
    for rec in iters:
        g = rec["generation"]
        details = rec["self_play"]["game_details"]
        draws = [d for d in details if d["result"] == "draw"]
        if draws:
            very_short = sum(1 for d in draws if d["length"] < 40)
            short_d = sum(1 for d in draws if 40 <= d["length"] < 60)
            medium_d = sum(1 for d in draws if 60 <= d["length"] < 100)
            long_d = sum(1 for d in draws if d["length"] >= 100)
            print(
                f"Gen {g:2d}: {len(draws):2d} draws -> "
                f"VeryShort(<40)={very_short}  Short(40-60)={short_d}  "
                f"Med(60-100)={medium_d}  Long(100+)={long_d}"
            )
    
    # Training efficiency
    print("\n--- TRAINING EFFICIENCY ---")
    total_games = sum(r["self_play"]["games"] for r in iters)
    total_decisive = sum(
        r["self_play"]["white_wins"] + r["self_play"]["black_wins"] for r in iters
    )
    total_positions = sum(r["positions_added"] for r in iters)
    final = iters[-1]
    
    print(f"Total games played:     {total_games}")
    print(f"Total decisive games:   {total_decisive} ({total_decisive/total_games*100:.1f}%)")
    print(f"Total positions added:  {total_positions}")
    print(f"Final buffer size:      {final['buffer_size']}")
    print(f"Total training steps:   {final['total_training_steps']}")
    print(f"Steps/position ratio:   {final['total_training_steps']/total_positions:.3f}")
    print(f"Positions/step ratio:   {total_positions/final['total_training_steps']:.1f}")
    
    # Key problems summary
    print("\n" + "=" * 85)
    print("DIAGNOSIS SUMMARY")
    print("=" * 85)
    
    # Check value loss trend
    vloss_start = iters[0]["value_loss"] if len(iters) > 0 else 0
    vloss_end = iters[-1]["value_loss"]
    
    # Check decisive game trend
    early_decisive = sum(
        r["self_play"]["white_wins"] + r["self_play"]["black_wins"]
        for r in iters[:3]
    ) / min(3, len(iters))
    late_decisive = sum(
        r["self_play"]["white_wins"] + r["self_play"]["black_wins"]
        for r in iters[-3:]
    ) / min(3, len(iters))
    
    # Check policy stagnation
    ploss_last3 = [r["policy_loss"] for r in iters[-3:]]
    ploss_range = max(ploss_last3) - min(ploss_last3)
    
    print(f"\n1. VALUE HEAD COLLAPSE:")
    print(f"   Value loss: {vloss_start:.6f} -> {vloss_end:.6f}")
    print(f"   Signal: {'COLLAPSED' if vloss_end < 0.01 else 'WEAK' if vloss_end < 0.05 else 'OK'}")
    print(f"   Problem: Model predicts ~0 (draw) for ALL positions = zero gradient for value head")
    
    print(f"\n2. DRAW DEATH SPIRAL:")
    print(f"   Early decisive avg: {early_decisive:.1f}/30 games")
    print(f"   Late decisive avg:  {late_decisive:.1f}/30 games")
    print(f"   Trend: {'COLLAPSING' if late_decisive < early_decisive * 0.5 else 'DECLINING' if late_decisive < early_decisive else 'STABLE'}")
    print(f"   Problem: Draws produce value=0 training signal -> model learns to draw -> more draws")
    
    print(f"\n3. POLICY STAGNATION:")
    print(f"   Last 3 gen policy range: {ploss_range:.4f}")
    print(f"   Status: {'STALLED' if ploss_range < 0.02 else 'STILL LEARNING'}")
    print(f"   Problem: Policy can't improve without quality (decisive) training data")
    
    print(f"\n4. BUFFER COMPOSITION:")
    # Estimate draw positions in buffer
    total_draw_positions = 0
    total_all_positions = 0
    for rec in iters:
        for game in rec["self_play"]["game_details"]:
            total_all_positions += game["positions"]
            if game["result"] == "draw":
                total_draw_positions += game["positions"]
    draw_pct = total_draw_positions / total_all_positions * 100 if total_all_positions else 0
    print(f"   Draw positions in buffer: ~{draw_pct:.1f}% (target: <60%)")
    print(f"   Problem: Buffer flooded with draw data, washing out win/loss signals")


if __name__ == "__main__":
    main()
