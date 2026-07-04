"""Quick RNN validation with improved injectors."""
import sys
sys.path.insert(0, r"C:\Users\Utilisateur\Documents\NeuralDBG")
from validate_combinatorial import *

print("RNN test with improved bug injectors...")
configs = rnn_configs(10)
results = []
for cfg in configs:
    r = evaluate_config(cfg)
    d = r.get("detected", 0)
    t = r.get("total", 0)
    print(f"  {cfg.name:40s} | base={r.get('baseline',-1):3d} | {d}/{t}")
    results.append(r)

total_d = sum(r.get("detected", 0) for r in results)
total_t = sum(r.get("total", 0) for r in results)
pct = 100 * total_d // max(total_t, 1)
print(f"\nRNN improved: {total_d}/{total_t} ({pct}%)")
print("Before: 117/240 (49%)")
print(f"Delta: {pct - 49:+d}%")
