"""Test the causal chain engine on a real bug scenario."""
import sys
sys.path.insert(0, r"C:\Users\Utilisateur\Documents\NeuralDBG")
import torch, torch.nn as nn
from neuraldbg import NeuralDbg

torch.manual_seed(42)

# Create a model with known bug: exploding gradients from huge inputs
model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
opt = torch.optim.SGD(model.parameters(), lr=0.1)
x_explode = torch.randn(4, 8) * 100  # huge inputs cause explosion
target = torch.randn(4, 2)

print("=" * 60)
print("CAUSAL CHAIN ENGINE — Live Test")
print("=" * 60)

with NeuralDbg(model) as dbg:
    for step in range(10):
        opt.zero_grad()
        loss = nn.MSELoss()(model(x_explode), target)
        loss.backward()
        dbg.step_iteration()
        dbg.record_loss(loss.item())
        opt.step()

    events = dbg.dump_events()
    print(f"\nTotal events: {len(events)}")

    # Flat hypotheses
    print(f"\n--- Flat hypotheses (current) ---")
    for h in dbg.explain_failure()[:3]:
        print(f"  [{h.confidence:.2f}] {h.description[:120]}")

    # Causal chains
    print(f"\n--- Causal chains (NEW) ---")
    chains = dbg.explain_causal()
    print(f"  Chains found: {len(chains)}")
    for i, c in enumerate(chains[:5]):
        print(f"\n  Chain {i+1} (conf={c.confidence:.3f}, len={c.length}):")
        print(f"    Root cause: {c.root_cause}")
        print(f"    Final symptom: {c.final_symptom}")
        print(f"    Path: {c.description[:160]}")
        for link in c.links[:4]:
            src_st = link.source_event.get('to_state','?')
            tgt_st = link.target_event.get('to_state','?')
            print(f"    [{link.rule}] {link.source_event.get('event_type')}({link.source_event.get('layer_name')})@{link.source_event.get('step')}[{src_st}] -> {link.target_event.get('event_type')}({link.target_event.get('layer_name')})@{link.target_event.get('step')}[{tgt_st}]")

print(f"\nDone. Causal chain engine operational.")
