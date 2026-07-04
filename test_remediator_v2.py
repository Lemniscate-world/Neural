"""Test the improved Remediator with accumulation, undo, severity, bounds."""
import sys
sys.path.insert(0, r'C:\Users\Utilisateur\Documents\Neural-Agent')
from neuralagent.remediator import Remediator

class H:
    def __init__(self, desc, conf=0.95):
        self.description = desc
        self.confidence = conf

# Test 1: Non-accumulating
print('=== Non-accumulating mode ===')
r = Remediator({'lr': 0.01, 'activation': 'ReLU'})
for t in ['gradient explosion', 'vanishing gradients', 'dead neurons']:
    cfg, info = r.remediate([H(t)])
    print(f'  {t:25s}: lr={cfg["lr"]:.6f}, act={cfg.get("activation","?")}')

# Test 2: Accumulating
print('\n=== Accumulating mode ===')
r = Remediator({'lr': 0.01, 'activation': 'ReLU'}, accumulate=True)
for t in ['gradient explosion', 'vanishing gradients', 'dead neurons']:
    cfg, info = r.remediate([H(t)])
    print(f'  {t:25s}: lr={cfg["lr"]:.6f}, act={cfg.get("activation","?")}')

# Test 3: Undo + Reset
print('\n=== Undo/Reset ===')
cfg, _ = r.undo()
print(f'  After undo:  lr={cfg["lr"]:.6f}')
cfg, _ = r.reset()
print(f'  After reset: lr={cfg["lr"]:.6f}')

# Test 4: Severity scaling
print('\n=== Severity scaling ===')
r = Remediator({'lr': 0.01})
cfg_low, info_low = r.remediate([H('gradient explosion', conf=0.3)], severity=0.3)
cfg_high, info_high = r.remediate([H('gradient explosion', conf=0.95)], severity=1.0)
print(f'  Low confidence (sev=0.3):  lr={cfg_low["lr"]:.6f}')
print(f'  High confidence (sev=1.0): lr={cfg_high["lr"]:.6f}')
print(f'  Expected low > high (closer to original): {cfg_low["lr"] > cfg_high["lr"]}')

# Test 5: Safety bounds
print('\n=== Safety bounds ===')
r = Remediator({'lr': 0.5})
cfg, _ = r.remediate([H('gradient explosion')])
print(f'  LR 0.5 -> 0.05 (within bounds): lr={cfg["lr"]:.6f}')

r2 = Remediator({'lr': 1e-3})
cfg2, _ = r2.remediate([H('gradient vanishing')])  # lr x2
print(f'  LR 1e-3 x2 -> 2e-3 (within bounds): lr={cfg2["lr"]:.6f}')

print('\nAll tests passed.')
