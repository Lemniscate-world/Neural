"""Test the Remediator with various NeuralDBG hypotheses."""
import sys
sys.path.insert(0, r'C:\Users\Utilisateur\Documents\Neural-Agent')
from neuralagent.remediator import Remediator

config = {'lr': 0.01, 'activation': 'ReLU', 'clip_grad_norm': 0}
r = Remediator(config)

tests = [
    'gradient explosion detected at Linear_0 step 100',
    'data_anomaly distribution_shift at Embedding',
    'dead neurons in ReLU_3 layer',
    'vanishing gradients at LayerNorm_2',
    'saturated activations in root layer',
    'optimizer instability diverging at step 200',
    'some completely unknown bug pattern',
]

class H:
    def __init__(self, desc):
        self.description = desc
        self.confidence = 0.95

for t in tests:
    new_config, info = r.remediate([H(t)])
    changes = []
    for k, v in new_config.items():
        if config.get(k) != v:
            changes.append(f'{k}: {config.get(k)} -> {v}')
    print(f'Input:  {t}')
    print(f'Config: {", ".join(changes) if changes else "no changes"}')
    print(f'Info:   {info[:100]}')
    print()
