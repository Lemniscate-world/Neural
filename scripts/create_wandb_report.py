"""Generate a W&B Report for the Fully Connected blog submission.

Usage:
    python scripts/create_wandb_report.py

Requires:
    pip install wandb neuraldbg torch

This script:
1. Runs a quick demo training with NeuralDBG + W&B
2. Logs diagnostic data to W&B
3. Prints instructions to create a Report from the logged data
"""

import torch
import torch.nn as nn
import wandb

from neuraldbg.integrations.wandb import NeuralDBGCallback


DEMO_PROJECT = "neuraldbg-ecosystem-demo"


class DemoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32),
            nn.Sigmoid(),
            nn.Linear(32, 32),
            nn.Sigmoid(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


def run_demo():
    wandb.init(project=DEMO_PROJECT, name="neuraldbg-demo")
    model = DemoModel()
    callback = NeuralDBGCallback(model, family="MLP", log_every_n_steps=25)

    with callback:
        for step in range(200):
            x = torch.randn(16, 16)
            loss = model(x).sum()
            loss.backward()
            callback.step(loss.item())

    report = callback.report()
    print(f"\nDiagnostic summary: {report['summary']}")
    print(f"Events: {report['total_events']}")
    print(f"Anomalies: {report['anomaly_events']}")
    print(f"Causal chains: {len(report['causal_chains'])}")

    wandb.finish()
    print(f"\nW&B run logged to project '{DEMO_PROJECT}'")
    print("\n--- Next step ---")
    print("1. Go to https://wandb.ai/YOUR_ORG/neuraldbg-ecosystem-demo")
    print("2. Open the run named 'neuraldbg-demo'")
    print("3. Click 'Create Report' in the top-right corner")
    print("4. Add the neuraldbg/* charts and tables to the report")
    print("5. Add the text from docs/posts/wandb_community.md")
    print("6. Click 'Publish' and share the report URL")
    print("7. Email the report URL to editor@wandb.com for review")


if __name__ == "__main__":
    run_demo()
