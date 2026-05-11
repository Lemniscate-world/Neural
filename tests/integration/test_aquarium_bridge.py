import torch
import torch.nn as nn
import os
import subprocess
import pytest
from neuraldbg import NeuralDbg

# Paths
SCHEMA_PATH = "neuraldbg/schema/events.json"
VALIDATOR_PATH = "infrastructure/scripts/validate_schema.py"
OUTPUT_DIR = "outputs/test_bridge"


def run_validator(data_path):
    cmd = [os.sys.executable, VALIDATOR_PATH, data_path, SCHEMA_PATH]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(autouse=True)
def cleanup_outputs():
    if os.path.exists(OUTPUT_DIR):
        import shutil

        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# SCENARIO A: Vanishing Gradients (Deep Tanh)
def test_scenario_vanishing_to_aquarium():
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.Tanh(),
        nn.Linear(10, 10),
        nn.Tanh(),
        nn.Linear(10, 10),
        nn.Tanh(),
        nn.Linear(10, 1),
    )

    # Initialize with very small weights to accelerate vanishing
    for p in model.parameters():
        p.data.fill_(0.01)

    x = torch.randn(5, 10)
    target = torch.randn(5, 1)

    with NeuralDbg(model, threshold_vanishing=1e-1) as dbg:
        for _ in range(5):
            dbg.step_iteration()
            out = model(x)
            loss = nn.MSELoss()(out, target)
            loss.backward()

        package_path = os.path.join(OUTPUT_DIR, "vanishing")
        dbg.export_aquarium_package(package_path)

        # Verify JSON exists and matches schema
        json_path = os.path.join(package_path, "events.json")
        assert os.path.exists(json_path)

        success, output = run_validator(json_path)
        assert success, f"Schema validation failed:\n{output}"


# SCENARIO B: Exploding Gradients (High Weight Init)
def test_scenario_exploding_to_aquarium():
    model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 1))

    # Very large weights to cause explosion
    for p in model.parameters():
        p.data.fill_(100.0)

    x = torch.randn(5, 10)
    target = torch.randn(5, 1)

    with NeuralDbg(model, threshold_exploding=1.0) as dbg:
        dbg.step_iteration()
        out = model(x)
        loss = nn.MSELoss()(out, target)
        loss.backward()

        package_path = os.path.join(OUTPUT_DIR, "exploding")
        dbg.export_aquarium_package(package_path)

        json_path = os.path.join(package_path, "events.json")
        success, output = run_validator(json_path)
        assert success, f"Schema validation failed:\n{output}"


# SCENARIO C: Dead ReLU Layer
def test_scenario_dead_relu_to_aquarium():
    class DeadModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(10, 10)
            self.relu = nn.ReLU()
            self.out = nn.Linear(10, 1)

            # Force ReLU to be dead by setting extreme negative bias
            self.lin.bias.data.fill_(-1000.0)

        def forward(self, x):
            return self.out(self.relu(self.lin(x)))

    model = DeadModel()
    x = torch.randn(5, 10)
    target = torch.randn(5, 1)

    with NeuralDbg(model) as dbg:
        dbg.step_iteration()
        out = model(x)
        loss = nn.MSELoss()(out, target)
        loss.backward()

        package_path = os.path.join(OUTPUT_DIR, "dead_relu")
        dbg.export_aquarium_package(package_path)

        json_path = os.path.join(package_path, "events.json")
        success, output = run_validator(json_path)
        assert success, f"Schema validation failed:\n{output}"
