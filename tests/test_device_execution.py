import pytest
import torch
from execution_optimization.execution import run_inference

@pytest.mark.parametrize("device", ["cpu", "gpu", "auto"])
def test_device_selection(device):
    model = torch.nn.Linear(10, 5)
    data = torch.randn(1, 10)
    execution_config = {"device": device}

    output = run_inference(model, data, execution_config)
    assert output.shape == (1, 5)
