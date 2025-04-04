import pytest
import torch
import numpy as np
from execution_optimization.execution import get_device, run_inference

@pytest.mark.parametrize("device_str, expected_type", [
    ("cpu", "cpu"),
    ("gpu", "cuda" if torch.cuda.is_available() else "cpu"),
    ("auto", "cuda" if torch.cuda.is_available() else "cpu"),
    ("cuda:0", "cuda"),
    ("cuda:1", "cuda" if torch.cuda.device_count() > 1 else "cpu"),
    ("tpu", "cpu"),  # Assuming TPU not available in test env, falls back to CPU
])
def test_get_device(device_str, expected_type):
    """Test device selection logic with various device strings."""
    device = get_device(device_str)
    assert device.type == expected_type
    
    # For CUDA devices with index, check the index when available
    if device_str.startswith("cuda:") and torch.cuda.is_available():
        index = int(device_str.split(":")[1])
        if index < torch.cuda.device_count():
            assert device.index == index

@pytest.mark.parametrize("execution_config", [
    {"device": "cpu"},
    {"device": "gpu"},
    {"device": "auto"},
    {"device": "cuda:0"},
    {},  # Test default behavior
])
def test_run_inference_device_config(execution_config):
    """Test that run_inference respects the device configuration."""
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2)
    )
    data = torch.randn(1, 10)
    
    output = run_inference(model, data, execution_config)
    
    # Verify output shape and type
    assert output.shape == (1, 2)
    assert output.device.type == "cpu"  # Output should be on CPU regardless of compute device

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_multi_gpu_execution():
    """Test execution across multiple GPUs if available."""
    if torch.cuda.device_count() < 2:
        pytest.skip("Need at least 2 GPUs for this test")
    
    # Create a model that spans multiple GPUs
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5).to("cuda:0"),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2).to("cuda:1")
    )
    data = torch.randn(1, 10)
    
    # This should handle the multi-GPU scenario
    execution_config = {"device": "auto"}
    output = run_inference(model, data, execution_config)
    
    assert output.shape == (1, 2)
    assert output.device.type == "cpu"  # Output should be on CPU

@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_batch_processing(batch_size):
    """Test inference with different batch sizes."""
    model = torch.nn.Linear(10, 5)
    data = torch.randn(batch_size, 10)
    execution_config = {"device": "auto"}
    
    output = run_inference(model, data, execution_config)
    assert output.shape == (batch_size, 5)
