"""
Unit tests for OOM Prevention and Memory Optimization features in NeuralDBG.
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
import pytest
from neuraldbg import NeuralDbg, EventType, DataHealth, TensorDiskCache


def test_tensor_disk_cache_basic():
    """Verify TensorDiskCache saves tensors and cleans them up properly."""
    cache = TensorDiskCache(cache_dir="artifacts/test_tensor_cache")
    t = torch.randn(5, 5)
    
    # Save tensor
    path_str = cache.save(t, prefix="test_tensor")
    path = Path(path_str)
    assert path.exists()
    
    # Reload and verify equality
    t_loaded = torch.load(path)
    assert torch.allclose(t, t_loaded)
    
    # Cleanup and verify deletion
    cache.cleanup()
    assert not path.exists()


def test_neuraldbg_context_cleanup():
    """Verify NeuralDbg context manager triggers cache cleanup."""
    model = nn.Linear(5, 2)
    with NeuralDbg(model) as dbg:
        t = torch.randn(10, 10)
        path_str = dbg.disk_cache.save(t, prefix="context_test")
        path = Path(path_str)
        assert path.exists()
    
    # After exiting context, file must be deleted
    assert not path.exists()


def test_anomaly_disk_offload():
    """Verify that anomalous tensors are written to disk cache."""
    model = nn.Linear(10, 5)
    dbg = NeuralDbg(model)
    dbg.step = 5

    # NaN input tensor
    nan_tensor = torch.tensor([[1.0, float("nan"), 3.0]])
    dbg._check_data_anomaly(nan_tensor, "layer1")

    # Find the data anomaly event
    anomaly_events = [
        e for e in dbg.events if e.event_type == EventType.DATA_ANOMALY
    ]
    assert len(anomaly_events) == 1
    event = anomaly_events[0]
    assert event.to_state == DataHealth.NAN_DETECTED.value
    
    # Verify path exists in metadata and on disk
    assert "tensor_cache_path" in event.metadata
    cache_path = Path(event.metadata["tensor_cache_path"])
    assert cache_path.exists()

    # Load and verify it is indeed the tensor with NaN
    loaded_tensor = torch.load(cache_path)
    assert torch.isnan(loaded_tensor).any()

    # Clean up
    dbg.disk_cache.cleanup()
    assert not cache_path.exists()


def test_compute_activation_stats_efficiency_and_correctness():
    """Verify our optimized activation stats compute correctly for different dtypes and shapes."""
    model = nn.Linear(5, 2)
    dbg = NeuralDbg(model)

    # Test cases: (tensor, expected_sparsity, expected_saturation)
    test_cases = [
        # Normal float32 tensor
        (torch.tensor([[0.0, 0.5], [0.98, -0.99]], dtype=torch.float32), 0.25, 0.5),
        # Float16 tensor
        (torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.96, 0.3]], dtype=torch.float16), 0.5, 1.0 / 6.0),
        # Bfloat16 tensor
        (torch.tensor([0.0, 0.99, -0.97], dtype=torch.bfloat16), 1.0 / 3.0, 2.0 / 3.0),
        # Empty tensor
        (torch.tensor([], dtype=torch.float32), 0.0, 0.0),
    ]

    for tensor, exp_sparsity, exp_sat in test_cases:
        stats = dbg._compute_activation_stats(tensor)
        if tensor.numel() == 0:
            assert stats["mean"] == 0.0
            assert stats["std"] == 0.0
            assert stats["sparsity"] == 0.0
            continue
            
        assert abs(stats["sparsity"] - exp_sparsity) < 1e-5
        assert abs(stats["saturation_ratio"] - exp_sat) < 1e-5
        
        # Verify correctness of standard stats
        eps_mean = 1e-3 if tensor.dtype in (torch.float16, torch.bfloat16) else 1e-5
        assert abs(stats["mean"] - tensor.float().mean().item()) < eps_mean
        assert abs(stats["min"] - tensor.float().min().item()) < eps_mean
        assert abs(stats["max"] - tensor.float().max().item()) < eps_mean
        eps_norm = 5e-3 if tensor.dtype in (torch.float16, torch.bfloat16) else 1e-4
        assert abs(stats["norm"] - tensor.float().norm().item()) < eps_norm
