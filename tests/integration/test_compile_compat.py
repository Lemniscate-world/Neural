"""
Integration tests for torch.compile compatibility.

Tests that NeuralDbg operates correctly with compiled models.
"""

import torch
import torch.nn as nn
import pytest
from neuraldbg import NeuralDbg

# Skip tests if torch.compile not available or C++ headers missing (python3.x-dev)
try:
    import sys
    import sysconfig
    import os

    _python_h = os.path.join(sysconfig.get_path("include"), "Python.h")
    compiled_available = (
        hasattr(torch, "compile")
        and sys.version_info < (3, 14)
        and os.path.exists(_python_h)
    )
except Exception:
    compiled_available = False


@pytest.mark.skipif(not compiled_available, reason="torch.compile not available")
class TestCompileCompatibility:
    """Tests for torch.compile compatibility."""

    def test_context_manager_with_compiled_model(self):
        """Test that NeuralDbg works with compiled model."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))

        # Compile the model
        compiled_model = torch.compile(model)

        # Should work as context manager
        with NeuralDbg(compiled_model) as dbg:
            assert dbg.is_monitoring
            assert len(dbg.hooks) > 0

    def test_forward_pass_with_compiled_model(self):
        """Test that forward pass works with compiled model."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        compiled_model = torch.compile(model)

        with NeuralDbg(compiled_model):
            # Run forward pass
            x = torch.randn(32, 10)
            output = compiled_model(x)

            # Should produce output
            assert output.shape == (32, 5)

            # Events may or may not be captured depending on compile behavior
            # The key is that it doesn't crash

    def test_backward_pass_with_compiled_model(self):
        """Test that backward pass works with compiled model."""
        model = nn.Sequential(nn.Linear(10, 20), nn.Tanh(), nn.Linear(20, 1))

        compiled_model = torch.compile(model)

        with NeuralDbg(compiled_model, threshold_vanishing=1e-4) as dbg:
            # Run training step
            x = torch.randn(16, 10)
            y = torch.randn(16, 1)

            output = compiled_model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()

            assert len(dbg.events) > 0

    def test_explain_failure_with_compiled_model(self):
        """Test that explain_failure works after training with compiled model."""
        model = nn.Sequential(
            nn.Linear(10, 50), nn.Tanh(), nn.Linear(50, 50), nn.Tanh(), nn.Linear(50, 1)
        )

        compiled_model = torch.compile(model)

        with NeuralDbg(compiled_model, threshold_vanishing=1e-3) as dbg:
            # Simulate training steps
            for step in range(10):
                x = torch.randn(8, 10) * 0.1
                y = torch.randn(8, 1) * 0.01

                output = compiled_model(x)
                loss = nn.MSELoss()(output, y)
                loss.backward()
                dbg.step = step

        # Should be able to query for explanations
        hypotheses = dbg.explain_failure("vanishing_gradients")
        # May or may not have hypotheses depending on compile behavior
        assert isinstance(hypotheses, list)


class TestNonCompiledModel:
    """Tests for non-compiled model (baseline)."""

    def test_standard_model_works(self):
        """Test that standard non-compiled model works."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        with NeuralDbg(model) as dbg:
            x = torch.randn(32, 10)
            output = model(x)
            loss = output.sum()
            loss.backward()

            assert len(dbg.events) > 0
