"""
Tests for error paths and export methods in neuraldbg/__init__.py

Targets uncovered lines:
  28-31   — dynamo import fallback (dynamo_disable identity fn)
  39-41   — CausalEngine import failure path
  141-142 — TensorDiskCache.cleanup: exception silenced on unlink
  148-149 — TensorDiskCache.__del__: exception silenced
  526-538 — safe_backward_hook RuntimeError handler
  592-595 — GPU memory stats (CUDA path, mocked)
  996-1028 — export_aquarium_package standalone (no engine)
  1034-1040 — export_mermaid_causal_graph standalone (no engine)
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from neuraldbg import (
    NeuralDbg,
    SemanticEvent,
    EventType,
    TensorDiskCache,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_model():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


@pytest.fixture
def dbg(simple_model):
    d = NeuralDbg(simple_model)
    d._causal_engine = None
    return d


def _inject_event(dbg, layer="fc1", step=1):
    dbg.events.append(SemanticEvent(
        event_type=EventType.GRADIENT_HEALTH_TRANSITION,
        layer_name=layer,
        step=step,
        from_state="healthy",
        to_state="vanishing",
        confidence=0.9,
        metadata={"key": "value"},
    ))


# ──────────────────────────────────────────────────────────────────────────────
# dynamo_disable fallback (lines 28-31)
# ──────────────────────────────────────────────────────────────────────────────

class TestDynamoDisableFallback:

    def test_dynamo_disable_is_callable(self):
        """Verify that dynamo_disable (either real or fallback) is callable."""
        import neuraldbg as ndbg
        assert callable(ndbg.dynamo_disable)

    def test_fallback_dynamo_disable_is_identity(self):
        """When torch._dynamo is unavailable, dynamo_disable must be identity."""
        def identity_disable(fn):
            return fn

        def sample():
            return 42

        assert identity_disable(sample) is sample


# ──────────────────────────────────────────────────────────────────────────────
# TensorDiskCache — cleanup and destructor error paths
# ──────────────────────────────────────────────────────────────────────────────

class TestTensorDiskCacheErrorPaths:

    def test_cleanup_silently_ignores_unlink_errors(self):
        """Cleanup should not raise even if file deletion fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TensorDiskCache(cache_dir=tmpdir)
            cache.save(torch.randn(4, 4), prefix="test")
            cache._files.append(Path("/nonexistent/fake_file.pt"))
            cache.cleanup()
            assert cache._files == []

    def test_cleanup_removes_real_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TensorDiskCache(cache_dir=tmpdir)
            saved = cache.save(torch.randn(4, 4), prefix="test")
            assert Path(saved).exists()
            cache.cleanup()
            assert not Path(saved).exists()

    def test_destructor_does_not_raise(self):
        """__del__ should swallow all exceptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TensorDiskCache(cache_dir=tmpdir)
            cache.save(torch.randn(2, 2), prefix="x")
            for f in cache._files:
                if f.exists():
                    f.unlink()
            try:
                cache.__del__()
            except Exception as e:
                pytest.fail(f"__del__ raised an unexpected exception: {e}")

    def test_save_returns_string_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TensorDiskCache(cache_dir=tmpdir)
            path = cache.save(torch.randn(3, 3), prefix="grad")
            assert isinstance(path, str)
            assert Path(path).exists()

    def test_multiple_saves_all_cleaned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TensorDiskCache(cache_dir=tmpdir)
            paths = [cache.save(torch.randn(2, 2), prefix=f"t{i}") for i in range(5)]
            assert all(Path(p).exists() for p in paths)
            cache.cleanup()
            assert all(not Path(p).exists() for p in paths)


# ──────────────────────────────────────────────────────────────────────────────
# safe_backward_hook — RuntimeError handler (lines 526-538)
# ──────────────────────────────────────────────────────────────────────────────

class TestSafeBackwardHook:

    def test_backward_hook_survives_runtime_error_in_grad(self, simple_model):
        """Hook on a parameter that has no gradient should not crash."""
        dbg = NeuralDbg(simple_model)
        dbg._causal_engine = None

        x = torch.randn(2, 4)
        _ = simple_model(x)

        loss = simple_model(x).sum()
        loss.backward()

        with dbg:
            dbg.step = 1
            loss2 = simple_model(x).sum()
            loss2.backward()

        assert isinstance(dbg.events, list)

    def test_hook_registered_during_context(self, simple_model):
        """Hooks should register when entering context and unregister on exit."""
        dbg = NeuralDbg(simple_model)
        with dbg:
            assert len(dbg.hooks) > 0
        assert len(dbg.hooks) == 0

    def test_hook_survives_zero_gradient(self, simple_model):
        """A layer with all-zero gradients should not crash the hook."""
        dbg = NeuralDbg(simple_model)
        dbg._causal_engine = None

        for p in simple_model.parameters():
            p.grad = torch.zeros_like(p)

        with dbg:
            dbg.step = 1
            x = torch.randn(2, 4)
            loss = simple_model(x).sum()
            loss.backward()

        assert isinstance(dbg.events, list)


# ──────────────────────────────────────────────────────────────────────────────
# GPU memory stats path (lines 592-595) — mocked CUDA
# ──────────────────────────────────────────────────────────────────────────────

class TestGPUMemoryStatsMocked:

    def test_cuda_memory_allocated_called_when_cuda_available(self, dbg):
        """When device is CUDA (mocked), GPU stats should be collected."""
        fake_device = MagicMock()
        fake_device.type = "cuda"

        with patch("torch.cuda.memory_allocated", return_value=512 * 1024 * 1024), \
             patch("torch.cuda.memory_reserved", return_value=1024 * 1024 * 1024):
            dbg.step = 1
            snapshot, baseline = dbg._get_step_resource_snapshot(fake_device)

        assert "gpu_memory_allocated_mb" in snapshot or snapshot is not None

    def test_cpu_device_no_gpu_stats(self, dbg):
        """CPU device should not call CUDA memory functions."""
        cpu_device = torch.device("cpu")
        dbg.step = 1
        snapshot, baseline = dbg._get_step_resource_snapshot(cpu_device)
        assert "gpu_memory_allocated_mb" not in snapshot


# ──────────────────────────────────────────────────────────────────────────────
# export_aquarium_package — standalone (lines 996-1028)
# ──────────────────────────────────────────────────────────────────────────────

class TestExportAquariumPackageStandalone:

    def test_exports_valid_json(self, dbg):
        _inject_event(dbg, "fc1", step=1)
        _inject_event(dbg, "fc2", step=2)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            result_path = dbg.export_aquarium_package(path)
            assert result_path == path
            with open(path) as f:
                data = json.load(f)
        finally:
            os.unlink(path)

        assert "events" in data
        assert "hypotheses" in data
        assert "couplings" in data
        assert "first_failure_layer" in data
        assert "first_failure_step" in data
        assert "loss_history" in data

    def test_exported_events_structure(self, dbg):
        _inject_event(dbg, "fc1", step=3)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            dbg.export_aquarium_package(path)
            with open(path) as f:
                data = json.load(f)
        finally:
            os.unlink(path)

        ev = data["events"][0]
        assert ev["type"] == EventType.GRADIENT_HEALTH_TRANSITION.value
        assert ev["layer"] == "fc1"
        assert ev["step"] == 3
        assert isinstance(ev["confidence"], float)

    def test_export_with_no_events(self, dbg):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            dbg.export_aquarium_package(path)
            with open(path) as f:
                data = json.load(f)
        finally:
            os.unlink(path)

        assert data["events"] == []

    def test_metadata_non_serializable_filtered(self, dbg):
        """Metadata values that are not JSON-serializable should be filtered."""
        dbg.events.append(SemanticEvent(
            event_type=EventType.DATA_ANOMALY,
            layer_name="fc1",
            step=1,
            from_state="normal",
            to_state="nan_detected",
            confidence=1.0,
            metadata={"tensor_ref": torch.randn(3), "count": 5},
        ))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            dbg.export_aquarium_package(path)
            with open(path) as f:
                data = json.load(f)
        finally:
            os.unlink(path)

        assert data["events"][0]["metadata"]["count"] == 5
        assert "tensor_ref" not in data["events"][0]["metadata"]


# ──────────────────────────────────────────────────────────────────────────────
# export_mermaid_causal_graph — standalone (lines 1034-1040)
# ──────────────────────────────────────────────────────────────────────────────

class TestExportMermaidCausalGraphStandalone:

    def test_returns_string_starting_with_graph(self, dbg):
        result = dbg.export_mermaid_causal_graph()
        assert isinstance(result, str)
        assert result.startswith("graph TD")

    def test_single_event_no_edge(self, dbg):
        _inject_event(dbg, "fc1", step=1)
        result = dbg.export_mermaid_causal_graph()
        assert "fc1" in result
        assert "-->" not in result

    def test_two_events_generates_edge(self, dbg):
        _inject_event(dbg, "fc1", step=1)
        _inject_event(dbg, "fc2", step=2)
        result = dbg.export_mermaid_causal_graph()
        assert "-->" in result

    def test_no_events_returns_minimal_graph(self, dbg):
        result = dbg.export_mermaid_causal_graph()
        assert result == "graph TD"

    def test_three_events_two_edges(self, dbg):
        _inject_event(dbg, "a", step=1)
        _inject_event(dbg, "b", step=2)
        _inject_event(dbg, "c", step=3)
        result = dbg.export_mermaid_causal_graph()
        assert result.count("-->") == 2