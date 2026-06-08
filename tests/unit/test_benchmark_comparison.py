"""Tests for benchmark_public/comparison.py — naive logger vs NeuralDBG."""

from benchmark_public.comparison import (
    compare_on_mha_bug,
    naive_loss_logger,
)


def test_naive_logger_misses_mha_bug():
    """W&B-style loss logger sees finite loss, no failure detected."""
    out = naive_loss_logger(
        "mha_fully_masked_row_BUG-001",
        step_losses=[0.008, 0.007, 0.009],
        post_step_param_nan=[True, True, True],  # gradients are NaN!
    )
    assert out.detected_failure is False
    assert "NaN gradients" in out.notes
    assert "not visible" in out.notes


def test_naive_logger_detects_nan_loss():
    """If loss itself goes NaN, the naive logger DOES detect it."""
    out = naive_loss_logger(
        "synthetic_nan_loss",
        step_losses=[0.5, 0.3, float("nan")],
        post_step_param_nan=[False, False, True],
    )
    assert out.detected_failure is True


def test_compare_on_mha_bug_neuraldbg_wins():
    """On the MHA bug, NeuralDBG (with FIX-001) detects, naive logger misses."""
    result = compare_on_mha_bug(
        step_losses=[0.008, 0.007, 0.009],
        post_step_param_nan=[True, True, True],
    )
    assert result["tools"]["neuraldbg_with_FIX-001"] == "DETECTED"
    assert result["tools"]["naive_loss_logger_(wandb_like)"] == "MISSED"
    assert result["winner"] == "neuraldbg_with_FIX-001"


def test_compare_on_mha_bug_tie_when_no_nan_grads():
    """If no NaN gradients, both tools are correct (no failure)."""
    result = compare_on_mha_bug(
        step_losses=[0.5, 0.3, 0.2],
        post_step_param_nan=[False, False, False],
    )
    assert result["tools"]["neuraldbg_with_FIX-001"] == "MISSED"
    assert result["tools"]["naive_loss_logger_(wandb_like)"] == "MISSED"
    assert result["winner"] == "tie"
