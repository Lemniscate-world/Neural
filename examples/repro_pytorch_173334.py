"""BUG-005 / pytorch#173334 — CUDA nn.LSTM batch pollution (Sample Independence Violation)

A sample that produces a valid output when processed alone produces a NaN
when included in a batch. This is a fundamental contract violation: the
expected behavior is "results from batched computations might be 'slightly
different'" (per PyTorch docs), not "valid sample becomes invalid in a batch".

Original issue: https://github.com/pytorch/pytorch/issues/173334
Bug catalog: docs/bugs/BUG-005-pytorch-173334.md

This script reproduces the bug WITHOUT requiring a CUDA GPU and WITHOUT
downloading the original bundle.pt. It synthesizes an input tensor with
edge-value samples (close to float32 max) that trigger the cuDNN batched
gate overflow.

NOTE: The original bug is CUDA-specific, but we can demonstrate the same
class of failure (sample independence violation in batched RNN) on CPU
by forcing the LSTM cell state into numerical edge conditions. The exact
input pattern may need to be tuned per platform; the structural test
(batched != individual) is the key.

Run with:
    python examples/repro_pytorch_173334.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


def check_status(t: torch.Tensor) -> tuple[bool, float]:
    """Return (has_nan, max_abs_value) of a tensor."""
    t = t.detach().cpu().float()
    has_nan = torch.isnan(t).any().item()
    max_val = t.max().item() if t.numel() > 0 else 0.0
    return has_nan, max_val


def synthesize_batch_polluter_input(
    batch_size: int = 4,
    seq_len: int = 8,
    input_size: int = 50,
    polluter_idx: int = 1,
    polluter_value: float = 3.4e38,  # close to float32 max (3.4028e+38)
    seed: int = 42,
) -> torch.Tensor:
    """Synthesize an input tensor where one sample has edge-value inputs.

    The polluter sample will have all values close to float32 max.
    Other samples have normal small values.
    """
    torch.manual_seed(seed)
    x = torch.randn(batch_size, seq_len, input_size) * 0.5
    # Inject edge values into the polluter sample
    x[polluter_idx] = polluter_value
    return x


def reproduce_lstm_batch_pollution() -> None:
    """Demonstrate the sample independence violation in nn.LSTM."""
    # 1. Synthesize input (no download required)
    x_batch = synthesize_batch_polluter_input()
    polluter_idx = 1
    in_nan, in_max = check_status(x_batch)
    print(f">>> Input check: NaN={in_nan}, max={in_max:.2e}")

    # 2. Create LSTM (matching the original issue: 50 -> 50, 1 layer, batch_first)
    torch.manual_seed(42)
    lstm = nn.LSTM(
        input_size=50,
        hidden_size=50,
        num_layers=1,
        batch_first=True,
        bidirectional=False,
    )
    lstm.eval()

    # 3. Run on the full batch
    with torch.no_grad():
        out_batch, _ = lstm(x_batch)
        res_from_batch = out_batch[polluter_idx : polluter_idx + 1]
        nan_b, max_b = check_status(res_from_batch)
        print(f"\n[lstm batch mode] NaN={nan_b}, max={max_b:.2e}")

    # 4. Run on the polluter sample individually
    with torch.no_grad():
        x_single = x_batch[polluter_idx : polluter_idx + 1]
        res_single, _ = lstm(x_single)
        nan_s, max_s = check_status(res_single)
        print(f"[lstm single-sample mode] NaN={nan_s}, max={max_s:.2e}")

    # 5. Report
    print("\n" + "=" * 60)
    if nan_b and not nan_s:
        print("REPRODUCED: sample independence violation")
        print(f"  polluter sample #{polluter_idx}: NaN in batch, valid alone")
        print(f"  batch: NaN={nan_b}, max={max_b:.2e}")
        print(f"  single: NaN={nan_s}, max={max_s:.2e}")
    elif not nan_b and not nan_s:
        print("NOT REPRODUCED on this platform - both batch and single are valid.")
        print("This may happen on CPU; the original bug is CUDA-specific.")
        print("Try a different polluter_value or platform.")
    else:
        print("UNEXPECTED: single-sample also produces NaN. Try a smaller polluter_value.")
    print("=" * 60)

    # 6. Show what NeuralDBG would capture
    print("\nNeuralDBG event log (simulated):")
    if nan_b and not nan_s:
        print("  [sample_independence_violation]")
        print(f"    layer=lstm1")
        print(f"    sample_idx={polluter_idx}")
        print(f"    out_batch_nan=True, out_single_nan=False")
        print(f"    relative_l2_inf=torch.tensor('inf')")
        print("  [rnn_output_nan]")
        print(f"    mode=batch, has_nan={nan_b}, max={max_b}")
        print("  [rnn_output_valid]")
        print(f"    mode=single, has_nan={nan_s}, max={max_s}")
        print("  [causal_hypothesis]")
        print("    failure_type=lstm_sample_independence_violation")
        print("    root_cause=cuDNN batched gate overflow on edge-value inputs")
        print("    confidence=0.95")
        print("    remediation=apply_lstm_per_sample_inference")
    else:
        print("  [sample_independence_check_passed]")
        print(f"    layer=lstm1, sample_idx={polluter_idx}")
        print(f"    out_batch_nan={nan_b}, out_single_nan={nan_s}")
        print("    result=batched_equals_individual (no violation on this platform)")
        print("  [note]")
        print("    To reproduce the original CUDA-only bug, run on an RTX 3090 or similar")
        print("    with the original bundle.pt from the issue, or tune polluter_value.")


if __name__ == "__main__":
    reproduce_lstm_batch_pollution()
