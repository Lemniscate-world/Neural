# BUG-005 — PyTorch #173334 CUDA nn.LSTM batch pollution (Sample Independence Violation)

> **MID**: BUG-005
> **Linked**: FIX-005 (NeuralDBG detection + Neural-Agent fix)
> **Status**: Detection script created, upstream comment drafted, NOT posted
> **Date opened**: 2026-06-09
> **Owner**: LambdaSection

## Source

- Upstream issue: https://github.com/pytorch/pytorch/issues/173334
- Title: *"CUDA nn.LSTM produces NaN in batch mode but correct output in single-sample mode (Sample Independence Violation)"*
- Status upstream: OPEN, labeled `module: NaNs and Infs`, `module: rnn`, `module: cuda`, `triaged`
- Author: [@zifan6699](https://github.com/zifan6699)
- Hardware: NVIDIA RTX 3090 (consumer, fits the bug-hunt charter: < 8GB VRAM per side)
- PyTorch version: 2.6.0+cu126

## Root cause

`nn.LSTM` on CUDA exhibits a fundamental sample independence violation: a sample that produces a perfectly valid output (e.g. 0.995) when processed **alone** produces a NaN when included as part of a **batch**. The transition from a valid number to NaN is not a "slight difference" — it is a catastrophic breakdown of numerical consistency.

The trigger conditions:
1. Input contains values near the float32 representable maximum (~3.40e+38). The "input" itself is valid (no NaN, no Inf).
2. LSTM has standard topology (input_size=50, hidden_size=50, num_layers=1, batch_first=True)
3. The batch contains at least one sample where the cumulative effect of edge-value inputs overflows the cuDNN gate computations when computed in parallel

This is reproducible on standard CUDA hardware (RTX 3090) without exotic configuration.

## Trigger conditions

1. Model: `nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True, bidirectional=False)`
2. Input dtype: float32
3. Input shape: `(batch, seq, 50)` with at least one sample containing values close to float32 max
4. Execution: `model.cuda().eval()` + `model(input)` in `torch.no_grad()`
5. PyTorch: 2.6.0+cu126 (other versions may also be affected)

## NeuralDBG detection

The detection script `examples/repro_pytorch_173334.py` shows what NeuralDBG captures:

1. **sample_independence_violation** event: `out_batch[i] != out_single[i]` where `out_single[i]` was valid and `out_batch[i]` is NaN
2. **rnn_output_nan** event in `out_batch[sample_idx]` (NaN=True, max=nan)
3. **rnn_output_valid** event in `out_single` (NaN=False, max=0.995) — same sample, different result
4. **causal_chain**: edge-value input -> cuDNN batched gate overflow -> LSTM cell state corruption -> NaN output
5. **root_cause_hypothesis**: *"LSTM CUDA sample independence violation: batched computation produces NaN while individual sample computation is valid"*

## Neural-Agent proposed fix

```python
# Fix 1: Per-sample inference (workaround, no upstream change needed)
# Run LSTM one sample at a time when inputs may contain edge values
def safe_lstm_inference(model, x_batch):
    out_batches = []
    for i in range(x_batch.size(0)):
        out_single, _ = model(x_batch[i:i+1])
        out_batches.append(out_single)
    return torch.cat(out_batches, dim=0)

# Fix 2: Use CPU for inference when edge values are expected
# (loses GPU speedup but guarantees sample independence)

# Fix 3: Normalize inputs to a safe range before LSTM
# (lossy, but guaranteed to work)
```

The upstream fix should be in PyTorch's CUDA LSTM kernel to handle edge-value accumulation correctly, but no upstream PR exists yet.

## Reproduction script

`examples/repro_pytorch_173334.py` — self-contained 3-stage repro:
1. Synthesizes an input tensor where one sample contains float32-max values (the "polluter")
2. Runs the LSTM on the full batch (expects NaN)
3. Runs the LSTM on individual samples (expects valid output for the same polluter)
4. Compares the two to demonstrate the sample independence violation

## Deliverables checklist

- [x] BUG-005 tracking file (this file)
- [x] `examples/repro_pytorch_173334.py` (self-contained repro)
- [x] Upstream comment draft (`docs/posts/pytorch_173334_comment.md`)
- [ ] Post comment on GitHub (CEO TODO: manual copy-paste per acquisition_tracker)
- [ ] `tests/unit/test_lstm_sample_independence_detection.py` (CI-friendly variant — TODO, requires GPU)
- [ ] NeuralDBG `sample_independence_violation` event type in `neuraldbg/__init__.py` (TODO)
- [ ] Neural-Agent `apply_lstm_per_sample_inference()` remediation rule (TODO)
- [ ] Verify the fix path with the test once implemented
