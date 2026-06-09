## Sample Independence Violation in `nn.LSTM` on CUDA (pytorch#173334)

Hi — I want to flag a fundamental contract violation in `nn.LSTM` on CUDA that
makes batched inference silently produce NaN where individual-sample inference
produces a valid result.

### Repro
(see the linked repro for the full script + bundle; the synthetic version
that triggers the same edge case without the original `bundle.pt` is
available at: `examples/repro_pytorch_173334.py`)

With `nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True)`
on CUDA, an input tensor where **one** sample has values near `3.40e+38` (valid
float32, no NaN/Inf) produces:

- **Single-sample** `lstm(x[1:2])` → valid output, `max ≈ 0.995`
- **Batched** `lstm(x)` → `out[1:2]` is NaN, `max = nan`

The input is identical in both cases. The expected behavior per PyTorch docs
is "slight differences" — not "valid sample becomes NaN inside a batch."

### Why this matters

The proposed workaround is to per-sample-loop, which is 10–50× slower and
breaks the whole point of using a GPU. Worse: the same workload passes unit
tests (which usually run small per-sample inputs) and silently produces
NaN-only training runs in production. This is the worst kind of bug:
undetectable without a sample-independence test.

### Suggested upstream fix

Three possible directions, in order of preference:

1. **Detect and warn** — at the start of the LSTM forward, scan the input
   for values with magnitude > some threshold (e.g. 1e30) and either:
   - raise a `UserWarning` with a pointer to the workaround, or
   - automatically fall back to per-sample computation (slower but correct)

2. **Harden cuDNN handle** — request `CUDNN_DATA_PARALLEL` or batch size
   of 1 from cuDNN when inputs exceed the safe range. (Harder, requires
   understanding the cuDNN contract for `cudnnRNNForward` on
   numerically-extreme inputs.)

3. **Document the limitation** — at least add a clear "do not use with
   inputs > 1e30 in batched mode" to the LSTM docstring, so users
   know to validate or normalize their inputs.

### What we built

We added a detector (`examples/repro_pytorch_173334.py`) that synthesizes
the failure pattern from scratch (no GPU required for the script) and
emits a `sample_independence_violation` event. The fix path is
`apply_lstm_per_sample_inference` in our `neural-agent` package: a
drop-in replacement that loops over samples when edge values are
detected, falling back to batched mode otherwise.

If a maintainer is interested, I can open a draft PR for the detection
warning (option 1) — it would be ~30 lines of Python.

Thanks for looking at this.
