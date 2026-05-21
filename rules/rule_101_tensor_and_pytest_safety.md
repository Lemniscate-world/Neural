# RULE 101: Tensor Operations and Test Suite Warning Governance

## Rule

ML/DL training hooks, anomaly tracking, and test suites must be defensive against
data-type crashes and warning noise.

## Implementation Standards

### 1. PyTorch Tensor Dtype Guarding

When monitoring activations or gradients:
- Check `torch.is_floating_point(t)` before float-only statistics such as
  `.mean()`, `.std()`, `.var()`, and epsilon comparisons.
- Use dtype-aware epsilon values to avoid underflow. Use about `1e-4` for
  float16/bfloat16 and about `1e-9` for float32/float64.
- Skip statistical computations on non-floating tensors such as token indices,
  labels, masks, or class IDs.

### 2. Pytest Warning Filters

Keep CI logs actionable:
- Filter third-party deprecation noise in `pyproject.toml` or `pytest.ini`.
- Keep internal warnings visible unless they are intentionally tested.

## Verification

When modifying tensor monitoring:
1. Verify floating-point checks guard every float-only operation.
2. Verify dtype-aware epsilon handling for float16 and bfloat16.
3. Run tests and confirm warning output is minimal and actionable.

## Enforcement

If training crashes on non-floating tensors, add a `torch.is_floating_point`
guard before the operation. If third-party warnings obscure CI output, add a
specific warning filter instead of hiding all warnings.
