[D] What I learned building a debugger for PyTorch training loops — and how it changed how I think about failure diagnosis

Hey r/ML,

I spent the last few months building a tool that hooks into PyTorch training loops to automatically detect and localize failures (vanishing gradients, exploding gradients, data anomalies). Along the way, I learned some things about training failure diagnosis that might be useful even if you never use the tool.

## The key insight: most training failures are local, not global

When your loss spikes or vanishes, the natural instinct is to look at the loss curve. But the loss is a **global aggregate** — it tells you *something* went wrong, but not *where*.

In my testing across hundreds of synthetic failure scenarios, the actual root cause is almost always **localized to a specific layer at a specific step**:

- Vanishing gradients: the failure starts at the deepest layer with saturated activations, then propagates backward
- Exploding gradients: the failure starts at the layer with the highest gradient norm, then propagates forward
- Data anomalies: the failure starts at the input layer, then corrupts everything downstream

The trick is to monitor **per-layer gradient norms** and detect **transitions** (healthy → vanishing), not absolute values.

## What actually matters in gradient monitoring

Most people monitor:
- Loss over time (too global)
- Gradient histograms (too noisy, too much data)
- Weight norms (slow to change, lagging indicator)

What I found works best:
- **Gradient norm transitions**: "Linear_3 went from healthy (0.12) to vanishing (0.00003) at step 47"
- **First occurrence tracking**: which layer failed *first* (this is usually the root cause)
- **Activation regime shifts**: when activations go from normal to saturated/dead

This is basically what NeuralDBG does under the hood — I open-sourced it recently and it's on PyPI (`pip install neuraldbg`) if anyone wants to try it. The key design choice was to extract **semantic events** (transitions) rather than raw tensors — this makes the output small enough to reason about.

## Practical takeaway you can use today

Even without any tool, you can add this to your training loop:

```python
# One-time gradient norm snapshot per layer
if step % 10 == 0:
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            if norm < 1e-6:
                print(f"WARNING: vanishing gradient at {name} step {step} (norm={norm:.2e})")
            elif norm > 1e3:
                print(f"WARNING: exploding gradient at {name} step {step} (norm={norm:.2e})")
```

This won't give you causal hypotheses, but it will catch 80% of training failures early.

## Questions for the community

1. How do you currently debug training failures? Print statements? TensorBoard? Something custom?
2. Have you found that failures are typically localized to specific layers, or more distributed?
3. What's your "go-to" debugging workflow when loss goes to NaN?

Curious to hear what works for people in practice.

---

Links (for those interested):
- GitHub: https://github.com/LambdaSection/NeuralDBG (MIT, open-source)
- Quickstart: `pip install neuraldbg`
