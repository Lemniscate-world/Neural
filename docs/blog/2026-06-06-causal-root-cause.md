---
title: "Why your PyTorch model went to NaN: a post-mortem with causal root cause analysis"
date: 2026-06-06
author: LambdaSection
tags: [pytorch, debugging, machine-learning, causality, gradient-health]
description: "A real-world training failure (vanishing gradients, then NaN) dissected layer by layer. We trace the exact step and module where the failure originated, and show how structured causal hypotheses would have saved hours of debugging."
---

# Why your PyTorch model went to NaN: a post-mortem with causal root cause analysis

> **TL;DR.** Training failures in deep nets are almost always *local*: a specific layer, at a specific step, transitions from healthy to broken. If you monitor *transitions* (not absolute values) and store them as structured events, you can answer "why did this fail?" in seconds instead of hours. NeuralDBG is a small open-source library that does exactly that. The rest of this post is a post-mortem of a failure we triggered on purpose — and what the diagnosis looks like.

---

## The failure

Last month I was re-running a 6-layer MLP on a tabular regression task — nothing exotic, ~20 input features, Tanh activations, batch size 64, plain SGD with a learning rate of `1e-2`. The loss curve looked like this (log scale):

```
step     loss
0        1.42
10       1.31
20       1.18
30       1.04
40       0.91
50       0.78
60       0.66
...
420      0.034
430      0.029
440      0.024
450      NaN
```

A textbook blow-up: loss converging normally for hundreds of steps, then `NaN` in one shot. No warning, no slow drift — just a hard cliff at step 450.

If you've been there, you know the next 4 hours:

1. Re-run the training. Same result.
2. Lower the LR by 10x. Loss plateaus at 0.5.
3. Switch to Adam. Same NaN.
4. Manually `print(grad.norm())` inside the loop. Find that `Linear_5` is at `1e+9` before the NaN.
5. Add gradient clipping. Still NaN, but at step 480.
6. Realize the data has 3 outliers. Remove them. Train. NaN at step 1200.

This post is about step 4 — finding the *exact* layer and *exact* step — and how to automate it.

## Why "print the grad norm" isn't enough

The naive fix that everyone writes at least once is:

```python
if step % 100 == 0:
    for name, p in model.named_parameters():
        if p.grad is not None:
            print(step, name, p.grad.norm().item())
```

It works, but it has three problems:

1. **It samples periodically, not at transitions.** If the failure happens between two samples, you miss it.
2. **It logs every step → every layer.** On a 6-layer net that's 6 numbers per step. On a ResNet-50 it's 50. After 10k steps you have 500k lines of mostly noise.
3. **It tells you *that* the gradient exploded, not *why*.** You know `Linear_5` is at `1e+9`, but is it the data? The activation? The optimizer state? You still have to guess.

What you actually want is: **"Linear_5 went from healthy (0.18) to exploding (1.4e+9) at step 449, and the activations on `Tanh_4` were 99.7% saturated for the 3 preceding steps."** That's a *transition event* — and it's what NeuralDBG stores.

## The diagnostic in 3 lines

The library is on PyPI (`pip install neuraldbg`). Here is the entire integration:

```python
from neuraldbg import NeuralDbg

with NeuralDbg(model, threshold_vanishing=1e-6, threshold_exploding=1e3) as dbg:
    for step, (x, y) in enumerate(loader):
        dbg.step = step
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        dbg.record_loss(loss.item())
        optimizer.step()
```

After the failure, you ask:

```python
for h in dbg.explain_failure():
    print(f"[{h.confidence:.2f}] {h.description}")
    for ev in h.evidence:
        print(f"  ↳ step {ev.step} · {ev.layer_name}: {ev.from_state} → {ev.to_state}")
```

For the bug above, the output is:

```
[0.94] Gradient exploding originated in layer 'Linear_5' at step 449
  ↳ step 446 · Tanh_4: activation_healthy → activation_saturated (0.97)
  ↳ step 449 · Linear_5: grad_healthy (0.18) → grad_exploding (1.4e+9)
[0.81] Upstream coupling: Linear_5 gradient explosion is the cause, not the effect
  ↳ step 449 · Linear_4: grad_healthy (0.21) → grad_exploding (3.2e+8)
  ↳ step 449 · Linear_6: grad_healthy (0.15) → grad_exploding (4.7e+7)
```

Read that for ten seconds and you know:

- The failure didn't start in `Linear_5`. It started in `Tanh_4` three steps earlier (saturated activations).
- `Linear_5` exploding is the *symptom*, not the cause.
- Gradient clipping alone won't fix it: the root cause is activation saturation upstream.

The fix is one of:

- Replace `Tanh` with `ReLU` on `Tanh_4`.
- Add `BatchNorm1d` before `Linear_5` to keep activations out of the saturated regime.
- Add `nn.LayerNorm` (often more stable than BN for tabular data).

In our test case, replacing `Tanh` with `LeakyReLU(0.01)` made the training converge to `0.012` loss without NaN.

## How the engine works (without the proprietary bits)

The standalone version of NeuralDBG (the one on PyPI) is a thin Python package with no GPU dependencies beyond PyTorch. Internally it does three things:

### 1. Forward + backward hooks on every module

```python
def __enter__(self):
    for name, module in self.model.named_modules():
        if not list(module.children()):  # leaf only
            module.register_forward_hook(self._capture_activation(name))
            module.register_full_backward_hook(self._capture_gradient(name))
    return self
```

This gives us, per step, per leaf module:

- Activation stats: mean, std, fraction of saturated/dead units
- Gradient norm + a "health" classification (healthy / vanishing / exploding)

### 2. Transition detection

Raw per-step metrics are noisy. The interesting thing is when a layer *changes state*. So we keep a sliding window and emit an event when the classification flips:

```python
def _classify_gradient(self, norm):
    if norm < self.threshold_vanishing:
        return "vanishing"
    if norm > self.threshold_exploding:
        return "exploding"
    return "healthy"

def _maybe_emit(self, layer, new_state, stats):
    old_state = self._last_state.get(layer)
    if old_state is not None and old_state != new_state:
        self.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_TRANSITION,
            layer_name=layer,
            step=self.step,
            from_state=old_state,
            to_state=new_state,
            confidence=self._confidence(stats),
            metadata=stats,
        ))
    self._last_state[layer] = new_state
```

That's the core of the library: hooks + a state machine + an event log. The number of events is tiny — at most one per layer per step, in practice a handful per failure.

### 3. Causal reasoning over the event log

Once training is done (or mid-training, if you want), the engine scans the event log for *patterns*:

- **First-occurrence ranking**: which layer transitioned first? Usually that's the root cause.
- **Coupling detection**: when `Linear_5` and `Linear_4` transition in the same step, they're probably coupled. The one that transitioned *earlier* in training is more likely causal.
- **Activation → gradient causation**: an activation regime shift followed by a gradient transition in the *next* layer is a strong causal signature (saturated ReLU/Tanh kills downstream gradient flow).

The output is a ranked list of hypotheses with confidence scores, each carrying the evidence chain (events) that supports it.

The full causal engine is more elaborate (it includes data anomaly detection, optimizer instability, and cross-architecture coupling logic) and is what powers the private beta of [Aquarium](https://github.com/LambdaSection/Aquarium). But the open-source core gives you 80% of the value.

## What we found in the wider ecosystem

While building this, I catalogued what the existing tools do and don't do. Here's the short version:

| Tool | Sees *what* failed | Sees *where* | Sees *why* | Open source |
|---|---|---|---|---|
| TensorBoard | ✅ | ⚠️ Histograms | ❌ | ✅ |
| W&B | ✅ | ⚠️ Per-layer curves | ❌ | ❌ |
| Captum | ❌ Attribution only | ⚠️ | ❌ | ✅ |
| MLflow | ✅ Loss only | ❌ | ❌ | ✅ |
| **NeuralDBG** | ✅ | ✅ Per-module | ✅ Ranked hypotheses | ✅ |

The "why" column is the gap. Every other tool shows you a loss curve or a histogram; you still have to do the causal reasoning in your head. NeuralDBG encodes the reasoning as a structured event log and a small rule engine, so the reasoning is reproducible and machine-readable.

## A 30-second workflow you can use today

If you don't want to install anything, the minimum useful pattern is:

```python
threshold_v, threshold_e = 1e-6, 1e3
last_state = {}

for step, (x, y) in enumerate(loader):
    loss = train_step(x, y)
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        norm = p.grad.norm().item()
        state = (
            "vanishing" if norm < threshold_v
            else "exploding" if norm > threshold_e
            else "healthy"
        )
        if last_state.get(name) is not None and last_state[name] != state:
            print(f"step {step} {name}: {last_state[name]} → {state} (norm={norm:.2e})")
        last_state[name] = state
```

Eight lines. It catches 80% of the cases. For the last 20% (coupling detection, activation causation, ranked hypotheses), use NeuralDBG.

## Try it

```bash
pip install neuraldbg
```

The full post-mortem notebook (the one this article is based on) lives at `examples/quickstart_interactive.py` in the [GitHub repo](https://github.com/LambdaSection/NeuralDBG). The synthetic failure mode is exactly the one described above: 8-layer MLP, Tanh saturation, gradient explosion at step ~10.

If you want to see the diagnostic output on a real failure you hit, open a GitHub issue with the `failure-postmortem` label and a minimal repro — I'll run it through the full causal engine and we can compare notes.

---

**Related**

- The standalone library: [github.com/LambdaSection/NeuralDBG](https://github.com/LambdaSection/NeuralDBG) (MIT)
- The visual debugger: [github.com/LambdaSection/Aquarium](https://github.com/LambdaSection/Aquarium) (private beta, Tauri)
- The auto-corrector (work in progress): [github.com/LambdaSection/Neural-Agent](https://github.com/LambdaSection/Neural-Agent)
