# v0.dev Landing Page Prompt - NeuralDBG

Build a clean landing page for **NeuralDBG**.

---

## Hero section
- **Headline:** A causal inference engine for deep learning that tells you WHY your model failed, not just WHAT went wrong.
- **Sub-headline:** For ML researchers and research engineers who spend hours debugging training failures without knowing the root cause.
- **CTA:** "Join waitlist" -> email form

---

## Problem section (3 pain points from interviews)
1. **"My training plateaued and I don't know why"** — TensorBoard shows metrics but doesn't explain the cause of failure.
2. **"I tried every hyperparameter change but nothing works"** — Without root cause analysis, debugging is guessing.
3. **"Debugging takes more time than actual research"** — Researchers spend 80% of time on debugging instead of model improvements.

---

## Solution section
| Feature | Benefit |
|---------|---------|
| **Semantic Event Extraction** | Captures meaningful transitions (vanishing gradients, dead neurons) instead of raw tensor data |
| **Causal Reasoning Engine** | Generates ranked hypotheses with confidence scores about why training failed |
| **Cross-layer Propagation Analysis** | Identifies how failures cascade through your network layers |

---

## How it works (3 steps)
1. **Wrap your model** — `with NeuralDbg(model) as dbg:`
2. **Run training** — NeuralDBG captures events automatically
3. **Get explanations** — `dbg.explain_failure("vanishing_gradients")` returns ranked causal hypotheses

---

## Social proof placeholder
> "Join 127 ML researchers waiting for early access"

---

## Tech specs
- PyTorch only
- Works with torch.compile
- 448 events captured on EfficientNet-B0
- 202 events captured on ViT-B/16
- Open source (MIT license)

---

## Footer
- Email capture: https://formsubmit.co/neuraldbg@proton.me
- No tracking, no cookies
- Built with: Static HTML + Tailwind (no React)

---

## Style requirements
- Clean, professional
- No gradients, no glassmorphism
- Monospace fonts for code
- Dark mode option

---

## Deployment
After export:
1. Add form action to email capture
2. Add to .gitignore until launch
3. Deploy: `npx -y surge {deploy_dir} neuraldbg.surge.sh`