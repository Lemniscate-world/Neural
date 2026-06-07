# NeuralDBG Blog

Long-form technical posts about PyTorch training failures, causal inference, and ML debugging.

## Posts

| Date | Title | Format | Status |
|---|---|---|---|
| 2026-06-06 | [Why your PyTorch model went to NaN](2026-06-06-causal-root-cause.md) | [HTML](2026-06-06-causal-root-cause.html) · [MD](2026-06-06-causal-root-cause.md) | ⚠️ Synthetic scenario (illustrative) |
| 2026-06-13 | [Post-mortem: PyTorch issue #41508](2026-06-13-pytorch-41508-postmortem.md) | [HTML](2026-06-13-pytorch-41508-postmortem.html) · [MD](2026-06-13-pytorch-41508-postmortem.md) | ✅ Real bug (open since 2020, 25+ participants) |

> The 2026-06-06 article uses a synthetic 6-layer MLP scenario for illustration. The 2026-06-13 article is the first **real-bug post-mortem** — see [REAL_BUGS.md](REAL_BUGS.md) for the policy.

## How posts are organized

Each post ships in two formats:

- **Markdown** (`.md`) — for GitHub, dev.to, Medium, and any static site generator.
- **HTML** (`.html`) — a self-contained page styled to match the NeuralDBG landing page. Drop it on GitHub Pages, Netlify, or any static host.

## Contributing

Open a PR with a new `.md` file. Naming convention: `YYYY-MM-DD-short-slug.md`.
