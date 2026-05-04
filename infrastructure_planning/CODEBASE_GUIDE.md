# NeuralDBG — Codebase Guide for Contributors
*Last updated: 2026-03-30*

This document is the entry point for anyone new to the NeuralDBG repository. Read this BEFORE touching any code.

---

## What is NeuralDBG?

NeuralDBG is a **causal inference engine for deep learning training dynamics**. It answers the question: *"Why did my model fail during training?"* -- not just *that* it failed.

It is NOT a traditional debugger. It does not pause execution or store tensors. Instead, it extracts **semantic events** (meaningful transitions like "gradients went from healthy to vanishing") and builds **causal hypotheses** ("this happened because of X, with 85% confidence").

---

## Repository Structure

```text
NeuralDBG/
|
|-- neuraldbg.py                 [CORE] The causal inference engine (474 lines)
|-- demo_vanishing_gradients.py  [DEMO] Shows the engine in action (152 lines)
|-- tests/                       [TESTS] Unit and integration tests
|-- infrastructure_planning/     [DEVOPS] Tasks and guides for infra work
|-- scripts/                     [DEVOPS] Automation scripts (validation sync, hooks)
|-- .githooks/                   [DEVOPS] Local git hooks for validation sync
|-- pyproject.toml               [CONFIG] Python project configuration
|-- requirements.txt             [DEPS] Python dependencies (if present)
|-- .github/workflows/           [CI/CD] GitHub Actions workflows
|-- PLAN.md                      [ROADMAP] MVP plan and build order
|-- AGENTS.md                    [RULES] AI agent contract (30+ rules)
|-- SESSION_SUMMARY.md           [LOG] Session history (EN/FR)
```

---

## Key Files Explained

### neuraldbg.py (The Core Engine)
This is the Hub of the project. It contains:

| Class | What it does |
|-------|--------------|
| `EventType` | Enum of semantic events (gradient transitions, activation shifts, etc.) |
| `ActivationHealth` | Enum (NORMAL, SATURATED, DEAD, ANOMALOUS) for state-based tracking. |
| `GradientHealth` | Enum (HEALTHY, VANISHING, EXPLODING, SATURATED). |
| `SemanticEvent` | Pure Python representation of a failure (vanishing gradients, dead neurons). |
| `CausalHypothesis` | Reasoning result with evidence and causal chain. |
| `NeuralDbg` | The main engine class. Wraps a PyTorch model with hooks to detect events |

**Key methods in `NeuralDbg`:**
- `__enter__` / `__exit__` : Context manager to wrap a training loop
- `_install_hooks` : Attaches PyTorch forward/backward hooks to every layer
- `_classify_gradient_health` : Categorizes a gradient norm as healthy/vanishing/exploding
- `_detect_gradient_transition` : Detects when gradient health changes between steps
- `explain_failure()`: Main entry point for causal reasoning.
- `get_root_causes()`: Identifies the first layer and step where a failure originated.
- `detect_coupled_failures()`: Identifies temporal and structural links between events.
- `export_mermaid_causal_graph()`: Visualizes the trace.

### demo_vanishing_gradients.py (The Demo)
This is a self-contained script that:
1. Creates a deep Tanh network (prone to vanishing gradients by design)
2. Creates problematic data (small scale, small learning rate)
3. Trains the model inside a `NeuralDbg` context manager
4. Prints the causal hypotheses explaining why gradients vanished

**To run:** `python demo_vanishing_gradients.py`

---

## Key Concepts (Glossary)

| Term | Definition |
|------|------------|
| **Vanishing Gradients** | When gradients become so small that early layers stop learning. Common with Tanh/Sigmoid activations in deep networks. |
| **Exploding Gradients** | The opposite: gradients grow exponentially, causing numerical instability. |
| **Semantic Event** | A high-level, meaningful transition in training (NOT raw tensor data). Example: "gradient health changed from HEALTHY to VANISHING at step 42 in layer 3". |
| **Causal Hypothesis** | A ranked explanation of WHY a failure occurred, with confidence score and evidence chain. |
| **Hook (PyTorch)** | A callback function attached to a neural network layer that fires during forward or backward passes. NeuralDBG uses hooks to observe training without modifying the model code. |
| **Post-Mortem Analysis** | Analyzing training AFTER it finishes (not during). Like an autopsy for a failed training run. |
| **Causal Compression** | Reducing thousands of events into a small set of root causes. |

---

## Architecture (Hub and Spokes)

```text
                    +-------------------+
                    |   neuraldbg.py    |  <-- HUB (pure logic, no I/O)
                    |   (Core Engine)   |
                    +-------------------+
                   /         |          \
                  /          |           \
  +-------------+   +---------------+   +------------------+
  | PyTorch     |   | Demo Scripts  |   | CI/CD Pipeline   |
  | (Hooks API) |   | (Usage)       |   | (GitHub Actions) |
  +-------------+   +---------------+   +------------------+
       SPOKE 1          SPOKE 2             SPOKE 3
   (Framework)       (Application)      (Infrastructure)
```

**For DevOps/MLOps:** Your work is in Spoke 3 (Infrastructure). You should NOT need to modify `neuraldbg.py` (the Hub). Your tasks involve wrapping the existing code with CI/CD, Docker, and experiment tracking.

---

## How to Get Started

1. Clone the repo
2. Create a virtual environment: `python -m venv .venv`
3. Install dependencies: `pip install -e .`
4. Run the demo: `python demo_vanishing_gradients.py`
5. Run tests: `pytest tests/`
6. Check your Linear issues for assigned tasks

---

## Developer Tooling — MCP Linear (Claude Code)

Linear est intégré dans Claude Code via le **MCP HTTP officiel de Linear**, défini au scope projet dans `.mcp.json`.

### Config active (`.mcp.json`)

```json
{
  "mcpServers": {
    "linear": {
      "type": "http",
      "url": "https://mcp.linear.app/mcp",
      "headers": {
        "Authorization": "Bearer ${LINEAR_API_KEY}"
      }
    }
  }
}
```

La variable `LINEAR_API_KEY` doit être définie dans l'environnement shell (ou dans `.env` local, non versionné).

### Règle de précédence importante

La config **scope projet** (`.mcp.json`) a priorité sur la config **scope user** (`~/.claude/settings.json`). Si tu définis un serveur `linear` dans les deux, seule la config projet sera utilisée. Ne pas dupliquer la config dans `settings.json`.

### Vérification

```bash
# Via le binaire Claude Code embarqué dans l'extension VS Code
/home/kami/.vscode/extensions/anthropic.claude-code-2.1.116-linux-x64/resources/native-binary/claude mcp list
# → linear: https://mcp.linear.app/mcp (HTTP) ✓ Connected

claude mcp get linear
# → Scope: Project config | Type: http | Status: ✓ Connected
```

### Recréer la config via CLI

```bash
claude mcp add --scope project --transport http linear https://mcp.linear.app/mcp \
  --header "Authorization: Bearer lin_api_..."
```

> **Note :** `linear-mcp-server` (package npx local) ne fonctionne pas avec Claude Code — utiliser uniquement l'endpoint HTTP officiel.
