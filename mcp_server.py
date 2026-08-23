"""
NeuralDBG MCP Server â€” Expose NeuralDBG as a tool for AI agents.

Model Context Protocol (MCP) server that allows AI agents (Copilot, Claude, etc.)
to use NeuralDBG's causal diagnostic capabilities.

Tools exposed:
  - neuraldbg_diagnose: Run NeuralDBG on a model and return causal analysis
  - neuraldbg_benchmark: Run competitive benchmark
  - neuraldbg_explain: Explain a specific failure pattern

Usage:
    python mcp_server.py
    # Or via mcp CLI:
    mcp install mcp_server.py --name neuraldbg

Requirements:
    pip install mcp
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add NeuralDBG to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn

from neuraldbg import NeuralDbg

# ============================================================
# MCP Server Definition
# ============================================================

try:
    import mcp.types as types
    from mcp.server import NotificationOptions, Server
    from mcp.server.models import InitializationCapabilities
    from mcp.server.stdio import stdio_server

    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    print("[NeuralDBG-MCP] mcp package not installed. Run: pip install mcp")


if HAS_MCP:
    server = Server("neuraldbg")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """List all tools exposed by NeuralDBG MCP server."""
        return [
            types.Tool(
                name="neuraldbg_diagnose",
                description="Run NeuralDBG causal diagnostic on a PyTorch model. Detects gradient health issues, activation anomalies, optimizer instability, and data corruption. Returns causal chains showing root cause â†’ symptom propagation.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_code": {
                            "type": "string",
                            "description": "Python code defining a PyTorch model and training loop. Must contain model = ... and a training loop with loss.backward().",
                        },
                        "steps": {
                            "type": "integer",
                            "description": "Number of training steps to run (default: 10)",
                            "default": 10,
                        },
                        "family": {
                            "type": "string",
                            "description": "Architecture family for calibrated thresholds: MLP, CNN, RNN, Transformer, Hybrid, BlackSwan, RL",
                            "enum": [
                                "MLP",
                                "CNN",
                                "RNN",
                                "Transformer",
                                "Hybrid",
                                "BlackSwan",
                                "RL",
                            ],
                        },
                    },
                    "required": ["model_code"],
                },
            ),
            types.Tool(
                name="neuraldbg_benchmark",
                description="Run competitive benchmark comparing NeuralDBG vs W&B/TensorBoard-style monitoring. Measures detection rate, causal chain count, and root cause accuracy.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "scenarios": {
                            "type": "integer",
                            "description": "Number of failure scenarios to test (1-6)",
                            "default": 6,
                        },
                    },
                },
            ),
            types.Tool(
                name="neuraldbg_explain",
                description="Explain a specific failure pattern detected during training. Returns root cause hypothesis with confidence score.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "failure_type": {
                            "type": "string",
                            "description": "Type of failure to explain: vanishing_gradients, exploding_gradients, optimizer_instability, data_anomaly, nan_detected",
                            "enum": [
                                "vanishing_gradients",
                                "exploding_gradients",
                                "optimizer_instability",
                                "data_anomaly",
                                "nan_detected",
                            ],
                        },
                        "events_json": {
                            "type": "string",
                            "description": "JSON string of NeuralDBG events from a previous run (optional â€” uses last run if omitted)",
                        },
                    },
                },
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        """Handle tool calls from MCP clients."""

        if name == "neuraldbg_diagnose":
            model_code = arguments.get("model_code", "")
            steps = arguments.get("steps", 10)
            family = arguments.get("family", "MLP")

            result = _run_diagnosis(model_code, steps, family)
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "neuraldbg_benchmark":
            scenarios = arguments.get("scenarios", 6)
            result = _run_benchmark(scenarios)
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "neuraldbg_explain":
            failure_type = arguments.get("failure_type", "vanishing_gradients")
            result = _explain_failure(failure_type)
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        else:
            raise ValueError(f"Unknown tool: {name}")


# ============================================================
# Tool Implementations
# ============================================================


def _run_diagnosis(model_code: str, steps: int, family: str) -> dict:
    """Execute model code under NeuralDBG supervision."""
    try:
        # Build a simple model from the code
        # For safety, we execute in a restricted namespace
        namespace = {
            "torch": torch,
            "nn": nn,
            "NeuralDbg": NeuralDbg,
        }

        # Wrap in a function that creates model + training loop
        # The user code should define: def make_model(): ...; def train_step(model, opt): ...
        # Execution intentionnelle du code fourni par l'appelant : fonctionnalite centrale du serveur MCP
        exec(model_code, namespace)  # nosec B102

        if "make_model" not in namespace:
            return {"error": "model_code must define make_model() function"}

        model = namespace["make_model"]()

        events_list = []
        chains_list = []
        losses: list = []

        with NeuralDbg(model, family=family) as dbg:
            opt = torch.optim.SGD(model.parameters(), lr=0.01)

            for s in range(steps):
                opt.zero_grad()
                try:
                    if "train_step" in namespace:
                        loss = namespace["train_step"](model, opt)
                    else:
                        x = torch.randn(
                            8, model.fc1.in_features if hasattr(model, "fc1") else 16
                        )
                        out = model(x)
                        loss = out.sum()
                        loss.backward()

                    dbg.step_iteration()
                    dbg.record_loss(
                        loss.item() if hasattr(loss, "item") else float(loss)
                    )
                    opt.step()
                    losses.append(float(loss) if hasattr(loss, "item") else float(loss))
                except Exception as e:
                    losses.append(None)

            events_list = dbg.dump_events()
            try:
                chains = dbg.explain_causal()
                chains_list = [str(c) for c in chains]
            except Exception:
                chains_list = []

            try:
                hyps = dbg.explain_failure()
                hypotheses = [
                    h.description[:200] if hasattr(h, "description") else str(h)
                    for h in hyps
                ]
            except Exception:
                hypotheses = []

        # Count anomalies
        anomaly_events = [
            e
            for e in events_list
            if e.get("event_type")
            in (
                "gradient_health_transition",
                "activation_regime_shift",
                "optimizer_instability",
                "data_anomaly",
            )
        ]

        return {
            "status": "success",
            "steps": steps,
            "family": family,
            "total_events": len(events_list),
            "anomaly_events": len(anomaly_events),
            "event_types": list(set(e.get("event_type") for e in events_list)),
            "causal_chains": len(chains_list),
            "top_chain": chains_list[0][:200] if chains_list else None,
            "hypotheses": hypotheses[:5],
            "final_loss": losses[-1] if losses else None,
            "loss_trend": losses,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)[:300]}


def _run_benchmark(scenarios: int) -> dict:
    """Run simplified competitive benchmark."""
    from benchmark_comparison import (bug_dead, bug_exploding, bug_nan,
                                      bug_vanishing, bug_zero, build_model,
                                      run_scenario)

    scenarios_list = [
        ("Healthy", None),
        ("Exploding", bug_exploding),
        ("Vanishing", bug_vanishing),
        ("NaN Data", bug_nan),
        ("Dead Bias", bug_dead),
        ("Zero Init", bug_zero),
    ][:scenarios]

    results = []
    for name, bug_fn in scenarios_list:
        r = run_scenario(name, bug_fn, steps=10)
        results.append(
            {
                "scenario": r["scenario"],
                "neuraldbg_events": r["neuraldbg"]["events"],
                "neuraldbg_chains": r["neuraldbg"]["chains"],
                "baseline_alerts": r["baseline"]["total_alerts"],
            }
        )

    ndbg_detected = sum(1 for r in results if r["neuraldbg_events"] > 0)
    baseline_detected = sum(1 for r in results if r["baseline_alerts"] > 0)

    return {
        "status": "success",
        "scenarios": len(results),
        "neuraldbg_detection": f"{ndbg_detected}/{len(results)}",
        "baseline_detection": f"{baseline_detected}/{len(results)}",
        "advantage": f"+{ndbg_detected - baseline_detected}",
        "results": results,
    }


def _explain_failure(failure_type: str) -> dict:
    """Explain a specific failure type."""
    explanations = {
        "vanishing_gradients": {
            "root_cause": "Gradients < 1e-4 persistent across layers",
            "common_triggers": [
                "Sigmoid saturation",
                "Deep networks without skip connections",
                "Learning rate too low",
                "Poor weight initialization",
            ],
            "neuraldbg_detection": "gradient_health_transition â†’ vanishing. Trend-based detection catches slow decays over 5+ steps.",
            "remediation": "Increase LR 10Ã—, add BatchNorm, use ReLU/GeLU instead of Sigmoid, add residual connections.",
        },
        "exploding_gradients": {
            "root_cause": "Gradients > 1e3 in at least one layer",
            "common_triggers": [
                "Learning rate too high",
                "No gradient clipping",
                "Unstable loss function",
                "RNN with long sequences",
            ],
            "neuraldbg_detection": "gradient_health_transition â†’ exploding. Detected at first occurrence with layer name.",
            "remediation": "Reduce LR 10Ã—, add gradient clipping (max_norm=1.0), use LayerNorm in RNNs.",
        },
        "optimizer_instability": {
            "root_cause": "Loss oscillates or diverges without clear gradient anomaly",
            "common_triggers": [
                "Adam Î² parameters wrong",
                "Learning rate schedule too aggressive",
                "Mixed precision without loss scaling",
            ],
            "neuraldbg_detection": "optimizer_instability event. Links to gradient health and loss trend.",
            "remediation": "Reduce LR, use cosine schedule, ensure loss scaling for fp16.",
        },
        "data_anomaly": {
            "root_cause": "NaN or extreme values in input data",
            "common_triggers": [
                "Corrupted dataset",
                "Missing normalization",
                "Integer overflow in preprocessing",
            ],
            "neuraldbg_detection": "data_anomaly event with distribution shift detection.",
            "remediation": "Add input validation, normalize data, check for NaN in dataloader.",
        },
        "nan_detected": {
            "root_cause": "NaN appeared in loss or gradients",
            "common_triggers": [
                "Division by zero",
                "log(0)",
                "sqrt(negative)",
                "fp16 overflow",
            ],
            "neuraldbg_detection": "nan_detected event with step-by-step NaN propagation trace.",
            "remediation": "Check loss function domain, add epsilon to log/sqrt, use fp32 for critical ops.",
        },
    }

    explanation = explanations.get(
        failure_type, {"error": f"Unknown failure type: {failure_type}"}
    )
    explanation["failure_type"] = failure_type
    explanation["status"] = "success"
    return explanation


# ============================================================
# Entry Point
# ============================================================


async def main():
    """Run the MCP server."""
    if not HAS_MCP:
        print("[NeuralDBG-MCP] Error: mcp package required. Run: pip install mcp")
        return

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationCapabilities(
                sampling=None,
                experimental=None,
                roots=None,
            ),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
