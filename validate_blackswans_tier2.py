"""
Tier 2 Black-Swan Architecture Tester
FlashAttention, Neural ODE, Quantized (INT8/INT4 simulation)

Extends validate_blackswans.py with 3 additional architecture families.

Usage: python validate_blackswans_tier2.py [--quick] [--full]
"""

import sys, json, random, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuraldbg import NeuralDbg
from validate_combinatorial import ArchConfig, BUGS, n_problematic

torch.manual_seed(42)
random.seed(42)


# ============================================================
# 1. FlashAttention-style model
# ============================================================

class FlashAttentionBlock(nn.Module):
    """Simplified FlashAttention-style block with causal masking.

    Uses memory-efficient attention (scaled_dot_product_attention)
    which is the PyTorch 2.0+ API that FlashAttention backends plug into.
    """

    def __init__(self, dim=64, num_heads=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # PyTorch 2.0+ scaled_dot_product_attention (FlashAttention backend)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(attn_out)


class FlashAttnModel(nn.Module):
    """Transformer-style model using FlashAttention blocks."""

    def __init__(self, dim=64, num_layers=3, num_heads=4, seq_len=32):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.attn_blocks = nn.ModuleList([
            FlashAttentionBlock(dim, num_heads) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.fc = nn.Linear(dim, 10)

    def forward(self, x):
        # x: [B, dim] -> expand to [B, seq_len, dim]
        B = x.shape[0]
        # Simulate sequence by repeating with positional noise
        x = self.embed(x)
        x = x.unsqueeze(1).expand(-1, 32, -1)  # [B, 32, dim]
        x = x + 0.02 * torch.randn_like(x)  # Positional jitter
        for attn, norm in zip(self.attn_blocks, self.norms):
            x = norm(x + attn(x))
        return self.fc(x.mean(dim=1))


def flash_configs(n=6):
    configs = []
    dims = [32, 64, 128]
    layers = [2, 3, 4]
    idx = 0
    for d in dims:
        for l in layers:
            if idx >= n:
                return configs
            configs.append(ArchConfig(
                family="FlashAttn", name=f"FlashAttn_d{d}_l{l}",
                depth=l, width=d, activation="gelu", norm="layernorm",
                skip=True, dropout=0.0,
            ))
            idx += 1
    return configs


# ============================================================
# 2. Neural ODE
# ============================================================

class ODEFunc(nn.Module):
    """ODE function: dz/dt = f(z, t)."""

    def __init__(self, dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.Tanh(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t, z):
        # t: scalar, z: [B, dim]
        return self.net(z)


class NeuralODELayer(nn.Module):
    """Neural ODE layer using Euler discretization.

    dz/dt = f(z, t), solved via fixed-step Euler over [0, 1].
    This is the simplest ODE solver; adaptive-step (Dopri5) exists
    in torchdiffeq but Euler captures the same diagnostic challenges.
    """

    def __init__(self, dim=32, num_steps=4):
        super().__init__()
        self.ode_func = ODEFunc(dim)
        self.num_steps = num_steps

    def forward(self, z0):
        z = z0
        dt = 1.0 / self.num_steps
        for i in range(self.num_steps):
            t = i * dt
            dz = self.ode_func(t, z)
            z = z + dt * dz  # Euler step
        return z


class NeuralODEModel(nn.Module):
    """Model with Neural ODE blocks."""

    def __init__(self, dim=32, num_layers=3, ode_steps=4):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.ode_layers = nn.ModuleList([
            NeuralODELayer(dim, ode_steps) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(dim, 10)

    def forward(self, x):
        x = F.gelu(self.embed(x))
        for ode in self.ode_layers:
            x = ode(x)
        return self.fc(x)


def neural_ode_configs(n=6):
    configs = []
    dims = [32, 64, 128]
    layers = [2, 3, 4]
    idx = 0
    for d in dims:
        for l in layers:
            if idx >= n:
                return configs
            configs.append(ArchConfig(
                family="NeuralODE", name=f"NeuralODE_d{d}_l{l}",
                depth=l, width=d, activation="tanh", norm=None,
                skip=False, dropout=0.0,
                extra={"ode_steps": 4},
            ))
            idx += 1
    return configs


# ============================================================
# 3. Quantized model (INT8/INT4 simulation)
# ============================================================

class FakeQuantizedLinear(nn.Module):
    """Linear layer simulating INT8 quantization.

    Uses fake quantization: weights are quantized/dequantized
    on each forward pass, but computations remain in fp32.
    This captures the numerical precision loss of real quantization.
    """

    def __init__(self, in_features, out_features, bits=8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bits = bits
        self.quant_min = -(2 ** (bits - 1))
        self.quant_max = (2 ** (bits - 1)) - 1

    def forward(self, x):
        # Fake-quantize weights
        w = self.linear.weight
        w_scale = w.abs().max() / self.quant_max
        if w_scale > 0:
            w_quant = torch.round(w / w_scale).clamp(self.quant_min, self.quant_max)
            w_dequant = w_quant * w_scale
        else:
            w_dequant = w

        return F.linear(x, w_dequant, self.linear.bias)


class QuantizedModel(nn.Module):
    """Model with simulated INT8 quantized layers."""

    def __init__(self, dim=64, num_layers=3, bits=8):
        super().__init__()
        self.embed = FakeQuantizedLinear(dim, dim, bits)
        self.layers = nn.ModuleList([
            FakeQuantizedLinear(dim, dim, bits) for _ in range(num_layers)
        ])
        self.activations = nn.ModuleList([nn.GELU() for _ in range(num_layers)])
        self.fc = FakeQuantizedLinear(dim, 10, bits)

    def forward(self, x):
        x = F.gelu(self.embed(x))
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return self.fc(x)


def quant_configs(n=6):
    configs = []
    dims = [32, 64, 128]
    layers = [2, 3, 4]
    bits_list = [8, 4]
    idx = 0
    for d in dims:
        for l in layers:
            for b in bits_list:
                if idx >= n:
                    return configs
                configs.append(ArchConfig(
                    family="Quantized", name=f"Quantized_d{d}_l{l}_int{b}",
                    depth=l, width=d, activation="gelu", norm=None,
                    skip=False, dropout=0.0,
                    extra={"bits": b},
                ))
                idx += 1
    return configs


# ============================================================
# Unified dispatch
# ============================================================

def train_tier2(cfg: ArchConfig, steps=8, bug=None):
    """Train a Tier 2 model with NeuralDBG hooks."""
    family = cfg.family

    # Build model
    if family == "FlashAttn":
        model = FlashAttnModel(dim=cfg.width, num_layers=cfg.depth)
    elif family == "NeuralODE":
        model = NeuralODEModel(dim=cfg.width, num_layers=cfg.depth,
                               ode_steps=cfg.extra.get("ode_steps", 4))
    elif family == "Quantized":
        model = QuantizedModel(dim=cfg.width, num_layers=cfg.depth,
                               bits=cfg.extra.get("bits", 8))
    else:
        raise ValueError(f"Unknown Tier 2 family: {family}")

    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Data: simple classification
    def make_data():
        if family == "FlashAttn":
            # FlashAttn takes [B, dim] and expands to sequence internally
            return torch.randn(16, cfg.width), torch.randint(0, 10, (16,))
        else:
            return torch.randn(16, cfg.width), torch.randint(0, 10, (16,))

    with NeuralDbg(model) as dbg:
        bug_applied = False
        for s in range(steps):
            x, y = make_data()
            if bug and s >= 3:
                if not bug_applied:
                    x_mod, _, opt, _ = bug(x, y, opt, model)
                    if isinstance(x_mod, torch.Tensor):
                        x = x_mod
                    bug_applied = True

            opt.zero_grad()
            try:
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                dbg.step_iteration()
                dbg.record_loss(loss.item())
                opt.step()
            except Exception:
                break

        events = dbg.dump_events()
        hyps = dbg.explain_failure()
        chains = dbg.explain_causal()
    return events, hyps, chains


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BLACK-SWAN ARCHITECTURE TESTER — Tier 2")
    print("FlashAttention + Neural ODE + Quantized (INT8/INT4)")
    print("=" * 60)

    all_configs = []
    all_configs.extend([("FlashAttn", c) for c in flash_configs(6)])
    all_configs.extend([("NeuralODE", c) for c in neural_ode_configs(6)])
    all_configs.extend([("Quantized", c) for c in quant_configs(6)])

    bugs_to_test = BUGS  # 6 standard bugs

    results = []
    for family, cfg in all_configs:
        print(f"\n  [{family}] {cfg.name}")

        # Baseline
        try:
            ev, _, _ = train_tier2(cfg, steps=8)
            baseline = n_problematic(ev)
        except Exception as e:
            print(f"    Baseline error: {e}")
            continue

        threshold = max(baseline + 2, 2)
        detected = 0
        total = 0

        for bug_name, bug_fn in bugs_to_test:
            try:
                ev, _, _ = train_tier2(cfg, steps=8, bug=bug_fn)
                n = n_problematic(ev)
                if n > threshold:
                    detected += 1
                total += 1
            except Exception:
                total += 1

        print(f"    Baseline: {baseline} | Detected: {detected}/{total}")
        results.append({
            "family": family, "name": cfg.name,
            "detected": detected, "total": total
        })

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS — Tier 2 Black-Swans")
    print(f"{'='*60}")

    by_family = {}
    for r in results:
        fam = r["family"]
        if fam not in by_family:
            by_family[fam] = {"detected": 0, "total": 0}
        by_family[fam]["detected"] += r["detected"]
        by_family[fam]["total"] += r["total"]

    grand_detected = 0
    grand_total = 0
    for fam, counts in sorted(by_family.items()):
        pct = counts["detected"] / max(counts["total"], 1) * 100
        print(f"  {fam:15s}: {counts['detected']}/{counts['total']} ({pct:.0f}%)")
        grand_detected += counts["detected"]
        grand_total += counts["total"]

    overall = grand_detected / max(grand_total, 1) * 100
    print(f"\n  OVERALL: {grand_detected}/{grand_total} ({overall:.0f}%)")

    # Save report
    report = {
        "tier": 2,
        "families": ["FlashAttn", "NeuralODE", "Quantized"],
        "overall_pct": overall,
        "by_family": {
            fam: {
                "detected": counts["detected"],
                "total": counts["total"],
                "pct": counts["detected"] / max(counts["total"], 1) * 100,
            }
            for fam, counts in by_family.items()
        },
        "details": results,
    }
    with open("blackswan_tier2_results.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Report: blackswan_tier2_results.json")
