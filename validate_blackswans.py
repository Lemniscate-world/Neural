"""Black-Swan Architecture Tester — Tier 1: GNN, MoE, Diffusion.

Extends validate_combinatorial.py with 3 novel architecture families
that represent known unknowns in the black-swan catalog.

Usage: python validate_blackswans.py [--quick] [--full]
"""

import sys, json, random
sys.path.insert(0, r"C:\Users\Utilisateur\Documents\NeuralDBG")

import torch, torch.nn as nn
import torch.nn.functional as F
from neuraldbg import NeuralDbg
from validate_combinatorial import *

torch.manual_seed(42)
random.seed(42)

PROBLEMATIC = {"data_anomaly", "nan_detected", "silent_corruption", "optimizer_instability"}


# ============================================================
# 1. Graph Neural Network (GCN)
# ============================================================

class GCNLayer(nn.Module):
    """Simple Graph Convolutional layer."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x, adj):
        # x: [B, N, D], adj: [N, N]
        x = torch.bmm(adj.unsqueeze(0).expand(x.size(0), -1, -1), x)
        return F.relu(self.linear(x))


class GNNModel(nn.Module):
    def __init__(self, in_dim=16, hidden=32, out_dim=2, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(in_dim, hidden))
        for _ in range(num_layers - 1):
            self.layers.append(GCNLayer(hidden, hidden))
        self.fc = nn.Linear(hidden, out_dim)
    
    def forward(self, x):
        # x[0]=node features [B,N,D], x[1]=adjacency [N,N]
        if isinstance(x, tuple):
            nodes, adj = x
        else:
            nodes = x
            adj = torch.eye(nodes.size(1))
        for layer in self.layers:
            nodes = layer(nodes, adj)
        return self.fc(nodes.mean(dim=1))


def gnn_configs(n=20):
    configs = []
    depths = [2, 3, 4]
    widths = [32, 64, 128]
    acts = ["relu", "gelu"]
    idx = 0
    for d in depths:
        for w in widths:
            for act in acts:
                if idx >= n:
                    return configs
                configs.append(ArchConfig(
                    family="GNN", name=f"GNN_d{d}_w{w}_{act}",
                    depth=d, width=w, activation=act, norm=None,
                    skip=False, dropout=0.0,
                    extra={"num_nodes": 8}))
                idx += 1
    return configs


def make_gnn_data(batch=8, num_nodes=8, dim=16):
    x = torch.randn(batch, num_nodes, dim)
    adj = torch.randint(0, 2, (num_nodes, num_nodes)).float()
    adj = (adj + adj.T) / 2  # symmetric
    adj.fill_diagonal_(0)
    return (x, adj), torch.randint(0, 2, (batch,))


# ============================================================
# 2. Mixture of Experts (MoE)
# ============================================================

class SparseMoE(nn.Module):
    """Sparse Mixture of Experts with top-k routing."""
    def __init__(self, dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # Routing
        logits = self.router(x)  # [B, num_experts]
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(topk_vals, dim=-1)
        
        # Expert computation
        out = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = topk_idx[:, i]
            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)
                if mask.any():
                    out[mask] += weights[mask, i:i+1] * self.experts[e_idx](x[mask])
        return out


class MoEModel(nn.Module):
    def __init__(self, dim=64, num_layers=3, num_experts=4):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.moe_layers = nn.ModuleList([
            SparseMoE(dim, num_experts, top_k=2) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.fc = nn.Linear(dim, 10)
    
    def forward(self, x):
        x = F.gelu(self.embed(x))
        for moe, norm in zip(self.moe_layers, self.norms):
            x = norm(x + moe(x))
        return self.fc(x)


def moe_configs(n=15):
    configs = []
    depths = [2, 3, 4]
    widths = [32, 64, 128]
    experts = [4, 8]
    idx = 0
    for d in depths:
        for w in widths:
            for e in experts:
                if idx >= n:
                    return configs
                configs.append(ArchConfig(
                    family="MoE", name=f"MoE_d{d}_w{w}_e{e}",
                    depth=d, width=w, activation="gelu", norm="layernorm",
                    skip=True, dropout=0.0,
                    extra={"num_experts": e}))
                idx += 1
    return configs


def make_moe_data(batch=16, width=64):
    return torch.randn(batch, width), torch.randint(0, 10, (batch,))


def build_moe(cfg: ArchConfig) -> nn.Module:
    return MoEModel(dim=cfg.width, num_layers=cfg.depth, num_experts=cfg.extra.get("num_experts", 4))


# ============================================================
# 3. Diffusion UNet
# ============================================================

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(1, dim)
        self.linear2 = nn.Linear(dim, dim)
    
    def forward(self, t):
        t = t.float().unsqueeze(-1) / 1000.0
        return F.silu(self.linear2(F.silu(self.linear1(t))))


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.norm2 = nn.BatchNorm2d(out_ch)
    
    def forward(self, x, t_emb):
        h = F.relu(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        return F.relu(self.norm2(self.conv2(h)))


class DiffusionUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, time_dim=64):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        self.enc1 = UNetBlock(in_ch, base_ch, time_dim)
        self.enc2 = UNetBlock(base_ch, base_ch*2, time_dim)
        self.bottleneck = UNetBlock(base_ch*2, base_ch*2, time_dim)
        self.dec2 = UNetBlock(base_ch*2, base_ch, time_dim)
        self.dec1 = nn.Conv2d(base_ch, in_ch, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x, t):
        t_emb = self.time_embed(t)
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)
        b = self.bottleneck(self.pool(e2), t_emb)
        d2 = F.interpolate(b, size=e2.shape[2:])
        d1 = F.interpolate(self.dec2(d2, t_emb), size=e1.shape[2:])
        return self.dec1(F.relu(d1 + e1))


def diffusion_configs(n=15):
    configs = []
    bases = [16, 32, 64]
    time_dims = [32, 64]
    idx = 0
    for base in bases:
        for td in time_dims:
            if idx >= n:
                return configs
            configs.append(ArchConfig(
                family="Diffusion", name=f"Diffusion_b{base}_t{td}",
                depth=4, width=base, activation="relu", norm="batchnorm",
                skip=True, dropout=0.0,
                extra={"base_ch": base, "time_dim": td}))
            idx += 1
    return configs


def make_diffusion_data(batch=4, channels=3, size=16):
    x = torch.randn(batch, channels, size, size)
    t = torch.randint(0, 1000, (batch,))
    return x, t


def build_diffusion(cfg: ArchConfig) -> nn.Module:
    return DiffusionUNet(in_ch=3, base_ch=cfg.extra.get("base_ch", 32),
                         time_dim=cfg.extra.get("time_dim", 64))


# ============================================================
# Unified dispatch
# ============================================================

BLACKSWAN_BUILDERS = {
    "GNN": (lambda cfg: GNNModel(in_dim=16, hidden=cfg.width, num_layers=cfg.depth),
            lambda batch=8, cfg=None: make_gnn_data(batch=batch)),
    "MoE": (build_moe, lambda batch=16, cfg=None: make_moe_data(batch=batch, width=cfg.width if cfg else 64)),
    "Diffusion": (build_diffusion, lambda batch=4, cfg=None: make_diffusion_data(batch=batch)),
}


def train_blackswan(cfg, steps=8, bug=None):
    builder, data_fn_raw = BLACKSWAN_BUILDERS[cfg.family]
    model = builder(cfg)
    data_fn = lambda: data_fn_raw(cfg=cfg)  # bind config
    
    if cfg.family == "Diffusion":
        return _train_diffusion(model, data_fn, steps, bug)
    else:
        return _train_classification(model, data_fn, steps, bug)


def _train_classification(model, data_fn, steps, bug):
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    with NeuralDbg(model) as dbg:
        bug_applied = False  # apply bug once
        for s in range(steps):
            x, y = data_fn()
            if bug and s >= 3:
                if not bug_applied:
                    # Use the imported bug functions from validate_combinatorial
                    # which handle RNN-specific gate corruption
                    x_mod, y_mod, opt, model = bug(x, y, opt, model)
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


def _train_diffusion(model, data_fn, steps, bug):
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    
    with NeuralDbg(model) as dbg:
        bug_applied = False
        for s in range(steps):
            x, t = data_fn()
            if bug and s >= 3:
                if not bug_applied:
                    x_mod, _, opt, _ = bug(x, torch.zeros(1), opt, model)
                    if isinstance(x_mod, torch.Tensor):
                        x = x_mod
                    bug_applied = True
            
            opt.zero_grad()
            try:
                noise = torch.randn_like(x)
                noisy = x + noise
                pred = model(noisy, t)
                loss = loss_fn(pred, noise)
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
    print("BLACK-SWAN ARCHITECTURE TESTER — Tier 1")
    print("GNN + MoE + Diffusion")
    print("=" * 60)
    
    all_configs = []
    all_configs.extend([("GNN", c) for c in gnn_configs(6)])
    all_configs.extend([("MoE", c) for c in moe_configs(6)])
    all_configs.extend([("Diffusion", c) for c in diffusion_configs(6)])
    
    bugs_to_test = [
        ("exploding", bug_exploding),
        ("nan_data", bug_nan),
        ("zero_init", bug_zero),
        ("vanishing", bug_vanishing),
        ("dead_bias", bug_dead),
        ("divergence", bug_divergence),
    ]
    
    results = []
    for family, cfg in all_configs:
        print(f"\n  [{family}] {cfg.name}")
        
        # Baseline
        try:
            ev, _, _ = train_blackswan(cfg, steps=8)
            baseline = n_problematic(ev)
        except Exception as e:
            print(f"    Baseline error: {e}")
            continue
        
        threshold = max(baseline + 2, 2)  # noisy archs
        detected = 0
        total = 0
        
        for bug_name, bug_fn in bugs_to_test:
            try:
                ev, _, _ = train_blackswan(cfg, steps=8, bug=bug_fn)
                n = n_problematic(ev)
                if n > threshold:
                    detected += 1
                total += 1
            except Exception:
                total += 1
        
        print(f"    Baseline: {baseline} | Detected: {detected}/{total}")
        results.append({"family": family, "name": cfg.name, "detected": detected, "total": total})
    
    # Summary
    print(f"\n{'='*60}")
    print("RESULTS — Tier 1 Black-Swans")
    print(f"{'='*60}")
    
    by_family = {}
    for r in results:
        fam = r["family"]
        if fam not in by_family:
            by_family[fam] = {"d": 0, "t": 0}
        by_family[fam]["d"] += r["detected"]
        by_family[fam]["t"] += r["total"]
    
    total_d = 0
    total_t = 0
    for fam, v in sorted(by_family.items()):
        pct = 100 * v["d"] // max(v["t"], 1)
        print(f"  {fam:15s}: {v['d']}/{v['t']} ({pct}%)")
        total_d += v["d"]
        total_t += v["t"]
    
    print(f"\n  OVERALL: {total_d}/{total_t} ({100*total_d//max(total_t,1)}%)")
    
    # Save
    report = {
        "families": ["GNN", "MoE", "Diffusion"],
        "overall": f"{total_d}/{total_t}",
        "by_family": {k: f"{v['d']}/{v['t']}" for k, v in by_family.items()},
        "results": results,
    }
    with open("blackswan_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report: blackswan_results.json")
