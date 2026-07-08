"""
Tier 4 Black-Swan Architecture Tester
RL (Actor-Critic) + RAG (Retrieval-Augmented Generation)

Usage: python validate_blackswans_tier4.py [--quick]
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
# 1. RL Actor-Critic
# ============================================================

class PolicyNetwork(nn.Module):
    """REINFORCE policy network — single head, pure policy gradient.

    No value baseline to avoid gradient redundancy that masked
    bug signals in the Actor-Critic version.
    """

    def __init__(self, state_dim=16, hidden_dim=64, num_actions=4, num_layers=2):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, state):
        return self.policy_head(self.backbone(state))


def rl_configs(n=6):
    configs = []
    dims = [32, 64, 128]
    depths = [2, 3]
    idx = 0
    for d in dims:
        for l in depths:
            if idx >= n: return configs
            configs.append(ArchConfig(
                family="RL", name=f"RL_d{d}_l{l}",
                depth=l, width=d, activation="relu", norm=None,
                skip=False, dropout=0.0,
                extra={"num_actions": 4}))
            idx += 1
    return configs


# ============================================================
# 2. RAG (Retrieval-Augmented Generation)
# ============================================================

class SimpleRetriever(nn.Module):
    """Simulated dense retriever: embedding + top-k selection."""

    def __init__(self, dim=64, num_docs=16):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.doc_embeddings = nn.Parameter(torch.randn(num_docs, dim) * 0.1)

    def forward(self, query, top_k=3):
        # query: [B, dim]
        q = F.normalize(self.query_proj(query), dim=-1)
        d = F.normalize(self.doc_embeddings, dim=-1)
        scores = q @ d.T  # [B, num_docs]
        _, top_indices = torch.topk(scores, top_k, dim=-1)
        # Return top-k document embeddings
        retrieved = self.doc_embeddings[top_indices]  # [B, top_k, dim]
        return retrieved, scores


class RAGModel(nn.Module):
    """Simple RAG: retrieve docs, attend over them, generate output."""

    def __init__(self, dim=64, num_docs=16, top_k=3):
        super().__init__()
        self.retriever = SimpleRetriever(dim, num_docs)
        self.top_k = top_k
        self.cross_attn_q = nn.Linear(dim, dim)
        self.cross_attn_k = nn.Linear(dim, dim)
        self.cross_attn_v = nn.Linear(dim, dim)
        self.generator = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, 10),
        )

    def forward(self, x):
        # x: [B, dim] — query
        docs, scores = self.retriever(x, self.top_k)  # [B, top_k, dim]
        # Cross-attention: query attends over retrieved docs
        q = self.cross_attn_q(x).unsqueeze(1)  # [B, 1, dim]
        k = self.cross_attn_k(docs)             # [B, top_k, dim]
        v = self.cross_attn_v(docs)             # [B, top_k, dim]
        attn = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(x.size(-1)), dim=-1)
        context = (attn @ v).squeeze(1)  # [B, dim]
        return self.generator(context)


def rag_configs(n=6):
    configs = []
    dims = [32, 64, 128]
    top_ks = [3, 4]
    idx = 0
    for d in dims:
        for k in top_ks:
            if idx >= n: return configs
            configs.append(ArchConfig(
                family="RAG", name=f"RAG_d{d}_k{k}",
                depth=2, width=d, activation="gelu", norm=None,
                skip=False, dropout=0.0,
                extra={"num_docs": 16, "top_k": k}))
            idx += 1
    return configs


# ============================================================
# Unified dispatch
# ============================================================

def train_tier4(cfg: ArchConfig, steps=8, bug=None):
    """Train a Tier 4 model with NeuralDBG hooks."""
    family = cfg.family

    if family == "RL":
        model = PolicyNetwork(state_dim=cfg.width, hidden_dim=cfg.width,
                              num_actions=cfg.extra.get("num_actions", 4),
                              num_layers=cfg.depth)
    elif family == "RAG":
        model = RAGModel(dim=cfg.width,
                         num_docs=cfg.extra.get("num_docs", 16),
                         top_k=cfg.extra.get("top_k", 3))
    else:
        raise ValueError(f"Unknown Tier 4 family: {family}")

    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    def make_data():
        return torch.randn(16, cfg.width), torch.randint(0, 10, (16,))

    with NeuralDbg(model) as dbg:
        bug_applied = False
        for s in range(steps):
            x, y = make_data()
            if bug and s >= 3 and not bug_applied:
                x_mod, _, opt, _ = bug(x, y, opt, model)
                if isinstance(x_mod, torch.Tensor):
                    x = x_mod
                bug_applied = True

            opt.zero_grad()
            try:
                if family == "RL":
                    # REINFORCE: policy gradient with simulated rewards
                    logits = model(x)
                    # Sample actions and compute log-probabilities
                    probs = F.softmax(logits, dim=-1)
                    log_probs = F.log_softmax(logits, dim=-1)
                    # Simulated rewards: higher for "correct" actions (y)
                    rewards = torch.zeros_like(probs)
                    rewards.scatter_(1, y.unsqueeze(1), 1.0)
                    # Policy gradient loss: -E[log_prob * reward]
                    loss = -(log_probs * rewards).sum(dim=-1).mean()
                else:
                    out = model(x)
                    loss = nn.CrossEntropyLoss()(out, y)
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
    print("BLACK-SWAN ARCHITECTURE TESTER — Tier 4")
    print("RL (Actor-Critic) + RAG (Retrieval-Augmented)")
    print("=" * 60)

    all_configs = []
    all_configs.extend([("RL", c) for c in rl_configs(6)])
    all_configs.extend([("RAG", c) for c in rag_configs(6)])

    bugs_to_test = BUGS

    results = []
    for family, cfg in all_configs:
        print(f"\n  [{family}] {cfg.name}")

        try:
            ev, _, _ = train_tier4(cfg, steps=8)
            baseline = n_problematic(ev)
        except Exception as e:
            print(f"    Baseline error: {e}")
            continue

        threshold = max(baseline + 2, 2)
        detected = 0
        total = 0

        for bug_name, bug_fn in bugs_to_test:
            try:
                ev, _, _ = train_tier4(cfg, steps=8, bug=bug_fn)
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
    print("RESULTS — Tier 4 Black-Swans")
    print(f"{'='*60}")

    by_family = {}
    for r in results:
        fam = r["family"]
        if fam not in by_family:
            by_family[fam] = {"detected": 0, "total": 0}
        by_family[fam]["detected"] += r["detected"]
        by_family[fam]["total"] += r["total"]

    grand_d, grand_t = 0, 0
    for fam, counts in sorted(by_family.items()):
        pct = counts["detected"] / max(counts["total"], 1) * 100
        print(f"  {fam:15s}: {counts['detected']}/{counts['total']} ({pct:.0f}%)")
        grand_d += counts["detected"]
        grand_t += counts["total"]

    overall = grand_d / max(grand_t, 1) * 100
    print(f"\n  OVERALL: {grand_d}/{grand_t} ({overall:.0f}%)")

    report = {
        "tier": 4,
        "families": ["RL", "RAG"],
        "overall_pct": overall,
        "by_family": {
            fam: {"detected": c["detected"], "total": c["total"],
                  "pct": c["detected"] / max(c["total"], 1) * 100}
            for fam, c in by_family.items()
        },
        "details": results,
    }
    with open("blackswan_tier4_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report: blackswan_tier4_results.json")
