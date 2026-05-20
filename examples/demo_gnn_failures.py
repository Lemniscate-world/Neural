#!/usr/bin/env python3
"""
GNN (GCN / GAT) failure scenarios demonstrating NeuralDBG causal inference.
Covers: oversmoothing (deep GCN), exploding gradients, NaN injection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from neuraldbg import NeuralDbg


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, use_norm=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(out_features)

    def forward(self, x, adj):
        h = self.linear(x)
        h = adj @ h
        if self.use_norm:
            h = self.norm(h)
        return F.relu(h)


class GCN(nn.Module):
    def __init__(self, in_features, hidden, out_classes, num_layers, use_norm=True):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(in_features, hidden, use_norm))
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden, hidden, use_norm))
        self.layers.append(GCNLayer(hidden, out_classes, use_norm=False))

    def forward(self, x, adj):
        for layer in self.layers[:-1]:
            x = layer(x, adj)
        x = self.layers[-1](x, adj)
        return x


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features * num_heads)
        self.attn = nn.Linear(out_features * num_heads, num_heads)

    def forward(self, x, adj):
        h = self.linear(x)
        attn_logits = self.attn(h)
        attn = F.softmax(attn_logits, dim=1)
        h = h * attn
        return F.relu(h.mean(dim=-2, keepdim=False) if self.num_heads > 1 else h)


class GAT(nn.Module):
    def __init__(self, in_features, hidden, out_classes, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(in_features, hidden))
        for _ in range(num_layers - 2):
            self.layers.append(GATLayer(hidden, hidden))
        self.layers.append(GATLayer(hidden, out_classes, num_heads=1))

    def forward(self, x, adj):
        for layer in self.layers[:-1]:
            x = layer(x, adj)
        x = self.layers[-1](x, adj)
        return x


def _make_adj(num_nodes, density=0.3):
    adj = (torch.rand(num_nodes, num_nodes) < density).float()
    adj = adj + torch.eye(num_nodes)
    d = adj.sum(dim=1, keepdim=True)
    d_inv_sqrt = d.pow(-0.5)
    d_inv_sqrt = torch.where(
        torch.isinf(d_inv_sqrt), torch.zeros_like(d_inv_sqrt), d_inv_sqrt
    )
    return d_inv_sqrt * adj * d_inv_sqrt.transpose(0, 1)


def _make_graph_data(num_nodes=50, in_features=16, out_classes=5):
    x = torch.randn(num_nodes, in_features)
    adj = _make_adj(num_nodes)
    labels = torch.randint(0, out_classes, (num_nodes,))
    return x, adj, labels


def train_gnn(model, x, adj, labels, num_steps=20, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    with NeuralDbg(model, threshold_vanishing=1e-4, threshold_exploding=1.0) as dbg:
        for step in range(num_steps):
            optimizer.zero_grad()
            dbg.step = step
            output = model(x, adj)
            loss = criterion(output, labels)
            loss.backward()
            dbg.record_loss(loss.item())
            optimizer.step()
    return dbg


def analyze_results(dbg):
    return {
        "hypotheses": dbg.explain_failure("vanishing_gradients")
        + dbg.explain_failure("exploding_gradients"),
        "opt_hypotheses": dbg.explain_failure("optimizer_instability"),
        "data_hypotheses": dbg.explain_failure("data_anomaly"),
        "couplings": dbg.detect_coupled_failures(),
        "events": dbg.events,
        "mermaid": dbg.export_mermaid_causal_graph(),
    }


def scenario_oversmoothing(num_steps=30):
    """Deep GCN (8 layers) without norm -> oversmoothing, vanishing gradients."""
    model = GCN(in_features=16, hidden=32, out_classes=5, num_layers=8, use_norm=False)
    x, adj, labels = _make_graph_data()
    return train_gnn(model, x, adj, labels, num_steps=num_steps, lr=1e-3)


def scenario_gnn_exploding(num_steps=30):
    """GCN with inflated weights -> exploding gradients."""
    model = GCN(in_features=16, hidden=32, out_classes=5, num_layers=3)
    for param in model.parameters():
        with torch.no_grad():
            param.mul_(100.0)
    x, adj, labels = _make_graph_data()
    return train_gnn(model, x, adj, labels, num_steps=num_steps, lr=1e-1)


def scenario_gnn_nan(num_steps=30):
    """GCN with NaN injected into weights -> data anomaly detection."""
    model = GCN(in_features=16, hidden=32, out_classes=5, num_layers=3)
    with torch.no_grad():
        for param in model.parameters():
            param.view(-1)[0] = float("nan")
            break
    x, adj, labels = _make_graph_data()
    return train_gnn(model, x, adj, labels, num_steps=num_steps, lr=1e-3)


def main():
    torch.manual_seed(42)
    print("[NeuralDBG] GNN (GCN/GAT) failure scenarios\n")

    for name, fn in [
        ("Deep GCN -> oversmoothing", scenario_oversmoothing),
        ("Inflated weights -> exploding", scenario_gnn_exploding),
        ("NaN injection -> data anomaly", scenario_gnn_nan),
    ]:
        dbg = fn(num_steps=20)
        results = analyze_results(dbg)
        print(f"\n{'=' * 60}")
        print(f"SCENARIO: {name}")
        print(f"{'=' * 60}")
        print(f"Events: {len(results['events'])}")
        for label, hyps in [
            ("Gradient hypotheses", results["hypotheses"]),
            ("Optimizer hypotheses", results["opt_hypotheses"]),
            ("Data anomaly", results["data_hypotheses"]),
        ]:
            if hyps:
                print(f"{label}:")
                for h in hyps:
                    print(f"  [{h.confidence:.2f}] {h.description}")
        if results["couplings"]:
            print("Coupled failures:")
            for c in results["couplings"]:
                d = c.get("step_difference", 0)
                print(
                    f"  {c['trigger']} -> {c['consequence']} (d={d}, {c['confidence']:.2f})"
                )

    print("\n[DONE] GNN scenarios complete.")


if __name__ == "__main__":
    main()
