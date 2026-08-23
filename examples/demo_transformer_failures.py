#!/usr/bin/env python3
"""
Transformer (GPT-style) failure scenarios demonstrating NeuralDBG causal inference.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from neuraldbg import NeuralDbg


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, scale_sqrt=True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model doit etre divisible par n_heads")
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(self.d_head) if scale_sqrt else 1.0

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / self.scale
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model)
        self.c_proj = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, use_norm=True, scale_sqrt=True):
        super().__init__()
        self.use_norm = use_norm
        self.attn = SelfAttention(d_model, n_heads, scale_sqrt)
        self.mlp = MLP(d_model)
        if use_norm:
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        if self.use_norm:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        else:
            x = x + self.attn(x)
            x = x + self.mlp(x)
        return x


class NanoGPT(nn.Module):
    def __init__(
        self,
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=3,
        seq_len=32,
        use_norm=True,
        scale_sqrt=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, use_norm, scale_sqrt)
                for _ in range(n_layers)
            ]
        )
        if use_norm:
            self.ln_f = nn.LayerNorm(d_model)
        self.use_norm = use_norm
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        if self.use_norm:
            x = self.ln_f(x)
        return self.lm_head(x)


def _make_loader(batch_size=4, seq_len=32, num_samples=100, vocab_size=100):
    X = torch.randint(0, vocab_size, (num_samples, seq_len))
    y = torch.randint(0, vocab_size, (num_samples, seq_len))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)


def train_transformer(model, dataloader, num_steps=20, lr=0.001, warmup_steps=0):
    """Train transformer with NeuralDBG monitoring."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    with NeuralDbg(model, threshold_vanishing=1e-4, threshold_exploding=1.0) as dbg:
        for step in range(num_steps):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                dbg.step = step

                # Linear warmup (or not, depending on warmup_steps)
                if step < warmup_steps:
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr * (step + 1) / warmup_steps
                else:
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr

                output = model(batch_x)
                loss = criterion(
                    output.view(-1, output.size(-1)),
                    batch_y.view(-1),
                )
                loss.backward()
                dbg.record_loss(loss.item())
                optimizer.step()
                break
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


def scenario_no_warmup(num_steps=30):
    """Transformer without LR warmup + high LR → gradient explosion."""
    model = NanoGPT(vocab_size=100, d_model=64, n_layers=3)
    loader = _make_loader(num_samples=50)
    return train_transformer(model, loader, num_steps=num_steps, lr=1e-2)


def scenario_no_norm(num_steps=30):
    """Transformer without LayerNorm → activation saturation."""
    model = NanoGPT(vocab_size=100, d_model=64, n_layers=3, use_norm=False)
    loader = _make_loader(num_samples=50)
    return train_transformer(model, loader, num_steps=num_steps, lr=1e-3)


def scenario_no_scale(num_steps=30):
    """Attention without sqrt(d_k) scaling → vanishing gradients."""
    model = NanoGPT(
        vocab_size=100, d_model=64, n_layers=3, scale_sqrt=False, use_norm=True
    )
    loader = _make_loader(num_samples=50)
    return train_transformer(model, loader, num_steps=num_steps, lr=1e-4)


def main():
    torch.manual_seed(42)
    print("[NeuralDBG] Transformer (GPT) failure scenarios\n")

    for name, fn in [
        ("No warmup + high LR -> exploding gradients", scenario_no_warmup),
        ("No LayerNorm -> activation saturation", scenario_no_norm),
        ("No attn scale -> vanishing gradients", scenario_no_scale),
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

    print("\n[DONE] Transformer scenarios complete.")


if __name__ == "__main__":
    main()
