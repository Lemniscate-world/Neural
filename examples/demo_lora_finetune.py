#!/usr/bin/env python3
"""
LLM fine-tuning (LoRA) failure scenarios demonstrating NeuralDBG causal inference.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from neuraldbg import NeuralDbg


class LoRALayer(nn.Module):
    """Low-rank adapter for a linear layer."""

    def __init__(self, in_dim, out_dim, rank=4, alpha=1.0):
        super().__init__()
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x):
        return (x @ self.lora_a @ self.lora_b) * self.scale


class LoRALinear(nn.Module):
    """Linear layer with optional LoRA adapter."""

    def __init__(self, in_dim, out_dim, use_lora=True, rank=4):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.use_lora = use_lora
        if use_lora:
            self.lora = LoRALayer(in_dim, out_dim, rank=rank)
        self.lora_enabled = True

    def forward(self, x):
        out = self.linear(x)
        if self.use_lora and self.lora_enabled:
            out = out + self.lora(x)
        return out


class LoRAGPT(nn.Module):
    """Minimal GPT with LoRA on attention projections."""

    def __init__(
        self, vocab_size=100, d_model=32, n_heads=2, n_layers=2, seq_len=16, lora_rank=4
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "ln1": nn.LayerNorm(d_model),
                        "attn_q": LoRALinear(d_model, d_model, rank=lora_rank),
                        "attn_k": LoRALinear(d_model, d_model, rank=lora_rank),
                        "attn_v": LoRALinear(d_model, d_model, rank=lora_rank),
                        "attn_o": LoRALinear(d_model, d_model, rank=lora_rank),
                        "ln2": nn.LayerNorm(d_model),
                        "mlp": nn.Sequential(
                            LoRALinear(d_model, d_model * 4, rank=lora_rank),
                            nn.GELU(),
                            LoRALinear(d_model * 4, d_model, rank=lora_rank),
                        ),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            q = block["attn_q"](block["ln1"](x))
            k = block["attn_k"](block["ln1"](x))
            v = block["attn_v"](block["ln1"](x))
            att = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.d_model), dim=-1)
            x = x + block["attn_o"](att @ v)
            x = x + block["mlp"](block["ln2"](x))
        return self.lm_head(self.ln_f(x))


def make_loader(batch_size=4, seq_len=16, vocab_size=100, num_samples=80):
    X = torch.randint(0, vocab_size, (num_samples, seq_len))
    y = torch.randint(0, vocab_size, (num_samples, seq_len))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size)


def train_lora(
    model,
    dataloader,
    num_steps=20,
    lr=0.001,
    freeze_base=True,
    nan_step=None,
    forgetting_step=None,
):
    """Train LoRA adapters with NeuralDBG monitoring."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    if freeze_base:
        for name, p in model.named_parameters():
            if "lora" not in name:
                p.requires_grad = False

    nan_injected = False
    with NeuralDbg(model, threshold_vanishing=1e-4, threshold_exploding=1.0) as dbg:
        for step in range(num_steps):
            for bx, by in dataloader:
                optimizer.zero_grad()
                dbg.step = step

                if nan_step is not None and step >= nan_step and not nan_injected:
                    model.blocks[0]["attn_q"].lora.lora_a.data[0, 0] = float("nan")
                    nan_injected = True

                if forgetting_step is not None and step >= forgetting_step:
                    for n, p in model.named_parameters():
                        if "lora" in n and p.grad is not None:
                            p.grad.data.zero_()

                out = model(bx)
                loss = criterion(out.reshape(-1, out.size(-1)), by.reshape(-1))
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
    }


def scenario_nan_lora(num_steps=20):
    """NaN injected into LoRA adapter -> data anomaly."""
    model = LoRAGPT()
    loader = make_loader()
    return train_lora(model, loader, num_steps=num_steps, lr=0.001, nan_step=5)


def scenario_exploding_lora(num_steps=20):
    """Very high LR for LoRA adapters -> gradient explosion."""
    model = LoRAGPT()
    loader = make_loader()
    return train_lora(model, loader, num_steps=num_steps, lr=100.0)


def scenario_forgetting_lora(num_steps=20):
    """Gradients zeroed -> catastrophic forgetting (vanishing)."""
    model = LoRAGPT()
    loader = make_loader()
    return train_lora(model, loader, num_steps=num_steps, lr=0.001, forgetting_step=3)


def main():
    torch.manual_seed(42)
    print("[NeuralDBG] LoRA fine-tuning failure scenarios\n")

    for name, fn in [
        ("NaN in LoRA adapter", scenario_nan_lora),
        ("Exploding gradients (LR=100)", scenario_exploding_lora),
        ("Gradient zeroing -> forgetting", scenario_forgetting_lora),
    ]:
        dbg = fn(num_steps=12)
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
                for h in hyps[:3]:
                    print(f"  [{h.confidence:.2f}] {h.description}")

    print("\n[DONE] LoRA scenarios complete.")


if __name__ == "__main__":
    main()
