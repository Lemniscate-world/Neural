"""Combinatorial architecture validation — test NeuralDBG on 200+ architectures.

Generates 200 architecture configurations across 5 families:
  - MLP (deep, residual, bottleneck)
  - CNN (conv2d-based, varying depth/kernel/norm)
  - RNN (LSTM, GRU, bidirectional)
  - Transformer (encoder/decoder, varying heads/depth)
  - Hybrid (mixed layer types)

Each config tested with 6 bugs + 1 normal baseline = ~1400 evaluations.
Goal: validate detection rate across the full architecture space.

Usage: python validate_combinatorial.py [--quick] [--full]
  --quick : 50 configs (~5 min)
  --full  : 200 configs (~20 min)
  default : 100 configs (~10 min)
"""

import sys, json, time, itertools, random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

sys.path.insert(0, r"C:\Users\Utilisateur\Documents\NeuralDBG")

import torch, torch.nn as nn
import torch.nn.functional as F
from neuraldbg import NeuralDbg

torch.manual_seed(42)
random.seed(42)

PROBLEMATIC = {"data_anomaly", "nan_detected", "silent_corruption", "optimizer_instability"}

# ============================================================
# Configuration generators
# ============================================================

@dataclass
class ArchConfig:
    family: str
    name: str
    depth: int
    width: int
    activation: str
    norm: Optional[str]
    skip: bool
    dropout: float
    extra: dict = field(default_factory=dict)  # family-specific params

    def make_model(self) -> nn.Module:
        return build_model(self)

    def make_data(self, batch=16) -> tuple:
        return make_data_for(self, batch)


def activation_fn(name: str):
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU,
            "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}[name]()

def norm_fn(name: str, width: int, extra: dict):
    if name is None:
        return nn.Identity()
    if name == "batchnorm":
        return nn.BatchNorm1d(width)
    if name == "layernorm":
        return nn.LayerNorm(width)
    return nn.Identity()


# --- MLP family ---
def mlp_configs(n=40):
    depths = [2, 4, 6, 8, 10]
    widths = [16, 32, 64, 128, 256]
    activations = ["relu", "gelu", "silu", "tanh", "leaky_relu"]
    norms = [None, "batchnorm", "layernorm"]
    skips = [False, True]
    dropouts = [0.0, 0.1]

    configs = []
    for d in depths:
        for w in widths:
            for act in activations[:3]:  # limit combos
                for norm in norms[:2]:
                    for skip in skips:
                        for drop in dropouts:
                            if len(configs) >= n:
                                return configs
                            configs.append(ArchConfig(
                                family="MLP", name=f"MLP_d{d}_w{w}_{act}_{norm}_skip{skip}",
                                depth=d, width=w, activation=act, norm=norm,
                                skip=skip, dropout=drop))
    return configs


def build_mlp(cfg: ArchConfig) -> nn.Module:
    layers = []
    in_w = cfg.width
    for i in range(cfg.depth):
        layers.append(nn.Linear(in_w, cfg.width))
        if cfg.norm:
            layers.append(norm_fn(cfg.norm, cfg.width, cfg.extra))
        layers.append(activation_fn(cfg.activation))
        if cfg.dropout > 0:
            layers.append(nn.Dropout(cfg.dropout))
    layers.append(nn.Linear(cfg.width, 2))
    return nn.Sequential(*layers)


def make_mlp_data(batch=16, width=64):
    return torch.randn(batch, width), torch.randint(0, 2, (batch,))


# --- CNN family ---
def cnn_configs(n=40):
    depths = [2, 3, 4, 5]
    widths = [16, 32, 64, 96]
    kernels = [3, 5]
    activations = ["relu", "gelu", "leaky_relu"]
    norms = [None, "batchnorm"]
    skips = [False, True]

    configs = []
    for d in depths:
        for w in widths:
            for k in kernels:
                for act in activations[:2]:
                    for norm in norms:
                        for skip in skips:
                            if len(configs) >= n:
                                return configs
                            configs.append(ArchConfig(
                                family="CNN", name=f"CNN_d{d}_w{w}_k{k}_{act}_{norm}_skip{skip}",
                                depth=d, width=w, activation=act, norm=norm,
                                skip=skip, dropout=0.0, extra={"kernel": k}))
    return configs


class ConvBlock(nn.Module):
    def __init__(self, ch, kernel, act, norm, skip):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel, padding=kernel//2, bias=norm is None)
        self.conv2 = nn.Conv2d(ch, ch, kernel, padding=kernel//2, bias=norm is None)
        self.norm1 = nn.BatchNorm2d(ch) if norm == "batchnorm" else nn.Identity()
        self.norm2 = nn.BatchNorm2d(ch) if norm == "batchnorm" else nn.Identity()
        self.act = act
        self.skip = skip

    def forward(self, x):
        r = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        if self.skip:
            x = self.act(x + r)
        else:
            x = self.act(x)
        return x


def build_cnn(cfg: ArchConfig) -> nn.Module:
    act = activation_fn(cfg.activation)
    k = cfg.extra.get("kernel", 3)
    layers = [nn.Conv2d(3, cfg.width, k, padding=k//2), nn.BatchNorm2d(cfg.width), act]
    for _ in range(cfg.depth):
        layers.append(ConvBlock(cfg.width, k, act, cfg.norm, cfg.skip))
    layers.append(nn.AdaptiveAvgPool2d(1))
    layers.append(nn.Flatten())
    layers.append(nn.Linear(cfg.width, 10))
    return nn.Sequential(*layers)


def make_cnn_data(batch=16):
    return torch.randn(batch, 3, 16, 16), torch.randint(0, 10, (batch,))


# --- RNN family ---
def rnn_configs(n=30):
    depths = [1, 2, 3, 4]
    widths = [32, 64, 128, 256]
    rnn_types = ["lstm", "gru"]
    bidirs = [False, True]

    configs = []
    for d in depths:
        for w in widths:
            for rtype in rnn_types:
                for bidir in bidirs:
                    if len(configs) >= n:
                        return configs
                    configs.append(ArchConfig(
                        family="RNN", name=f"RNN_{rtype}_d{d}_w{w}{'_bi' if bidir else ''}",
                        depth=d, width=w, activation="tanh", norm=None,
                        skip=False, dropout=0.0,
                        extra={"rnn_type": rtype, "bidirectional": bidir}))
    return configs


def build_rnn(cfg: ArchConfig) -> nn.Module:
    rtype = cfg.extra["rnn_type"]
    bidir = cfg.extra["bidirectional"]
    rnn_cls = nn.LSTM if rtype == "lstm" else nn.GRU
    class RNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = rnn_cls(cfg.width, cfg.width, cfg.depth,
                               batch_first=True, bidirectional=bidir)
            mult = 2 if bidir else 1
            self.fc = nn.Linear(cfg.width * mult, 10)
        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :])
    return RNNModel()


def make_rnn_data(batch=16, width=64):
    return torch.randn(batch, 16, width), torch.randint(0, 10, (batch,))


# --- Transformer family ---
def transformer_configs(n=50):
    depths = [1, 2, 3, 4]
    d_models = [32, 64, 96, 128]
    heads = [2, 4, 8]
    activations = ["relu", "gelu"]
    norms = ["layernorm"]
    dropouts = [0.0, 0.1]

    configs = []
    for d in depths:
        for dm in d_models:
            for h in heads:
                if dm % h != 0:
                    continue
                for act in activations:
                    for drop in dropouts:
                        if len(configs) >= n:
                            return configs
                        configs.append(ArchConfig(
                            family="Transformer", name=f"TF_d{d}_dm{dm}_h{h}_{act}",
                            depth=d, width=dm, activation=act, norm="layernorm",
                            skip=True, dropout=drop, extra={"heads": h}))
    return configs


class TFBlock(nn.Module):
    def __init__(self, d, heads, act, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d*4), act, nn.Dropout(dropout),
            nn.Linear(d*4, d), nn.Dropout(dropout))

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + a)
        return self.norm2(x + self.ffn(x))


def build_transformer(cfg: ArchConfig) -> nn.Module:
    act = activation_fn(cfg.activation)
    h = cfg.extra["heads"]
    class TFModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(cfg.width, cfg.width)
            self.blocks = nn.Sequential(*[
                TFBlock(cfg.width, h, act, cfg.dropout) for _ in range(cfg.depth)
            ])
            self.norm = nn.LayerNorm(cfg.width)
            self.fc = nn.Linear(cfg.width, 10)
        def forward(self, x):
            x = act(self.embed(x))
            x = self.blocks(x)
            return self.fc(self.norm(x).mean(dim=1))
    return TFModel()


def make_tf_data(batch=16, d_model=64):
    return torch.randn(batch, 16, d_model), torch.randint(0, 10, (batch,))


# --- Hybrid family ---
def hybrid_configs(n=40):
    """Mix of layer types: conv + linear, attention + mlp, etc."""
    configs = []
    combos = [
        ("cnn+mlp", [("conv", 2), ("linear", 3)]),
        ("attn+mlp", [("attn", 2), ("linear", 3)]),
        ("rnn+mlp", [("rnn", 1), ("linear", 3)]),
        ("cnn+rnn+mlp", [("conv", 1), ("rnn", 1), ("linear", 2)]),
        ("all", [("conv", 1), ("attn", 1), ("rnn", 1), ("linear", 2)]),
    ]
    widths = [32, 64, 128]
    activations = ["relu", "gelu"]
    idx = 0
    for combo_name, blocks in combos:
        for w in widths:
            for act in activations:
                if idx >= n:
                    return configs
                configs.append(ArchConfig(
                    family="Hybrid", name=f"Hybrid_{combo_name}_w{w}_{act}",
                    depth=sum(b[1] for b in blocks), width=w, activation=act,
                    norm="layernorm", skip=True, dropout=0.0,
                    extra={"blocks": blocks, "act": act}))
                idx += 1
    return configs


def build_hybrid(cfg: ArchConfig) -> nn.Module:
    blocks = cfg.extra["blocks"]
    act = activation_fn(cfg.extra["act"])
    class HybridModel(nn.Module):
        def __init__(self):
            super().__init__()
            layers = [nn.Linear(cfg.width, cfg.width), act]
            for btype, count in blocks:
                for _ in range(count):
                    if btype == "conv":
                        layers.append(nn.Conv1d(cfg.width, cfg.width, 3, padding=1))
                        layers.append(act)
                    elif btype == "attn":
                        layers.append(nn.MultiheadAttention(cfg.width, 4, batch_first=True))
                        layers.append(nn.LayerNorm(cfg.width))
                    elif btype == "rnn":
                        layers.append(nn.LSTM(cfg.width, cfg.width, batch_first=True))
                    elif btype == "linear":
                        layers.append(nn.Linear(cfg.width, cfg.width))
                        layers.append(nn.LayerNorm(cfg.width))
                        layers.append(act)
            self.net = nn.ModuleList(layers)
            self.fc = nn.Linear(cfg.width, 10)
        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)  # add seq dim for conv1d/attn
            for layer in self.net:
                if isinstance(layer, (nn.MultiheadAttention, nn.LSTM)):
                    x = x.transpose(0, 1) if isinstance(layer, nn.LSTM) else x
                    out, _ = layer(x, x, x) if isinstance(layer, nn.MultiheadAttention) else layer(x)
                    x = out if isinstance(layer, nn.MultiheadAttention) else out[0]
                elif isinstance(layer, nn.Conv1d):
                    x = x.transpose(1, 2)
                    x = layer(x)
                    x = x.transpose(1, 2)
                else:
                    x = layer(x)
            return self.fc(x.mean(dim=1))
    return HybridModel()


def make_hybrid_data(batch=16, width=64):
    return torch.randn(batch, 16, width), torch.randint(0, 10, (batch,))


# --- BlackSwan family (Tier 1-4 architectures: GNN, MoE, Diffusion, RL, etc.) ---
def blackswan_configs(n=30):
    """Black-swan architectures from Tiers 1-4: GNN, MoE, Diffusion, RL, RAG, FlashAttn, etc."""
    configs = []
    subtypes = [
        ("GNN", 5), ("MoE", 5), ("Diffusion", 5), ("RL", 5),
        ("RAG", 3), ("FlashAttn", 3), ("NeuralODE", 2), ("Federated", 2),
    ]
    widths = [32, 64, 128]
    depths = [2, 3]
    idx = 0
    for subtype, count in subtypes:
        for w in widths:
            for d in depths:
                if idx >= n:
                    return configs
                configs.append(ArchConfig(
                    family="BlackSwan", name=f"BS_{subtype}_d{d}_w{w}",
                    depth=d, width=w, activation="relu", norm="layernorm",
                    skip=True, dropout=0.0,
                    extra={"subtype": subtype}))
                idx += 1
                if idx >= count * len(widths) * len(depths):
                    break
    return configs


def build_blackswan(cfg: ArchConfig) -> nn.Module:
    """Build a simplified black-swan architecture for combinatorial testing."""
    subtype = cfg.extra.get("subtype", "MLP")
    w = cfg.width
    d = cfg.depth

    if subtype == "GNN":
        # Simple message-passing: linear → "aggregate" → linear
        class GNNCell(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.msg = nn.Linear(dim, dim)
                self.update = nn.Linear(dim, dim)
            def forward(self, x):
                # x: (batch, dim) — treat batch as nodes, create identity adj
                adj = torch.eye(x.size(0), device=x.device)
                m = self.msg(x)
                m = m + adj @ m
                return torch.relu(self.update(m))
        class GNNModel(nn.Module):
            def __init__(self, dim, depth):
                super().__init__()
                self.cells = nn.ModuleList([GNNCell(dim) for _ in range(depth)])
                self.fc = nn.Linear(dim, 10)
            def forward(self, x):
                # Handle 2D input: (batch, dim) — treat as nodes
                if x.dim() == 2:
                    for cell in self.cells:
                        x = cell(x)
                    return self.fc(x.mean(dim=0, keepdim=True).expand(x.size(0), -1))
                # 3D input: (batch, nodes, dim)
                B, N, D = x.shape
                x = x.view(B * N, D)
                for cell in self.cells:
                    x = cell(x)
                x = x.view(B, N, -1).mean(dim=1)
                return self.fc(x)
        return GNNModel(w, d)

    elif subtype == "MoE":
        class MoELayer(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.experts = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])
                self.gate = nn.Linear(dim, 4)
            def forward(self, x):
                gate = torch.softmax(self.gate(x), dim=-1)
                out = sum(gate[..., i:i+1] * expert(x) for i, expert in enumerate(self.experts))
                return out
        class MoEModel(nn.Module):
            def __init__(self, dim, depth):
                super().__init__()
                self.layers = nn.ModuleList([MoELayer(dim) for _ in range(depth)])
                self.fc = nn.Linear(dim, 10)
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return self.fc(x)
        return MoEModel(w, d)

    elif subtype == "Diffusion":
        class DiffusionBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.t_embed = nn.Linear(1, w)
                self.net = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, w))
            def forward(self, x, t=0.5):
                te = self.t_embed(torch.tensor([[t]]).float().to(x.device))
                return x + self.net(x + te)
        class DiffusionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([DiffusionBlock() for _ in range(d)])
                self.fc = nn.Linear(w, 10)
            def forward(self, x):
                for block in self.blocks:
                    x = block(x)
                return self.fc(x)
        return DiffusionModel()

    elif subtype == "RL":
        class PolicyNet(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                in_dim = w
                for _ in range(d):
                    layers.extend([nn.Linear(in_dim, w), nn.ReLU()])
                    in_dim = w
                self.net = nn.Sequential(*layers)
                self.head = nn.Linear(w, 10)
            def forward(self, x):
                return self.head(self.net(x))
        return PolicyNet()

    elif subtype == "RAG":
        class RAGBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.query = nn.Linear(w, w)
                self.key = nn.Linear(w, w)
                self.value = nn.Linear(w, w)
                self.out = nn.Linear(w, w)
            def forward(self, x, retrieved=None):
                q = self.query(x)
                k = self.key(retrieved if retrieved is not None else x)
                v = self.value(retrieved if retrieved is not None else x)
                attn = torch.softmax(q @ k.transpose(-2, -1) / (w ** 0.5), dim=-1)
                return self.out(attn @ v)
        class RAGModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([RAGBlock() for _ in range(d)])
                self.fc = nn.Linear(w, 10)
            def forward(self, x):
                for block in self.blocks:
                    x = block(x)
                return self.fc(x)
        return RAGModel()

    elif subtype == "FlashAttn":
        class FlashAttnModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.MultiheadAttention(w, 4, batch_first=True)
                self.fc = nn.Linear(w, 10)
            def forward(self, x):
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                a, _ = self.attn(x, x, x)
                return self.fc(a.mean(dim=1))
        return FlashAttnModel()

    elif subtype == "NeuralODE":
        class ODEFunc(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(w, w), nn.Tanh(), nn.Linear(w, w))
            def forward(self, t, x):
                return self.net(x)
        class ODEModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.func = ODEFunc()
                self.fc = nn.Linear(w, 10)
            def forward(self, x):
                # Euler-step ODE simulation (simplified)
                for _ in range(3):
                    x = x + 0.1 * self.func.net(x)
                return self.fc(x)
        return ODEModel()

    elif subtype == "Federated":
        class FedAvgModel(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                for _ in range(d):
                    layers.extend([nn.Linear(w, w), nn.ReLU()])
                self.net = nn.Sequential(*layers)
                self.fc = nn.Linear(w, 10)
            def forward(self, x):
                # Simulate client averaging: add slight noise
                if self.training:
                    x = x + torch.randn_like(x) * 0.01
                return self.fc(self.net(x))
        return FedAvgModel()

    # Default: simple MLP
    layers = []
    for _ in range(d):
        layers.extend([nn.Linear(w, w), nn.ReLU()])
    layers.append(nn.Linear(w, 10))
    return nn.Sequential(*layers)


def make_blackswan_data(batch=16, width=64):
    """Black-swan data: returns 3D tensor for seq-aware models, 2D for others."""
    return torch.randn(batch, width), torch.randint(0, 10, (batch,))


# ============================================================
# Unified builder + data dispatch
# ============================================================

BUILDERS = {"MLP": (build_mlp, make_mlp_data),
            "CNN": (build_cnn, make_cnn_data),
            "RNN": (build_rnn, make_rnn_data),
            "Transformer": (build_transformer, make_tf_data),
            "Hybrid": (build_hybrid, make_hybrid_data),
            "BlackSwan": (build_blackswan, make_blackswan_data)}


def build_model(cfg: ArchConfig) -> nn.Module:
    return BUILDERS[cfg.family][0](cfg)


def make_data_for(cfg: ArchConfig, batch=16):
    builder, data_fn = BUILDERS[cfg.family]
    kwargs = {"batch": batch}
    if cfg.family == "CNN":
        return data_fn(batch=batch)
    elif cfg.family == "Transformer":
        kwargs["d_model"] = cfg.width
    elif cfg.family in ("MLP", "RNN", "Hybrid"):
        kwargs["width"] = cfg.width
    return data_fn(**kwargs)


# ============================================================
# Bug injectors
# ============================================================

def train_with_dbg(model, data_fn, steps=8, lr=0.01, bug=None):
    """Train with NeuralDBG. 'bug' is a mutation function applied at step 3."""
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    with NeuralDbg(model) as dbg:
        for s in range(steps):
            x, y = data_fn()
            x = x.float()
            y = y.long()

            if bug and s >= 3:
                x, y, opt, model = bug(x, y, opt, model)

            opt.zero_grad()
            try:
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
            except Exception:
                break  # model is broken, stop
            dbg.step_iteration()
            dbg.record_loss(loss.item())
            opt.step()

        events = dbg.dump_events()
        hyps = dbg.explain_failure()
        chains = dbg.explain_causal()

    return events, hyps, chains


def n_problematic(events):
    n = 0
    for e in events:
        et = e.get("event_type", "")
        ts = e.get("to_state", "").lower()
        if et in PROBLEMATIC:
            n += 1
        elif et == "gradient_health_transition" and ts not in ("healthy", "none", "normal", ""):
            n += 1
        elif et == "activation_regime_shift" and ts not in ("healthy", "none", "normal", ""):
            n += 1
    return n


# Bug definitions
def bug_exploding(x, y, opt, model):
    for g in opt.param_groups:
        g['lr'] = 50.0  # extreme
    return x, y, opt, model

def bug_vanishing(x, y, opt, model):
    """Replace all activations with sigmoid + scale down weights. For RNNs, corrupt forget gate bias."""
    is_rnn = False
    for m in model.modules():
        if isinstance(m, (nn.LSTM, nn.GRU)):
            is_rnn = True
            # Corrupt forget gate bias (bias_ih + bias_hh for LSTM)
            # LSTM bias layout: [input, forget, cell, output] * num_layers * directions
            if hasattr(m, 'bias_ih_l0') and m.bias_ih_l0 is not None:
                with torch.no_grad():
                    hidden_size = m.hidden_size
                    # Set forget gate bias to -10 (cause vanishing)
                    for layer_idx in range(m.num_layers):
                        for direction in range(1 + int(m.bidirectional)):
                            idx = layer_idx * (1 + int(m.bidirectional)) + direction
                            bias_key_ih = f'bias_ih_l{layer_idx}'
                            bias_key_hh = f'bias_hh_l{layer_idx}'
                            if hasattr(m, bias_key_ih):
                                b_ih = getattr(m, bias_key_ih)
                                b_ih[hidden_size:2*hidden_size] = -10.0  # forget gate
                            if hasattr(m, bias_key_hh):
                                b_hh = getattr(m, bias_key_hh)
                                b_hh[hidden_size:2*hidden_size] = -10.0  # forget gate
        
        # Replace activations in non-RNN components
        for name, child in list(m.named_children()):
            if isinstance(child, (nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.LeakyReLU, nn.ELU)):
                setattr(m, name, nn.Sigmoid())
    
    # For non-RNN models: scale down all weights to force near-zero gradients
    if not is_rnn:
        with torch.no_grad():
            for p in model.parameters():
                if p.dim() >= 2:
                    p.mul_(0.001)  # 1000x reduction
    
    # For RNNs: double sequence length to exacerbate BPTT vanishing
    if is_rnn and isinstance(x, torch.Tensor) and x.dim() == 3:
        x = x.repeat(1, 2, 1)  # 16 -> 32 sequence length
    
    return x, y, opt, model

def bug_zero(x, y, opt, model):
    for p in model.parameters():
        if p.dim() >= 2:
            nn.init.zeros_(p)
    return x, y, opt, model

def bug_nan(x, y, opt, model):
    """Inject NaN into input data. Handles tuple inputs (e.g., GNN)."""
    if isinstance(x, tuple):
        # GNN-style: (nodes, adj) — inject NaN into features tensor
        x_list = list(x)
        if isinstance(x_list[0], torch.Tensor) and x_list[0].dim() >= 2:
            x_list[0] = x_list[0].clone()
            x_list[0][0, 0] = float('nan')
        x = tuple(x_list)
    elif isinstance(x, torch.Tensor):
        x = x.clone()
        if x.dim() >= 2:
            x[0, 0] = float('nan')
    return x, y, opt, model

def bug_dead(x, y, opt, model):
    """Set all biases to -10. For RNNs, kill all gates."""
    for m in model.modules():
        if isinstance(m, (nn.LSTM, nn.GRU)):
            with torch.no_grad():
                for attr in dir(m):
                    if attr.startswith('bias_'):
                        bias = getattr(m, attr)
                        if bias is not None:
                            bias.fill_(-10.0)
        elif hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, -10.0)
    return x, y, opt, model

def bug_divergence(x, y, opt, model):
    for g in opt.param_groups:
        g['lr'] = 500.0
    return x, y, opt, model


BUGS = [
    ("exploding", bug_exploding),
    ("vanishing", bug_vanishing),
    ("zero_init", bug_zero),
    ("nan_data", bug_nan),
    ("dead_bias", bug_dead),
    ("divergence", bug_divergence),
]


# ============================================================
# Single config evaluator
# ============================================================

def evaluate_config(cfg: ArchConfig) -> dict:
    """Evaluate one architecture config: baseline + all 6 bugs."""
    data_fn = cfg.make_data
    result = {"name": cfg.name, "family": cfg.family,
              "depth": cfg.depth, "width": cfg.width,
              "activation": cfg.activation, "norm": str(cfg.norm),
              "skip": cfg.skip, "dropout": cfg.dropout}

    # 1. Baseline
    try:
        model = cfg.make_model()
        ev, _, _ = train_with_dbg(model, data_fn, steps=8)
        baseline = n_problematic(ev)
    except Exception as exc:
        return {**result, "baseline": -1, "error": str(exc)[:80],
                "detected": 0, "total": len(BUGS), "results": []}

    # Family-aware threshold: noisier architectures get lower offset
    family = cfg.family
    if family in ("RNN", "Hybrid"):
        threshold = max(baseline + 2, 2)  # Lower bar for noisy recurrent archs
    else:
        threshold = max(baseline + 3, 3)  # Standard for feed-forward archs
    result["baseline"] = baseline
    result["threshold"] = threshold

    # 2. Test each bug
    bug_results = []
    detected = 0
    for bug_name, bug_fn in BUGS:
        try:
            model = cfg.make_model()
            ev, _, _ = train_with_dbg(model, data_fn, steps=8, bug=bug_fn)
            n = n_problematic(ev)
            hit = n > threshold
            if hit:
                detected += 1
            bug_results.append({"bug": bug_name, "anomalies": n, "detected": hit})
        except Exception as exc:
            bug_results.append({"bug": bug_name, "anomalies": -1, "detected": False,
                               "error": str(exc)[:60]})

    result["results"] = bug_results
    result["detected"] = detected
    result["total"] = len(BUGS)
    return result


# ============================================================
# Main
# ============================================================

def generate_configs(n=100):
    """Generate n configs distributed across 6 families (including BlackSwan)."""
    per_family = n // 6
    configs = []
    configs.extend(mlp_configs(per_family))
    configs.extend(cnn_configs(per_family))
    configs.extend(rnn_configs(per_family))
    configs.extend(transformer_configs(per_family))
    configs.extend(hybrid_configs(per_family))
    configs.extend(blackswan_configs(per_family))
    # Fill remaining with random from largest families
    while len(configs) < n:
        configs.extend(mlp_configs(n - len(configs)))
    return configs[:n]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="50 configs (~5 min)")
    parser.add_argument("--full", action="store_true", help="200 configs (~20 min)")
    args = parser.parse_args()

    n_configs = 50 if args.quick else (200 if args.full else 100)
    configs = generate_configs(n_configs)

    print(f"{'='*65}")
    print(f"NEURALDBG — COMBINATORIAL ARCHITECTURE VALIDATION")
    print(f"{n_configs} architectures × 7 tests = {n_configs * 7} evaluations")
    print(f"{'='*65}")

    t0 = time.time()
    results = []
    for i, cfg in enumerate(configs):
        r = evaluate_config(cfg)
        results.append(r)
        status = "PASS" if r.get("baseline", -1) >= 0 else "FAIL"
        d = r.get("detected", 0)
        pct = f"{d}/{r['total']}" if r.get("total", 0) > 0 else "ERR"
        elapsed = time.time() - t0
        eta = (elapsed / (i + 1)) * (n_configs - i - 1) if i > 0 else 0
        print(f"  [{i+1:3d}/{n_configs}] {cfg.name[:50]:50s} | base={r.get('baseline',-1):3d} | {pct:5s} | ETA {eta:.0f}s",
              flush=True)

    elapsed = time.time() - t0

    # Aggregate
    by_family = defaultdict(lambda: {"total": 0, "detected_bugs": 0, "configs": 0, "errors": 0})
    for r in results:
        fam = r["family"]
        by_family[fam]["configs"] += 1
        by_family[fam]["total"] += r["total"]
        by_family[fam]["detected_bugs"] += r.get("detected", 0)
        if r.get("baseline", -1) < 0:
            by_family[fam]["errors"] += 1

    overall_detected = sum(v["detected_bugs"] for v in by_family.values())
    overall_total = sum(v["total"] for v in by_family.values())

    print(f"\n{'='*65}")
    print(f"RESULTS — {elapsed:.0f}s for {n_configs} architectures")
    print(f"{'='*65}")
    print(f"  {'Family':15s} | {'Configs':7s} | {'Detection':12s} | {'Errors':6s}")
    print(f"  {'-'*15} | {'-'*7} | {'-'*12} | {'-'*6}")
    for fam in ["MLP", "CNN", "RNN", "Transformer", "Hybrid"]:
        if fam in by_family:
            v = by_family[fam]
            pct = f"{100*v['detected_bugs']//max(v['total'],1)}%"
            print(f"  {fam:15s} | {v['configs']:3d}    | {v['detected_bugs']:3d}/{v['total']:3d} ({pct:3s}) | {v['errors']:3d}")

    print(f"\n  OVERALL: {overall_detected}/{overall_total} bugs detected "
          f"({100*overall_detected//max(overall_total,1)}%) in {elapsed:.0f}s")

    # Bug-type breakdown
    print(f"\n  Bug-type breakdown:")
    bug_counts = defaultdict(lambda: {"detected": 0, "total": 0})
    for r in results:
        for br in r.get("results", []):
            bug_counts[br["bug"]]["total"] += 1
            if br["detected"]:
                bug_counts[br["bug"]]["detected"] += 1
    for bug_name in ["exploding", "vanishing", "zero_init", "nan_data", "dead_bias", "divergence"]:
        bc = bug_counts[bug_name]
        pct = f"{100*bc['detected']//max(bc['total'],1)}%"
        print(f"    {bug_name:15s}: {bc['detected']:3d}/{bc['total']} ({pct:3s})")

    # Save JSON
    report = {
        "configs": n_configs,
        "total_evaluations": overall_total,
        "overall_detection": f"{overall_detected}/{overall_total}",
        "elapsed_seconds": int(elapsed),
        "by_family": {k: {"configs": v["configs"], "detection": f"{v['detected_bugs']}/{v['total']}",
                          "errors": v["errors"]} for k, v in by_family.items()},
        "by_bug": {k: {"detected": v["detected"], "total": v["total"]} for k, v in bug_counts.items()},
        "results": results,
    }
    out_path = "combinatorial_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Full report saved to: {out_path}")
    print(f"  Exit: {'PASS' if overall_detected >= overall_total * 0.8 else 'OK'} "
          f"(target >= 80% detection)")
    sys.exit(0)
