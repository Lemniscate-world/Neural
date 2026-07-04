"""Real-architecture validation: NeuralDBG on production-grade model families.

Tests NeuralDBG detection + causal chains on:
  1. Mini ResNet (CNN, conv2d + batchnorm + residual blocks)
  2. Mini Transformer (MultiHeadAttention + positional encoding + FFN)

Each architecture: 5 bugs + 1 normal (must not false-positive).

Usage: python validate_real_architectures.py
"""

import sys, types
sys.path.insert(0, r"C:\Users\Utilisateur\Documents\NeuralDBG")

import torch, torch.nn as nn
import torch.nn.functional as F
from neuraldbg import NeuralDbg

torch.manual_seed(42)

PROBLEMATIC = {"data_anomaly", "nan_detected", "silent_corruption", "optimizer_instability"}


# ============================================================
# 1. Mini ResNet
# ============================================================

class ResidualConvBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)


class MiniResNet(nn.Module):
    def __init__(self, in_ch=3, num_cls=10, base=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base), nn.ReLU())
        self.block1 = ResidualConvBlock(base)
        self.block2 = ResidualConvBlock(base)
        self.block3 = ResidualConvBlock(base)
        self.block4 = ResidualConvBlock(base)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(base, num_cls)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x); x = self.block2(x)
        x = self.block3(x); x = self.block4(x)
        return self.fc(self.pool(x).flatten(1))


# ============================================================
# 2. Mini Transformer
# ============================================================

class PosEnc(nn.Module):
    def __init__(self, d, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-torch.log(torch.tensor(10000.0)) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, d, heads=4, ff=128, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn   = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(ff, d), nn.Dropout(dropout))

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + a)
        return self.norm2(x + self.ffn(x))


class MiniTransformer(nn.Module):
    def __init__(self, vocab=256, d=64, heads=4, blocks=3, num_cls=10):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos   = PosEnc(d)
        self.enc   = nn.Sequential(*[TransformerBlock(d, heads) for _ in range(blocks)])
        self.norm  = nn.LayerNorm(d)
        self.fc    = nn.Linear(d, num_cls)

    def forward(self, x):
        x = self.pos(self.embed(x))
        x = self.enc(x)
        return self.fc(self.norm(x).mean(dim=1))


# ============================================================
# Data
# ============================================================

def img_data(b=32, c=3, s=8, nc=10):
    return torch.randn(b, c, s, s), torch.randint(0, nc, (b,))

def seq_data(b=32, sl=32, v=256, nc=10):
    return torch.randint(0, v, (b, sl)), torch.randint(0, nc, (b,))


# ============================================================
# Training + NeuralDBG
# ============================================================

def train(model, data_fn, steps=10, lr=0.01, mut=None, inj=None, inj_step=3):
    """Train with NeuralDBG. mut=pre-train mutation, inj=per-step bug injector."""
    if mut:
        mut(model)

    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    with NeuralDbg(model) as dbg:
        for s in range(steps):
            x, y = data_fn()
            if inj and s >= inj_step:
                x, y = inj(x, y)

            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            dbg.step_iteration()
            dbg.record_loss(loss.item())
            opt.step()

        events = dbg.dump_events()
        hyps   = dbg.explain_failure()
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


# ============================================================
# Bug injectors  (return modified x, y)
# ============================================================

def inj_nan(x, y):
    x = x.clone()
    x[0, 0] = float('nan')
    return x, y

def inj_nan_seq(x, y):
    """Corrupt the first token to trigger embedding lookup issues."""
    x = x.clone()
    # Set first token to 0 (valid) but we'll corrupt via model mutation instead
    return x, y

def mut_nan_embedding(m):
    """Inject NaN into embedding weights to simulate data corruption."""
    for mod in m.modules():
        if isinstance(mod, nn.Embedding):
            with torch.no_grad():
                mod.weight[0, 0] = float('nan')

# Pre-training mutations (modify model in-place)
def mut_saturate_resnet(m):
    for mod in m.modules():
        if isinstance(mod, ResidualConvBlock):
            orig = mod.forward
            def sat_fwd(self, x, _orig=orig):
                r = x
                x = torch.sigmoid(self.bn1(self.conv1(x)))
                x = self.bn2(self.conv2(x))
                return torch.sigmoid(x + r)
            mod.forward = types.MethodType(sat_fwd, mod)

def mut_saturate_transformer(m):
    for mod in m.modules():
        if isinstance(mod, TransformerBlock):
            mod.ffn[1] = nn.Sigmoid()

def mut_zero_init(m):
    for p in m.parameters():
        if p.dim() >= 2:
            nn.init.zeros_(p)

def mut_dead_bias(m):
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.LayerNorm)) and mod.bias is not None:
            nn.init.constant_(mod.bias, -10.0)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("NEURALDBG -- REAL ARCHITECTURE VALIDATION")
    print("Mini ResNet + Mini Transformer | 5 bugs + 1 normal each")
    print("=" * 65)

    all_r = []

    # --- Mini ResNet ---
    print("\n--- Mini ResNet (CNN, 4 residual blocks) ---")
    me, _, _ = train(MiniResNet(), img_data, steps=10)
    bl_resnet = n_problematic(me)
    th = max(bl_resnet + 5, 10)
    print(f"  Baseline: {bl_resnet} anomalies | Threshold: {th}\n")

    resnet_tests = [
        ("Exploding LR",      None,                  None,       True),
        ("Vanishing (sigmoid)", mut_saturate_resnet,  None,       True),
        ("Zero init",         mut_zero_init,          None,       True),
        ("NaN in data",       None,                  inj_nan,    True),
        ("Dead bias",         mut_dead_bias,          None,       True),
        ("Normal (no bug)",   None,                  None,       False),
    ]

    for label, mut, inj, is_bug in resnet_tests:
        lr = 10.0 if label == "Exploding LR" else 0.01
        model = MiniResNet()
        events, hyps, chains = train(model, img_data, steps=10, lr=lr, mut=mut, inj=inj)
        n = n_problematic(events)
        detected = n > th
        top_chain = f"{chains[0].root_cause} -> {chains[0].final_symptom}" if chains else "no chain"
        if not is_bug:
            status = "OK" if not detected else "FP!"
        elif detected:
            status = "PASS"
        else:
            status = "MISS"
        print(f"  {label:28s} | {status:5s} | {n:2d} anom | {top_chain[:55]}")
        all_r.append({"arch": "ResNet", "bug": label, "status": status, "anomalies": n})

    # --- Mini Transformer ---
    print("\n--- Mini Transformer (3 encoder blocks) ---")
    me, _, _ = train(MiniTransformer(), seq_data, steps=10)
    bl_tf = n_problematic(me)
    th_tf = max(bl_tf + 5, 10)
    print(f"  Baseline: {bl_tf} anomalies | Threshold: {th_tf}\n")

    tf_tests = [
        ("Exploding LR",        None,                       None,              True),
        ("Vanishing (sigmoid)", mut_saturate_transformer,   None,              True),
        ("Zero init",           mut_zero_init,              None,              True),
        ("NaN in data",         mut_nan_embedding,          None,              True),
        ("Dead bias",           mut_dead_bias,              None,              True),
        ("Normal (no bug)",     None,                       None,              False),
    ]

    for label, mut, inj, is_bug in tf_tests:
        lr = 10.0 if label == "Exploding LR" else 0.01
        model = MiniTransformer()
        events, hyps, chains = train(model, seq_data, steps=10, lr=lr, mut=mut, inj=inj)
        n = n_problematic(events)
        detected = n > th_tf
        top_chain = f"{chains[0].root_cause} -> {chains[0].final_symptom}" if chains else "no chain"
        if not is_bug:
            status = "OK" if not detected else "FP!"
        elif detected:
            status = "PASS"
        else:
            status = "MISS"
        print(f"  {label:28s} | {status:5s} | {n:2d} anom | {top_chain[:55]}")
        all_r.append({"arch": "Transformer", "bug": label, "status": status, "anomalies": n})

    # --- Summary ---
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    for r in all_r:
        print(f"  {r['arch']:15s} | {r['bug']:25s} | {r['status']:5s} | {r['anomalies']:2d}")

    n_pass = sum(1 for r in all_r if r['status'] == 'PASS')
    n_ok   = sum(1 for r in all_r if r['status'] == 'OK')
    n_fp   = sum(1 for r in all_r if r['status'] == 'FP!')
    n_miss = sum(1 for r in all_r if r['status'] == 'MISS')
    n_bugs = sum(1 for r in all_r if r['bug'] != 'Normal (no bug)')
    n_norm = sum(1 for r in all_r if r['bug'] == 'Normal (no bug)')

    print(f"\n  Detection: {n_pass}/{n_bugs} bugs ({100*n_pass//max(n_bugs,1)}%)")
    print(f"  False positives: {n_fp}/{n_norm}")
    print(f"  Missed: {n_miss}")

    sys.exit(0 if n_fp == 0 else 1)
