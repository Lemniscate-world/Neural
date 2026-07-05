"""Architecture Fuzzer — Tier 2 Black-Swan Discovery.

Randomly generates valid PyTorch models, trains them with random bugs,
and records any crashes, unexpected behaviors, or detection failures.
Discovers unknown failure modes in the architecture space.

Usage: python arch_fuzzer.py [--runs 100] [--seed 42]
"""

import sys, json, random, time, itertools
from collections import defaultdict
from dataclasses import dataclass, field

sys.path.insert(0, r"C:\Users\Utilisateur\Documents\NeuralDBG")

import torch, torch.nn as nn
import torch.nn.functional as F
from neuraldbg import NeuralDbg

# ============================================================
# Fuzzer components
# ============================================================

LAYER_TYPES = [
    "linear", "conv1d", "conv2d", "lstm", "gru",
    "multihead_attention", "batchnorm1d", "batchnorm2d", "layernorm",
    "dropout", "skip_connection",
]

ACTIVATIONS = ["ReLU", "GELU", "SiLU", "Tanh", "LeakyReLU", "ELU", "Sigmoid"]

BUGS = [
    ("exploding", lambda opt: setattr(opt.param_groups[0], 'lr', random.uniform(10, 500))),
    ("vanishing", lambda m: _saturate_model(m)),
    ("nan_data", lambda x: x.index_put_([torch.tensor([0]), torch.tensor([0])], torch.tensor(float('nan'))) if x.dim() >= 2 else x),
    ("zero_init", lambda m: [nn.init.zeros_(p) for p in m.parameters() if p.dim() >= 2]),
    ("dead_bias", lambda m: [nn.init.constant_(p, -random.uniform(1, 20)) for p in m.parameters() if p.dim() == 1 and 'bias' in str(type(p)).lower()] or None),
    ("divergence", lambda opt: setattr(opt.param_groups[0], 'lr', random.uniform(100, 1000))),
    ("mixed_precision", lambda m: m.half()),
]

def _saturate_model(model):
    """Replace random activations with Sigmoid to cause vanishing."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.LeakyReLU, nn.ELU)):
            if random.random() < 0.5:
                parent = model
                for part in name.split('.')[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, name.split('.')[-1], nn.Sigmoid())


@dataclass
class FuzzResult:
    model_name: str
    layers: list
    bug: str
    events: int
    crashed: bool
    error: str = ""
    detected: bool = False


# ============================================================
# Random model generator
# ============================================================

def generate_random_model(max_layers=8, input_dim=None, output_dim=2):
    """Generate a random valid PyTorch model."""
    if input_dim is None:
        input_dim = random.choice([8, 16, 32, 64, 128])

    layers = []
    current_dim = input_dim
    current_channels = None
    is_image = False
    has_rnn = False

    num_layers = random.randint(2, max_layers)
    for i in range(num_layers):
        ltype = random.choice(LAYER_TYPES)

        if ltype == "linear":
            out_dim = random.choice([16, 32, 64, 128, 256])
            layers.append(("linear", {"in": current_dim, "out": out_dim}))
            current_dim = out_dim
            current_channels = None

        elif ltype == "conv1d":
            out_ch = random.choice([16, 32, 64])
            kernel = random.choice([3, 5, 7])
            layers.append(("conv1d", {"in": current_dim, "out": out_ch, "kernel": kernel}))
            current_dim = out_ch
            is_image = True

        elif ltype == "conv2d":
            if not is_image:
                continue  # skip if we haven't established image mode
            out_ch = random.choice([16, 32, 64])
            kernel = random.choice([3, 5])
            layers.append(("conv2d", {"in": current_dim, "out": out_ch, "kernel": kernel}))
            current_dim = out_ch

        elif ltype == "lstm":
            hidden = random.choice([16, 32, 64, 128])
            layers.append(("lstm", {"input": current_dim, "hidden": hidden}))
            current_dim = hidden
            has_rnn = True

        elif ltype == "gru":
            hidden = random.choice([16, 32, 64, 128])
            layers.append(("gru", {"input": current_dim, "hidden": hidden}))
            current_dim = hidden
            has_rnn = True

        elif ltype == "multihead_attention":
            if current_dim < 8:
                continue
            heads = random.choice([2, 4, 8])
            if current_dim % heads != 0:
                current_dim = (current_dim // heads) * heads
            layers.append(("mha", {"embed_dim": current_dim, "heads": heads}))
            # output dim stays same for MHA

        elif ltype == "batchnorm1d":
            layers.append(("bn1d", {"dim": current_dim}))

        elif ltype == "batchnorm2d":
            if is_image:
                layers.append(("bn2d", {"dim": current_dim}))

        elif ltype == "layernorm":
            layers.append(("ln", {"dim": current_dim}))

        elif ltype == "dropout":
            layers.append(("dropout", {"p": random.uniform(0.1, 0.5)}))

        elif ltype == "skip_connection":
            pass  # handled in forward

        # Add activation after some layers
        if random.random() < 0.6:
            act = random.choice(ACTIVATIONS)
            layers.append(("activation", {"type": act}))

    return layers, current_dim, is_image, has_rnn


def build_model_from_spec(spec, input_dim, output_dim):
    """Build a PyTorch model from a layer specification."""
    layers_spec, final_dim, is_image, has_rnn = spec

    class FuzzModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.is_image = is_image
            self.has_rnn = has_rnn
            self.modules_list = nn.ModuleList()
            self.layer_info = []

            current_dim = input_dim
            current_ch = input_dim

            for ltype, params in layers_spec:
                if ltype == "linear":
                    m = nn.Linear(params["in"], params["out"])
                    self.modules_list.append(m)
                    self.layer_info.append(("linear", m))
                    current_dim = params["out"]
                    current_ch = params["out"]
                elif ltype == "conv1d":
                    m = nn.Conv1d(params["in"], params["out"], params["kernel"], padding=params["kernel"]//2)
                    self.modules_list.append(m)
                    self.layer_info.append(("conv1d", m))
                    current_dim = params["out"]
                    current_ch = params["out"]
                elif ltype == "conv2d":
                    m = nn.Conv2d(params["in"], params["out"], params["kernel"], padding=params["kernel"]//2)
                    self.modules_list.append(m)
                    self.layer_info.append(("conv2d", m))
                    current_dim = params["out"]
                    current_ch = params["out"]
                elif ltype == "lstm":
                    m = nn.LSTM(params["input"], params["hidden"], batch_first=True)
                    self.modules_list.append(m)
                    self.layer_info.append(("lstm", m))
                    current_dim = params["hidden"]
                elif ltype == "gru":
                    m = nn.GRU(params["input"], params["hidden"], batch_first=True)
                    self.modules_list.append(m)
                    self.layer_info.append(("gru", m))
                    current_dim = params["hidden"]
                elif ltype == "mha":
                    m = nn.MultiheadAttention(params["embed_dim"], params["heads"], batch_first=True)
                    self.modules_list.append(m)
                    self.layer_info.append(("mha", m))
                elif ltype == "bn1d":
                    m = nn.BatchNorm1d(params["dim"])
                    self.modules_list.append(m)
                    self.layer_info.append(("bn1d", m))
                elif ltype == "bn2d":
                    m = nn.BatchNorm2d(params["dim"])
                    self.modules_list.append(m)
                    self.layer_info.append(("bn2d", m))
                elif ltype == "ln":
                    m = nn.LayerNorm(params["dim"])
                    self.modules_list.append(m)
                    self.layer_info.append(("ln", m))
                elif ltype == "dropout":
                    m = nn.Dropout(params["p"])
                    self.modules_list.append(m)
                    self.layer_info.append(("dropout", m))
                elif ltype == "activation":
                    act_cls = getattr(nn, params["type"])
                    m = act_cls()
                    self.modules_list.append(m)
                    self.layer_info.append(("activation", m))

            self.fc = nn.Linear(current_dim, output_dim)

        def forward(self, x):
            if self.is_image and x.dim() == 2:
                x = x.unsqueeze(-1)  # [B, D] -> [B, D, 1] for conv1d
            for ltype, module in self.layer_info:
                if ltype in ("lstm", "gru"):
                    if x.dim() == 2:
                        x = x.unsqueeze(1)  # add seq dim
                    out, _ = module(x)
                    x = out[:, -1, :]  # last timestep
                elif ltype == "mha":
                    if x.dim() == 2:
                        x = x.unsqueeze(1)
                    a, _ = module(x, x, x)
                    x = a.mean(dim=1)
                elif ltype in ("conv1d", "conv2d"):
                    if x.dim() == 2:
                        x = x.unsqueeze(-1)  # [B,D] -> [B,D,1]
                    x = module(x)
                    x = x.flatten(1).mean(dim=0, keepdim=True).expand(x.size(0), -1) if x.dim() > 2 else x
                else:
                    x = module(x)
            return self.fc(x)

    return FuzzModel()


def make_fuzz_data(model, batch=8):
    """Generate input data compatible with the model."""
    # Try to infer input shape from model
    try:
        first_layer = model.layer_info[0][1]
        if isinstance(first_layer, nn.Linear):
            dim = first_layer.in_features
            return torch.randn(batch, dim), torch.randint(0, 2, (batch,))
        elif isinstance(first_layer, (nn.Conv1d, nn.Conv2d)):
            ch = first_layer.in_channels
            if isinstance(first_layer, nn.Conv2d):
                return torch.randn(batch, ch, 8, 8), torch.randint(0, 2, (batch,))
            return torch.randn(batch, ch, 16), torch.randint(0, 2, (batch,))
    except:
        pass
    return torch.randn(batch, 16), torch.randint(0, 2, (batch,))


# ============================================================
# Fuzzer runner
# ============================================================

def fuzz_one(seed=None):
    """Run one fuzzing iteration with random architecture + bug."""
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    try:
        spec = generate_random_model(max_layers=6)
        model = build_model_from_spec(spec, input_dim=16, output_dim=2)
        data_fn = lambda: make_fuzz_data(model)

        bug_name, bug_fn = random.choice(BUGS)
        x, y = data_fn()

        with NeuralDbg(model) as dbg:
            opt = torch.optim.SGD(model.parameters(), lr=random.uniform(0.001, 0.1))
            for s in range(8):
                x, y = data_fn()
                if s >= 3:
                    if bug_name == "exploding" or bug_name == "divergence":
                        bug_fn(opt)
                    elif bug_name == "nan_data":
                        x = bug_fn(x)
                    elif bug_name in ("zero_init", "dead_bias", "mixed_precision"):
                        if s == 3:
                            bug_fn(model)
                    elif bug_name == "vanishing":
                        bug_fn(model)

                opt.zero_grad()
                try:
                    loss = nn.CrossEntropyLoss()(model(x), y)
                    loss.backward()
                    dbg.step_iteration()
                    dbg.record_loss(loss.item())
                    opt.step()
                except Exception as e:
                    return FuzzResult(
                        model_name=f"fuzz_{seed}",
                        layers=str(spec[0]),
                        bug=bug_name,
                        events=0,
                        crashed=True,
                        error=str(e)[:100],
                    )

            events = dbg.dump_events()
            n = len(events)
            return FuzzResult(
                model_name=f"fuzz_{seed}",
                layers=str(spec[0]),
                bug=bug_name,
                events=n,
                crashed=False,
                detected=n > 2,
            )
    except Exception as e:
        return FuzzResult(
            model_name=f"fuzz_{seed}",
            layers="?",
            bug="?",
            events=0,
            crashed=True,
            error=str(e)[:100],
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=50, help="Number of fuzz iterations")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    print("=" * 60)
    print(f"ARCHITECTURE FUZZER — {args.runs} iterations")
    print("Tier 2: Unknown Unknowns — Black-Swan Discovery")
    print("=" * 60)

    results = []
    crashes = []
    t0 = time.time()

    for i in range(args.runs):
        seed = args.seed + i
        r = fuzz_one(seed)
        results.append(r)

        status = "CRASH" if r.crashed else ("DETECT" if r.detected else "ok")
        print(f"  [{i+1:3d}/{args.runs}] {status:6s} | bug={r.bug:15s} | events={r.events:3d} "
              f"| {'ERROR: '+r.error if r.crashed else ''}", flush=True)

        if r.crashed:
            crashes.append(r)

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS — {elapsed:.0f}s")
    print(f"{'='*60}")

    n_crashed = sum(1 for r in results if r.crashed)
    n_detected = sum(1 for r in results if r.detected)
    n_ok = sum(1 for r in results if not r.crashed and not r.detected)

    print(f"  Total runs:     {len(results)}")
    print(f"  Crashes:        {n_crashed} ({100*n_crashed//max(len(results),1)}%)")
    print(f"  Detected:       {n_detected} ({100*n_detected//max(len(results),1)}%)")
    print(f"  OK (no crash):  {n_ok} ({100*n_ok//max(len(results),1)}%)")

    if crashes:
        print(f"\n  CRASH DETAILS:")
        for r in crashes[:5]:
            print(f"    bug={r.bug} | error={r.error[:80]}")

    # Per-bug stats
    bug_stats = defaultdict(lambda: {"total": 0, "crashed": 0, "detected": 0})
    for r in results:
        bug_stats[r.bug]["total"] += 1
        if r.crashed:
            bug_stats[r.bug]["crashed"] += 1
        if r.detected:
            bug_stats[r.bug]["detected"] += 1

    print(f"\n  Per-bug breakdown:")
    for bug, s in sorted(bug_stats.items()):
        print(f"    {bug:18s}: {s['crashed']} crashes / {s['total']} ({s['detected']} detected)")

    # Save report
    report = {
        "runs": args.runs,
        "seed": args.seed,
        "elapsed": int(elapsed),
        "crashes": n_crashed,
        "detected": n_detected,
        "ok": n_ok,
        "bug_stats": {k: dict(v) for k, v in bug_stats.items()},
        "crash_details": [{"bug": r.bug, "error": r.error, "layers": r.layers} for r in crashes],
    }
    with open("fuzz_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: fuzz_report.json")

    # Discovery: any new patterns?
    if n_crashed > 0:
        print(f"\n  BLACK-SWAN DISCOVERY: {n_crashed} crashes found!")
        print(f"  These architectures/bugs cause NeuralDBG to crash.")
        print(f"  Review fuzz_report.json for details.")
    else:
        print(f"\n  No crashes found — NeuralDBG handles all fuzzed archs.")
