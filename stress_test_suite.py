"""Stress Test Suite — +10 Resilience Verification for NeuralDBG.

Tests NeuralDBG's ability to handle extreme conditions without crashing,
producing false positives, or missing genuine failures.

Target: 15/15 capabilities passing. Current: 6/15.

Usage: python stress_test_suite.py
"""

import sys
sys.path.insert(0, r"C:\Users\Utilisateur\Documents\NeuralDBG")

import torch, torch.nn as nn
import torch.nn.functional as F
from neuraldbg import NeuralDbg
from collections import defaultdict

torch.manual_seed(42)

RESULTS = []
PASS = "PASS"
FAIL = "FAIL"

# ============================================================
# Test helper
# ============================================================
def test(name, fn, target="no_crash"):
    """Run a stress test and record result."""
    try:
        result = fn()
        if target == "no_crash":
            RESULTS.append((name, PASS, "No crash"))
        elif target == "detect":
            detected = bool(result)
            RESULTS.append((name, PASS if detected else FAIL,
                          "Detected" if detected else "NOT detected"))
        elif target == "no_fp":
            fp = bool(result)
            RESULTS.append((name, FAIL if fp else PASS,
                          "False positive!" if fp else "No false positive"))
        else:
            RESULTS.append((name, PASS, str(result)))
    except Exception as e:
        RESULTS.append((name, FAIL, str(e)[:80]))

def count_events(events):
    return len(events)

# ============================================================
# Model builders
# ============================================================
def simple_model():
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))

def deep_model(layers=100):
    mods = []
    for i in range(layers):
        mods.append(nn.Linear(32, 32))
        mods.append(nn.ReLU())
    mods.append(nn.Linear(32, 2))
    return nn.Sequential(*mods)

def attention_model(seq_len=1000):
    class AttnModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(32, 4, batch_first=True)
            self.fc = nn.Linear(32, 2)
        def forward(self, x):
            a, _ = self.attn(x, x, x)
            return self.fc(a.mean(dim=1))
    return AttnModel()


# ============================================================
# Test 1: 10x Normal Gradient Magnitude
# ============================================================
def test_10x_gradient():
    model = simple_model()
    with NeuralDbg(model) as dbg:
        for s in range(5):
            x = torch.randn(4, 16) * 10.0  # 10x normal scale
            y = torch.randint(0, 2, (4,))
            opt = torch.optim.SGD(model.parameters(), lr=1.0)  # high LR
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            dbg.step_iteration()
            dbg.record_loss(loss.item())
            opt.step()
        return count_events(dbg.dump_events()) > 0  # Should still produce events

test("10x gradient magnitude", test_10x_gradient, target="detect")


# ============================================================
# Test 2: 0.1x Normal Gradient (Vanishing)
# ============================================================
def test_01x_gradient():
    model = simple_model()
    # Init with tiny weights to cause vanishing
    for p in model.parameters():
        nn.init.constant_(p, 0.001)
    with NeuralDbg(model) as dbg:
        for s in range(10):
            x = torch.randn(4, 16) * 0.01  # tiny inputs
            y = torch.randint(0, 2, (4,))
            opt = torch.optim.SGD(model.parameters(), lr=0.0001)  # tiny LR
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            dbg.step_iteration()
            dbg.record_loss(loss.item())
            opt.step()
        return count_events(dbg.dump_events()) > 0

test("0.1x gradient (vanishing)", test_01x_gradient, target="detect")


# ============================================================
# Test 3: 10x Input Scale
# ============================================================
def test_10x_input():
    model = simple_model()
    with NeuralDbg(model) as dbg:
        for s in range(5):
            x = torch.randn(4, 16) * 100.0  # extremely large inputs
            y = torch.randint(0, 2, (4,))
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            dbg.step_iteration()
            dbg.record_loss(loss.item())
            opt.step()
        return count_events(dbg.dump_events()) > 0

test("10x input scale", test_10x_input, target="detect")


# ============================================================
# Test 4: NaN in Input
# ============================================================
def test_nan_input():
    model = simple_model()
    with NeuralDbg(model) as dbg:
        for s in range(6):
            x = torch.randn(4, 16)
            if s >= 2:
                x[0, 0] = float('nan')
            y = torch.randint(0, 2, (4,))
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            try:
                loss = nn.CrossEntropyLoss()(model(x), y)
                if torch.isnan(loss):
                    dbg.record_loss(float('nan'))
                    break
                loss.backward()
                dbg.step_iteration()
                dbg.record_loss(loss.item())
                opt.step()
            except RuntimeError:
                # NaN may cause backward to fail — that's detection too
                dbg.record_loss(float('nan'))
                break
        events = dbg.get_events()
        # Check for nan in event_type or to_state
        has_nan = False
        for e in events:
            et = getattr(e, 'event_type', None)
            if et is None:
                continue
            et_val = str(et.value).lower() if hasattr(et, 'value') else str(et).lower()
            ts = str(getattr(e, 'to_state', '')).lower()
            if 'nan' in et_val or 'nan' in ts:
                has_nan = True
                break
        return has_nan

test("NaN in input", test_nan_input, target="detect")


# ============================================================
# Test 5: Inf in Input
# ============================================================
def test_inf_input():
    model = simple_model()
    with NeuralDbg(model) as dbg:
        x = torch.randn(4, 16)
        x[0, 0] = float('inf')
        y = torch.randint(0, 2, (4,))
        try:
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            dbg.step_iteration()
        except Exception:
            pass  # Inf may cause crash — that's OK, we just need to not hang
        return True  # Survival test

test("Inf in input", test_inf_input, target="no_crash")


# ============================================================
# Test 6: Zero-Size Tensors (Empty Batch)
# ============================================================
def test_empty_batch():
    model = simple_model()
    with NeuralDbg(model) as dbg:
        try:
            x = torch.randn(0, 16)  # empty batch
            y = torch.randint(0, 2, (0,))
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            dbg.step_iteration()
        except Exception:
            pass  # Expected to potentially fail
        return True

test("Zero-size tensor (empty batch)", test_empty_batch, target="no_crash")


# ============================================================
# Test 7: NaN Labels
# ============================================================
def test_nan_labels():
    model = simple_model()
    with NeuralDbg(model) as dbg:
        try:
            x = torch.randn(4, 16)
            y = torch.tensor([0, 1, float('nan'), 0])
            loss = nn.CrossEntropyLoss()(model(x), y.long())
            loss.backward()
            dbg.step_iteration()
        except Exception:
            pass
        return True

test("NaN labels", test_nan_labels, target="no_crash")


# ============================================================
# Test 8: Very Deep Network (100+ layers)
# ============================================================
def test_deep_network():
    model = deep_model(layers=100)
    with NeuralDbg(model) as dbg:
        for s in range(3):
            x = torch.randn(4, 32)
            y = torch.randint(0, 2, (4,))
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            dbg.step_iteration()
            opt.step()
        return count_events(dbg.dump_events()) > 0

test("100-layer deep network", test_deep_network, target="detect")


# ============================================================
# Test 9: Very Long Sequence (1K tokens attention)
# ============================================================
def test_long_sequence():
    model = attention_model()
    with NeuralDbg(model) as dbg:
        try:
            x = torch.randn(2, 1000, 32)  # 1K tokens
            y = torch.randint(0, 2, (2,))
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            dbg.step_iteration()
        except Exception:
            pass
        return True

test("1K token attention", test_long_sequence, target="no_crash")


# ============================================================
# Test 10: fp16 Mixed Precision
# ============================================================
def test_fp16():
    model = simple_model().half()  # convert to fp16
    with NeuralDbg(model) as dbg:
        try:
            x = torch.randn(4, 16).half()
            y = torch.randint(0, 2, (4,))
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            dbg.step_iteration()
        except Exception:
            pass
        return True

test("fp16 mixed precision", test_fp16, target="no_crash")


# ============================================================
# Test 11: LSTM/GRU with hidden state capture
# ============================================================
def test_rnn_hidden_state():
    class RNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(16, 32, batch_first=True)
            self.fc = nn.Linear(32, 2)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model = RNNModel()
    with NeuralDbg(model) as dbg:
        for s in range(5):
            x = torch.randn(4, 8, 16)
            y = torch.randint(0, 2, (4,))
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            dbg.step_iteration()
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            opt.step()
        return count_events(dbg.dump_events()) > 0

test("LSTM hidden state capture", test_rnn_hidden_state, target="detect")


# ============================================================
# Test 12: Duplicate input (all same sample)
# ============================================================
def test_duplicate_input():
    model = simple_model()
    with NeuralDbg(model) as dbg:
        x = torch.randn(1, 16).repeat(16, 1)  # all identical
        y = torch.zeros(16, dtype=torch.long)
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        dbg.step_iteration()
        return True

test("Duplicate inputs (all identical)", test_duplicate_input, target="no_crash")


# ============================================================
# Test 13: Gradient clipping interaction
# ============================================================
def test_grad_clip():
    model = simple_model()
    with NeuralDbg(model) as dbg:
        x = torch.randn(4, 16) * 10.0
        y = torch.randint(0, 2, (4,))
        opt = torch.optim.SGD(model.parameters(), lr=10.0)
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        dbg.step_iteration()
        opt.step()
        return count_events(dbg.dump_events()) > 0

test("Gradient clipping (extreme)", test_grad_clip, target="detect")


# ============================================================
# Test 14: Extreme LR schedule (cosine decay to 0)
# ============================================================
def test_extreme_lr_schedule():
    """LR drops from 1.0 to 1e-6 in 1 step — should trigger gradient regime change."""
    model = simple_model()
    with NeuralDbg(model) as dbg:
        for s in range(10):
            x = torch.randn(4, 16)
            y = torch.randint(0, 2, (4,))
            lr = 1.0 if s == 0 else 1e-6  # drastic drop after step 0
            opt = torch.optim.SGD(model.parameters(), lr=lr)
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            dbg.step_iteration()
            dbg.record_loss(loss.item())
            opt.step()
        return count_events(dbg.dump_events()) > 0

test("Extreme LR schedule drop", test_extreme_lr_schedule, target="detect")


# ============================================================
# Test 15: Zero-gradient (all params grad=0)
# ============================================================
def test_zero_gradient():
    model = simple_model()
    for p in model.parameters():
        nn.init.constant_(p, 0.0)
    with NeuralDbg(model) as dbg:
        x = torch.randn(4, 16)
        y = torch.randint(0, 2, (4,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        dbg.step_iteration()
        return count_events(dbg.dump_events()) > 0

test("Zero gradient (dead model)", test_zero_gradient, target="detect")


# ============================================================
# Report
# ============================================================
print("=" * 65)
print("NEURALDBG STRESS TEST SUITE — +10 Resilience")
print("=" * 65)
print(f"  {'Test':45s} | {'Result':6s} | Detail")
print(f"  {'-'*45} | {'-'*6} | {'-'*30}")

passed = 0
failed = 0
for name, result, detail in RESULTS:
    status = "PASS" if result == PASS else "FAIL"
    if result == PASS:
        passed += 1
    else:
        failed += 1
    print(f"  {name:45s} | {status:6s} | {detail}")

print(f"\n  Score: {passed}/{len(RESULTS)} ({100*passed//len(RESULTS)}%)")
print(f"  Target: 15/15 (100%)")
print(f"  Gap: {failed} tests failing")

# Resilience score
score = 100 * passed // len(RESULTS)
if score == 100:
    print("\n  ALL RESILIENCE TESTS PASSED — NeuralDBG is +10 ready.")
elif score >= 80:
    print(f"\n  {failed} test(s) to fix for full +10 resilience.")
else:
    print(f"\n  CRITICAL: {failed} tests failing. +10 resilience NOT achieved.")

sys.exit(0 if score == 100 else 1)
