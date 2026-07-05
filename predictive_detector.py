"""Tier 3 — Predictive Anomaly Detector (Zero-Config Black-Swan Detection).

Learns "normal" training dynamics from healthy runs across architectures.
Flags ANY deviation without pre-defining failure modes.

Strategy:
  1. Collect healthy training statistics from many architectures
  2. Build statistical profiles (mean, std, percentiles)
  3. For any new run, compute z-scores
  4. Flag deviations >3σ as potential black-swans

Usage: python predictive_detector.py [--train] [--detect export.json]
  --train : Collect healthy training profiles (run first)
  --detect : Detect anomalies in a NeuralDBG export
"""

import sys, json, math
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, r"C:\Users\Utilisateur\Documents\NeuralDBG")

import torch, torch.nn as nn
import numpy as np
from neuraldbg import NeuralDbg
from validate_combinatorial import *

torch.manual_seed(42)

PROFILE_PATH = "healthy_training_profile.json"


# ============================================================
# Profile collection
# ============================================================

def collect_healthy_profiles(n_configs=30):
    """Train healthy models across architectures and collect statistics."""
    print(f"Collecting healthy training profiles from {n_configs} architectures...")
    
    # Collect configs from all families
    configs = []
    configs.extend(mlp_configs(8))
    configs.extend(cnn_configs(6))
    configs.extend(rnn_configs(6))
    configs.extend(transformer_configs(5))
    configs.extend(hybrid_configs(5))
    configs = configs[:n_configs]
    
    all_stats = []
    
    for i, cfg in enumerate(configs):
        try:
            model = cfg.make_model()
            data_fn = cfg.make_data
            
            with NeuralDbg(model) as dbg:
                opt = torch.optim.SGD(model.parameters(), lr=0.01)
                loss_fn = nn.CrossEntropyLoss()
                
                run_stats = {"family": cfg.family, "name": cfg.name, "steps": []}
                
                for s in range(10):
                    x, y = data_fn()
                    opt.zero_grad()
                    loss = loss_fn(model(x), y)
                    loss.backward()
                    dbg.step_iteration()
                    dbg.record_loss(loss.item())
                    opt.step()
                    
                    # Collect per-step statistics from dbg
                    events = dbg.dump_events()
                    grad_norms = []
                    act_means = []
                    for e in events:
                        meta = getattr(e, 'metadata', {})
                        if 'current_norm' in meta:
                            grad_norms.append(meta['current_norm'])
                        if hasattr(e, 'metadata') and 'current_saturation' in meta:
                            act_means.append(meta.get('current_saturation', 0))
                    
                    step_stats = {
                        "step": s,
                        "loss": loss.item(),
                        "grad_norms": grad_norms[-5:] if grad_norms else [],
                        "act_saturation": act_means[-5:] if act_means else [],
                        "n_events": len(events),
                    }
                    run_stats["steps"].append(step_stats)
                
                all_stats.append(run_stats)
                print(f"  [{i+1:2d}/{n_configs}] {cfg.name[:50]}", flush=True)
                
        except Exception as e:
            print(f"  [{i+1:2d}/{n_configs}] SKIP: {e}")
    
    # Compute aggregate statistics
    profile = _compute_profile(all_stats)
    
    with open(PROFILE_PATH, "w") as f:
        json.dump(profile, f, indent=2, default=str)
    
    print(f"\n  Profile saved: {PROFILE_PATH}")
    print(f"  Architectures: {len(all_stats)}")
    print(f"  Stats: loss_mean={profile['loss']['mean']:.3f}, grad_mean={profile['grad_norm']['mean']:.3f}")
    return profile


def _compute_profile(all_stats):
    """Compute statistical profile from collected runs."""
    # Aggregate across all steps and runs
    losses = []
    grad_norms = []
    act_sats = []
    n_events = []
    
    for run in all_stats:
        for step in run["steps"]:
            losses.append(step["loss"])
            grad_norms.extend(step["grad_norms"])
            act_sats.extend(step["act_saturation"])
            n_events.append(step["n_events"])
    
    def stats(arr):
        if not arr:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        a = np.array(arr)
        a = a[np.isfinite(a)]  # remove NaN/Inf
        if len(a) == 0:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "p99": float(np.percentile(a, 99)),
            "n_samples": len(a),
        }
    
    return {
        "loss": stats(losses),
        "grad_norm": stats(grad_norms),
        "act_saturation": stats(act_sats),
        "n_events": stats(n_events),
        "n_architectures": len(all_stats),
    }


# ============================================================
# Anomaly detection
# ============================================================

def detect_anomalies(export_path, profile=None):
    """Detect anomalies in a NeuralDBG export using learned profile."""
    if profile is None:
        try:
            with open(PROFILE_PATH) as f:
                profile = json.load(f)
        except FileNotFoundError:
            print("No profile found. Run with --train first.")
            return []
    
    with open(export_path) as f:
        data = json.load(f)
    
    events = data.get("events", [])
    if not events:
        return []
    
    # Extract metrics from export
    grad_norms = []
    act_sats = []
    for e in events:
        meta = e.get("metadata", {})
        if "current_norm" in meta:
            grad_norms.append(meta["current_norm"])
        if "current_saturation" in meta:
            act_sats.append(meta["current_saturation"])
    
    n_events = len(events)
    
    # Compute z-scores
    anomalies = []
    
    def check_metric(name, value, prof_key, threshold=3.0):
        p = profile.get(prof_key, {})
        mean = p.get("mean", 0)
        std = p.get("std", 1)
        if std == 0:
            return None
        z = abs(value - mean) / std
        if z > threshold:
            return {"metric": name, "value": value, "mean": mean, "std": std, "z_score": round(z, 1)}
        return None
    
    # Check event count
    r = check_metric("event_count", n_events, "n_events")
    if r:
        anomalies.append(r)
    
    # Check mean gradient norm
    if grad_norms:
        mean_gn = sum(grad_norms) / len(grad_norms)
        r = check_metric("grad_norm_mean", mean_gn, "grad_norm")
        if r:
            anomalies.append(r)
        
        # Check max gradient norm
        max_gn = max(grad_norms)
        r = check_metric("grad_norm_max", max_gn, "grad_norm", threshold=5.0)
        if r:
            anomalies.append(r)
    
    # Check activation saturation
    if act_sats:
        mean_as = sum(act_sats) / len(act_sats)
        r = check_metric("act_saturation", mean_as, "act_saturation")
        if r:
            anomalies.append(r)
    
    # Add profile metadata
    for a in anomalies:
        a["profile_n_archs"] = profile.get("n_architectures", 0)
    
    return anomalies


def predict_black_swan(events, profile=None):
    """Quick prediction: does this look like a black-swan?"""
    anomalies = detect_anomalies(events, profile)
    if anomalies:
        return {
            "black_swan_detected": True,
            "confidence": min(0.95, 0.5 + 0.1 * len(anomalies)),
            "anomalies": anomalies,
        }
    return {"black_swan_detected": False, "anomalies": []}


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Collect healthy training profiles")
    parser.add_argument("--detect", type=str, help="Detect anomalies in a NeuralDBG export JSON")
    parser.add_argument("--configs", type=int, default=30, help="Number of configs for training")
    args = parser.parse_args()
    
    if args.train:
        profile = collect_healthy_profiles(n_configs=args.configs)
        print(f"\n  Profile ready. Run with --detect <export.json> to find anomalies.")
    
    elif args.detect:
        anomalies = detect_anomalies(args.detect)
        if anomalies:
            print(f"\n  [!] BLACK-SWAN DETECTED — {len(anomalies)} anomalies:")
            for a in anomalies:
                print(f"    {a['metric']}: {a['value']:.3f} (z={a['z_score']}, baseline {a['mean']:.3f}+-{a['std']:.3f})")
        else:
            print(f"\n  [OK] No anomalies detected — training looks normal.")
    
    else:
        print("Usage: python predictive_detector.py --train  (collect profiles)")
        print("       python predictive_detector.py --detect export.json  (find anomalies)")
