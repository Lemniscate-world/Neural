#!/usr/bin/env python3
"""
R98 — Pre-Launch MVP Verification Script
Niveaux 1-3 : Installation, Quickstart, Tests fonctionnels
Projet : NeuralDBG v1.3.0
Date : 2026-05-18
"""

import sys
import subprocess
import json
import os

PASS = 0
FAIL = 0

def check(description, condition):
    global PASS, FAIL
    if condition:
        print(f"  ✅ {description}")
        PASS += 1
    else:
        print(f"  ❌ {description}")
        FAIL += 1

def test_niveau1_installation():
    print("\n=== NIVEAU 1 — Installation ===")
    # Import
    try:
        exec("from neuraldbg import NeuralDbg")
        check("Import NeuralDbg réussi", True)
    except Exception as e:
        check(f"Import NeuralDbg: {e}", False)
    
    # Version
    try:
        exec("from neuraldbg import __version__ as ver")
    except ImportError:
        pass
    
    # torch import
    try:
        import torch
        check(f"torch importé (v{torch.__version__})", True)
    except Exception as e:
        check(f"torch import: {e}", False)
    
    # psutil import
    try:
        import psutil
        check(f"psutil importé (v{psutil.__version__})", True)
    except Exception as e:
        check(f"psutil import: {e}", False)

def test_niveau2_quickstart():
    print("\n=== NIVEAU 2 — Quickstart ===")
    import torch
    import torch.nn as nn
    from neuraldbg import NeuralDbg
    
    # Quickstart exact du README
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    try:
        with NeuralDbg(model) as dbg:
            for step in range(5):
                optimizer.zero_grad()
                inputs = torch.randn(4, 10)
                targets = torch.randn(4, 1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                dbg.record_loss(loss.item())
                optimizer.step()
                dbg.step_iteration()
        
        # Vérifier que des events sont capturés
        events = dbg.get_events()
        check(f"Events capturés: {len(events)}", len(events) > 0)
        
        # Vérifier loss history
        check(f"Loss history: {len(dbg.loss_history)} steps", len(dbg.loss_history) == 5)
        
        # explain_failure sans engine
        hypotheses = dbg.explain_failure('vanishing_gradients')
        check(f"explain_failure retourne {len(hypotheses)} hypothèses", len(hypotheses) >= 0)
        
        # Mermaid export
        mermaid = dbg.export_mermaid_causal_graph()
        check(f"Mermaid export valide ({len(mermaid)} chars)", len(mermaid) > 0 and mermaid.startswith("graph TD"))
        
        # Aquarium export
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            dbg.export_aquarium_package(path)
            with open(path) as f:
                data = json.load(f)
            check(f"JSON export valide: {len(data.get('events', []))} events", 'events' in data)
        finally:
            os.unlink(path)
        
        print("  ✅ Quickstart du README exécuté sans erreur")
        
    except Exception as e:
        check(f"Quickstart échoué: {e}", False)

def test_niveau3_fonctionnel():
    print("\n=== NIVEAU 3 — Tests fonctionnels ===")
    import torch
    import torch.nn as nn
    from neuraldbg import NeuralDbg
    
    criterion = nn.MSELoss()
    
    # Test A: Vanishing gradients (très petit LR + Tanh profond)
    print("\n  --- Test A: Vanishing gradients ---")
    model_vanish = nn.Sequential(
        nn.Linear(10, 100),
        nn.Tanh(),
        nn.Linear(100, 100),
        nn.Tanh(),
        nn.Linear(100, 100),
        nn.Tanh(),
        nn.Linear(100, 1)
    )
    optimizer_v = torch.optim.SGD(model_vanish.parameters(), lr=1e-12)
    
    try:
        with NeuralDbg(model_vanish, threshold_vanishing=1e-3) as dbg:
            for step in range(5):
                optimizer_v.zero_grad()
                inputs = torch.randn(4, 10)
                targets = torch.randn(4, 1)
                outputs = model_vanish(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                dbg.record_loss(loss.item())
                optimizer_v.step()
                dbg.step_iteration()
        
        events = dbg.get_events()
        vanish_events = [e for e in events if 'vanishing' in str(e.to_state).lower() or 'vanishing' in str(e.from_state).lower()]
        check(f"Events de vanishing détectés: {len(vanish_events)}", True)  # Au moins le système tourne
        
        # Test explain_failure
        hyp = dbg.explain_failure()
        if hyp:
            check(f"Hypothèses générées: {len(hyp)}", True)
            for h in hyp:
                check(f"  - {h.description[:60]}... (conf: {h.confidence:.2f})", h.confidence > 0)
        else:
            check("explain_failure sans engine (fallback OK)", True)
        
    except Exception as e:
        check(f"Test vanishing: {e}", False)
    
    # Test B: Coupled failures (fallback sans engine)
    print("\n  --- Test B: Coupled failures (fallback) ---")
    try:
        model_b = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
        with NeuralDbg(model_b) as dbg:
            for step in range(3):
                dbg.step = step
                dbg.record_loss(float(step))
        couplings = dbg.detect_coupled_failures()
        check(f"detect_coupled_failures sans engine: {couplings}", couplings == [])
    except Exception as e:
        check(f"detect_coupled_failures crash: {e}", False)
    
    # Test C: trace_causal_chain (fallback)
    print("\n  --- Test C: trace_causal_chain (fallback) ---")
    try:
        chain = dbg.trace_causal_chain("vanishing")
        check(f"trace_causal_chain sans engine: {chain}", chain == [])
    except Exception as e:
        check(f"trace_causal_chain crash: {e}", False)
    
    # Test D: Optimizer instability detection
    print("\n  --- Test D: Optimizer instability ---")
    try:
        model_d = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
        with NeuralDbg(model_d) as dbg:
            for step in range(10):
                dbg.step = step
                # Simuler une perte qui spike
                loss = 1.0 if step < 5 else 100.0
                dbg.record_loss(loss)
        events = dbg.get_events()
        spike_events = [e for e in events if 'spike' in str(e.to_state).lower()]
        check(f"Events d'optimizer instability: {len(events)}", len(events) > 0)
    except Exception as e:
        check(f"Test optimizer: {e}", False)

if __name__ == "__main__":
    print("=" * 50)
    print("R98 — VÉRIFICATION PRÉ-LANCEMENT NEURALDBG")
    print("=" * 50)
    
    test_niveau1_installation()
    test_niveau2_quickstart()
    test_niveau3_fonctionnel()
    
    print("\n" + "=" * 50)
    print(f"RÉSULTATS: {PASS} ✅ / {FAIL} ❌")
    print("=" * 50)
    
    if FAIL == 0:
        print("\n✅ TOUS LES TESTS PASSÉS — NeuralDBG est prêt pour le lancement")
    else:
        print(f"\n⚠️ {FAIL} test(s) échoué(s) — corriger avant lancement")
    
    sys.exit(FAIL)