#!/usr/bin/env python
"""
Enhanced Causal Reasoning Demo
================================
Demonstrates Phase 2 enhancements:
- Cross-layer propagation analysis
- Temporal pattern detection
- Multi-factor confidence scoring
"""

import torch
import torch.nn as nn
import importlib.util
import os

from neuraldbg import NeuralDbg

script_dir = os.path.dirname(os.path.abspath(__file__))
enhanced_path = os.path.join(script_dir, "neuraldbg", "enhanced_causality.py")
spec = importlib.util.spec_from_file_location("enhanced_causality", enhanced_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load enhanced causality module from {enhanced_path}")
enhanced_causality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enhanced_causality)
enhance_with_granger_style = enhanced_causality.enhance_with_granger_style


def create_problematic_network():
    """Create a network with cascading failure pattern."""
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    return model


def run_enhanced_analysis():
    """Run analysis with enhanced causal reasoning."""
    print("=" * 70)
    print("ENHANCED CAUSAL REASONING DEMO")
    print("=" * 70)
    
    model = create_problematic_network()
    
    # Simulate training with problematic pattern
    torch.manual_seed(42)
    
    print("\nSimulating training with cascading failure pattern...")
    
    with NeuralDbg(model) as dbg:
        for step in range(50):
            # Create varied input that causes instability
            x = torch.randn(16, 784) * (0.8 + 0.4 * (step % 5) / 5)
            y = torch.randint(0, 10, (16,))
            
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            
            dbg.record_loss(loss.item())
    
    print(f"Events captured: {len(dbg.events)}")
    
    # Standard analysis
    print("\n" + "-" * 70)
    print("STANDARD EXPLANATION")
    print("-" * 70)
    standard = dbg.explain_failure("vanishing_gradients")
    print(f"Hypotheses: {len(standard)}")
    for h in standard[:3]:
        print(f"  - {h.description[:80]}...")
        print(f"    Confidence: {h.confidence:.2f}")
    
    # Enhanced analysis
    print("\n" + "-" * 70)
    print("ENHANCED EXPLANATION (Granger-style)")
    print("-" * 70)
    enhanced = enhance_with_granger_style(dbg.events)
    print(f"Hypotheses: {len(enhanced)}")
    for h in enhanced[:5]:
        print(f"  - {h.description[:80]}...")
        print(f"    Confidence: {h.confidence:.2f} (multi-factor)")
        if h.causal_chain:
            print(f"    Causal chain: {h.causal_chain}")
    
    # Comparison
    print("\n" + "-" * 70)
    print("COMPARISON")
    print("-" * 70)
    print(f"Standard hypotheses: {len(standard)}")
    print(f"Enhanced hypotheses: {len(enhanced)}")
    print(f"\nEnhanced adds:")
    print("  - Cross-layer propagation detection")
    print("  - Temporal pattern analysis")
    print("  - Multi-factor confidence scoring")
    
    print("\n" + "=" * 70)
    print("ENHANCEMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_enhanced_analysis()
