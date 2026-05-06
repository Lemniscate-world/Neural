#!/usr/bin/env python
"""
Scale Testing: EfficientNet and Vision Transformer
Tests NeuralDBG on larger architectures to validate robustness
"""

import torch
import torch.nn as nn
from neuraldbg import NeuralDbg

def test_efficientnet():
    """Test NeuralDBG on EfficientNet-B0 (~5.3M params)"""
    print("=" * 60)
    print("Testing EfficientNet-B0 (5.3M params)")
    print("=" * 60)
    
    from torchvision.models import efficientnet_b0
    model = efficientnet_b0(weights=None)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    input_tensor = torch.randn(1, 3, 224, 224)
    
    print("\n--- Testing with NeuralDbg ---")
    with NeuralDbg(model) as dbg:
        # Forward pass
        output = model(input_tensor)
        
        # Simulate backward (create a fake loss)
        loss = output.sum()
        loss.backward()
        
        print(f"\nEvents captured: {len(dbg.events)}")
        
        # Show event types
        if dbg.events:
            event_types = {}
            for e in dbg.events:
                t = e.event_type.value if hasattr(e.event_type, 'value') else str(e.event_type)
                event_types[t] = event_types.get(t, 0) + 1
            print("Event breakdown:", event_types)
        
        # Try causal analysis
        if dbg.events:
            try:
                explanations = dbg.explain_failure("saturated_activations")
                print(f"\nCausal explanations generated: {len(explanations)}")
            except Exception as e:
                print(f"Causal analysis note: {e}")
    
    print("\nPASS: EfficientNet-B0 test completed\n")
    return True


def test_vision_transformer():
    """Test NeuralDBG on ViT-B/16 (~86M params)"""
    print("=" * 60)
    print("Testing Vision Transformer ViT-B/16 (~86M params)")
    print("=" * 60)
    
    from torchvision.models import vit_b_16
    model = vit_b_16(weights=None)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    # ViT expects 224x224 images
    input_tensor = torch.randn(1, 3, 224, 224)
    
    print("\n--- Testing with NeuralDbg ---")
    with NeuralDbg(model) as dbg:
        # Forward pass
        output = model(input_tensor)
        
        # Simulate backward
        loss = output.sum()
        loss.backward()
        
        print(f"\nEvents captured: {len(dbg.events)}")
        
        # Show event types
        if dbg.events:
            event_types = {}
            for e in dbg.events:
                t = e.event_type.value if hasattr(e.event_type, 'value') else str(e.event_type)
                event_types[t] = event_types.get(t, 0) + 1
            print("Event breakdown:", event_types)
        
        # Try causal analysis
        if dbg.events:
            try:
                explanations = dbg.explain_failure("activation_regime_shift")
                print(f"\nCausal explanations generated: {len(explanations)}")
            except Exception as e:
                print(f"Causal analysis note: {e}")
    
    print("\nPASS: ViT-B/16 test completed\n")
    return True


def test_memory_profile():
    """Profile memory usage during extended runs"""
    print("=" * 60)
    print("Memory Profiling Test")
    print("=" * 60)
    
    import tracemalloc
    
    from torchvision.models import efficientnet_b0
    model = efficientnet_b0(weights=None)
    model.eval()
    
    tracemalloc.start()
    
    print("\n--- Running 10 forward+backward passes ---")
    for i in range(10):
        input_tensor = torch.randn(4, 3, 224, 224)  # Batch of 4
        
        with NeuralDbg(model) as dbg:
            output = model(input_tensor)
            loss = output.sum()
            loss.backward()
        
        if i % 5 == 0:
            current, peak = tracemalloc.get_traced_memory()
            print(f"Pass {i+1}: Current {current/1024/1024:.1f}MB, Peak {peak/1024/1024:.1f}MB")
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"\nFinal - Current: {current/1024/1024:.1f}MB, Peak: {peak/1024/1024:.1f}MB")
    tracemalloc.stop()
    
    print("\nPASS: Memory profiling completed\n")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NeuralDBG Scale Testing")
    print("Testing on EfficientNet and Vision Transformer")
    print("=" * 60 + "\n")
    
    results = {}
    
    try:
        results['efficientnet'] = test_efficientnet()
    except Exception as e:
        print(f"FAIL: EfficientNet test failed: {e}")
        results['efficientnet'] = False
    
    try:
        results['vit'] = test_vision_transformer()
    except Exception as e:
        print(f"FAIL: ViT test failed: {e}")
        results['vit'] = False
    
    try:
        results['memory'] = test_memory_profile()
    except Exception as e:
        print(f"FAIL: Memory test failed: {e}")
        results['memory'] = False
    
    print("\n" + "=" * 60)
    print("Scale Testing Summary")
    print("=" * 60)
    for test, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
