"""BUG-009 / pytorch#187227 — SDPA 32-bit offset overflow in mem-efficient attention

SDPA with attn_bias triggers int32 overflow for large tensors (>2^31 elements).
Silent correctness bug — no crash, just wrong attention outputs.

NOTE: Requires CUDA GPU with large VRAM to reproduce (>16GB for 2^31 elements).
This script documents the pattern and expected failure mode.

Original issue: https://github.com/pytorch/pytorch/issues/187227
Bug catalog: docs/bugs/BUG-009-pytorch-187227.md
"""

import torch
import sys

print("=" * 60)
print("BUG-009: SDPA 32-bit offset overflow")
print(f"PyTorch: {torch.__version__}")
print("=" * 60)

if not torch.cuda.is_available():
    print("\n[SKIP] CUDA not available. This bug requires GPU with >16GB VRAM.")
    print("Documenting expected failure mode below.")
else:
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB)")
    if vram_gb < 16:
        print("VRAM < 16GB — bug may not trigger (needs >2^31 elements).")

print("""
Expected failure mode:
1. Create attention with attn_bias where total elements > 2^31
2. SDPA mem-efficient kernel uses 32-bit offset internally
3. Integer overflow causes wrong memory access
4. Silent incorrect attention outputs (no crash, no NaN)

NeuralDBG would detect:
1. ATTENTION_OUTPUT_MISMATCH: SDPA vs eager attention differ
2. NUMERICAL_OVERFLOW event at the SDPA boundary
3. CAUSAL_CHAIN: large tensor > int32 -> offset overflow -> wrong attention -> training corruption
""")
