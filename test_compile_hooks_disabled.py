
import torch
import torch.nn as nn
import torch._dynamo as dynamo
import pytest
import sys

if sys.version_info >= (3, 14) or not hasattr(torch, "compile"):
    pytest.skip(
        "torch.compile is not supported in this Python/PyTorch environment",
        allow_module_level=True,
    )

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.lin(x)

@dynamo.disable
def hook_fn(module, input, output):
    print(f"Disabled Hook fired for {module}")

model = SimpleModel()
model.lin.register_forward_hook(hook_fn)

x = torch.randn(1, 10)
print("--- Running Compiled (aot_eager) with Disabled Hook ---")
compiled_model = torch.compile(model, backend="aot_eager")
compiled_model(x)
