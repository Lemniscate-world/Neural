
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.lin(x)

def hook_fn(module, input, output):
    print(f"Hook fired for {module}")

model = SimpleModel()
model.lin.register_forward_hook(hook_fn)

print("--- Running Eager ---")
x = torch.randn(1, 10)
model(x)

print("--- Running Compiled (aot_eager) ---")
compiled_model = torch.compile(model, backend="aot_eager")
compiled_model(x)

print("--- Running Compiled (inductor if possible) ---")
try:
    compiled_model_ind = torch.compile(model)
    compiled_model_ind(x)
except Exception as e:
    print(f"Inductor failed: {e}")
