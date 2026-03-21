
import torch
import torch.nn as nn
import torch._dynamo as dynamo

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
