
import torch
import torch.nn as nn
import pytest
import time
from neuraldbg import NeuralDbg

# 1. Output-Only Gradients Scenario (UserWarning Test)
def test_output_only_gradients():
    model = nn.Linear(10, 1)
    # Input does NOT require grad
    x = torch.randn(1, 10)
    
    # FIRST TEST EAGER
    with NeuralDbg(model) as dbg:
        out = model(x)
        out.sum().backward()
        events = dbg.get_events()
        print(f"\nEager Events: {len(events)}")
        assert len(events) > 0, "Should capture events in Eager mode"

    # RESET
    model = nn.Linear(10, 1)
    
    # THEN TEST COMPILED
    with NeuralDbg(model) as dbg:
        # Compile AFTER installing hooks
        compiled_model = torch.compile(model, backend="aot_eager")
        out = compiled_model(x)
        loss = out.sum()
        
        # This should trigger the UserWarning but STILL fire hooks
        loss.backward()
        
        # Verify event capture
        events = dbg.get_events()
        print(f"Compiled Events: {len(events)}")
        assert len(events) > 0, "Should capture events even if only outputs have gradients"

# 2. High-Density Hooks (Performance Benchmark)
def test_high_density_hook_performance():
    class DeepModel(nn.Module):
        def __init__(self, layers=20):
            super().__init__()
            self.net = nn.Sequential(*[nn.Linear(10, 10) for _ in range(layers)])
            
        def forward(self, x):
            return self.net(x)

    model = DeepModel(layers=20)
    x = torch.randn(1, 10)
    
    # Warmup
    torch.compile(model, backend="aot_eager")(x).sum().backward()

    # Benchmark without NeuralDbg
    start = time.perf_counter()
    for _ in range(10):
        model(x).sum().backward()
    baseline = time.perf_counter() - start

    # Benchmark with NeuralDbg
    with NeuralDbg(model) as dbg:
        compiled_model = torch.compile(model, backend="aot_eager")
        start = time.perf_counter()
        for _ in range(10):
            compiled_model(x).sum().backward()
        hooked_time = time.perf_counter() - start

    print(f"\nBaseline: {baseline:.4f}s | Hooked: {hooked_time:.4f}s")
    # We expect overhead but it shouldn't be astronomical (>10x)
    assert hooked_time < baseline * 20, "Overhead too high (>20x)"

# 3. Compiler Disable Workaround
def test_compiler_disable_persistence():
    class SubModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(10, 10)
            
        @torch.compiler.disable
        def forward(self, x):
            return self.lin(x)

    class MainModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = SubModule()
            self.last = nn.Linear(10, 1)
            
        def forward(self, x):
            x = self.sub(x)
            return self.last(x)

    model = MainModel()
    x = torch.randn(1, 10)
    
    with NeuralDbg(model) as dbg:
        compiled_model = torch.compile(model, backend="aot_eager")
        out = compiled_model(x)
        out.sum().backward()
        
        events = dbg.get_events()
        # Events should be captured for BOTH the compiled 'last' layer 
        # and the disabled 'sub' layer
        sub_events = [e for e in events if 'sub' in e.layer_name]
        last_events = [e for e in events if 'last' in e.layer_name]
        
        assert len(sub_events) > 0, "Should capture events in @torch.compiler.disable regions"
        assert len(last_events) > 0, "Should capture events in compiled regions"

# 4. Backend Parity Mock (AOT_Eager vs Inductor imitation)
def test_graph_break_recovery():
    class GraphBreakModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(10, 10)
            self.lin2 = nn.Linear(10, 1)
            
        def forward(self, x):
            x = self.lin1(x)
            # Force a graph break
            torch._dynamo.graph_break()
            x = self.lin2(x)
            return x

    model = GraphBreakModel()
    x = torch.randn(1, 10)
    
    with NeuralDbg(model) as dbg:
        compiled_model = torch.compile(model, backend="aot_eager")
        out = compiled_model(x)
        out.sum().backward()
        
        events = dbg.get_events()
        # Hooks should survive across graph breaks
        lin1_events = [e for e in events if 'lin1' in e.layer_name]
        lin2_events = [e for e in events if 'lin2' in e.layer_name]
        
        assert len(lin1_events) > 0, "Should capture events before graph break"
        assert len(lin2_events) > 0, "Should capture events after graph break"

# 5. Distributed Wrapper Simulation (DataParallel)
def test_dataparallel_wrapping():
    model = nn.Linear(10, 1)
    # Simulate a wrapper like DataParallel
    dp_model = nn.DataParallel(model, device_ids=[]) # Mocking DP on CPU
    x = torch.randn(2, 10)
    
    with NeuralDbg(dp_model) as dbg:
        # Note: DataParallel might copy weights, testing if hooks persist
        out = dp_model(x)
        out.sum().backward()
        
        events = dbg.get_events()
        assert len(events) > 0, "Should capture events inside DataParallel wrapper"


# 6. Inplace Operation Compatibility (ResNet-style residual connections)
def test_inplace_ops_backward_hook_compatibility():
    """Verify NeuralDBG works with models that use inplace operations.

    ResNet-style architectures use two patterns that conflict with
    register_full_backward_hook:
      1. ReLU(inplace=True) -- modifies activation tensors inplace
      2. out += identity   -- inplace add in residual connections

    This test builds a minimal residual block that reproduces both
    patterns and verifies that NeuralDBG captures events without
    RuntimeError.
    """
    class ResidualBlock(nn.Module):
        """Minimal residual block with inplace ReLU and inplace add."""
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out += identity  # inplace add (the key conflict point)
            out = self.relu(out)
            return out

    class SmallResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU(inplace=True)
            self.block1 = ResidualBlock(16)
            self.block2 = ResidualBlock(16)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 10)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.block1(x)
            x = self.block2(x)
            x = self.pool(x).flatten(1)
            return self.fc(x)

    model = SmallResNet()
    x = torch.randn(4, 3, 8, 8)
    labels = torch.randint(0, 10, (4,))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Run 3 training steps with NeuralDBG -- should NOT raise RuntimeError
    with NeuralDbg(model) as dbg:
        for step in range(3):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            dbg.step_iteration()

        events = dbg.get_events()
        # Should capture activation events from the residual blocks
        assert len(events) > 0, "Should capture events from ResNet-style model"

        # Verify we got events from inside the residual blocks
        block_events = [e for e in events if 'block' in e.layer_name]
        assert len(block_events) > 0, (
            "Should capture events from inside residual blocks"
        )
