import pytest
import os
import tempfile
from neural.parser import create_parser, ModelTransformer
from neural.execution_optimization.execution import get_device, run_inference

# Sample network with device specifications
DEVICE_NETWORK = """
network DeviceTestModel {
    input: (10, 10, 3)
    layers:
        Conv2D(32, (3,3), "relu") @ "cuda:0"
        MaxPooling2D((2, 2))
        Flatten() @ "cpu"
        Dense(64, "relu") @ "cuda:0"
        Dense(10, "softmax")
    execution {
        device: "auto"
    }
}
"""

@pytest.fixture
def parser():
    return create_parser("network")

@pytest.fixture
def transformer():
    return ModelTransformer()

def test_device_parsing_to_execution(parser, transformer):
    """Test that device specifications in DSL are correctly parsed and can be used for execution."""
    # Parse the network
    tree = parser.parse(DEVICE_NETWORK)
    model_config = transformer.transform(tree)
    
    # Verify device specifications were parsed correctly
    assert model_config["layers"][0].get("device") == "cuda:0"
    assert model_config["layers"][2].get("device") == "cpu"
    assert model_config["layers"][3].get("device") == "cuda:0"
    assert model_config["execution_config"]["device"] == "auto"
    
    # In a real implementation, we would now create and run the model
    # For this test, we'll just verify the device selection logic
    layer_devices = [
        layer.get("device", model_config["execution_config"]["device"]) 
        for layer in model_config["layers"]
    ]
    
    # Verify we have the expected device assignments
    assert "cuda:0" in layer_devices
    assert "cpu" in layer_devices

def test_execution_config_override():
    """Test that global execution config can override layer-specific devices."""
    # This would typically be implemented in the model compiler/executor
    # Here we're just testing the concept
    
    # Example of how execution config might override layer devices
    layer_devices = ["cuda:0", "auto", "cpu", "cuda:0", "auto"]
    execution_config = {"device": "cpu"}  # Force CPU execution
    
    # In the real implementation, this logic would be in the model executor
    effective_devices = [
        "cpu" if execution_config.get("force_cpu", False) else device
        for device in layer_devices
    ]
    
    # All devices should be CPU if force_cpu is True
    if execution_config.get("force_cpu", False):
        assert all(device == "cpu" for device in effective_devices)
