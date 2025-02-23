import pytest
import torch
import numpy as np
import os
import sys

# Add the parent directory of 'neural' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from neural.automatic_hyperparameter_optimization.AHPO import TestModel, train_model, get_data, objective

def test_model_forward():
    model = TestModel()
    x = torch.randn(32, 784)
    assert model(x).shape == (32, 10)

def test_training_loop():
    model = TestModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loader = get_data(32, train=True)
    val_loader = get_data(32, train=False)  # Add validation loader
    loss = train_model(model, optimizer, train_loader, val_loader, epochs=1)  # Updated signature
    assert isinstance(loss, float)

def test_hpo_objective():
    class MockTrial:
        def suggest_categorical(self, name, choices):
            if name == "batch_size":
                return 32
            elif name == "optimizer":
                return "Adam"
        
        def suggest_float(self, name, low, high, log=False):
            return 0.001  # Fixed learning rate
    
    trial = MockTrial()
    loss = objective(trial)
    assert 0 <= loss < float("inf")

def test_parsed_hpo_config():
    from neural.parser.parser import ModelTransformer, create_parser
    
    config = '''
    network TestNet {
        input: (28,28,1)
        layers:
            Dense(units=HPO(range(32, 256)), activation="relu")
        loss: "cross_entropy"
        optimizer: Adam(learning_rate=HPO(log_range(0.0001, 0.1)))
    }
    '''
    parser = create_parser()
    model = ModelTransformer().transform(parser.parse(config))
    assert "hpo" in model["layers"][0]["params"]["units"]