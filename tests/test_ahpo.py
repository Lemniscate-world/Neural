import pytest
import torch
import numpy as np
import os
import sys

# Add the parent directory of 'neural' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neural.automatic_hyperparameter_optimization.hpo import optimize_and_return
from neural.code_generation.code_generator import generate_optimized_dsl
from neural.automatic_hyperparameter_optimization.hpo import train_model, get_data, objective

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

def test_hpo_integration():
    config = """
network HPOExample {
    input: (28,28,1)
    layers:
        Dense(HPO(choice(128, 256)))
        Dropout(HPO(range(0.3, 0.7, step=0.1)))
        Output(10, "softmax")
    loss: "cross_entropy"
    optimizer: "Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))"
}
"""
    best_params = optimize_and_return(config, n_trials=1)
    assert 'dense_units' in best_params
    assert 'dropout_rate' in best_params
    assert 'learning_rate' in best_params
    optimized = generate_optimized_dsl(config, best_params)
    assert 'HPO' not in optimized