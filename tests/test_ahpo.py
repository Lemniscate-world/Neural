import pytest
import torch
from AHPO import TestModel, train_model, validate_optimizer, get_data

def test_model_forward():
    model = TestModel()
    x = torch.randn(32, 784)
    assert model(x).shape == (32, 10)

def test_training_loop():
    model = TestModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loader = get_data(32)
    loss = train_model(model, optimizer, train_loader, epochs=1)
    assert isinstance(loss, float)

def test_optimizer_validation():
    # Valid config
    validate_optimizer({
        "type": "Adam",
        "params": {"learning_rate": 0.001, "beta_1": 0.9}
    })
    
    # Invalid optimizer type
    with pytest.raises(ValueError):
        validate_optimizer({"type": "Ranger", "params": {}})
    
    # Invalid param
    with pytest.raises(ValueError):
        validate_optimizer({
            "type": "SGD",
            "params": {"momentum": 0.9, "invalid_param": 1.0}
        })

def test_hpo_objective():
    import optuna
    from AHPO import objective
    
    # Mock trial
    class MockTrial:
        def suggest_categorical(self, name, choices):
            return choices[0]
        def suggest_float(self, name, low, high, log=False):
            return 0.001
    
    trial = MockTrial()
    loss = objective(trial)
    assert isinstance(loss, float)