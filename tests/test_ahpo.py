import pytest
import torch
import torch.nn as nn
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neural.hpo.hpo import optimize_and_return, create_dynamic_model  # Updated import
from neural.hpo.hpo import train_model, get_data, objective
from neural.parser.parser import ModelTransformer, DSLValidationError
from neural.code_generation.code_generator import generate_optimized_dsl

class MockTrial:
    def suggest_categorical(self, name, choices):
        return 32 if name == "batch_size" else choices[0]
    def suggest_float(self, name, low, high, step=None, log=False):
        return low if not log else 0.001
    def suggest_int(self, name, low, high):
        return low

def mock_data_loader(dataset_name, input_shape, batch_size, train=True):
    class MockDataset:
        def __init__(self):
            self.data = torch.randn(100, *input_shape)
            self.targets = torch.randint(0, 10, (100,))
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
    return torch.utils.data.DataLoader(MockDataset(), batch_size=batch_size, shuffle=train)

# 1. Enhanced Forward Pass Tests
def test_model_forward_flat_input():
    config = "network Test { input: (28,28,1) layers: Dense(128) Output(10) }"
    model_dict, hpo_params = ModelTransformer().parse_network_with_hpo(config)
    model = create_dynamic_model(model_dict, MockTrial(), hpo_params, backend='pytorch')
    x = torch.randn(32, *model_dict['input']['shape'])  # [32, 28, 28, 1]
    x = x.permute(0, 3, 1, 2)  # [32, 1, 28, 28]
    output = model(x)
    assert output.shape == (32, 10), f"Expected (32, 10), got {output.shape}"
    
def test_model_forward_conv2d():
    config = "network Test { input: (28,28,1) layers: Conv2D(filters=16, kernel_size=3) Flatten() Dense(128) Output(10) }"
    model_dict, hpo_params = ModelTransformer().parse_network_with_hpo(config)
    model = create_dynamic_model(model_dict, MockTrial(), hpo_params, backend='pytorch')
    # Generate input in NHWC, then permute to NCHW
    x = torch.randn(32, *model_dict['input']['shape'])  # [32, 28, 28, 1]
    x = x.permute(0, 3, 1, 2)  # [32, 1, 28, 28]
    output = model(x)
    assert output.shape == torch.Size([32, 10]), f"Expected output shape [32, 10], got {output.shape}"

# 2. Enhanced Training Loop Tests
@patch('neural.automatic_hyperparameter_optimization.hpo.get_data', mock_data_loader)
def test_training_loop_convergence():
    config = "network Test { input: (28,28,1) layers: Dense(128) Output(10) }"
    model_dict, hpo_params = ModelTransformer().parse_network_with_hpo(config)
    model = create_dynamic_model(model_dict, MockTrial(), hpo_params, backend='pytorch')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Use mocked get_data instead of None
    train_loader = mock_data_loader('mock_dataset', model_dict['input']['shape'], batch_size=32, train=True)
    val_loader = mock_data_loader('mock_dataset', model_dict['input']['shape'], batch_size=32, train=False)
    loss = train_model(model, optimizer, train_loader, val_loader, backend='pytorch')
    assert isinstance(loss[0], float) and 0 <= loss[0] < 10, f"Loss not reasonable: {loss[0]}"
    assert 0 <= loss[1] <= 1, f"Accuracy not in range: {loss[1]}"

def test_training_loop_invalid_optimizer():
    config = "network Test { input: (28,28,1) layers: Dense(128) Output(10) }"
    model_dict, hpo_params = ModelTransformer().parse_network_with_hpo(config)
    model = create_dynamic_model(model_dict, MockTrial(), hpo_params, backend='pytorch')  # Updated
    with pytest.raises(AttributeError):
        train_model(model, "invalid_optimizer", None, None, backend='pytorch')

# 3. Enhanced HPO Objective Tests
@patch('neural.automatic_hyperparameter_optimization.hpo.get_data', mock_data_loader)
def test_hpo_objective_multi_objective():
    config = "network Test { input: (28,28,1) layers: Dense(128) Output(10) loss: 'cross_entropy' optimizer: 'Adam' }"
    trial = MockTrial()
    loss, acc = objective(trial, config, 'MNIST', backend='pytorch')
    assert 0 <= loss < float("inf"), f"Loss out of range: {loss}"
    assert -1 <= acc <= 0, f"Accuracy out of range: {acc}"

@patch('neural.automatic_hyperparameter_optimization.hpo.get_data', mock_data_loader)
def test_hpo_objective_with_hpo_params():
    config = "network Test { input: (28,28,1) layers: Dense(HPO(choice(64, 128))) Output(10) optimizer: 'Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))' }"
    trial = MockTrial()
    loss, acc = objective(trial, config, 'MNIST', backend='pytorch')
    assert 0 <= loss < float("inf")
    assert -1 <= acc <= 0

# 4. Enhanced Parser Tests
def test_parsed_hpo_config_all_types():
    config = """
    network Test {
        input: (28,28,1)
        layers:
            Dense(units=HPO(choice(32, 64)), activation="relu")
            Dropout(HPO(range(0.1, 0.5, step=0.1)))
            Output(HPO(log_range(10, 20)))
    }
    """
    model_dict, hpo_params = ModelTransformer().parse_network_with_hpo(config)
    assert len(hpo_params) == 3
    assert hpo_params[0]['hpo']['type'] == 'categorical'
    assert hpo_params[1]['hpo']['type'] == 'range'
    assert hpo_params[2]['hpo']['type'] == 'log_range'

def test_parser_invalid_config():
    config = "network Test { input: (28,28,1) layers: Dense(-1) }"
    with pytest.raises(DSLValidationError, match="must be positive"):
        ModelTransformer().parse_network_with_hpo(config)

# 5. Enhanced HPO Integration Tests
@patch('neural.automatic_hyperparameter_optimization.hpo.get_data', mock_data_loader)
def test_hpo_integration_full_pipeline():
    config = """
    network Example {
        input: (28,28,1)
        layers:
            Dense(HPO(choice(128, 256)))
            Dropout(HPO(range(0.3, 0.7, step=0.1)))
            Output(10, "softmax")
        loss: "cross_entropy"
        optimizer: "Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))"
    }
    """
    best_params = optimize_and_return(config, n_trials=3, dataset_name='MNIST', backend='pytorch')
    assert set(best_params.keys()) == {'batch_size', 'dense_units', 'dropout_rate', 'learning_rate'}
    optimized = generate_optimized_dsl(config, best_params)
    assert 'HPO' not in optimized
    model_dict, hpo_params = ModelTransformer().parse_network_with_hpo(optimized)
    assert not hpo_params
    model = create_dynamic_model(model_dict, MockTrial(), hpo_params, backend='pytorch')  # Updated
    assert model(torch.randn(32, 28, 28, 1)).shape == (32, 10)

# 6. Additional Tests
def test_code_generator_invalid_params():
    config = "network Test { input: (28,28,1) layers: Dense(128) }"
    invalid_params = {'unknown_param': 42}
    with pytest.raises(KeyError):
        generate_optimized_dsl(config, invalid_params)

@patch('neural.automatic_hyperparameter_optimization.hpo.get_data', mock_data_loader)
def test_hpo_edge_case_no_layers():
    config = "network Test { input: (28,28,1) layers: Output(10) }"
    best_params = optimize_and_return(config, n_trials=1, dataset_name='MNIST', backend='pytorch')
    assert 'batch_size' in best_params
    optimized = generate_optimized_dsl(config, best_params)
    model_dict, _ = ModelTransformer().parse_network_with_hpo(optimized)
    model = create_dynamic_model(model_dict, MockTrial(), [], backend='pytorch')  # Updated
    assert model(torch.randn(32, 28, 28, 1)).shape == (32, 10)