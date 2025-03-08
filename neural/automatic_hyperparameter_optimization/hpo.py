# AHPO.py (updated)
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from neural.parser.parser import ModelTransformer, create_parser

# Data Loading
def get_data(batch_size, train=True):
    return torch.utils.data.DataLoader(
        MNIST(root='./data', train=train, transform=ToTensor(), download=True),
        batch_size=batch_size, shuffle=train
    )

# Dynamic Model from Parsed DSL
class DynamicModel(nn.Module):
    def __init__(self, model_dict, trial, hpo_params):
        super().__init__()
        self.layers = nn.ModuleList()
        input_shape = model_dict['input']['shape']  # (28, 28, 1)
        in_features = input_shape[0] * input_shape[1] * input_shape[2]  # 784
        
        for layer in model_dict['layers']:
            params = layer['params'].copy()
            if layer['type'] == 'Dense':
                if 'hpo' in params['units']:
                    hpo = next(h for h in hpo_params if h['layer_type'] == 'Dense' and h['param_name'] == 'units')
                    if hpo['hpo']['type'] == 'categorical':
                        units = trial.suggest_categorical('dense_units', hpo['hpo']['values'])
                    params['units'] = units
                self.layers.append(nn.Linear(in_features, params['units']))
                if params.get('activation') == 'relu':
                    self.layers.append(nn.ReLU())
                in_features = params['units']
            elif layer['type'] == 'Dropout':
                if 'hpo' in params['rate']:
                    hpo = next(h for h in hpo_params if h['layer_type'] == 'Dropout' and h['param_name'] == 'rate')
                    if hpo['hpo']['type'] == 'range':
                        rate = trial.suggest_float('dropout_rate', hpo['hpo']['start'], hpo['hpo']['end'], step=hpo['hpo']['step'])
                    params['rate'] = rate
                self.layers.append(nn.Dropout(params['rate']))
            elif layer['type'] == 'Output':
                self.layers.append(nn.Linear(in_features, params['units']))
                if params.get('activation') == 'softmax':
                    self.layers.append(nn.Softmax(dim=1))
                in_features = params['units']

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten (batch, 28, 28, 1) -> (batch, 784)
        for layer in self.layers:
            x = layer(x)
        return x

# Training Loop
def train_model(model, optimizer, train_loader, val_loader, device='cpu', epochs=1):
    model.to(device)
    criterion = nn.CrossEntropyLoss()  # Matches MNIST classification
    model.train()
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
    return val_loss / len(val_loader)

# HPO Objective
def objective(trial, config):
    transformer = ModelTransformer()
    model_dict, hpo_params = transformer.parse_network_with_hpo(config)
    
    # Hyperparameters from DSL and trial
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    train_loader = get_data(batch_size, train=True)
    val_loader = get_data(batch_size, train=False)
    
    optimizer_config = model_dict['optimizer']
    if 'hpo' in optimizer_config['params']['learning_rate']:
        hpo = next(h for h in hpo_params if h['param_name'] == 'learning_rate')
        lr = trial.suggest_float("learning_rate", hpo['hpo']['low'], hpo['hpo']['high'], log=True)
    else:
        lr = optimizer_config['params'].get('learning_rate', 0.001)
    
    model = DynamicModel(model_dict, trial, hpo_params)
    optimizer = getattr(optim, optimizer_config['type'])(model.parameters(), lr=lr)
    
    val_loss = train_model(model, optimizer, train_loader, val_loader)
    return val_loss if val_loss < 1000 else float("inf")

# Run Optimization
def optimize_and_return(config, n_trials=10):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, config), n_trials=n_trials)
    return study.best_params

# Test with your HPOExample (modified for MNIST)
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
best_params = optimize_and_return(config)
print(f"Best params: {best_params}")