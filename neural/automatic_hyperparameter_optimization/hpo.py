import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor
from neural.parser.parser import ModelTransformer, create_parser


def get_data(dataset_name, input_shape, batch_size, train=True):
    datasets = {'MNIST': MNIST, 'CIFAR10': CIFAR10}  
    dataset = datasets.get(dataset_name, MNIST)
    return torch.utils.data.DataLoader(
        dataset(root='./data', train=train, transform=ToTensor(), download=True),
        batch_size=batch_size, shuffle=train
    )

# Dynamic Model from Parsed DSL
class DynamicModel(nn.Module):
    """ Constructs a PyTorch model from model_dict, sampling HPO values 
    (e.g., dense_units, dropout_rate) using Optuna’s trial."""
    def __init__(self, model_dict, trial, hpo_params):
        super().__init__()
        self.layers = nn.ModuleList()
        self.input_shape = model_dict['input']['shape']
        self.flat_size = prod(self.input_shape)  # Dynamic flattening
        in_features = self.flat_size
        
        for layer in model_dict['layers']:
            params = layer['params'].copy()
            if layer['type'] == 'Conv2D':
                filters = params['filters'] if 'filters' in params else trial.suggest_int('conv_filters', 16, 64)
                kernel_size = params.get('kernel_size', 3)
                self.layers.append(nn.Conv2d(in_channels, filters, kernel_size))
                in_channels = filters
                self.needs_flatten = True
            elif layer['type'] == 'Flatten':
                self.layers.append(nn.Flatten())
                in_features = in_channels * input_shape[1] * input_shape[2]  # Post-conv
                self.needs_flatten = False
            elif layer['type'] == 'Dense':
                if 'hpo' in params['units']:
                    hpo = next(h for h in hpo_params if h['layer_type'] == 'Dense' and h['param_name'] == 'units')
                    units = trial.suggest_categorical('dense_units', hpo['hpo']['values'])
                    params['units'] = units
                self.layers.append(nn.Linear(in_features, params['units']))
                if params.get('activation') == 'relu':
                    self.layers.append(nn.ReLU())
                in_features = params['units']
            elif layer['type'] == 'Dropout':
                if 'hpo' in params['rate']:
                    hpo = next(h for h in hpo_params if h['layer_type'] == 'Dropout' and h['param_name'] == 'rate')
                    rate = trial.suggest_float('dropout_rate', hpo['hpo']['start'], hpo['hpo']['end'], step=hpo['hpo']['step'])
                    params['rate'] = rate
                self.layers.append(nn.Dropout(params['rate']))
            elif layer['type'] == 'Output':
                if 'hpo' in params['units']:
                    hpo = next(h for h in hpo_params if h['layer_type'] == 'Output' and h['param_name'] == 'units')
                    units = trial.suggest_categorical('output_units', hpo['hpo']['values'])
                    params['units'] = units
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
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return val_loss / len(val_loader), correct / total

# HPO Objective
def objective(trial, config):
    """ Parses config, builds model, trains it, and returns validation loss for Optuna to minimize."""
    model_dict, hpo_params = ModelTransformer().parse_network_with_hpo(config)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    train_loader = get_data(dataset_name, model_dict['input']['shape'], batch_size, True)
    val_loader = get_data(dataset_name, model_dict['input']['shape'], batch_size, False)
    
    optimizer_config = model_dict['optimizer']
    if 'hpo' in optimizer_config['params']['learning_rate']:
        hpo = next(h for h in hpo_params if h['param_name'] == 'learning_rate')
        lr = trial.suggest_float("learning_rate", hpo['hpo']['low'], hpo['hpo']['high'], log=True)
    else:
        lr = optimizer_config['params'].get('learning_rate', 0.001)
    
    model = DynamicModel(model_dict, trial, hpo_params)
    optimizer = getattr(optim, optimizer_config['type'])(model.parameters(), lr=lr)
    
    val_loss, val_acc = train_model(model, optimizer, train_loader, val_loader)
    return val_loss, -val_acc

# Run Optimization

def optimize_and_return(config, n_trials=10, dataset_name='MNIST'):
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(lambda trial: objective(trial, config, dataset_name), n_trials=n_trials)
    return study.best_trials[0].params