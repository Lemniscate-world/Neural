import optuna
import pysnooper
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor
from neural.parser.parser import ModelTransformer, create_parser

@pysnooper.snoop()
def get_data(dataset_name, input_shape, batch_size, train=True):
    datasets = {'MNIST': MNIST, 'CIFAR10': CIFAR10}  
    dataset = datasets.get(dataset_name, MNIST)
    return torch.utils.data.DataLoader(
        dataset(root='./data', train=train, transform=ToTensor(), download=True),
        batch_size=batch_size, shuffle=train
    )

def prod(iterable):
    result = 1
    for x in iterable:
        result *= x
    return result

@pysnooper.snoop()
class DynamicModel(nn.Module):
    def __init__(self, model_dict, trial, hpo_params):
        super().__init__()
        self.layers = nn.ModuleList()
        input_shape = model_dict['input']['shape']
        self.needs_flatten = len(input_shape) > 2
        in_channels = input_shape[-1] if len(input_shape) > 2 else 1
        in_features = prod(input_shape)  # Always compute product, forward handles flattening

        for layer in model_dict['layers']:
            params = layer['params'].copy()
            if layer['type'] == 'Conv2D':
                filters = params.get('filters', trial.suggest_int('conv_filters', 16, 64))
                kernel_size = params.get('kernel_size', 3)
                self.layers.append(nn.Conv2d(in_channels, filters, kernel_size))
                h_out = (input_shape[1] - kernel_size + 1)  # Assuming stride=1, padding=0
                w_out = (input_shape[2] - kernel_size + 1)
                input_shape = (h_out, w_out, filters)
                in_channels = filters
                in_features = None  # Reset for subsequent layers if needed
            elif layer['type'] == 'Flatten':
                self.layers.append(nn.Flatten())
                in_features = prod(input_shape)
                self.needs_flatten = False
            elif layer['type'] == 'Dense':
                if 'hpo' in params['units']:
                    hpo = next(h for h in hpo_params if h['layer_type'] == 'Dense' and h['param_name'] == 'units')
                    units = trial.suggest_categorical('dense_units', hpo['hpo']['values'])
                    params['units'] = units
                # Ensure in_features is not None
                if in_features is None:
                    raise ValueError("Input features must be defined for Dense layer.")
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
                # Check if 'units' is a dictionary with HPO configuration
                if isinstance(params.get('units'), dict) and 'hpo' in params['units']:
                    hpo = next(h for h in hpo_params if h['layer_type'] == 'Output' and h['param_name'] == 'units')
                    units = trial.suggest_categorical('output_units', hpo['hpo']['values'])
                    params['units'] = units
                self.layers.append(nn.Linear(in_features, params['units']))
                if params.get('activation') == 'softmax':
                    self.layers.append(nn.Softmax(dim=1))
                in_features = params['units']

    def forward(self, x):
        if self.needs_flatten:
            x = x.view(x.size(0), *self.input_shape[-3:])  # Preserve channels
        else:
            x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x

@pysnooper.snoop()
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

def objective(trial, config, dataset_name='MNIST'):
    model_dict, hpo_params = ModelTransformer().parse_network_with_hpo(config)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    train_loader = get_data(dataset_name, model_dict['input']['shape'], batch_size, True)
    val_loader = get_data(dataset_name, model_dict['input']['shape'], batch_size, False)
    
    optimizer_config = model_dict['optimizer']
    learning_rate_param = optimizer_config['params'].get('learning_rate', 0.001)

    if isinstance(learning_rate_param, dict) and 'hpo' in learning_rate_param:
        hpo = learning_rate_param['hpo']
        if hpo['type'] == 'log_range':
            lr = trial.suggest_float("learning_rate", hpo['low'], hpo['high'], log=True)
    elif isinstance(learning_rate_param, str) and 'HPO(log_range' in learning_rate_param:
        try:
            hpo = next(h for h in hpo_params if h['layer_type'] == 'optimizer' and h['param_name'] == 'learning_rate')
            lr = trial.suggest_float("learning_rate", hpo['hpo']['low'], hpo['hpo']['high'], log=True)
        except StopIteration:
            raise ValueError("HPO for learning_rate not found in hpo_params; parsing failed.")
    else:
        lr = float(learning_rate_param)
    
    model = DynamicModel(model_dict, trial, hpo_params)
    optimizer = getattr(optim, optimizer_config['type'])(model.parameters(), lr=lr)
    
    val_loss, val_acc = train_model(model, optimizer, train_loader, val_loader)
    return val_loss, -val_acc

@pysnooper.snoop()
def optimize_and_return(config, n_trials=10, dataset_name='MNIST'):
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(lambda trial: objective(trial, config, dataset_name), n_trials=n_trials)
    return study.best_trials[0].params