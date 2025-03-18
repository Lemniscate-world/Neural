import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor
from neural.parser.parser import ModelTransformer
import keras

# Data Loader
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

# Factory Function
def create_dynamic_model(model_dict, trial, hpo_params, backend='pytorch'):
    if backend == 'pytorch':
        return DynamicPTModel(model_dict, trial, hpo_params)
    elif backend == 'tensorflow':
        return DynamicTFModel(model_dict, trial, hpo_params)
    raise ValueError(f"Unsupported backend: {backend}")

# Dynamic Models
class DynamicPTModel(nn.Module):
    def __init__(self, model_dict, trial, hpo_params):
        super().__init__()
        self.layers = nn.ModuleList()
        input_shape = model_dict['input']['shape']
        self.needs_flatten = len(input_shape) > 2
        in_channels = input_shape[-1] if len(input_shape) > 2 else 1
        in_features = prod(input_shape)

        for layer in model_dict['layers']:
            params = layer['params'].copy()
            if layer['type'] == 'Conv2D':
                filters = params.get('filters', trial.suggest_int('conv_filters', 16, 64))
                kernel_size = params.get('kernel_size', 3)
                self.layers.append(nn.Conv2d(in_channels, filters, kernel_size))
                h_out = (input_shape[1] - kernel_size + 1)
                w_out = (input_shape[2] - kernel_size + 1)
                input_shape = (h_out, w_out, filters)
                in_channels = filters
                in_features = None
            elif layer['type'] == 'Flatten':
                self.layers.append(nn.Flatten())
                in_features = prod(input_shape)
                self.needs_flatten = False
            elif layer['type'] == 'Dense':
                units = params['units']
                if isinstance(units, dict) and 'hpo' in units:  # HPO case
                    hpo = next(h for h in hpo_params if h['layer_type'] == 'Dense' and h['param_name'] == 'units')
                    units = trial.suggest_categorical('dense_units', hpo['hpo']['values'])
                if in_features is None:
                    raise ValueError("Input features must be defined for Dense layer.")
                self.layers.append(nn.Linear(in_features, units))
                if params.get('activation') == 'relu':
                    self.layers.append(nn.ReLU())
                in_features = units
            elif layer['type'] == 'Dropout':
                rate = params['rate']
                if isinstance(rate, dict) and 'hpo' in rate:  # HPO case
                    hpo = next(h for h in hpo_params if h['layer_type'] == 'Dropout' and h['param_name'] == 'rate')
                    rate = trial.suggest_float('dropout_rate', hpo['hpo']['start'], hpo['hpo']['end'], step=hpo['hpo']['step'])
                self.layers.append(nn.Dropout(rate))
            elif layer['type'] == 'Output':
                units = params['units']
                if isinstance(units, dict) and 'hpo' in units:  # HPO case
                    hpo = next(h for h in hpo_params if h['layer_type'] == 'Output' and h['param_name'] == 'units')
                    units = trial.suggest_categorical('output_units', hpo['hpo']['values'])
                self.layers.append(nn.Linear(in_features, units))
                if params.get('activation') == 'softmax':
                    self.layers.append(nn.Softmax(dim=1))

    def forward(self, x):
        if self.needs_flatten:
            x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x

class DynamicTFModel(tf.keras.Model):
    def __init__(self, model_dict, trial, hpo_params):
        super().__init__()
        self.layers_list = []
        input_shape = model_dict['input']['shape']
        in_features = prod(input_shape)
        for layer in model_dict['layers']:
            params = layer['params'].copy()
            if layer['type'] == 'Dense':
                if 'hpo' in params['units']:
                    hpo = next(h for h in hpo_params if h['layer_type'] == 'Dense' and h['param_name'] == 'units')
                    units = trial.suggest_categorical('dense_units', hpo['hpo']['values'])
                    params['units'] = units
                self.layers_list.append(tf.keras.layers.Dense(params['units'], activation='relu' if params.get('activation') == 'relu' else None))
                in_features = params['units']
            elif layer['type'] == 'Dropout':
                if 'hpo' in params['rate']:
                    hpo = next(h for h in hpo_params if h['layer_type'] == 'Dropout' and h['param_name'] == 'rate')
                    rate = trial.suggest_float('dropout_rate', hpo['hpo']['start'], hpo['hpo']['end'], step=hpo['hpo']['step'])
                    params['rate'] = rate
                self.layers_list.append(tf.keras.layers.Dropout(params['rate']))
            elif layer['type'] == 'Output':
                if isinstance(params.get('units'), dict) and 'hpo' in params['units']:
                    hpo = next(h for h in hpo_params if h['layer_type'] == 'Output' and h['param_name'] == 'units')
                    units = trial.suggest_categorical('output_units', hpo['hpo']['values'])
                    params['units'] = units
                self.layers_list.append(tf.keras.layers.Dense(params['units'], activation='softmax' if params.get('activation') == 'softmax' else None))

    def call(self, inputs):
        x = tf.reshape(inputs, [inputs.shape[0], -1])  # Flatten input
        for layer in self.layers_list:
            x = layer(x)
        return x

# Training Method
def train_model(model, optimizer, train_loader, val_loader, backend='pytorch', epochs=1):
    if backend == 'pytorch':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        for _ in range(epochs):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        return val_loss / len(val_loader), correct / total
    elif backend == 'tensorflow':
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        val_loss, correct, total = 0.0, 0, 0
        for data, target in val_loader:
            data = tf.convert_to_tensor(data.numpy())
            target = tf.convert_to_tensor(target.numpy())
            output = model(data, training=False)
            val_loss += loss_fn(target, output).numpy()
            pred = tf.argmax(output, axis=1)
            correct += tf.reduce_sum(tf.cast(pred == target, tf.int32)).numpy()
            total += target.shape[0]
        return val_loss / len(val_loader), correct / total

# HPO Objective
def objective(trial, config, dataset_name='MNIST', backend='pytorch'):
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
        else:
            lr = float(learning_rate_param)
    else:
        lr = float(learning_rate_param)

    model = create_dynamic_model(model_dict, trial, hpo_params, backend)
    if backend == 'pytorch':
        optimizer = getattr(optim, optimizer_config['type'])(model.parameters(), lr=lr)
    elif backend == 'tensorflow':
        optimizer = tf.keras.optimizers.get({'class_name': optimizer_config['type'], 'config': {'learning_rate': lr}})

    val_loss, val_acc = train_model(model, optimizer, train_loader, val_loader, backend)
    return val_loss, -val_acc  # Negative accuracy for minimization

# Optimize and Return
def optimize_and_return(config, n_trials=10, dataset_name='MNIST', backend='pytorch'):
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(lambda trial: objective(trial, config, dataset_name, backend), n_trials=n_trials)
    print("Model optimized with best parameters:", study.best_trials[0].params)
    return study.best_trials[0].params