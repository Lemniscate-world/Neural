import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor
from neural.parser.parser import ModelTransformer
import keras
from neural.shape_propagation.shape_propagator import ShapePropagator
from neural.execution_optimization.execution import get_device
import copy

# Data Loader
def get_data(dataset_name, input_shape, batch_size, train=True, backend='pytorch'):
    datasets = {'MNIST': MNIST, 'CIFAR10': CIFAR10}
    dataset = datasets.get(dataset_name, MNIST)(root='./data', train=train, transform=ToTensor(), download=True)
    if backend == 'pytorch':
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
    elif backend == 'tensorflow':
        data = dataset.data.numpy() / 255.0  # Normalize
        targets = dataset.targets.numpy()
        if len(data.shape) == 3:  # Add channel dimension
            data = data[..., None]  # [N, H, W] → [N, H, W, 1]
        return tf.data.Dataset.from_tensor_slices((data, targets)).batch(batch_size)

def prod(iterable):
    result = 1
    for x in iterable:
        result *= x
    return result

# Factory Function
def create_dynamic_model(model_dict, trial, hpo_params, backend='pytorch'):
    resolved_model_dict = copy.deepcopy(model_dict)
    print(f"Pre-resolve model_dict: {resolved_model_dict}")

    # Resolve HPO parameters in layers
    for layer in resolved_model_dict['layers']:
        if 'params' in layer and layer['params']:
            for param_name, param_value in layer['params'].items():
                if isinstance(param_value, dict) and 'hpo' in param_value:
                    hpo = param_value['hpo']
                    if hpo['type'] == 'categorical':
                        layer['params'][param_name] = trial.suggest_categorical(f"{layer['type']}_{param_name}", hpo['values'])
                    elif hpo['type'] == 'range':
                        layer['params'][param_name] = trial.suggest_float(
                            f"{layer['type']}_{param_name}", 
                            hpo['start'], 
                            hpo['end'], 
                            step=hpo.get('step', None)
                        )
                    elif hpo['type'] == 'log_range':
                        layer['params'][param_name] = trial.suggest_float(
                            f"{layer['type']}_{param_name}", 
                            hpo['low'], 
                            hpo['high'], 
                            log=True
                        )

    # Resolve HPO parameters in optimizer
    if 'optimizer' in resolved_model_dict and resolved_model_dict['optimizer']:
        for param_name, param_value in resolved_model_dict['optimizer']['params'].items():
            if isinstance(param_value, dict) and 'hpo' in param_value:
                hpo = param_value['hpo']
                if hpo['type'] == 'log_range':
                    resolved_model_dict['optimizer']['params'][param_name] = trial.suggest_float(
                        f"opt_{param_name}", 
                        hpo['low'], 
                        hpo['high'], 
                        log=True
                    )

    print(f"Post-resolve model_dict: {resolved_model_dict}")
    if backend == 'pytorch':
        return DynamicPTModel(resolved_model_dict, trial, hpo_params)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    

def resolve_hpo_params(model_dict, trial, hpo_params):
    import copy
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    resolved_dict = copy.deepcopy(model_dict)
    
    # logger.debug(f"Original layers: {resolved_dict['layers']}")
    for i, layer in enumerate(resolved_dict['layers']):
        if 'params' in layer and layer['params'] is not None and 'units' in layer['params'] and isinstance(layer['params']['units'], dict) and 'hpo' in layer['params']['units']:
            hpo = layer['params']['units']['hpo']
            key = f"{layer['type']}_units_{i}"
            if hpo['type'] == 'categorical':
                layer['params']['units'] = trial.suggest_categorical(key, hpo['values'])
            elif hpo['type'] == 'log_range':
                layer['params']['units'] = trial.suggest_float(key, hpo['low'], hpo['high'], log=True)
            # logger.debug(f"Layer {i} resolved units: {layer['params']['units']}")
    
    if resolved_dict['optimizer'] and 'params' in resolved_dict['optimizer']:
        # Clean up optimizer type
        opt_type = resolved_dict['optimizer']['type']
        if '(' in opt_type:
            resolved_dict['optimizer']['type'] = opt_type[:opt_type.index('(')].capitalize()  # 'adam(...)' -> 'Adam'
        # logger.debug(f"Cleaned optimizer type: {resolved_dict['optimizer']['type']}")
        
        for param, val in resolved_dict['optimizer']['params'].items():
            if isinstance(val, dict) and 'hpo' in val:
                hpo = val['hpo']
                if hpo['type'] == 'log_range':
                    resolved_dict['optimizer']['params'][param] = trial.suggest_float(
                        f"opt_{param}", hpo['low'], hpo['high'], log=True
                    )
                # logger.debug(f"Optimizer resolved {param}: {resolved_dict['optimizer']['params'][param]}")
    
    # logger.debug(f"Resolved dict: {resolved_dict}")
    return resolved_dict


# Dynamic Models
class DynamicPTModel(nn.Module):
    def __init__(self, model_dict, trial, hpo_params):
        super().__init__()
        self.model_dict = model_dict
        self.layers = nn.ModuleList()
        self.shape_propagator = ShapePropagator(debug=True)
        input_shape_raw = model_dict['input']['shape']  # (28, 28, 1)
        input_shape = (None, input_shape_raw[-1], *input_shape_raw[:-1])  # (None, 1, 28, 28)
        current_shape = input_shape
        in_channels = input_shape[1]  # 1
        in_features = None

        print(f"Initial shape: {current_shape}")
        for layer in model_dict['layers']:
            params = layer['params'] if layer['params'] is not None else {}
            params = params.copy()
            print(f"Before propagate: {layer['type']}, current_shape={current_shape}")

            # Compute in_features from current (input) shape before propagation
            if layer['type'] in ['Dense', 'Output'] and in_features is None:
                in_features = prod(current_shape[1:])  # Use input shape
                self.layers.append(nn.Flatten())
                print(f"{layer['type']} (implicit Flatten): in_features={in_features}")

            # Propagate shape after setting in_features
            current_shape = self.shape_propagator.propagate(current_shape, layer, framework='pytorch')
            print(f"After propagate: {layer['type']}, current_shape={current_shape}")

            if layer['type'] == 'Conv2D':
                filters = params.get('filters', trial.suggest_int('conv_filters', 16, 64))
                kernel_size = params.get('kernel_size', 3)
                self.layers.append(nn.Conv2d(in_channels, filters, kernel_size))
                in_channels = filters
            elif layer['type'] == 'MaxPooling2D':
                pool_size = params.get('pool_size', trial.suggest_int('maxpool2d_pool_size', 2, 3))
                stride = params.get('stride', pool_size)
                print(f"MaxPooling2D: pool_size={pool_size}, stride={stride}")
                self.layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=stride))
            elif layer['type'] == 'Flatten':
                self.layers.append(nn.Flatten())
                in_features = prod(current_shape[1:])
                print(f"Flatten: in_features={in_features}")
            elif layer['type'] == 'Dense':
                units = params['units'] if 'units' in params else trial.suggest_int('dense_units', 64, 256)
                if in_features <= 0:
                    raise ValueError(f"Invalid in_features for Dense: {in_features}")
                print(f"Dense: in_features={in_features}, units={units}")
                self.layers.append(nn.Linear(in_features, units))
                in_features = units
            elif layer['type'] == 'Dropout':
                rate = params['rate'] if 'rate' in params else trial.suggest_float('dropout_rate', 0.3, 0.7, step=0.1)
                print(f"Dropout: rate={rate}")
                self.layers.append(nn.Dropout(p=rate))
            elif layer['type'] == 'Output':
                units = params['units'] if 'units' in params else 10
                if in_features <= 0:
                    raise ValueError(f"Invalid in_features for Output: {in_features}")
                print(f"Output: in_features={in_features}, units={units}")
                self.layers.append(nn.Linear(in_features, units))
                in_features = units
            elif layer['type'] == 'LSTM':
                input_size = current_shape[-1]
                units = params.get('units', trial.suggest_int('lstm_units', 32, 256))
                num_layers = params.get('num_layers', 1)
                if isinstance(params.get('num_layers'), dict) and 'hpo' in params.get('num_layers'):
                    num_layers = trial.suggest_int('lstm_num_layers', 1, 3)
                print(f"LSTM: input_size={input_size}, units={units}, num_layers={num_layers}")
                self.layers.append(nn.LSTM(input_size, units, num_layers=num_layers, batch_first=True))
                in_features = units
            elif layer['type'] == 'BatchNormalization':
                momentum = params.get('momentum', trial.suggest_float('bn_momentum', 0.8, 0.99))
                print(f"BatchNormalization: momentum={momentum}")
                self.layers.append(nn.BatchNorm2d(in_channels))
            elif layer['type'] == 'Transformer':
                d_model = params.get('d_model', trial.suggest_int('transformer_d_model', 64, 512))
                nhead = params.get('nhead', trial.suggest_int('transformer_nhead', 4, 8))
                num_encoder_layers = params.get('num_encoder_layers', trial.suggest_int('transformer_encoder_layers', 1, 4))
                num_decoder_layers = params.get('num_decoder_layers', trial.suggest_int('transformer_decoder_layers', 1, 4))
                dim_feedforward = params.get('dim_feedforward', trial.suggest_int('transformer_ff_dim', 128, 1024))
                print(f"Transformer: d_model={d_model}, nhead={nhead}, encoder_layers={num_encoder_layers}, decoder_layers={num_decoder_layers}, ff_dim={dim_feedforward}")
                self.layers.append(nn.Transformer(d_model=d_model,
                                                  nhead=nhead,
                                                  num_encoder_layers=num_encoder_layers,
                                                  num_decoder_layers=num_decoder_layers,
                                                  dim_feedforward=dim_feedforward))
                in_features = d_model
            else:
                raise ValueError(f"Unsupported layer type: {layer['type']}")
        print("Final layers:", self.layers)

    def forward(self, x):
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
def train_model(model, optimizer, train_loader, val_loader, backend='pytorch', epochs=1, execution_config=None):
    if backend == 'pytorch':
        device = get_device(execution_config.get("device", "auto") if execution_config else "auto")
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
        preds, targets = [], []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                preds.extend(pred.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        # Compute precision and recall
        preds_np = np.array(preds)
        targets_np = np.array(targets)
        precision = precision_score(targets_np, preds_np, average='macro')
        recall = recall_score(targets_np, preds_np, average='macro')
        
        return val_loss / len(val_loader), correct / total, precision, recall
    

# HPO Objective
def objective(trial, config, dataset_name='MNIST', backend='pytorch'):
    import torch.optim as optim
    model_dict, hpo_params = ModelTransformer().parse_network_with_hpo(config)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    train_loader = get_data(dataset_name, model_dict['input']['shape'], batch_size, True)
    val_loader = get_data(dataset_name, model_dict['input']['shape'], batch_size, False)
    
    model = create_dynamic_model(model_dict, trial, hpo_params, backend)
    optimizer_config = model.model_dict['optimizer']
    
    learning_rate_param = optimizer_config['params'].get('learning_rate', 0.001)
    if isinstance(learning_rate_param, dict) and 'hpo' in learning_rate_param:
        hpo = learning_rate_param['hpo']
        if hpo['type'] == 'log_range':
            lr = trial.suggest_float("learning_rate", hpo['low'], hpo['high'], log=True)
        else:
            lr = float(learning_rate_param)
    else:
        lr = float(learning_rate_param)
    
    if backend == 'pytorch':
        optimizer = getattr(optim, optimizer_config['type'])(model.parameters(), lr=lr)
    
    # Pass optimizer correctly
    loss, acc, precision, recall = train_model(model, optimizer, train_loader, val_loader, backend=backend)
    return loss, acc, precision, recall



# Optimize and Return
def optimize_and_return(config, n_trials=10, dataset_name='MNIST', backend='pytorch'):
    study = optuna.create_study(directions=["minimize", "minimize", "maximize", "maximize"])
    
    def objective_wrapper(trial):
        # Parse the config once per trial
        model_dict, hpo_params = ModelTransformer().parse_network_with_hpo(config)
        
        # Resolve batch_size from training_config or HPO
        training_config = model_dict.get('training_config', {})
        batch_size = training_config.get('batch_size', 32)  # Default if not specified
        
        if isinstance(batch_size, dict) and 'hpo' in batch_size:
            hpo = batch_size['hpo']
            if hpo['type'] == 'categorical':
                batch_size = trial.suggest_categorical("batch_size", hpo['values'])
            elif hpo['type'] == 'range':
                batch_size = trial.suggest_int("batch_size", hpo['start'], hpo['end'], step=hpo.get('step', 1))
            elif hpo['type'] == 'log_range':
                batch_size = trial.suggest_int("batch_size", hpo['low'], hpo['high'], log=True)
        elif isinstance(batch_size, list):
            batch_size = trial.suggest_categorical("batch_size", batch_size)
        
        # Ensure batch_size is an integer
        batch_size = int(batch_size)
        
        train_loader = get_data(dataset_name, model_dict['input']['shape'], batch_size, True, backend)
        val_loader = get_data(dataset_name, model_dict['input']['shape'], batch_size, False, backend)
        
        # Create model and resolve all HPO parameters in one place
        model = create_dynamic_model(model_dict, trial, hpo_params, backend)
        
        # Get optimizer configuration from resolved model_dict, default if None
        optimizer_config = model.model_dict['optimizer']
        if optimizer_config is None:
            optimizer_config = {'type': 'Adam', 'params': {'learning_rate': 0.001}}
        elif 'params' not in optimizer_config or not optimizer_config['params']:
            optimizer_config['params'] = {'learning_rate': 0.001}
        
        lr = optimizer_config['params']['learning_rate']  # Already resolved by create_dynamic_model or default
        
        if backend == 'pytorch':
            optimizer = getattr(optim, optimizer_config['type'])(model.parameters(), lr=lr)
        
        # Train and evaluate
        loss, acc, precision, recall = train_model(model, optimizer, train_loader, val_loader, backend=backend)
        return loss, acc, precision, recall

    study.optimize(objective_wrapper, n_trials=n_trials)
    
    # Normalize the best parameters
    best_params = study.best_trials[0].params
    normalized_params = {
        'batch_size': best_params.get('batch_size', 32),  # Use consistent naming
    }
    if 'Dense_units' in best_params:
        normalized_params['dense_units'] = best_params['Dense_units']
    if 'Dropout_rate' in best_params:
        normalized_params['dropout_rate'] = best_params['Dropout_rate']
    if 'opt_learning_rate' in best_params:
        normalized_params['learning_rate'] = best_params['opt_learning_rate']
    else:
        normalized_params['learning_rate'] = 0.001  # Default from optimizer_config
    
    return normalized_params