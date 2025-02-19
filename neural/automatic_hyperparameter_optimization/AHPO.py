import optuna
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# Fix: Make get_data accept batch_size as a parameter.
def get_data(batch_size):
    return torch.utils.data.DataLoader(
        MNIST(root='./data', train=True, transform=ToTensor(), download=True),
        batch_size=batch_size, shuffle=True
    )

# Placeholder TestModel: a simple fully connected network for MNIST.
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)  # MNIST images flattened to 784 features, 10 outputs

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Placeholder training function: runs one epoch and returns total loss.
def train_model(model, optimizer, train_loader, device='cpu', epochs=1):
    model.to(device)
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss

def objective(trial):
    # Suggest hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    learning_rate = trial.suggest_loguniform("learning_rate", 0.0001, 0.1)

    # Get training data with the specified batch_size
    train_loader = get_data(batch_size)

    # Instantiate model and optimizer
    model = TestModel()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)
    
    # Train the model for 1 epoch (adjust as needed)
    loss = train_model(model, optimizer, train_loader, epochs=1)
    
    return loss

# Run HPO
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# Print best hyperparameters
print("Best params:", study.best_params)


### Optimizer Validation ###

VALID_OPTIMIZERS = {"Adam", "SGD", "RMSprop"}
VALID_PARAMS = {
    "Adam": {"learning_rate", "beta_1", "beta_2", "epsilon"},
    "SGD": {"learning_rate", "momentum", "nesterov"}
}

def validate_optimizer(config):
    opt_type = config["type"]
    if opt_type not in VALID_OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {opt_type}")
    
    # Check keys of the parameters dictionary.
    invalid_params = set(config["params"].keys()) - VALID_PARAMS[opt_type]
    if invalid_params:
        raise ValueError(f"Invalid params for {opt_type}: {invalid_params}")
