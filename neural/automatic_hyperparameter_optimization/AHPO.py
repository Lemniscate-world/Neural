import optuna
import torch.optim as optim
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from neural.parser.parser import ModelTransformer  # Assume parser.py is in same directory

# --- Data Loading ---
def get_data(batch_size):
    return torch.utils.data.DataLoader(
        MNIST(root='./data', train=True, transform=ToTensor(), download=True),
        batch_size=batch_size, shuffle=True
    )

# --- Model Definition ---
class TestModel(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(TestModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

# --- Training Loop ---
def train_model(model, optimizer, train_loader, device='cpu', epochs=1):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss

# --- HPO Core ---
def objective(trial):
    # Suggest hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    lr = trial.suggest_float("learning_rate", 1e-4, 0.1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    
    # Model/optimizer setup
    model = TestModel()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    # Training
    train_loader = get_data(batch_size)
    loss = train_model(model, optimizer, train_loader, epochs=1)
    return loss

# --- Optimization ---
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# --- Optimizer Validation ---
VALID_OPTIMIZERS = {"Adam", "SGD", "RMSprop"}
VALID_PARAMS = {
    "Adam": {"learning_rate", "beta_1", "beta_2", "epsilon"},
    "SGD": {"learning_rate", "momentum", "nesterov"}
}

def validate_optimizer(config):
    opt_type = config["type"]
    if opt_type not in VALID_OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {opt_type}")
    invalid_params = set(config["params"].keys()) - VALID_PARAMS.get(opt_type, set())
    if invalid_params:
        raise ValueError(f"Invalid params for {opt_type}: {invalid_params}")