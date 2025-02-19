import optuna
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# --- Data Loading ---
def get_data(batch_size, train=True):
    return torch.utils.data.DataLoader(
        MNIST(root='./data', train=train, transform=ToTensor(), download=True),
        batch_size=batch_size, shuffle=train
    )

# --- Model Definition ---
class TestModel(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

# --- Training Loop ---
def train_model(model, optimizer, train_loader, val_loader, device='cpu', epochs=1):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    model.train()
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
    return val_loss / len(val_loader)

# --- HPO Core ---
def objective(trial):
    try:
        # Hyperparameters
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        lr = trial.suggest_float("learning_rate", 1e-4, 0.1, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
        
        # Data
        train_loader = get_data(batch_size, train=True)
        val_loader = get_data(batch_size, train=False)
        
        # Model
        model = TestModel()
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        
        # Training
        val_loss = train_model(model, optimizer, train_loader, val_loader)
        
        if np.isnan(val_loss) or val_loss > 1000:
            raise optuna.TrialPruned()
            
        return val_loss
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return float("inf")

# --- Optimization ---
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)