import optuna
import torch.optim as optim
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

def get_data():
    return torch.utils.data.DataLoader(
        MNIST(root='./data', train=True, transform=ToTensor(), download=True),
        batch_size=batch_size, shuffle=True
    )

def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    learning_rate = trial.suggest_loguniform("learning_rate", 0.0001, 0.1)

    train_loader = get_data()  # Use new function

    model = TestModel()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)
    
    loss = train_model(model, optimizer, train_loader)
    
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
    
    invalid_params = set(config["params"]) - VALID_PARAMS[opt_type]
    if invalid_params:
        raise ValueError(f"Invalid params for {opt_type}: {invalid_params}")
