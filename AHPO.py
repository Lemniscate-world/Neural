import optuna
import torch.optim as optim

# Define objective function to optimize
def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    learning_rate = trial.suggest_loguniform("learning_rate", 0.0001, 0.1)

    # Load dataset
    train_loader = torch.utils.data.DataLoader(MNIST(), batch_size=batch_size, shuffle=True)

    # Define model & optimizer
    model = TestModel()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)

    # Train model
    loss = train_model(model, optimizer, train_loader)

    return loss

# Run HPO
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# Print best hyperparameters
print("Best params:", study.best_params)
