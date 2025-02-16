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
