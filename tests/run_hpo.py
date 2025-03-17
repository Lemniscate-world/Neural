from neural.hpo.hpo import optimize_and_return
from neural.code_generation.code_generator import generate_optimized_dsl

config = """
network HPOExample {
    input: (28,28,1)
    layers:
        Dense(HPO(choice(128, 256)))
        Dropout(HPO(range(0.3, 0.7, step=0.1)))
        Output(10, "softmax")
    loss: "cross_entropy"
    optimizer: "Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))"
}
"""
best_params = optimize_and_return(config, n_trials=5)
optimized_config = generate_optimized_dsl(config, best_params)
print(optimized_config)