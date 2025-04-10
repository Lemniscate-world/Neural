# Neural Hyperparameter Optimization (HPO)

<p align="center">
  <img src="../../docs/images/hpo_workflow.png" alt="HPO Workflow" width="600"/>
</p>

## Overview

The Hyperparameter Optimization (HPO) module automates the process of finding optimal hyperparameters for neural network models. It uses advanced optimization techniques to efficiently search the hyperparameter space and identify configurations that maximize model performance.

## Key Features

1. **Cross-Framework Optimization**: Optimize hyperparameters for TensorFlow, PyTorch, and JAX models.
2. **Efficient Search Strategies**: Implement Bayesian optimization, evolutionary algorithms, and other efficient search methods.
3. **Distributed Execution**: Run optimization trials in parallel across multiple machines.
4. **Early Stopping**: Automatically terminate underperforming trials to save resources.
5. **Visualization**: Generate visualizations of the optimization process and results.
6. **Integration with Neural DSL**: Seamlessly integrate with Neural DSL models.

## Components

### 1. HPO Engine (`hpo.py`)

The main component that:
- Defines the hyperparameter search space
- Orchestrates the optimization process
- Manages trial execution
- Analyzes and reports results

### 2. Search Strategies

Implementations of various search algorithms:
- **Bayesian Optimization**: Uses Gaussian processes to model the objective function.
- **Tree-structured Parzen Estimator (TPE)**: Builds probabilistic models of performance.
- **Evolutionary Algorithms**: Uses genetic algorithms for optimization.
- **Random Search**: Provides a baseline for comparison.
- **Grid Search**: Exhaustively searches a predefined grid of hyperparameters.

### 3. Trial Executors

Components for executing optimization trials:
- **Local Executor**: Runs trials on the local machine.
- **Distributed Executor**: Distributes trials across multiple machines.
- **Cloud Executor**: Executes trials on cloud platforms.

### 4. Result Analyzers

Tools for analyzing optimization results:
- **Performance Analyzer**: Analyzes model performance across trials.
- **Importance Analyzer**: Identifies the most important hyperparameters.
- **Visualization Generator**: Creates visualizations of the optimization process.

## Usage

### Basic Usage

```python
from neural.hpo.hpo import optimize_and_return

# Neural DSL code
neural_code = """
network MNIST {
  input: (28, 28, 1)
  layers:
    Conv2D(32, kernel_size=3, activation="relu")
    MaxPooling2D(pool_size=2)
    Flatten()
    Dense(128, activation="relu")
    Output(10, activation="softmax")

  optimizer: Adam(learning_rate=0.001)
  batch_size: 64
}
"""

# Define the hyperparameter search space
search_space = {
    "learning_rate": {"type": "float", "min": 0.0001, "max": 0.01, "log": True},
    "batch_size": {"type": "categorical", "values": [32, 64, 128, 256]},
    "dropout_rate": {"type": "float", "min": 0.1, "max": 0.5}
}

# Run hyperparameter optimization
best_params = optimize_and_return(
    neural_code,
    n_trials=50,
    dataset_name="MNIST",
    backend="tensorflow",
    search_space=search_space
)

print("Best hyperparameters:", best_params)
```

### CLI Usage

```bash
neural compile model.neural --backend tensorflow --hpo --dataset MNIST
```

### Advanced Configuration

```python
from neural.hpo.hpo import HPOptimizer
from neural.hpo.strategies import BayesianStrategy

# Create an HPO optimizer with custom configuration
optimizer = HPOptimizer(
    strategy=BayesianStrategy(),
    n_trials=100,
    timeout=3600,  # 1 hour timeout
    early_stopping=True,
    parallel_trials=4,
    storage="sqlite:///hpo_results.db"
)

# Run optimization
best_params = optimizer.optimize(
    neural_code,
    dataset_name="MNIST",
    backend="tensorflow",
    search_space=search_space,
    metric="val_accuracy",
    direction="maximize"
)
```

## Hyperparameter Types

The HPO module supports various hyperparameter types:

1. **Numerical Parameters**:
   - **Float**: Continuous values (e.g., learning rate)
   - **Integer**: Discrete values (e.g., batch size)

2. **Categorical Parameters**:
   - **Discrete Choices**: Select from a list of options (e.g., activation functions)

3. **Conditional Parameters**:
   - Parameters that depend on other parameters (e.g., layer-specific parameters)

## Integration with Code Generation

The HPO module integrates with the Code Generation module to generate optimized code:

```python
from neural.code_generation.code_generator import generate_optimized_dsl

# Generate optimized DSL code with the best hyperparameters
optimized_dsl = generate_optimized_dsl(neural_code, best_params)

# Generate executable code with the optimized hyperparameters
from neural.code_generation.code_generator import generate_code
from neural.parser.parser import create_parser, ModelTransformer

parser = create_parser()
tree = parser.parse(optimized_dsl)
model_data = ModelTransformer().transform(tree)
optimized_code = generate_code(model_data, backend="tensorflow")
```

## Visualization

The HPO module provides visualization tools for analyzing optimization results:

```python
from neural.hpo.visualization import plot_optimization_history, plot_param_importance

# Plot optimization history
plot_optimization_history(optimizer.trials, metric="val_accuracy")

# Plot parameter importance
plot_param_importance(optimizer.trials)
```

## Extension Points

The HPO module is designed to be extensible:

1. **Custom Search Strategies**: Implement new search algorithms.
2. **Custom Trial Executors**: Add support for new execution environments.
3. **Custom Metrics**: Define custom performance metrics for optimization.
4. **Integration with External HPO Libraries**: Integrate with libraries like Optuna, Ray Tune, or HyperOpt.

## Resources

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Bayesian Optimization](https://arxiv.org/abs/1807.02811)
- [Neural DSL Reference](../../docs/DSL.md)
- [HPO Tutorial](../../docs/tutorials/hpo.md)
