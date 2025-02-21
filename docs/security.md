# Security in Neural

## Overview
Neural uses Flask and Flask-SocketIO for its API and WebSocket endpoints, with security measures to protect data and access.

## Authentication
- **HTTP Basic Auth**: The `/trace` API endpoint and WebSocket connections require basic authentication.
- **Configuration**: Credentials are stored in `config.yaml`:
  ```yaml
  auth:
    username: "admin"
    password: "your_secure_password"

    
## Hacky Mode
- **Purpose**: Inspired by "HackingNeuralNetworks," `--hacky` mode analyzes gradient leakage and simulates adversarial inputs to detect security vulnerabilities.
- **Usage**: `neural debug my_model.neural --hacky`
- **Features**:
  - **Gradient Leakage**: Checks for NaN gradients that could expose model internals.
  - **Adversarial Inputs**: Simulates noisy inputs to assess robustness.
- **Security Note**: Use in controlled environments—exposing vulnerabilities could aid attackers if not mitigated.