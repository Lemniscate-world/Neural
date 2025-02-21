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