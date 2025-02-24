# Changelog

## [0.1.0] - 21-02-2025

### Added

- Initial release with DSL parser, CLI, and NeuralDbg dashboard.
- No-code interface for model building.
- ONNX export and TensorBoard integration.
  
### Known Issues

- Bugs in shape propagation (under investigation).

## [0.1.1] - 22-02-2025

### Fixed

- Gantt chart name assertion in `test_update_trace_graph_gantt`.
- Heatmap data generation in `test_update_trace_graph_heatmap`.
- Type errors in `test_update_flops_memory_chart`, `test_update_dead_neurons`, `test_update_anomaly_chart`.
- WebSocket test missing `socketio` parameter in `test_websocket_connection`.
- KeyError for `kernel_size` in `test_model_comparison`.
- Invalid data handling in `test_update_trace_graph_invalid_data`.
- Tensor flow import in `test_tensor_flow_visualization`.
- NameError in `test_websocket_connection`
- Tensor flow import in `test_tensor_flow_visualization`.
- Dashboard theme check in `test_dashboard_theme`.
- Dashboard visualization dependency in `test_dashboard_visualization`.
- Box plot layer order mismatch in `test_update_trace_graph_box`.

## [Unreleased]

### Added

- `--hacky` mode for security analysis.

## [0.1.2] 

### Fixed

Parser:
  - Fixed MaxPooling2D strides parsing.

  - Resolved Conv2D layer parsing to ensure filters, kernel_size, and activation are captured (test: conv2d-relu).

  - Addressed AttributeError in conv2d method by using _extract_value helper for parameter handling (test: conv2d-tanh).

- CLI: Fixed test_compile_command errors (imports, file creation, data types, exit codes).

- WebSocket: Patched connection refusal (server setup).

- Dashboard: Fixed Selenium ERR_CONNECTION_REFUSED during visualization.

- Code Generator: Resolved NoneType error in TensorFlow code generation.