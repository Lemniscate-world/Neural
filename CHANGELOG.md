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

## [Unreleased]

### Added

- `--hacky` mode for security analysis.