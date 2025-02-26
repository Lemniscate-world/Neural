# Changelog

## [0.2.0] - 25-02-2025

### Added
- **DSL Semantic Validation**: Custom error handling with severity levels (ERROR, WARNING, etc.) for granular error reporting.
- **Layer-Specific Checks**:
  - Dropout rate range validation (0 ≤ rate ≤ 1).
  - Conv2D filters/kernel_size, Dense units, MaxPooling parameters, and RNN/Embedding/Transformer dimensions must be positive integers.
  - BatchNormalization axis must be an integer.
- **CLI Enhancements**:
  - Global `--verbose` flag and structured logging with timestamps.
  - `--dry-run` mode for compile command.
  - Expanded `debug` command with backend simulation and step confirmation.
  - `no-code` command to launch GUI dashboard.
- **Documentation**: Added DSL syntax rules and error examples to docs.

### Fixed
- **Parser Errors**:
  - `test_layer_parsing[dropout-invalid-rate]`: Now raises error for invalid rates.
  - `test_layer_parsing[transformer]`: Default params added for TransformerEncoder (num_heads=8, ff_dim=512).
  - `test_layer_parsing[conv2d-zero-kernel]`: Kernel size validation upgraded to ERROR severity.
  - `test_cli.py::test_version_command`: Exit code corrected.
  - `test_network_parsing[invalid-validation-split]`: Validation split clamped to [0,1].
- **CLI Robustness**:
  - Unified file extension checks.
  - Wrapped parsing errors in try-except blocks to prevent silent failures.
- **Position Tracking**: Lark errors now include line/column details for debugging.

### Improved
- **Error Messaging**: Clearer DSL validation errors (e.g., "Conv2D kernel_size must be positive integers").
- **CLI Usability**: Progress bars, cached visualization, and backend flexibility (TensorFlow/PyTorch/ONNX).
- **Logging Configuration**: Severity levels mapped to standard logging modules (DEBUG, INFO, etc.).

---

## [0.1.2] - 24-02-2025
### Fixed
- MaxPooling2D strides parsing.
- Conv2D layer parameter extraction (filters, kernel_size, activation).
- CLI test errors (imports, file creation, exit codes).
- Dashboard connection issues and code generator NoneType errors.

---

## [0.1.1] - 22-02-2025
### Fixed
- Test suite stability (Gantt/heatmap assertions, WebSocket parameters, TensorFlow imports).
- KeyError handling in model comparison and invalid data flows.

---

## [0.1.0] - 21-02-2025
### Added
- Initial release with DSL parser, CLI, and NeuralDbg dashboard.
- ONNX export and TensorBoard integration.