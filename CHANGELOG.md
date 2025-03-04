# Changelog

## [0.2.1] - 04-03-2025

### Added
- **Macros for the DSL**:
  - Introduced `define` blocks to simplify reusable layer structures.
  - Allows parameter overrides in macro references.
  - Improved error messages for macro expansion.
- **Basic PyTorch Training Loop**:
  - Added a simple training loop for PyTorch, requiring user-provided DataLoader.
- **JSON Schema for Code Editors**:
  - Introduced `neural-schema.json` for syntax validation and autocompletion.

### Fixed
- **TensorFlow Code Generation**:
  - Fixed optimizer import handling (`Adam` is now imported explicitly).
  - Corrected loss function extraction from model data.
  - Ensured formatting consistency in `model.compile()`.
- **Layer Multiplication Bug**:
  - Fixed incorrect dictionary key (`multiply` → `*`).
- **Macro Parsing Errors**:
  - Macros now store correct layer definitions.
  - Fixed grammar conflicts between standard layer names and macros.
- **Dashboard Test Issues**:
  - Fixed title assertion errors.
  - Improved resource cleanup.

### Improved
- **Error Handling**:
  - Better distinction between custom layers and macros.
  - Clearer messages when parsing macros and layer structures.
- **Logging**:
  - Replaced `print()` statements with `logger.warning()` for unsupported PyTorch layers.
- **Nested Configurations**:
  - Layers can now contain sub-layers using `{}` (useful for Transformer and Residual networks).

### Known Issues
- **Neural is still in an early, very buggy state**. This release is primarily to showcase progress.
- Macro support is functional but requires further testing with complex architectures.

---

⚠️ **Neural is a work in progress! Expect bugs and missing features.** Feedback is welcome!



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