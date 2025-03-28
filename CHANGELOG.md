# Changelog

## [0.2.4] - 23-03-2025

### Added
- **Unified Model Creation**: Added `create_dynamic_model` factory function in `hpo.py` for PyTorch/TensorFlow backends (#434).
- **Training Loops in HPO**: Integrated training loops into `hpo.py` for full pipeline testing (#434).
- **Execution Optimization**: Added `execution_optimization` to `train_model` via `execution.py` with `get_device(preferred_device="auto")` for flexible device selection (#428).
- **TensorFlow Data Loader**: Added support for TensorFlow-compatible data loaders in `DynamicPTModel` (#427).
- **Precision Score**: Enhanced `train_model` in `hpo.py` to report precision alongside loss and accuracy (#428).

### Fixed
- **Test Failure: test_hpo_integration_full_pipeline (#434)**:
  - Updated `test_ahpo.py` to use `create_dynamic_model` and correct class names.
  - Fixed HPO logic in `objective` to respect the `backend` parameter.
  - Corrected HPO string parsing: `hpo_str` now reconstructs from `hpo['hpo']` instead of assuming `hpo['node']` is a Tree.
  - Fixed `generate_optimized_dsl` in `code_generator.py` for accurate HPO replacement.
  - Resolved key mismatch: Use `hpo['param_name']` (e.g., `'learning_rate'`) for optimizers, prefixed keys (e.g., `'dense_units'`) for layers.
  - Targeted HPO replacement: Iterate `hpo_params`, replace exact lines, and use `break` to prevent overwrites.
  - Skipped missing keys in `best_params` gracefully.
  - Parser update: `_parse_hpo` captures original string representations (e.g., `'1e-4'`).
  - Code generator uses original strings for exact config matches.
  - Renamed `HPOExample` network to `Example` for test consistency.
  - Refactored `DynamicPTModel` HPO handling for `Dense`, `Dropout`, and `Output` layers.

- **Test Failure: test_model_forward_conv2d (#427)**:
  - Updated `DynamicPTModel.__init__` to convert `input_shape_raw` to channels-first `(None, 1, 28, 28)` for PyTorch.
  - Adjusted test input tensor to match PyTorch’s `channels_first` format via permutation.

- **Test Failure: test_model_forward_flat_input**:
  - Fixed `in_features` calculation in `DynamicPTModel.__init__` for non-flattened inputs.
  - Computed `in_features` from input shape before `ShapePropagator.propagate` overwrites `current_shape`.
  - Avoided overriding `params['units']` with `trial.suggest_int`.

- **Test Failure: test_training_loop_convergence (#428)**:
  - Updated test to use `mock_data_loader` instead of passing `None` loaders.
  - Ensured `get_data` is called in the test with mocked loaders.

- **Test Failure: test_training_loop_invalid_optimizer (#429)**:
  - Added `MockDataset` and `MockDataLoader` to fix invalid optimizer test case.

### Improved
- **ShapePropagator**: Ensured `in_features` computation precedes shape propagation to avoid incorrect shape usage.
- **Train Model**: Unified PyTorch/TensorFlow training with device optimization and precision metrics (#428).

### Known Issues
- HPO replacement logic may still miss edge cases with complex configs.
- Limited validation for TensorFlow data loader compatibility.

---

## [0.2.3] - 16-03-2025

### Added
- **Multi-Framework HPO**: Extended `DynamicModel` for TensorFlow (`DynamicTFModel`) alongside PyTorch (#434).
- **Layer Support**: Added `LayerNormalization`, `InstanceNormalization`, `GroupNormalization`, `SqueezeExcitation`, `Attention` to parser (#105, #106, #107, #118, #307).
- **Macro & Custom Layer Enhancements**: Improved macro parsing, added device specification support (#136, #327, #328).

### Fixed
- **HPO Integration**: Corrected optimizer HPO parsing, fixed `in_features` calculation for 3D inputs (#434).
- **Layer Validation**: Enhanced `MaxPooling2D`, `BatchNormalization`, `Dropout`, `Conv2D` validation (#179, #363, #367, #368).
- **Parser Bugs**: Fixed `Concatenate`, `Activation`, `Lambda`, `Embedding` parameter handling (#140, #329, etc.).
- **Test Failures**: Resolved issues listed in your document (e.g., #136, #105, #179).

### Improved
- **Train Model**: Unified PyTorch/TensorFlow evaluation in `train_model` (#434).
- **Error Handling**: Better `VisitError` wrapping, detailed messages with line/column (#159).

---

## [0.2.2] - 05-03-2025

### Fixed
- **Layer Parameter Parsing**:
  - Unified parameter merging for `Dense`, `LSTM`, `GRUCell`, and `GaussianNoise` layers (#98, #110, #126, #355)
  - Resolved nested list flattening in `GaussianNoise(stddev=...)` (#126)
  - Fixed `STRING` token regex conflicts in activation functions (#154)
- **Validation & Error Handling**:
  - Added strict positive integer checks for `Dense.units` and `Conv2D.filters` (#159)
  - Fixed `VisitError` wrapping to expose raw `DSLValidationError` context (#159)
- **HPO Support**:
  - Corrected HPO grammar rules (`HPO(choice(...))` (#297)
  - Added HPO tracking for `units` and `activation` in `Dense` layers (#131, #297)
- **Macro System**:
  - Fixed macro parameter override logic during expansion

### Improved
- **Parameter Merging**:
  - Recursive list flattening for all layers (e.g., `[[{'units': 64}]]` → `{'units': 64}`)
  - Positional/named parameter unification (supports both `Dense(128)` and `Dense(units=128)`)
- **Error Messaging**:
  - Added line/column numbers to validation errors (e.g., `ERROR at line 5: Dense units must be positive`)
  - Expanded documentation with explicit error examples
- **Grammar Robustness**:
  - Resolved `NUMBER`/`FLOAT`/`INT` token conflicts (#342)
  - Simplified `param_style1` rules to prevent nested parentheses ambiguity

### Known Issues
- Limited PyTorch layer support (WIP)
- Macros with nested layer blocks may cause parser instability
- HPO `log_range()` requires explicit casting for integer parameters

---

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

---

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