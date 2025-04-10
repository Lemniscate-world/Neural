# Neural Tests

<p align="center">
  <img src="../docs/images/testing_pyramid.png" alt="Testing Pyramid" width="600"/>
</p>

## Overview

This directory contains the test suite for the Neural framework. The tests are organized into different categories based on their scope and purpose, following the testing pyramid approach with unit tests at the base, integration tests in the middle, and end-to-end tests at the top.

## Test Categories

### 1. Unit Tests

Unit tests focus on testing individual components in isolation:

- **Parser Tests**: Test the Neural DSL parser and transformer
- **Shape Propagation Tests**: Test the shape propagation and validation
- **Code Generation Tests**: Test the code generation for different backends
- **Visualization Tests**: Test the visualization components
- **HPO Tests**: Test the hyperparameter optimization components
- **CLI Tests**: Test the command-line interface components

### 2. Integration Tests

Integration tests focus on testing the interaction between components:

- **Parser-Shape Propagation Integration**: Test the integration between the parser and shape propagation
- **Parser-Code Generation Integration**: Test the integration between the parser and code generation
- **Shape Propagation-Visualization Integration**: Test the integration between shape propagation and visualization
- **CLI-Component Integration**: Test the integration between the CLI and other components

### 3. End-to-End Tests

End-to-end tests focus on testing complete workflows:

- **Compilation Workflow**: Test the complete compilation workflow from DSL to executable code
- **Visualization Workflow**: Test the complete visualization workflow
- **Debugging Workflow**: Test the complete debugging workflow
- **HPO Workflow**: Test the complete hyperparameter optimization workflow

### 4. Performance Tests

Performance tests focus on measuring and ensuring performance:

- **Startup Time Tests**: Test the startup time of the CLI
- **Memory Usage Tests**: Test the memory usage of different components
- **Execution Time Tests**: Test the execution time of different operations
- **Scalability Tests**: Test the scalability of the framework with large models

### 5. Regression Tests

Regression tests focus on preventing regressions:

- **Bug Regression Tests**: Test fixes for previously identified bugs
- **Feature Regression Tests**: Test previously implemented features
- **Performance Regression Tests**: Test performance improvements

## Test Structure

Each test file follows a consistent structure:

```python
# Import necessary modules
import unittest
from neural.module import Component

class TestComponent(unittest.TestCase):
    def setUp(self):
        # Set up test environment
        self.component = Component()

    def tearDown(self):
        # Clean up after tests
        pass

    def test_feature_one(self):
        # Test a specific feature
        result = self.component.feature_one()
        self.assertEqual(result, expected_result)

    def test_feature_two(self):
        # Test another feature
        result = self.component.feature_two()
        self.assertEqual(result, expected_result)

    # More test methods...

if __name__ == '__main__':
    unittest.main()
```

## Running Tests

### Running All Tests

To run all tests:

```bash
python -m unittest discover tests
```

### Running a Specific Test Category

To run a specific test category:

```bash
python -m unittest discover tests/unit
python -m unittest discover tests/integration
python -m unittest discover tests/end_to_end
```

### Running a Specific Test File

To run a specific test file:

```bash
python -m unittest tests/unit/test_parser.py
```

### Running a Specific Test Method

To run a specific test method:

```bash
python -m unittest tests.unit.test_parser.TestParser.test_parse_network
```

## Test Coverage

We use the `coverage` tool to measure test coverage:

```bash
# Install coverage
pip install coverage

# Run tests with coverage
coverage run -m unittest discover tests

# Generate coverage report
coverage report -m

# Generate HTML coverage report
coverage html
```

The coverage report shows which parts of the codebase are covered by tests and which parts need more testing.

## Continuous Integration

The tests are automatically run in a continuous integration (CI) environment on every pull request and push to the main branch. The CI pipeline includes:

1. Running all tests
2. Measuring test coverage
3. Running linters and static analyzers
4. Building and testing the documentation
5. Building and testing the package

## Writing Tests

When writing tests, follow these guidelines:

1. **Test One Thing**: Each test should test one specific feature or behavior
2. **Isolation**: Tests should be isolated from each other and not depend on external state
3. **Determinism**: Tests should be deterministic and produce the same result every time
4. **Readability**: Tests should be readable and easy to understand
5. **Performance**: Tests should be fast to run
6. **Coverage**: Tests should cover all code paths, including edge cases and error conditions

## Test Fixtures

Test fixtures are used to set up the test environment and provide test data:

- **Model Fixtures**: Sample Neural DSL models for testing
- **Shape Fixtures**: Sample tensor shapes for testing
- **Code Fixtures**: Sample generated code for testing
- **Visualization Fixtures**: Sample visualizations for testing

## Mock Objects

Mock objects are used to isolate components during testing:

- **Mock Parser**: Mock the parser for testing other components
- **Mock Shape Propagator**: Mock the shape propagator for testing other components
- **Mock Code Generator**: Mock the code generator for testing other components
- **Mock Visualizer**: Mock the visualizer for testing other components

## Resources

- [Python unittest Documentation](https://docs.python.org/3/library/unittest.html)
- [Coverage Documentation](https://coverage.readthedocs.io/)
- [Test-Driven Development](https://en.wikipedia.org/wiki/Test-driven_development)
- [Testing Pyramid](https://martinfowler.com/bliki/TestPyramid.html)
