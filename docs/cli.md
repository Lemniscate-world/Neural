# Neural CLI Documentation

## Command Overview
```bash
neural [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]
```

## Global Options
- `--verbose, -v`: Enable verbose logging
- `--help, -h`: Show help message

---

## Commands

### `compile` - Convert neural files to framework code
```bash
neural compile [OPTIONS] FILE
```

**Options**:
- `--backend, -b`: Target framework (`tensorflow`, `pytorch`, `onnx`) [default: tensorflow]
- `--output, -o`: Output file path
- `--dry-run`: Preview code without saving

**Example**:
```bash
neural compile model.neural --backend pytorch --output model.py
```

**Supported Files**:
- `.neural`, `.nr`: Network definitions
- `.rnr`: Research configurations

---

### `run` - Execute compiled models
```bash
neural run [OPTIONS] FILE
```

**Options**:
- `--backend, -b`: Runtime backend (`tensorflow`, `pytorch`) [default: tensorflow]

**Example**:
```bash
neural run model_pytorch.py --backend pytorch
```

---

### `visualize` - Generate architecture diagrams
```bash
neural visualize [OPTIONS] FILE
```

**Options**:
- `--format, -f`: Output format (`html`, `png`, `svg`) [default: html]
- `--cache/--no-cache`: Use cached visualizations [default: True]

**Example**:
```bash
neural visualize model.neural --format png --no-cache
```

**Outputs**:
- HTML: Interactive visualizations
- PNG/SVG: Static architecture diagrams

---

### `clean` - Remove generated files
```bash
neural clean
```

**Removes**:
- Generated `.py`, `.png`, `.svg`, `.html` files
- `.neural_cache` directory

---

### `version` - Show version info
```bash
neural version
```

**Displays**:
- CLI version
- Python version
- Dependency versions

---

### `debug` - Debugging tools
```bash
neural debug [OPTIONS] FILE
```

**Options**:
- `--gradients`: Analyze gradient flow
- `--dead-neurons`: Detect inactive neurons
- `--anomalies`: Find training anomalies
- `--step`: Interactive step debugging
- `--backend, -b`: Runtime backend (`tensorflow`, `pytorch`) [default: tensorflow]

**Example**:
```bash
neural debug model.neural --gradients --step --backend pytorch
```

---

### `no-code` - Launch visual builder
```bash
neural no-code [OPTIONS]
```

**Options**:
- `--port`: Web interface port [default: 8051]

**Example**:
```bash
neural no-code --port 8051
```

---

## Error Handling
- **Validation Errors**: Shown with line/column numbers
- **Exit Codes**: Non-zero for failures
- **Logging**: Timestamped logs with severity levels

---

## Caching System
- Visualizations cached in `.neural_cache`
- Cache keyed by file hash
- Disable with `--no-cache`
