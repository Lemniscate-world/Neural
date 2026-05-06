# Repository Structure

NeuralDBG organizes code and documentation following a clear, scalable layout.

## Directory Layout

```
neuraldbg/
├── neuraldbg/                  # Main package source code
│   ├── enhanced_causality.py
│   ├── schema/
│   └── ...
├── tests/                      # Unit tests
│   └── ...
│
├── docs/
│   ├── architecture/           # Design, system architecture
│   │   ├── INFERENCE_FLOW.md
│   │   ├── logic_graph.md
│   │   └── GAD.md
│   ├── guides/                 # User guides & tutorials
│   │   └── AI_GUIDELINES.md
│   ├── api/                    # API documentation (generated)
│   ├── STRUCTURE.md            # This file
│   └── research/               # Research papers, experiments
│
├── examples/                   # Example scripts & notebooks
│
├── infrastructure/
│   ├── docker/                 # Docker configuration
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   ├── scripts/                # Setup & deployment scripts
│   │   ├── bootstrap.sh
│   │   ├── install_hooks.sh
│   │   └── validate_schema.py
│   └── ci-cd/                  # CI/CD workflows (GitHub Actions)
│
├── config/                     # Project configuration files
│   ├── .pre-commit-config.yaml
│   ├── .cursorrules
│   └── copilot-instructions.md
│
├── .github/                    # GitHub-specific (workflows, templates)
│
├── [root essentials]
│   ├── README.md
│   ├── LICENSE.md
│   ├── SECURITY.md             # Security policy (MUST be at root)
│   ├── CHANGELOG.md
│   ├── ROADMAP.md
│   ├── pyproject.toml          # Package metadata
│   ├── Makefile
│   ├── requirements.txt
│   └── ...
```

## File Organization Rules

### Root Level (Minimal)
Only place files here that are critical for project visibility:
- `README.md` - Main project description
- `LICENSE.md` - License
- `SECURITY.md` - Security policy (required by kuro rules)
- `CHANGELOG.md` - Version history
- `ROADMAP.md` - Project roadmap
- `pyproject.toml` - Package configuration
- `.gitignore` - Git ignore rules
- Core config files: `Makefile`, `docker-compose.yml` (if critical)

### Documentation (`docs/`)
- **architecture/**: System design, inference flows, causality graphs
- **guides/**: User guides, tutorials, best practices
- **api/**: Generated API documentation
- **research/**: Research notes, experiments, evidence matrices

### Source Code (`neuraldbg/`)
- Main package implementation
- Keep it clean: only production code

### Tests (`tests/`)
- Unit tests, integration tests
- Follow same structure as `neuraldbg/`

### Infrastructure (`infrastructure/`)
- **docker/**: All Docker-related files
- **scripts/**: Deployment, setup, validation scripts
- **ci-cd/**: GitHub Actions workflows, CI configuration

### Configuration (`config/`)
- Pre-commit hooks
- IDE settings (cursor rules, copilot instructions)
- Tool configurations

### Examples (`examples/`)
- Runnable example scripts
- Demo notebooks
- Usage patterns

## Migration Notes

Migrated from flat root structure to organized hierarchy on 2026-05-06:
- Architecture docs → `docs/architecture/`
- Guides → `docs/guides/`
- Docker files → `infrastructure/docker/`
- Scripts → `infrastructure/scripts/`
- Tool configs → `config/`
- Removed duplicate venvs and build artifacts

## Maintenance

When adding new files:
1. Identify the file category (doc, example, script, etc.)
2. Place in appropriate subdirectory
3. Update this structure document if creating new top-level directories
4. Keep root level clean (fewer than 10 non-config files)
