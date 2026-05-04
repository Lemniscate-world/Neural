# Guide d'Installation Qodo - Code Reviews AI

Ce document explique comment installer et configurer Qodo pour permettre a votre equipier DevOps/MLOps d'acceder aux code reviews AI.

## Option 1: Installation Locale (CLI)

### Prerequisites
- Python 3.10+ 
- Git

### Installation

```bash
# Via pip
pip install qodo

# Via npm
npm install -g qodo
```

### Configuration

Creer un fichier `.qodo/qodo.toml` a la racine du projet:

```toml
[general]
language = "fr"

[review]
max_comments = 50
severity_threshold = "low"

[review.paths]
include = ["src/", "tests/", "*.py"]
exclude = ["venv/", ".venv/", "node_modules/"]
```

## Option 2: VS Code Extension

1. Ouvrir VS Code
2. Aller dans Extensions (Ctrl+Shift+X)
3. Rechercher "Qodo AI Code Review"
4. Cliquer sur Installer

## Option 3: GitHub Actions (Automatique)

Creer `.github/workflows/qodo-review.yml`:

```yaml
name: Qodo Code Review
on:
  pull_request:
    types: [opened, synchronize]
  push:
    branches: [main, develop]

jobs:
  qodo-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Qodo Review
        uses: Codium-ai/pr-agent-github-action@main
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
```

## Option 4: MCP Server (Alternative)

Si Qodo n'est pas disponible, utiliser un MCP server alternatif:

### Installation du MCP Server

```bash
# Creer le repertoire MCP
mkdir -p ~/.mcp/servers

# Clone ou telecharge le serveur MCP desire
git clone https://github.com/your-org/mcp-code-review-server.git ~/.mcp/servers/code-review
```

### Configuration VS Code

Dans `settings.json`:

```json
{
  "mcpServers": {
    "code-review": {
      "command": "node",
      "args": ["/path/to/mcp-code-review-server/index.js"]
    }
  }
}
```

## Verification

Verifier que Qodo est installe:

```bash
qodo --version
```

## Utilisation

```bash
# Review d'un fichier
qodo review path/to/file.py

# Review de tout le projet
qodo review --all

# Review d'un diff
git diff | qodo review --diff
```

## Support

Pour toute question, consulter la documentation officielle ou contacter l'equipe lead.
