# Linear dans Cursor (MCP) — Methode A a Z

Ce guide decrit la methode utilisee dans ce repo pour connecter Linear proprement.
Objectif: connexion stable pour tous les collaborateurs, sans cle en dur dans git.

## Resume de la methode utilisee

- Integration Linear activee dans Cursor UI (plugin/integration)
- MCP configure en HTTP dans le fichier utilisateur `~/.cursor/mcp.json`
- Cle API chargee via `LINEAR_API_KEY` (variable d'environnement persistante)
- Cursor lance depuis un shell qui contient cette variable

## Prerequis

- Cursor recent (support MCP)
- Compte Linear + cle API personnelle
- Shell `bash`
- Node non requis avec la methode HTTP (recommandee)

## Etape A — Creer la cle API Linear

1. Ouvre Linear.
2. Va dans `Settings -> API -> Personal API keys`.
3. Cree une cle et copie-la.

Important: considere cette cle comme un mot de passe.

## Etape B — Verifier/activer Linear dans l'UI Cursor

Sur certaines installations, il faut d'abord activer/installer l'integration Linear dans Cursor UI.

1. Ouvre `Cursor -> Settings`.
2. Cherche `Linear` dans les integrations/plugins/outils.
3. Active l'integration Linear si elle est desactivee.
4. Redemarre Cursor.

Indice local dans ce repo: `.cursor/settings.json` contient `"plugins.linear.enabled": true`.

## Etape C — Rendre la cle disponible pour tous les agents (persistant)

Creer un fichier secret local (hors repo) et le charger automatiquement:

```bash
mkdir -p ~/.config/secrets
cat > ~/.config/secrets/linear.env <<'EOF'
export LINEAR_API_KEY='your_linear_api_key_here'  # pragma: allowlist secret
EOF
chmod 600 ~/.config/secrets/linear.env
```

Ajouter le chargement auto dans `bash`:

```bash
grep -q "linear.env" ~/.bashrc || echo '[ -f ~/.config/secrets/linear.env ] && source ~/.config/secrets/linear.env' >> ~/.bashrc
grep -q "linear.env" ~/.profile || echo '[ -f ~/.config/secrets/linear.env ] && source ~/.config/secrets/linear.env' >> ~/.profile
source ~/.bashrc
```

Verification:

```bash
echo ${#LINEAR_API_KEY}
```

Le resultat doit etre `> 1`.

## Etape D — Configurer MCP HTTP au niveau utilisateur

Fichier a utiliser (recommande): `~/.cursor/mcp.json` (configuration utilisateur partagee entre projets).

Note importante:
- Dans certains cas, Cursor peut aussi utiliser une config par projet (`.cursor/mcp.json`).
- En cas de doute, utilise le chemin de configuration affiche dans `Settings -> Tools & MCP`.

Configuration:

```json
{
  "mcpServers": {
    "linear": {
      "url": "https://mcp.linear.app/mcp",
      "headers": {
        "Authorization": "Bearer ${env:LINEAR_API_KEY}"
      }
    }
  }
}
```

Regles de securite:
- Ne jamais mettre la cle en dur dans `mcp.json`.
- Garder `.env` hors git. Le fichier `~/.cursor/mcp.json` est hors repo par nature.

## Etape E — Lancer Cursor correctement

Toujours lancer Cursor depuis un shell ou `LINEAR_API_KEY` est chargee:

```bash
cd /chemin/vers/NeuralDBG
cursor .
```

Puis faire un redemarrage complet de Cursor.

## Etape F — Valider la connexion

Verification minimale:
- Les outils Linear apparaissent dans l'assistant Cursor.
- Une action de lecture fonctionne (ex: lister teams / afficher `me`).

Verification API (optionnelle):

```bash
curl -sS https://api.linear.app/graphql \
  -H "Authorization: $LINEAR_API_KEY" \
  -H "Content-Type: application/json" \
  --data '{"query":"{ viewer { id name email } }"}'
```

Si la connexion est bonne, la reponse contient `data.viewer`.

## Depannage

1. `echo ${#LINEAR_API_KEY}` renvoie `0`
- La variable n'est pas chargee dans ce shell.
- Refaire `source ~/.bashrc` ou ouvrir un nouveau terminal.

2. Outils Linear absents dans Cursor
- Verifier l'etape B (integration Linear activee).
- Verifier `~/.cursor/mcp.json` (ou le fichier MCP indique par Cursor dans l'UI).
- Fermer puis relancer completement Cursor.

3. Erreur `Unauthorized` / `403`
- Cle invalide ou mauvais workspace.
- Regenerer la cle dans Linear et recharger le shell.

4. Methode fallback (si HTTP indisponible dans ta version Cursor)
- Utiliser un serveur MCP local `npx @modelcontextprotocol/server-linear`.
- Cette methode requiert Node.js 18+.

## Securite

- Ne jamais committer une vraie cle API.
- Ne jamais partager la cle en capture ecran, PR, ticket, log.
- Si une cle fuit, la revoquer immediatement et en creer une nouvelle.
