# Linear dans Cursor (MCP) — Guide d'installation

Ce document explique comment connecter Cursor a Linear pour que l'assistant puisse lire/creer des issues, lister les teams, etc.

Objectif: avoir une integration Linear fonctionnelle, sans jamais committer de secret.

## Concepts (simple)

- **MCP (Model Context Protocol)**: un "connecteur" qui permet a Cursor d'appeler des outils externes (Linear, GitHub, etc.).
- **Serveur MCP**: le programme (ou service HTTP) que Cursor utilise pour parler a Linear.
- **Cle API Linear**: un secret qui autorise l'acces a ton workspace Linear. Elle doit rester privee.

## Prerequis

- Cursor (version recente avec MCP)
- Un compte Linear + une cle API
- Selon la methode:
  - **Methode Command (npx)**: Node.js 18+ et `npx`
  - **Methode HTTP**: aucun Node requis (si Cursor supporte le transport HTTP "streamable")

## Etape 1 — Creer une cle API Linear

1. Ouvre Linear.
2. Va dans **Settings** > **API** > **Personal API keys**.
3. Cree une cle, puis copie-la.

Important: traite cette cle comme un mot de passe.

## Etape 2 — Choisir une methode de connexion dans Cursor

Cursor peut se connecter a Linear de deux manieres principales. Choisis la plus simple pour toi.

### Methode A (recommandee) — Configuration via l'UI de Cursor

1. Ouvre Cursor > **Settings**.
2. Va dans **Tools & MCP**.
3. Ajoute/active un serveur MCP Linear.
4. Renseigne l'authentification (cle API) dans l'UI si Cursor propose un champ pour ca.
5. Redemarre completement Cursor (quitter puis relancer).

Avantages:
- Simple, pas besoin de gerer un fichier JSON a la main.
- Souvent stocke dans un endroit plus "safe" que le depot.

### Methode B — Fichier `.cursor/mcp.json` (serveur "command" avec npx)

Cette methode lance un serveur MCP local via `npx`.

Le projet contient deja un exemple de config dans:

- `.cursor/mcp.json` (mais `.cursor/` est dans `.gitignore`, donc **non versionne**)

Exemple minimal:

```json
{
  "mcpServers": {
    "linear": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-linear"],
      "env": {
        "LINEAR_API_KEY": ""
      }
    }
  }
}
```

Ensuite, tu fournis la cle via l'environnement (voir Etape 3).

### Methode C — Serveur MCP HTTP (si tu utilises le serveur officiel cloud)

Certains environnements/registries utilisent un serveur MCP Linear expose en HTTP.
L'exemple le plus courant ressemble a:

- URL: `https://mcp.linear.app/mcp`
- Transport: `http` / `streamableHttp` (selon l'UI Cursor)

Dans ce cas, l'auth se fait souvent via un header (ou un champ "API key" dans l'UI).

## Etape 3 — Mettre la cle API sans la committer

Regle: **ne jamais mettre la cle dans un fichier versionne**.

### Option 1 (simple) — Variable d'environnement dans le shell

Dans ton terminal:

```bash
export LINEAR_API_KEY="lin_api_xxx"
```

Puis lance Cursor depuis ce meme terminal:

```bash
cursor .
```

### Option 2 — Utiliser un `.env` local (non versionne)

1. Cree un fichier `.env` (il est ignore par git).
2. Ajoute:

```
LINEAR_API_KEY=lin_api_xxx
```

3. Charge le `.env` puis lance Cursor:

```bash
cd /chemin/vers/NeuralDBG
set -a && source .env && set +a
cursor .
```

Note: Cursor ne charge pas automatiquement le `.env` du projet. Il faut le `source` avant de lancer Cursor (ou utiliser l'UI).

## Etape 4 — Verifier que la connexion marche

Tu peux tester avec une action "lecture" (safe):

- Lister les teams
- Afficher l'utilisateur courant ("me")

Dans Cursor, si tout est correct, tu verras des outils Linear disponibles dans l'assistant.

## Depannage (si ca ne marche pas)

### 1) Node / npx introuvable (methode B)

Verifie:

```bash
node --version
npx --version
```

Si c'est vide ou "command not found", installe Node.js 18+ puis redemarre Cursor.

### 2) Cle API invalide / permissions

- Regenerer une nouvelle cle dans Linear si tu as un doute.
- Verifier que tu utilises la bonne cle pour le bon workspace.

### 3) Cursor ne voit pas la variable

- Si tu utilises `.env`, assure-toi de lancer Cursor depuis le terminal apres `source .env`.
- Sinon, configure l'env directement dans l'UI MCP de Cursor.

## Securite (a lire)

- Ne colle jamais ta cle dans un fichier commite (README, docs, scripts).
- Si une cle a ete exposee (meme "juste une seconde"), **revoque-la** et cree-en une nouvelle.

