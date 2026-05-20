# RULE 98: Pre-Launch MVP Verification Protocol

## Trigger
Avant tout lancement public (Show HN, Product Hunt, release blog, etc.).

## Objectif
Vérifier que le MVP fait CE QU'IL DIT et pas seulement qu'il s'installe. Un import réussi ne prouve pas que le produit marche.

## Niveaux de vérification

### Niveau 1 — Installation (obligatoire, J-7)
- [ ] `pip install {package}` dans un venf frais
- [ ] Import réussi : `from {package} import {main_class}`
- [ ] Version correspond au CHANGELOG
- [ ] Dépendances installées correctement (vérifier torch, psutil, etc.)

### Niveau 2 — Quickstart (obligatoire, J-3)
- [ ] Le quickstart du README peut être copié-collé et s'exécute sans erreur
- [ ] Le quickstart produit un résultat visible (pas juste "pas d'erreur")
- [ ] Si le produit doit détecter des défaillances : tester UN scénario qui échoue volontairement

### Niveau 3 — Tests fonctionnels (obligatoire, J-1)
Exécuter un script qui test LES PROMESSES DU README :
- [ ] **Cas normal** : le produit fait ce qui est annoncé sur un cas simple
- [ ] **Cas d'échec** : le produit détecte correctement un problème
- [ ] **Export** : les formats de sortie (JSON, graph, etc.) sont valides
- [ ] **Fallback** : les fonctionnalités sans engine propriétaire ne cassent pas

### Niveau 4 — CI & Qualité (obligatoire, J-1)
- [ ] GitHub Actions / CI sont verts
- [ ] Tests unitaires passent (pytest)
- [ ] Badges README (PyPI, license, Python version) sont visibles
- [ ] Aucune issue critique ouverte

### Niveau 5 — Final (jour J, 2h avant)
- [ ] `pip install {package}` dans un venf FRAIS (pas le même que la veille)
- [ ] Quickstart exécuté une dernière fois
- [ ] URL GitHub accessible
- [ ] README lisible sur mobile
- [ ] License bien affichée

## Script de test minimal (template)

Ce script doit être adapté à chaque projet et exécuté avant lancement :

```python
"""Pre-launch verification script for {PROJECT}."""
import sys
import subprocess

def test_install():
    """Niveau 1"""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "{package}"],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"Install failed: {result.stderr}"
    print("✅ pip install OK")

def test_import():
    """Niveau 1"""
    exec("from {package} import {main_class}")
    print("✅ Import OK")

def test_quickstart():
    """Niveau 2 - adapter au projet"""
    # Copier-coller le quickstart du README ici
    pass

def test_failure_scenario():
    """Niveau 3 - tester que le produit détecte une défaillance"""
    pass

if __name__ == "__main__":
    test_install()
    test_import()
    print("✅ Niveaux 1-2 passés")
    print("⚠️ Niveaux 3-5 à exécuter manuellement (nécessite entraînement PyTorch)")
```

## Fichier de sortie
Les résultats de vérification sont stockés dans `docs/verification_report_{VERSION}_{DATE}.md`.

## Sanction
Si un lancement est fait sans vérification complète (N1 à N5) :
- L'agent DOIT le signaler dans SESSION_SUMMARY.md comme violation
- Le launch plan doit être mis à jour avec les vérifications manquantes
- Le lancement suivant doit inclure une vérification complète

## Fichiers associés
- `docs/launch_plan_{PROJECT}.md` — le launch plan contient la check-list
- `docs/verification_report_*.md` — rapports de vérification