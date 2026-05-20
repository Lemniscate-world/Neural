# RULE 100: Session Compliance — Vérification Obligatoire en Début de Session

## Trigger
Au TOUT DÉBUT de chaque session de travail, avant toute autre action.

## Obligation
L'agent DOIT :
1. Lire `rules/COMPLIANCE_CHECKLIST.md`
2. Vérifier TOUTES les règles 🔴🔴 MANDATORY
3. Signaler dans le premier message de la session l'état de conformité
4. Si une règle mandatory est en 🔴 VIOLATED, NE PAS commencer à travailler avant correction

## Format du message de début de session
```
📋 Compliance Check — {DATE}
🔴🔴 Mandatory: X/16 ✅ | ⏳ Y pending | 🔴 Z violated
🔵 Optional: X/XX ✅
⚠️ Violations: (liste si existe)
```

## Sanction
Si un agent commence à travailler sans avoir vérifié la compliance checklist :
- La session est marquée comme VIOLATED dans SESSION_SUMMARY.md
- L'agent DOIT documenter pourquoi la vérification a été sautée

## Fichiers associés
- `rules/COMPLIANCE_CHECKLIST.md` — la checklist à vérifier
- `SESSION_SUMMARY.md` — où logger les violations