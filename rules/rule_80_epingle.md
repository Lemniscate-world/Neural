# RULE 80: Epingle Projets Auto-Update — Full Detail

## Purpose

Epingle_Projets.md is the single source of truth for lambda-Section portfolio status.
It must always reflect the REAL current state of all projects.
The user handles Google Drive sync manually.

## Trigger

After ANY session that changes:
- A project progress % (even -1% or +1%)
- A project status (Prototypage → Actif, Actif → En Pause, etc.)
- A new project added to any section
- A project pivoted, archived, or killed

## File location (master)

`~/Documents/kuro-rules/Epingle_Projets.md`

## Formatting Rules — MANDATORY

### Structure
- Each λ-Section uses a **table format** with columns : `| Projet | Progression | Statut | Description |`
- Section header format : `## λ-Section-N — Name`
- Section subtitle in blockquote : `> Thématique`
- Sections separated by `---`

### Naming
- Project names in Epingle_Projets.md MUST match `projects.txt` exactly (case-sensitive)
- If a mismatch is detected, fix Epingle_Projets.md to match projects.txt
- NEVER create a second entry for the same project in a different section

### Encoding
- File MUST be UTF-8 with proper French accents (é, è, ê, ç, à, ô, etc.)
- NO unaccented French text (e.g., "reseaux" → "réseaux", "dedie" → "dédié")

### Anti-Duplicate Rule
- Each project name MUST appear in exactly ONE section table
- If a project spans multiple sections, list it in its PRIMARY section and add a `> Note` in the secondary section referencing the primary
- BEFORE adding any project, search the entire file to verify it doesn't already exist

## Status Values

| Value | Meaning |
|-------|---------|
| Prototypage | Concept/early spike, no validated problem yet |
| Recherche | Mom Test / desk research phase active |
| Validation | Problem validated, solution being tested |
| Actif | In active development |
| En Pause | Temporarily paused |
| Archive | Archived, kept as proof of concept |
| Pivot | Pivoted, original direction abandoned |
| Outil | Internal tool, no progress % tracked |
| Externe | Third-party, no ownership |

## What to Update

1. Progress % — use R3 pessimistic formula, never inflate
2. Status tag if it changed
3. Description if the project pivoted or repositioned
4. Date of last update in the header blockquote

## What NOT to Change

- Section structure (λ-Section-1 through λ-Section-15 + Externes + Réflexion)
- Links in the Liens section
- Other projects not touched this session

## Pre-Update Checklist — MANDATORY

```
BEFORE updating Epingle_Projets.md:
1. SEARCH: Does the project already exist in the file? (anti-duplicate)
2. MATCH: Does the project name match projects.txt? (naming consistency)
3. VERIFY: Are all accents correct in the new text? (encoding)
4. CHECK: Is R90 table still accurate? (monthly deliverables)
5. FORMAT: Is the entry using table format, not list format? (structure)
```

## Enforcement

IF session changes project % but Epingle_Projets.md not updated:
  ACTION: Update before closing session
  DO NOT: Leave the portfolio doc stale
  DO NOT: Round up progress % — use R3 pessimistic formula
  DO NOT: Use list format (`- Project X%`) — use table format
  DO NOT: Create duplicate entries across sections
  DO NOT: Write unaccented French

## Example

BEFORE:
```
| NeuroDose | 0% | Prototypage | — |
```
AFTER:
```
| **NeuroDose** | 15% | Actif | Cognitive Supplement Tracker & Visualizer. Optimisation santé cognitive via modélisation pharmacocinétique. |
```

## References

- R3: Pessimistic progress formula
- R85: Portfolio completeness verification
- R86: Kuro Guardian surveillance
- R90: Monthly deliverables recall

---

**Créé** : 2026-05-05
**Mis à jour** : 2026-05-12 (Hardened: anti-duplicate, encoding, naming, table format)
**Enforcement** : MANDATORY