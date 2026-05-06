# RULE 80: Epingle Projets Auto-Update - Full Detail

## Purpose
Epingle_Projets.md is the single source of truth for lambda-Section portfolio status.
It must always reflect the REAL current state of all projects.
The user handles Google Drive sync manually.

## Trigger
After ANY session that changes:
- A project progress % (even -1% or +1%)
- A project status (Prototypage -> Actif, Actif -> En Pause, etc.)
- A new project added to any section
- A project pivoted, archived, or killed

## File location (master)
~/Documents/kuro-rules/Epingle_Projets.md

## Per-project update format
[Project Name] [X%] [Status] - [One sentence description, updated if pivoted]

## Status values
| Value | Meaning |
|-------|---------|
| (Prototypage) | Concept/early spike, no validated problem yet |
| (Recherche) | Mom Test / desk research phase active |
| (Actif) | In active development |
| (En Pause) | Temporarily paused |
| (Archive) | Archived, kept as proof of concept |
| (Pivot) | Pivoted, original direction abandoned |

## What to update
1. Progress % next to project name - use R3 pessimistic formula, never inflate
2. Status tag if it changed
3. Description line if the project pivoted or repositioned
4. Date of last update at the top of the file

## What NOT to change
- Section structure (lambda-Section-1 through lambda-Section-15)
- Links to Google Docs per project
- Formatting (bold, links) unless explicitly requested
- Other projects not touched this session

## Enforcement
IF session changes project % but Epingle_Projets.md not updated:
  ACTION: Update before closing session
  DO NOT: Leave the portfolio doc stale
  DO NOT: Round up progress % - use R3 pessimistic formula

## Example
BEFORE: NeuroDose 0% (Prototypage)
AFTER:  NeuroDose 15% (Actif) - Advanced nootropic metrics tracker, Mom Test complete