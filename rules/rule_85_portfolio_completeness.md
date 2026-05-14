# RULE 85: Portfolio Completeness Verification — MANDATORY

## Rule

AI agents MUST verify that `Epingle_Projets.md` contains ALL lambda-Section projects before declaring a portfolio update complete.

## Verification Steps

### MANDATORY: Pre-Update Scan
```
ACTION: List all directories in ~/Documents with AGENTS.md or .git
VERIFY: Each directory is either:
  - Listed in Epingle_Projets.md with percentage and description
  - Listed in "Autres Projets En Reflexion" section
  - Explicitly excluded (non-project directories)

ACTION: Perform Ownership & Deep Status Check (R87)
VERIFY: Check git remotes and README markers. Classify as:
  - Owned lambda-Section project (Lemniscate-world)
  - External project (Demeter-Financial-Labs)
  - Archived/Obsolete project
  - Special (Obsidian Vault, Tool, Fork)
```

### Exclusions Allowed
- `.windsurf`, `.cursor`, `.vscode`, `.config`
- `kuro-rules` (master rules repository)
- `vcpkg`, `WindowsPowerShell`, `MATLAB`
- `Documents` system folders (Adobe, Ableton, etc.)
- Projects explicitly marked as "archived" or "abandoned"

### Required Information Per Project
Each project in Epingle_Projets.md MUST have:
1. **Name**: Directory name
2. **Percentage**: 0%, 3-15% (active), 30%+ (advanced), 50%+ (mature)
3. **Status**: Prototypage, Validation, Actif, Recherche
4. **Description**: One-line description of what the project does
5. **Correct Section**: Matched to appropriate lambda-Section

## Enforcement

### When Updating Epingle_Projets.md
AI MUST:
1. Scan ~/Documents for all project directories
2. Compare with current Epingle_Projets.md entries
3. Identify missing projects
4. Read README.md, decision.md, or PLAN.md to determine:
   - Project type and section
   - Current percentage
   - Accurate description
5. Verify Ownership (R87) before adding to any "Owned" section.
6. Add missing projects with complete information and correct ownership classification.
7. Report to user: "Added X projects, identified Y external repositories."

### When User Asks "Tout est à jour ?"
AI MUST:
1. Scan ~/Documents
2. List ALL projects found
3. Compare with Epingle_Projets.md
4. Report: Found / Missing / Needs verification
5. Propose additions for missing projects

### Compliance Checklist
- [ ] ~/Documents scanned
- [ ] All AGENTS.md directories identified
- [ ] Missing projects listed
- [ ] Each missing project: section determined, % estimated, description written
- [ ] Epingle_Projets.md updated
- [ ] Date updated: 2026-MM-DD
- [ ] Commit with message including added projects

## Violation Examples

**VIOLATION**: User asks "tout concorde ?" and AI answers "oui" without scanning.
**CORRECT**: Scan first, then answer with evidence: "37 projets dans ~/Documents, 35 listés, 2 manquants identifiés".

**VIOLATION**: Adding project to Epingle without description.
**CORRECT**: "Hermes 10% (Validation) - Outil de coordination des livraisons pour commercants du Grand Lome..."

**VIOLATION**: Odin/Sagittarius missing for months despite existing in ~/Documents.
**CORRECT**: Detected during portfolio update, added immediately with proper classification.

## Rationale

The portfolio (Epingle_Projets.md) is the SINGLE SOURCE OF TRUTH for lambda-Section's 35+ projects. Missing projects create:
- Forgotten work streams
- Resource allocation errors
- Validation pipeline gaps (R84)
- Investor communication failures (R83)

Systematic verification prevents "dark projects" that consume effort without visibility.

## References

- R80: Epingle_Projets.md maintenance rules
- R83: Discord investor summary (depends on complete portfolio)
- R84: Validation automation (depends on knowing all projects needing validation)
- Master copy: ~/Documents/kuro-rules/Epingle_Projets.md

---

**Created**: 2026-05-05
**Applies to**: All portfolio updates, project reviews, "status check" requests
**Enforcement**: MANDATORY — Violations MUST be reported to user
