# RULE 87: Deep Intelligence & Ownership Verification — MANDATORY

## Rule

AI agents MUST verify the ownership and operational status of every repository before performing synchronization or reporting progress.

## Verification Protocol

### 1. Ownership Verification (MANDATORY)
Before syncing rules or adding to "Owned" sections in `Epingle_Projets.md`:
```bash
git remote -v
```
**Ownership Signals**:
- **OWNED**: Remotes containing `github.com/Lemniscate-world/`, `github.com/Lemniscate-SHA-256/`, or `github.com/pbakaus/`.
- **EXTERNAL**: Remotes containing `github.com/Demeter-Financial-Labs/` or other external orgs.
- **CLARIFY**: If no remote or unknown org, ask the user.

**Action**: 
- NEVER run `sync-rules.ps1` or similar broad sync tools on EXTERNAL repositories.
- EXTERNAL repositories must be listed in a separate section in `Epingle_Projets.md`.

### 2. Deep Status & Description Inspection (MANDATORY)
AI MUST read the first 100 lines of `README.md` for EVERY project entry. 
**Anti-Hallucination Protocol**:
- DO NOT assume a project's purpose from its name (e.g., `NeuroDose` is NOT for "network deployment", it is for "biohacking").
- DO NOT declare a project ARCHIVED unless explicitly stated in the README or confirm with user.
- VERIFY keywords: `ARCHIVED`, `OBSOLETE`, `NO LONGER MAINTAINED`, `Successor:`.
- Detect if the repo is a **Vault** (contains `.obsidian/`), a **Library**, or a **Template**.

### 3. Error Prevention & Hardening
- **Projects.txt Validation**: Ensure `projects.txt` ONLY contains OWNED projects.
- **Double-Check Audit**: When a project is found in `~/Documents` but missing in `Epingle_Projets.md`, perform Ownership Verification BEFORE adding it.

## Enforcement

```
IF project is EXTERNAL:
  ACTION: Do not sync governance rules (AGENTS.md, etc.)
  ACTION: List in "Projets Tiers / Externes" in Epingle_Projets.md

IF project is ARCHIVED:
  ACTION: Mark as (Archive) in Epingle_Projets.md
  ACTION: Stop active progress tracking (set to fixed percentage)

IF project is VAULT:
  ACTION: Mark as (Obsidian Notes)
  ACTION: Exclude from coding rule sync
```

## Violation Examples

**VIOLATION**: Syncing `kuro-rules` to `DevDemeterDAO` (External).
**CORRECT**: Isolate `DevDemeterDAO` in the External section, skip rule sync.

**VIOLATION**: Reporting `Neural-Again` as "Actif" when README says "ARCHIVED".
**CORRECT**: Update status to (Archive) and mention the successor `TokenWise`.

## Rationale
Prevents legal/operational friction by respecting repo ownership boundaries and optimizes AI focus by identifying dead or static projects.

---
**Created**: 2026-05-05
**Applies to**: All portfolio audits, rule synchronizations, and status reports.
**Enforcement**: MANDATORY.
