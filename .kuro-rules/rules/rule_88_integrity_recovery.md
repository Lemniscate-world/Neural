# RULE 88: File Integrity & Recovery — MANDATORY

## Rule

AI agents MUST implement and maintain measures to prevent, detect, and recover from file corruption (mojibake, accidental overwrites, or data loss) across the master governance framework and all linked repositories.

## Integrity Measures

### 1. Pre-Commit Integrity Check
Before any commit to `kuro-rules`:
- **UTF-8 Validation**: Ensure no non-UTF-8 characters or mojibake were introduced (R41).
- **Structure Check**: Verify that `AGENTS.md` and `Epingle_Projets.md` maintain their required markdown structures.
- **Link Check**: Verify that all rule files referenced in `AGENTS.md` exist in `rules/`.

### 2. Redundancy & Backups
- **Local Mirror**: The `kuro-rules` repository is the source of truth, but agents should encourage the user to maintain a cloud-synced backup (e.g., Google Drive/Dropbox) of the `~/Documents/kuro-rules` folder.
- **Session Backups**: Every session end MUST result in a `SESSION_SUMMARY.md` update, which serves as a chronological recovery log of all changes.

### 3. Corruption Recovery
If corruption is detected (mojibake, corrupted metadata):
- **Immediate STOP**: Do not perform further edits to the corrupted file.
- **Git Revert**: Use `git checkout [file]` or `git restore [file]` to return to the last known good state.
- **Diff Analysis**: Compare the corrupted version with the previous version to identify the cause of corruption.

### 4. Synchronization Safety
- **Sync Isolation (R87)**: Only sync to repositories listed in `projects.txt` and verified as OWNED.
- **Atomic Sync**: The `sync-rules.ps1` script should ideally perform a verify-then-write operation to ensure rules are not corrupted during transit.

## Enforcement

```
IF mojibake or encoding error detected:
  ACTION: Trigger R41 enforcement.
  ACTION: Revert to last commit immediately.

IF AGENTS.md structure is broken:
  ACTION: Repair header and table formatting before next sync.

IF a sync fails on a specific repo:
  ACTION: Do not force. Report the failure and check for file lock or corruption.
```

## Violation Examples

**VIOLATION**: Committing `Epingle_Projets.md` with corrupted non-UTF8 characters.
**CORRECT**: Detect corruption during edit, revert, and re-apply changes with proper encoding.

**VIOLATION**: Overwriting `AGENTS.md` with a truncated version due to an AI failure.
**CORRECT**: Verify file length and key markers (# Last updated) before finalizing the write.

## Rationale
The governance framework is the brain of the entire portfolio. Any corruption in the brain leads to catastrophic execution errors across all projects.

---
**Created**: 2026-05-05
**Applies to**: Master rules, all synchronizations, and metadata management.
**Enforcement**: MANDATORY.
