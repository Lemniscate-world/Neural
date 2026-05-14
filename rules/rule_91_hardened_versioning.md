# RULE 91: Hardened Versioning Integrity

## Rule

Versioning MUST be treated as a critical security and traceability asset. Every release version MUST be verified against three sources of truth before being finalized.

## Rationale

Manual versioning is prone to error and "fragility". By enforcing a multi-point verification, we ensure that the version reported in the code matches the version tagged in Git and documented in the Changelog.

## Verification Checklist

Before finalizing a version (e.g., in a `make release` command):
1. **Source Sync**: Check that the version string in `pyproject.toml` (or equivalent) matches the requested version.
2. **Changelog Entry**: Verify that `CHANGELOG.md` has an entry for the new version with the current date.
3. **Tag Verification**: Check that no existing Git tag matches the new version (avoiding overwrites).
4. **Environment Cleanliness**: Ensure `git status` is clean before tagging.

## Enforcement

```
IF versioning mismatch detected:
  ACTION: STOP release process
  ACTION: Report mismatch to user
  ACTION: Fix source of truth (pyproject.toml) before retrying

IF release command is run:
  ACTION: Automatically check CHANGELOG.md for corresponding entry
  ACTION: If missing, prompt user to update Changelog first
```

---

**Created**: 2026-05-11
**Trigger**: User reported "fragile" versioning system
**Enforcement**: MANDATORY
