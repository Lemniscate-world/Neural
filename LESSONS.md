# LESSONS.md -- NeuralDBG

This file captures technical lessons learned and recurring problems solved during development, as per **Rule 89**.

## [2026-05-11] -- Infrastructure & Hardening Session

### Problem: Linux-Centric Infrastructure on Windows
- **Issue**: The `Makefile` and scripts in `infrastructure/scripts/` used hardcoded Linux paths (`/tmp/`, `.venv/bin/`) and tools (`sed -i`) that failed on the user's native Windows environment.
- **Solution**: Refactored `Makefile` and `ensure_venv.sh` to use platform-agnostic commands and Python-based logic.
- **Rule Created**: **Rule 93: Cross-Platform Reliability**.

### Problem: Fragile Versioning System
- **Issue**: Versioning was handled manually and inconsistently, leading to mismatches between `pyproject.toml`, Git tags, and `CHANGELOG.md`.
- **Solution**: Implemented a multi-point verification protocol and a Python-based release script.
- **Rule Created**: **Rule 91: Hardened Versioning Integrity**.

### Problem: Accidental File Deletion during Refactoring
- **Issue**: Important examples (like `demo_vanishing_gradients.py`) were accidentally deleted or moved during repo restructuring.
- **Solution**: Implemented **Rule 88: Integrity Recovery** and mapped the repository structure in `STRUCTURE.md`.

---
**Status**: ACTIVE tracking.
