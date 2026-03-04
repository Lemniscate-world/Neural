# Rules for AI (AntiGravity / Cursor / Copilot)

> Copie des règles du projet — à appliquer par tout assistant IA (AntiGravity, Cursor, etc.)

---

## 0. Sync with kuro-rules
- When updating rules, sync to `~/Documents/kuro-rules`.

## 1. Explain as if First Time — Always
- Assume **zero prior knowledge**. Re-explain AI, ML, concepts, math as if the user knows nothing.
- The user codes while learning for the first time. Define terms, use simple analogies, break down formulas.

---

## 2. Always Update README & Changelog
- Every feature or fix must update `README.md` (usage, examples, API if changed).
- Update `CHANGELOG.md` (create if missing) with conventional commit-style entries.

## 3. Zero Friction for Users
- Tools must work out of the box. Minimal config, clear defaults.
- Provide copy-paste examples that run without extra setup.
- No hidden requirements; document any prerequisite explicitly.

## 4. Solve Real Pain Points
- Before building: *"Does this fix a real user pain?"*
- No speculative features; validate need first.
- Prefer solving existing problems over adding new capabilities.

## 5. Security & Quality Tooling
- CI must include **CodeQL**, **SonarQube**, and **Codacy** (or equivalent).
- No shortcuts on static analysis. Fail builds on critical issues.

## 6. One Project at a Time
- Work on **one project at a time**. No parallel development.
- Roadmap : `PROJECTS.md` (racine)


## CI/CD Debugging First (MANDATORY)
- If GitHub Actions is red on the active branch/PR, stop unrelated work.
- Inspect failing checks/logs first (`gh run list`, `gh run view --log`, `gh pr checks`).
- Apply minimal safe fix, re-run relevant validations, then continue feature work.
