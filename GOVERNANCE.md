# Governance

## Overview

NeuralDBG is maintained by **LambdaSection** as an open-source MIT project. We follow a lightweight maintainer model inspired by PyTorch Ecosystem requirements.

## Maintainers

| Name | GitHub | Role | Since |
|------|--------|------|-------|
| Jacques-Charles SENOUVO (Kuro) | `@Lemniscate-world` | Lead Maintainer, BDFL | 2025 |
| P3niel | `@P3niel` | Maintainer — docs, integrations, community | 2025 |

Core maintainers have merge rights and are listed in `.github/CODEOWNERS`. A second maintainer satisfies the Ecosystem WG requirement of ≥2 core maintainers.

Historical contributors: see `https://github.com/LambdaSection/NeuralDBG/graphs/contributors` and `CHANGELOG.md`.

## Decision Making

- **Trivial** (docs, typos, CI): single maintainer approval.
- **Standard** (features, bug fixes): PR + 1 maintainer review, CI green.
- **Major** (API breaking, license, governance): issue/RFC + approval from all core maintainers.

Lazy consensus: if no objection within 72h after review, the PR may be merged.

## Roles

- **Maintainer**: review/merge PRs, cut releases, triage issues, enforce Code of Conduct.
- **Contributor**: anyone submitting PRs/issues. Recognized after 5 merged PRs or significant feature.
- **Emeritus**: former maintainers retaining advisory role.

Becoming a maintainer: sustained contributions (≥3 months, ≥10 merged PRs or equivalent) + nomination by existing maintainer + unanimous approval.

## Release Methodology

- **Versioning**: SemVer (`MAJOR.MINOR.PATCH`), documented in `CHANGELOG.md`.
- **Cadence**: Monthly minor releases; patches as needed for security/bug fixes.
- **Process**: tag `vX.Y.Z` → `publish.yml` builds and publishes to PyPI → GitHub Release notes.
- **Support**: latest minor (`1.5.x`) actively maintained; see `SECURITY.md` for supported versions.
- **Changelog**: Keep a Changelog format.

## Communication

- Issues/PRs: GitHub
- Security: `SECURITY.md`
- Email: `lemniscate_zero@proton.me` (primary), `neuraldbg@lemniscate.ai`
- Code of Conduct: `CODE_OF_CONDUCT.md`
- Contributing: `CONTRIBUTING.md`

## Ecosystem

Related repos:

- `LambdaSection/NeuralDBG` (this repo) — MIT, public
- `LambdaSection/Aquarium` — causal chain visualizer. Currently private during active UX iteration; public viewer is `docs/aquarium.html` in this repo (zero-dependency). Will be made public when stable; see `docs/ecosystem.md`.

## Amending Governance

Changes to this document require PR + approval from all core maintainers.
