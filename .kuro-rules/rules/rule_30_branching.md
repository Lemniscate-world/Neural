# RULE 30: Branch Naming Convention

## Convention
[scope]/[issue-id]-[short-description]

## Scopes
| Scope | Usage |
|-------|-------|
| ceo/ | Strategic dev & rule management (CEO only) |
| devops/ | CI/CD, Docker, infra |
| mlops/ | Experiment tracking, data versioning |
| feat/ | New features (dev) |
| fix/ | Bug fixes |
| docs/ | Documentation |
| test/ | Test coverage |
| sec/ | Security fixes |

## Rules
- Nobody works on main directly
- Create branch BEFORE any work begins
- One branch per contributor per task
- Branch maps to a Linear issue

## Examples
ceo/KUR-42-semantic-event-structures
devops/KUR-15-docker-compose-setup
feat/KUR-31-mom-test-automation