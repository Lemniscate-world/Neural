# RULE 86: Kuro — Project Surveillance & Memory

## Rule

AI agents MUST assist in maintaining the Kuro system for continuous project surveillance and memory management across all lambda-Section projects.

## Verification

### When Starting Any Session
```
CHECK: Kuro daemon status
IF not running: NOTIFY user to start daemon
IF running: READ latest activity log from Kuro DB
```

### When Ending Any Session
```
ACTION: Ensure SESSION_SUMMARY.md is updated
ACTION: Kuro will auto-parse within 1 hour
VERIFY: Session appears in Kuro activity stream
```

## Kuro Integration Points

### 1. Session Awareness
- Kuro tracks ALL SESSION_SUMMARY.md updates
- Kuro detects editor used (Windsurf, Cursor, VS Code, etc.)
- Kuro extracts progress %, tests status, blockers
- Kuro alerts if same blocker appears 3+ sessions

### 2. Project Health Monitoring
```
ALERT if:
  - No SESSION_SUMMARY update for 7 days (Active project)
  - No SESSION_SUMMARY update for 14 days (Validation project)
  - Progress % stuck for 3+ sessions
  - "Blocker: Need new idea" for 14+ days (PIVOT status)
```

### 3. Epingle_Projets.md Sync
```
VERIFY: Epingle_Projets.md date vs Kuro last scan
ACTION: If Epingle older than 7 days, suggest update
ACTION: Run deep analysis (read SESSION_SUMMARY files) before update
```

### 4. Validation Pipeline Monitoring (R84)
```
CHECK: Projects in "Validation" status
ALERT if:
  - Landing page not deployed within 7 days of validation start
  - < 5 responses collected after 14 days
  - No Supabase dashboard accessed for 7 days
  - Kuro will track these automatically
```

## AI Agent Responsibilities

### During Portfolio Updates (R85 + R86)
1. **Scan Phase**: Use Kuro activity log as HINT only.
2. **Verify Phase**: READ actual SESSION_SUMMARY.md files.
3. **Ownership Phase (R87)**: Filter out external repositories from active tracking unless specified.
4. **Analysis Phase**: Extract real progress, not just declared.
5. **Update Phase**: Sync corrected data to Epingle_Projets.md, respecting ownership boundaries.

### During Project Work
1. **Start**: Check Kuro for project context
2. **During**: Write detailed SESSION_SUMMARY (Kuro parses)
3. **End**: Confirm Kuro will capture the session

### Kuro Data Usage
```python
# AI can query Kuro for context
GET /api/projects/{name}/activity  # Last 30 days
GET /api/projects/{name}/velocity   # Sessions/week trend
GET /api/alerts                     # Current warnings
GET /api/milestones                 # Upcoming deadlines
```

## Dashboard Integration

### GUI Elements AI Should Populate
- **Project Cards**: Progress %, last activity, next milestone (from Kuro DB)
- **Alert Panel**: Blockers requiring human intervention (Kuro detected)
- **Timeline View**: Session history with editor tags (Kuro tracked)
- **Velocity Chart**: Activity trend per project (Kuro calculated)

### AI-Generated Insights
```
"NeuralDBG velocity increased 40% last 2 weeks"
"Charmed stagnant 60 days - confirm PIVOT status?"
"EchoX approaching L3 milestone - prepare pilot"
"3 projects need SESSION_SUMMARY updates"
```

## Technical Architecture

### Kuro Stack (Target)
- **Daemon**: Rust/Python file watcher
- **Database**: SQLite local storage
- **API**: REST/GraphQL for AI queries
- **GUI**: Tauri/React or PyQt6 dashboard
- **Notifications**: Windows toast / Linux notify-send

### Data Flow
```
User edits SESSION_SUMMARY.md
    ↓
Guardian detects change (fsnotify/inotify)
    ↓
Guardian parses markdown, extracts metrics
    ↓
Guardian updates SQLite DB
    ↓
AI queries Guardian API for context
    ↓
AI provides informed assistance
```

## Compliance

### R85 + R86 Combined Checklist
- [ ] ~/Documents scanned for all projects
- [ ] SESSION_SUMMARY.md files read and parsed
- [ ] Kuro - R86: Kuro Protocol (ce fichier)
- Epingle_Projets.md: Source de vérité
- Kuro DB: Activity trackingh accurate %
- [ ] Descriptions include technical specifics
- [ ] Date stamp: "SESSION_SUMMARY parsed"
- [ ] Missing projects added to appropriate sections

### Violations
**VIOLATION**: Updating Epingle_Projets.md without reading SESSION_SUMMARY files.
**CORRECT**: Deep analysis as demonstrated 2026-05-05 — read 15+ SESSION_SUMMARY files, extract real progress.

**VIOLATION**: Assuming progress % without verification.
**CORRECT**: NeuralDBG declared 30% but SESSION_SUMMARY showed 72% — corrected to 72%.

**VIOLATION**: Missing Charmed PIVOT status.
**CORRECT**: SESSION_SUMMARY revealed pivot 2026-02-27 — updated from 50% Actif to 5% Pivot.

## References

- R80: Epingle_Projets.md maintenance
- R83: Discord investor summary
- R84: Validation automation
- R85: Portfolio completeness verification
- Project: ~/Documents/kuro/
- Vision: ~/Documents/kuro/ (merged with project folder)

---

**Created**: 2026-05-05
**Applies to**: All lambda-Section projects, all AI sessions
**Enforcement**: RECOMMENDED (until Guardian MVP deployed)
**Next Phase**: Build Guardian daemon (Rust/Python) when project prioritized
