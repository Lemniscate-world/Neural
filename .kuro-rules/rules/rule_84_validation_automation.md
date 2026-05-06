# RULE 84: Validation Pipeline Automation

Trigger: any project requiring field validation before product build.

---

## Mandate

Any validation pipeline MUST be automated end-to-end. No exceptions.

---

## Forbidden (R84 violations)

- Manual CSV exports from Google Forms
- Google Sheets tracking without API
- Spreadsheet-based response management
- Form endpoints without programmatic access
- Manual copy-paste of responses
- Dashboards that require manual refresh or export

---

## Required Architecture

```
Landing Page (HTML/JS)
    |
    v
Backend-as-a-Service API (Supabase / Firebase / equivalent)
    |
    +---> PostgreSQL / NoSQL Database
    +---> Realtime Dashboard (HTML/JS)
    +---> Python Analytics Script
    +---> Discord / Slack Webhook Alerts (optional)
```

### Minimum Components

| Component | Purpose | Free Option |
|-----------|---------|-------------|
| Web form | Collect responses | Custom HTML + JS |
| Database | Store responses | Supabase (500MB) |
| Dashboard | Visualize progress | Custom HTML + Supabase REST API |
| Analytics | Generate insights | Python + requests |
| Export | CSV/JSON for analysis | Python pandas |

---

## Setup Workflow

### Step 1: Create BaaS Project

Use Playwright MCP or manual setup:
1. Navigate to https://supabase.com
2. Create free project
3. Copy Project URL and Anon Key
4. Store keys in `.env` (never commit)

### Step 2: Deploy Schema

Run SQL from `supabase-schema.sql`:
- Table with computed fields (qualified, high_volume, willing_to_pay)
- RLS policies for anonymous insert
- Indexes for dashboard queries
- View `response_stats` for quick aggregation

### Step 3: Configure Landing

Update `landing/config.js`:
```javascript
window.HERMES_SUPABASE_URL = "https://xxx.supabase.co";
window.HERMES_SUPABASE_KEY = "eyJ...";
```

### Step 4: Deploy Static Files

Options:
- Netlify Drop (drag-and-drop zip)
- Vercel (`vercel --prod`)
- GitHub Pages
- Cloudflare Pages

### Step 5: Test End-to-End

Use Playwright MCP:
1. Open deployed URL
2. Fill form with test data
3. Submit
4. Verify response in Supabase dashboard
5. Verify dashboard updates
6. Run Python script and verify CSV export

### Step 6: Automate Monitoring

Optional but recommended:
- GitHub Action that runs `analyze_responses.py` daily
- Discord webhook alert when new qualified lead arrives
- Weekly auto-generated report pushed to Google Drive

---

## Verification Checklist

- [ ] Form submission works from mobile browser
- [ ] Response appears in database within 5 seconds
- [ ] Dashboard shows updated stats without refresh
- [ ] Python script connects and exports CSV
- [ ] R83 investor summary references the pipeline status
- [ ] No manual steps required between submission and analysis

---

## Failure Mode

If the pipeline has any manual step:
- The session is incomplete
- No distribution should start until fixed
- Document the blocker in SESSION_SUMMARY.md

---

## MCP/Agent Integration

### Available Tools

| Tool | Use Case |
|------|----------|
| Supabase MCP | **BEST**: Create projects, execute SQL, manage tables, get keys via natural language |
| Playwright MCP | Navigate Supabase web UI, test forms |
| Puppeteer MCP | Alternative browser automation |
| Python script | Analytics, export, reporting |
| GitHub Actions | CI/CD for validation pipeline |

### Supabase MCP Server (Recommended)

Official MCP server: `https://mcp.supabase.com/mcp`

Capabilities:
- `create_project` - Create new Supabase project (after auth)
- `execute_sql` - Run schema SQL directly
- `list_tables` - Verify tables created
- `get_project_url` - Extract API URL
- `get_publishable_keys` - Extract anon key
- `apply_migration` - Deploy schema changes

Configuration:
```json
{
  "mcpServers": {
    "supabase": {
      "type": "http",
      "url": "https://mcp.supabase.com/mcp"
    }
  }
}
```

### What CANNOT be automated (security)

- Initial Supabase account creation (requires email verification)
- OAuth authentication flow (requires browser interaction once)
- MFA/2FA verification

### What CAN be automated (with MCP)

- Project creation (after initial auth)
- Database schema deployment via `execute_sql`
- Configuration extraction (URL, keys)
- Form testing end-to-end
- Dashboard verification
- Analytics report generation
- Webhook notifications
- Daily/weekly reporting

---

## Template Files

Every validation project should include:

```
landing/
  index.html              # Presentation page
  formulaire-supabase.html # Form with Supabase POST
  dashboard.html          # Real-time stats
  config.js               # Supabase credentials (gitignored)
scripts/
  analyze_responses.py    # Analytics + CSV export
supabase-schema.sql       # Database schema
AUTOMATION_SETUP.md       # Setup instructions
.windsurf/workflows/
  validation-automation.md # This workflow
```

---

## Cost

Supabase free tier: 500MB storage, 75K connections/day.
Sufficient for 10,000+ responses.
