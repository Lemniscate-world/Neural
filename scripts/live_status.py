import urllib.request, json

def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json"})
    try:
        return json.loads(urllib.request.urlopen(req).read())
    except Exception as e:
        return {"_error": str(e)}

# 1. NeuralDBG stats
ndbg = get("https://api.github.com/repos/LambdaSection/NeuralDBG")
print(f"NeuralDBG: {ndbg.get('stargazers_count', 0)} stars, {ndbg.get('forks_count', 0)} forks, {ndbg.get('open_issues_count', 0)} open issues")

# 2. PRs upstream status
for num in [186786, 188053, 188066]:
    pr = get(f"https://api.github.com/repos/pytorch/pytorch/pulls/{num}")
    print(f"PR #{num}: state={pr.get('state')}, merged={pr.get('merged')}, draft={pr.get('draft')}")

# 3. Workflow runs for sync-portfolio on kuro-rules
runs = get("https://api.github.com/repos/Lemniscate-world/kuro-rules/actions/runs?per_page=5")
for r in runs.get("workflow_runs", [])[:5]:
    print(f"Workflow: {r['name']} status={r['conclusion']} ({r['created_at'][:10]})")

# 4. Portfolio page check
import socket
try:
    socket.setdefaulttimeout(5)
    sock = socket.create_connection(("lemniscate-world.github.io", 443), timeout=5)
    print(f"Portfolio site: reachable")
except Exception as e:
    print(f"Portfolio site: {e}")
