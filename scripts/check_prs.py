import urllib.request, json

def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json"})
    try: return json.loads(urllib.request.urlopen(req).read())
    except: return {}

for num in [188053, 188066]:
    pr = get(f"https://api.github.com/repos/pytorch/pytorch/pulls/{num}")
    state = pr.get("state", "?")
    merged = pr.get("merged", "?")
    draft = pr.get("draft", "?")
    created = pr.get("created_at", "?")[:10]
    print(f"PR #{num}: state={state}, merged={merged}, draft={draft}, created={created}")
    comments = pr.get("comments") if isinstance(pr.get("comments"), list) else []
    for c in comments[-3:]:
        user = c.get("user", {}).get("login", "?")
        body = c.get("body", "")[:200]
        print(f"  Comment by {user}: {body}")
    print()
