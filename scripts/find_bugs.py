import urllib.request, json

queries = [
    ("gradient+explosion+created:>2026-05-01", "Gradient explosion"),
    ("silent+NaN+backward+created:>2026-05-01", "Silent NaN backward"),
    ("gradient+incorrect+Silent+created:>2026-05-01", "Silent wrong gradient"),
]

for q, label in queries:
    url = (
        f"https://api.github.com/search/issues"
        f"?q=repo:pytorch/pytorch+is:issue+is:open+{q}"
        f"&sort=created&order=desc&per_page=4"
    )
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json"})
        data = json.loads(urllib.request.urlopen(req).read())
        print(f"\n--- {label} ({data.get('total_count',0)}) ---")
        for item in data.get("items", [])[:4]:
            labels = [l["name"] for l in item.get("labels", [])]
            print(f"  #{item['number']} {item['title'][:90]}")
            print(f"    {item['created_at'][:10]} | {item.get('comments',0)}c | {', '.join(labels[:3])}")
    except Exception as e:
        print(f"  Error: {e}")
