import urllib.request, json

url = (
    "https://api.github.com/search/issues"
    "?q=repo:pytorch/pytorch+is:issue+is:open+NaN+gradient"
    "&sort=created&order=desc&per_page=8"
)
req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json"})
try:
    data = json.loads(urllib.request.urlopen(req).read())
    print(f"Total results: {data.get('total_count', 0)}")
    for item in data.get("items", []):
        print(f"#{item['number']} {item['title'][:100]}")
        print(f"  Created: {item['created_at'][:10]}  Comments: {item.get('comments', '?')}")
        print()
except Exception as e:
    print(f"Error: {e}")
