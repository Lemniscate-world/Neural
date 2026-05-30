#!/usr/bin/env python3
"""Post a GitHub Discussion via gh CLI."""

import json
import subprocess
import sys
import tempfile
import os

BODY_FILE = sys.argv[1] if len(sys.argv) > 1 else "docs/discussion_body.md"
TITLE = "NeuralDBG v1.3.1 - Causal inference engine for PyTorch training failures"
CATEGORY = "Show and tell"

body = open(BODY_FILE, "r", encoding="utf-8").read().strip()


def gh_graphql(query):
    r = subprocess.run(
        ["gh", "api", "graphql", "-f", f"query={query}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(r.stdout)


# Get repo node ID
data = gh_graphql('{ repository(owner:"LambdaSection", name:"NeuralDBG") { id } }')
repo_id = data["data"]["repository"]["id"]

# Get category ID
data = gh_graphql(
    '{ repository(owner:"LambdaSection", name:"NeuralDBG") { discussionCategories(first:10) { nodes { id name } } } }'
)
cats = data["data"]["repository"]["discussionCategories"]["nodes"]
cat_id = next(c["id"] for c in cats if c["name"] == CATEGORY)

# Write variables to temp file
variables = {
    "input": {
        "repositoryId": repo_id,
        "categoryId": cat_id,
        "title": TITLE,
        "body": body,
    }
}

with tempfile.NamedTemporaryFile(
    mode="w", suffix=".json", delete=False, encoding="utf-8"
) as f:
    json.dump(variables, f, ensure_ascii=False)
    var_file = f.name

MUTATION = "mutation CreateDiscussion($input: CreateDiscussionInput!) { createDiscussion(input: $input) { discussion { url number } } }"

try:
    r = subprocess.run(
        ["gh", "api", "graphql", "-f", f"query={MUTATION}", "-F", f"@{var_file}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    result = json.loads(r.stdout)
    if "errors" in result:
        print(f"ERROR: {result['errors']}")
        sys.exit(1)
    url = result["data"]["createDiscussion"]["discussion"]["url"]
    number = result["data"]["createDiscussion"]["discussion"]["number"]
    print(f"Discussion #{number} created: {url}")
finally:
    os.unlink(var_file)
