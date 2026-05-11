#!/usr/bin/env python3
"""
Linear Sync - Team workflow automation for Linear issues.

This script provides a robust, editor-agnostic way to interact with Linear
using the REST/GraphQL API. Works for any team member regardless of editor.

Usage:
    python scripts/linear_sync.py list                    # List open issues
    python scripts/linear_sync.py status <issue-id>       # Get issue status
    python scripts/linear_sync.py update <issue-id> --done # Update issue state
    python scripts/linear_sync.py sync                     # Full team sync

Environment:
    LINEAR_API_KEY    - Your Linear personal API key
    LINEAR_TEAM_ID    - Team identifier (e.g., "MLO" for MLOps team)
"""

import os
import sys
import json
import argparse
from http.client import HTTPSConnection, HTTPException
from dataclasses import dataclass
from typing import Optional

LINEAR_API_HOST = "api.linear.app"
LINEAR_API_PATH = "/graphql"
LINEAR_API_KEY = os.environ.get("LINEAR_API_KEY", "")
LINEAR_TEAM_ID = os.environ.get("LINEAR_TEAM_ID", "")


@dataclass
class LinearIssue:
    identifier: str
    id: str
    title: str
    state: str
    priority: int
    assignee: Optional[str] = None
    url: str = ""


def linear_request(query: str, variables: dict = None) -> dict:
    """Execute a GraphQL query against Linear API."""
    if not LINEAR_API_KEY:
        print("ERROR: LINEAR_API_KEY environment variable not set.")
        print("Get your API key from: Linear -> Settings -> API")
        sys.exit(1)

    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    data = json.dumps(payload).encode("utf-8")

    headers = {
        "Authorization": LINEAR_API_KEY,
        "Content-Type": "application/json",
        "User-Agent": "NeuralDBG-linear-sync/1.0",
    }
    connection = HTTPSConnection(LINEAR_API_HOST, timeout=30)

    try:
        connection.request("POST", LINEAR_API_PATH, body=data, headers=headers)
        response = connection.getresponse()
        response_body = response.read().decode("utf-8")
        if response.status >= 400:
            print(f"ERROR: Linear API returned {response.status}: {response_body}")
            sys.exit(1)
        result = json.loads(response_body)
    except (HTTPException, OSError) as e:
        print(f"ERROR: Cannot reach Linear API: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Linear API returned invalid JSON: {e}")
        sys.exit(1)
    finally:
        connection.close()

    if "errors" in result:
        print(f"ERROR: GraphQL errors: {result['errors']}")
        sys.exit(1)

    return result.get("data", {})


def list_issues(state: str = "open", limit: int = 20) -> list[LinearIssue]:
    """List issues, optionally filtered by state."""
    state_filter = (
        "filter: {state: {eq: open}}"
        if state == "open"
        else "filter: {state: {eq: done}}"
    )

    query = f"""
    query {{
        issues({state_filter}, first: {limit}) {{
            nodes {{
                identifier
                id
                title
                priority
                url
                state {{ name }}
                assignee {{ name }}
            }}
        }}
    }}
    """

    data = linear_request(query)
    nodes = data.get("issues", {}).get("nodes", [])

    return [
        LinearIssue(
            identifier=n["identifier"],
            id=n["id"],
            title=n["title"],
            state=n["state"]["name"] if n.get("state") else "Unknown",
            priority=n.get("priority", 0),
            assignee=n["assignee"]["name"] if n.get("assignee") else None,
            url=n.get("url", ""),
        )
        for n in nodes
    ]


def get_issue(issue_id: str) -> LinearIssue:
    """Get a single issue by identifier (e.g., 'MLO-1')."""
    query = """
    query($identifier: String!) {
        issue(identifier: $identifier) {
            identifier
            id
            title
            priority
            url
            state { name }
            assignee { name }
            description
            createdAt
            updatedAt
        }
    }
    """

    data = linear_request(query, {"identifier": issue_id})
    issue = data.get("issue")

    if not issue:
        print(f"ERROR: Issue {issue_id} not found")
        sys.exit(1)

    return LinearIssue(
        identifier=issue["identifier"],
        id=issue["id"],
        title=issue["title"],
        state=issue["state"]["name"] if issue.get("state") else "Unknown",
        priority=issue.get("priority", 0),
        assignee=issue["assignee"]["name"] if issue.get("assignee") else None,
        url=issue.get("url", ""),
    )


def update_issue_state(issue_id: str, new_state: str) -> dict:
    """Update issue state (e.g., 'Done', 'In Progress')."""
    state_mapping = {
        "done": "Done",
        "in_progress": "In Progress",
        "canceled": "Canceled",
        "backlog": "Backlog",
    }

    if new_state not in state_mapping:
        print(
            f"ERROR: Unknown state '{new_state}'. Valid: {list(state_mapping.keys())}"
        )
        sys.exit(1)

    mutation = """
    mutation($identifier: String!, $state: String!) {
        issueUpdate(identifier: $identifier, input: {stateName: $state}) {
            success
            issue { identifier state { name } }
        }
    }
    """

    linear_state_name = state_mapping[new_state]
    data = linear_request(
        mutation,
        {"identifier": issue_id, "state": linear_state_name},
    )
    result = data.get("issueUpdate", {})

    if result.get("success"):
        print(f"[OK] {issue_id} -> {linear_state_name}")
        return result
    else:
        print(f"ERROR: Failed to update {issue_id}")
        sys.exit(1)


def create_issue(
    title: str,
    description: str = "",
    team_id: str = None,
    priority: int = 0,
) -> dict:
    """Create a new issue."""
    if not team_id and not LINEAR_TEAM_ID:
        print("ERROR: LINEAR_TEAM_ID not set. Provide --team or set LINEAR_TEAM_ID")
        sys.exit(1)

    mutation = """
    mutation($input: IssueCreateInput!) {
        issueCreate(input: $input) {
            success
            issue {
                identifier
                title
                url
            }
        }
    }
    """

    variables = {
        "input": {
            "title": title,
            "description": description,
            "teamId": team_id or LINEAR_TEAM_ID,
            "priority": priority,
        }
    }

    data = linear_request(mutation, variables)
    result = data.get("issueCreate", {})

    if result.get("success"):
        issue = result["issue"]
        print(f"[OK] Created {issue['identifier']}: {issue['title']}")
        print(f"      {issue['url']}")
        return result
    else:
        print("ERROR: Failed to create issue")
        sys.exit(1)


def print_issue(issue: LinearIssue, verbose: bool = False):
    """Print an issue in a formatted way."""
    priority_label = {0: "P0", 1: "P1", 2: "P2", 3: "P3", 4: "P4"}.get(
        issue.priority,
        "P?",
    )

    print(f"\n{'=' * 60}")
    print(f"  {issue.identifier} {priority_label}")
    print(f"{'=' * 60}")
    print(f"  Title:    {issue.title}")
    print(f"  State:    {issue.state}")
    print(f"  Assignee: {issue.assignee or 'Unassigned'}")
    print(f"  URL:      {issue.url}")

    if verbose and hasattr(issue, "description"):
        print("\n  Description:")
        print(f"  {issue.description or '(none)'}")

    print()


def cmd_list(args):
    """List issues command."""
    issues = list_issues(state=args.state, limit=args.limit)

    if not issues:
        print(f"No {args.state} issues found.")
        return

    print(f"\n{'=' * 65}")
    print(f"  Linear Issues ({args.state}) - {len(issues)} found")
    print(f"{'=' * 65}")

    for issue in issues:
        print_issue(issue)


def cmd_status(args):
    """Get issue status command."""
    issue = get_issue(args.issue_id)
    print_issue(issue, verbose=args.verbose)


def cmd_update(args):
    """Update issue state command."""
    update_issue_state(args.issue_id, args.new_state)


def cmd_create(args):
    """Create issue command."""
    create_issue(
        title=args.title,
        description=args.description or "",
        team_id=args.team,
        priority=args.priority or 0,
    )


def cmd_sync(args):
    """Sync command - show team status overview."""
    print("\n[Linear Sync] Fetching team status...\n")

    open_issues = list_issues(state="open", limit=50)

    if not open_issues:
        print("No open issues. Team is all caught up!")
        return

    print(f"Open Issues: {len(open_issues)}")
    print("-" * 65)

    by_state = {}
    for issue in open_issues:
        by_state.setdefault(issue.state, []).append(issue)

    for state, issues in sorted(by_state.items()):
        print(f"\n{state} ({len(issues)}):")
        for issue in issues:
            assignee = issue.assignee or "Unassigned"
            print(f"  [{issue.identifier}] {issue.title[:50]} - {assignee}")


def main():
    parser = argparse.ArgumentParser(
        description="Linear team workflow automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/linear_sync.py list --state open
  python scripts/linear_sync.py status MLO-1
  python scripts/linear_sync.py update MLO-1 --state done
  python scripts/linear_sync.py create "Fix bug" --priority 1
  python scripts/linear_sync.py sync

Environment:
  LINEAR_API_KEY    Your Linear personal API key
  LINEAR_TEAM_ID    Default team identifier (e.g., MLO)

Get API key: Linear app -> Settings -> API -> Personal API keys
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List issues")
    list_parser.add_argument("--state", choices=["open", "done"], default="open")
    list_parser.add_argument("--limit", type=int, default=20)
    list_parser.set_defaults(func=cmd_list)

    status_parser = subparsers.add_parser("status", help="Get issue details")
    status_parser.add_argument("issue_id", help="Issue identifier (e.g., MLO-1)")
    status_parser.add_argument("--verbose", "-v", action="store_true")
    status_parser.set_defaults(func=cmd_status)

    update_parser = subparsers.add_parser("update", help="Update issue state")
    update_parser.add_argument("issue_id", help="Issue identifier")
    update_parser.add_argument(
        "--state",
        dest="new_state",
        required=True,
        choices=["done", "in_progress", "backlog", "canceled"],
    )
    update_parser.set_defaults(func=cmd_update)

    create_parser = subparsers.add_parser("create", help="Create issue")
    create_parser.add_argument("title", help="Issue title")
    create_parser.add_argument("--description", "-d", default="")
    create_parser.add_argument("--team", "-t", default=None)
    create_parser.add_argument("--priority", "-p", type=int, default=0)
    create_parser.set_defaults(func=cmd_create)

    sync_parser = subparsers.add_parser("sync", help="Team sync overview")
    sync_parser.set_defaults(func=cmd_sync)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
