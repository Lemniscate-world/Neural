# scripts/create_issues.py
import xml.etree.ElementTree as ET
import os
import requests
from github import Github

# GitHub token (store in .env or GitHub Secrets)
GITHUB_TOKEN = os.environ.get("ISSUES_TOKEN")
REPO = "Lemniscate-SHA-256/Neural"

def parse_pytest_results(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    issues = []
    for testcase in root.findall(".//testcase"):
        failure = testcase.find("failure")
        if failure is not None:
            test_name = testcase.get("name")
            file = testcase.get("file")
            line = failure.get("line", "unknown")
            message = failure.text or "No failure message"
            issues.append({
                "title": f"Test Failure: {test_name}",
                "body": f"**Test Failure Details**\n"
                        f"- **Test Name**: {test_name}\n"
                        f"- **File**: {file}\n"
                        f"- **Line**: {line}\n"
                        f"- **Error**: {message}\n\n"
                        f"**Explanation**: This bug indicates an issue in the {file} code. Please investigate and fix the parser or related logic.\n\n"
                        f"**Comments**: Any additional context or reproduction steps? Assign to @Lemniscate-SHA-256 for review."
            })
    return issues

def create_github_issues(issues):
    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(REPO)
    for issue in issues:
        try:
            repo.create_issue(title=issue["title"], body=issue["body"])
            print(f"Created issue: {issue['title']}")
        except Exception as e:
            print(f"Failed to create issue for {issue['title']}: {e}")

    def issue_exists(repo, title):
        for issue in repo.get_issues(state="open"):
            if issue.title == title:
                return True
        return False

    if not issue_exists(repo, issue["title"]):
        repo.create_issue(title=issue["title"], body=issue["body"])


if __name__ == "__main__":
    create_github_issues(parse_pytest_results("test-results.xml"))