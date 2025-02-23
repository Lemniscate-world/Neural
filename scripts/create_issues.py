# scripts/create_issues.py
import xml.etree.ElementTree as ET
import os
import requests
from github import Github

# GitHub token (store in .env or GitHub Secrets)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO = "Lemniscate-SHA-256/Neural"

def parse_pytest_results(xml_path):
    xml_path = os.path.join(
        os.environ.get('GITHUB_WORKSPACE', ''), 
        'test-results.xml'
    )
    if not os.path.exists(xml_path):
        xml_path = 'test-results.xml'
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
    print(f"Processing {len(issues)} potential issues")
    
    g = Github(os.environ['GITHUB_TOKEN'])
    repo = g.get_repo(REPO)
    
    for idx, issue in enumerate(issues, 1):
        print(f"\n--- Processing Issue {idx}/{len(issues)} ---")
        print(f"Title: {issue['title']}")
        print(f"Body: {issue['body'][:100]}...")
        
        try:
            exists = any(issue.title.lower() == issue['title'].lower() 
                        for issue in repo.get_issues(state='open'))
            if exists:
                print("↻ Existing issue found - skipping")
                continue
                
            new_issue = repo.create_issue(
                title=issue["title"],
                body=issue["body"],
                labels=['bug', 'ci']
            )
            print(f"✓ Created issue #{new_issue.number}")
        except Exception as e:
            print(f"✖ Error: {str(e)}")

if __name__ == "__main__":
    create_github_issues(parse_pytest_results("test-results.xml"))