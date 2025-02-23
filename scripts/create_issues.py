import xml.etree.ElementTree as ET
import os
from github import Github

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO = "Lemniscate-SHA-256/Neural"

def parse_pytest_results(xml_path):
    xml_path = os.path.join(os.environ.get("GITHUB_WORKSPACE", ""), "test-results.xml")
    if not os.path.exists(xml_path):
        xml_path = "test-results.xml"

    tree = ET.parse(xml_path)
    root = tree.getroot()
    issues = []

    for testcase in root.findall(".//testcase"):
        failure = testcase.find("failure")
        if failure is not None:
            test_name = testcase.get("name")
            classname = testcase.get("classname", "unknown").replace(".", "/") + ".py"
            message = failure.text or "No failure message"

            issues.append({
                "title": f"Test Failure: {test_name}",
                "body": f"## Test Failure Details\n"
                        f"- **Test Name:** {test_name}\n"
                        f"- **File:** `{classname}`\n"
                        f"- **Error:** {message}\n\n"
                        f"**Explanation:** This test failure needs investigation.\n\n"
                        f"**Comments:** Any additional context? Assigning to @Lemniscate-SHA-256 for review.",
            })
    
    return issues

def create_github_issues(issues):
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        raise ValueError("Missing GITHUB_TOKEN environment variable")
    
    g = Github(token)
    repo = g.get_repo("Lemniscate-SHA-256/Neural")
    
    for issue in issues:
        # Check for existing issues with similar titles
        existing_issues = list(repo.get_issues(state='open'))
        exists = any(
            issue["title"] in existing.title 
            for existing in existing_issues
        )
        
        if not exists:
            try:
                repo.create_issue(
                    title=issue["title"][:120],  # Truncate long titles
                    body=issue["body"],
                    labels=['bug', 'ci'],
                    assignees=['Lemniscate-SHA-256']
                )
                print(f"Created issue: {issue['title']}")
            except Exception as e:
                print(f"Error creating issue: {str(e)}")

if __name__ == "__main__":
    create_github_issues(parse_pytest_results("test-results.xml"))
