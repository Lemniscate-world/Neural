# update_complexity_readme.py
import os
import subprocess
from glob import glob

# Directory to scan (root of repo by default)
BASE_DIR = "."

# Function to get complexity metrics for a single file
def get_complexity_metrics(file_path):
    try:
        # Cyclomatic complexity
        cc_output = subprocess.check_output(["radon", "cc", file_path, "-s"], text=True)
        # Raw metrics (LOC, comments, etc.)
        raw_output = subprocess.check_output(["radon", "raw", file_path], text=True)

        # Parse cyclomatic complexity
        cc_lines = cc_output.splitlines()
        avg_complexity = next((line.split()[-1] for line in cc_lines if "Average complexity" in line), "N/A")

        # Parse raw metrics
        raw_lines = raw_output.splitlines()
        loc = next((line.split()[-1] for line in raw_lines if "LOC" in line), "N/A")
        comments = next((line.split()[-1] for line in raw_lines if "comments" in line), "N/A")

        return {
            "file": os.path.relpath(file_path, BASE_DIR),
            "avg_complexity": avg_complexity,
            "loc": loc,
            "comments": comments
        }
    except subprocess.CalledProcessError as e:
        print(f"Error analyzing {file_path}: {e}")
        return {"file": file_path, "avg_complexity": "Error", "loc": "Error", "comments": "Error"}

# Scan all Python files and analyze complexity
def analyze_all_files():
    # Find all .py files recursively, excluding hidden dirs and venv
    py_files = [
        f for f in glob(f"{BASE_DIR}/**/*.py", recursive=True)
        if not any(part.startswith('.') or 'venv' in part for part in f.split(os.sep))
    ]
    if not py_files:
        return [{"file": "No Python files found", "avg_complexity": "N/A", "loc": "N/A", "comments": "N/A"}]
    
    return [get_complexity_metrics(file) for file in py_files]

# Generate README content
complexity_data = analyze_all_files()

readme_content = """# Code Complexity Analysis

Below is the complexity analysis for all Python files in this repository, generated using `radon`. Cyclomatic complexity measures the number of linearly independent paths through the code (lower is simpler).

| File | Avg. Cyclomatic Complexity | Lines of Code (LOC) | Comment Lines |
|------|----------------------------|---------------------|---------------|
"""

for data in complexity_data:
    readme_content += f"| {data['file']} | {data['avg_complexity']} | {data['loc']} | {data['comments']} |\n"

readme_content += f"""
*Generated on: {os.popen('date').read().strip()}*
"""

# Write to README.md
with open("README.md", "w") as f:
    f.write(readme_content)

print("README.md updated with complexity analysis for all Python files!")
