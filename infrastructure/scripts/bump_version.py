import sys
import os
import re


def bump_version(file_path, new_version):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Update pyproject.toml version
    if file_path.endswith("pyproject.toml"):
        pattern = r'^version = ".*"'
        replacement = f'version = "{new_version}"'
        new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    else:
        print(f"Error: Unsupported file type {file_path}")
        sys.exit(1)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"Successfully bumped version in {file_path} to {new_version}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python bump_version.py <file_path> <new_version>")
        sys.exit(1)

    bump_version(sys.argv[1], sys.argv[2])
