import subprocess
import sys
from pathlib import Path

def main():
    print("[Sync Summary] Converting SESSION_SUMMARY.md to SESSION_SUMMARY.docx...")
    md_path = Path("SESSION_SUMMARY.md")
    docx_path = Path("SESSION_SUMMARY.docx")
    
    if not md_path.exists():
        print(f"Error: {md_path} not found.")
        sys.exit(1)
        
    try:
        # Check if pandoc is installed
        subprocess.run(["pandoc", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: pandoc is not installed or not in PATH.")
        print("Please install pandoc to generate the Word backup, e.g. via: winget install pandoc")
        print("Skipping document conversion.")
        sys.exit(0)
        
    try:
        subprocess.run(["pandoc", str(md_path), "-o", str(docx_path)], check=True)
        print(f"Success: Converted to {docx_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during pandoc conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
