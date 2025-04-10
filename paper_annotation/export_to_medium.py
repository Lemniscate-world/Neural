#!/usr/bin/env python3
import os
import argparse
import subprocess
import re
from datetime import datetime

def export_to_medium(notebook_path):
    """Export a Jupyter notebook to Medium-ready markdown"""
    # Create exports directory if it doesn't exist
    os.makedirs("exports", exist_ok=True)
    
    # Get the notebook filename without extension
    notebook_name = os.path.splitext(os.path.basename(notebook_path))[0]
    
    # Export directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = os.path.join("exports", f"{notebook_name}_{timestamp}")
    os.makedirs(export_dir, exist_ok=True)
    
    # Run nbconvert to export to markdown
    cmd = [
        "jupyter", "nbconvert", 
        notebook_path, 
        "--to", "markdown", 
        "--output-dir", export_dir
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully exported to {export_dir}")
        
        # Create a Medium-ready version with front matter
        md_file = os.path.join(export_dir, f"{os.path.basename(notebook_name)}.md")
        
        with open(md_file, "r") as f:
            content = f.read()
        
        # Extract title from the first heading
        title_match = re.search(r"# (.+)", content)
        title = title_match.group(1) if title_match else "Paper Annotation"
        
        # Add Medium front matter
        medium_content = f"""---
title: "{title}"
published: false
tags: machinelearning,deeplearning,neuralnetworks,paperreview
---

{content}

---

*This article was created using [Neural DSL](https://github.com/Lemniscate-SHA-256/Neural), an open-source domain-specific language for neural networks.*
"""
        
        medium_file = os.path.join(export_dir, f"{os.path.basename(notebook_name)}_medium.md")
        with open(medium_file, "w") as f:
            f.write(medium_content)
        
        print(f"Medium-ready markdown created at {medium_file}")
        
        # Copy images to the export directory
        images_dir = os.path.join(notebook_path.replace(".ipynb", "_files"))
        if os.path.exists(images_dir):
            subprocess.run(["cp", "-r", images_dir, export_dir])
            print(f"Copied images from {images_dir} to {export_dir}")
        
        return medium_file
    
    except subprocess.CalledProcessError as e:
        print(f"Error exporting notebook: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a Jupyter notebook to Medium-ready markdown")
    parser.add_argument("notebook", help="Path to the Jupyter notebook")
    
    args = parser.parse_args()
    
    export_to_medium(args.notebook)
