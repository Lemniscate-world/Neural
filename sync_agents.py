import os
import shutil

# Master copy path
master_agents_md = r"c:\Users\Utilisateur\Documents\kuro-rules\AGENTS.md"

# List of projects (URIs) provided by the system
# We'll use the current directory's parent to find siblings, 
# but specifically target the projects from the corpus mapping if possible.
# For this script, we'll scan known locations.

projects = [
    r"c:\Users\Utilisateur\Documents\NeuralDBG",
    r"c:\Users\Utilisateur\Documents\kuro-rules",
    # Add other projects here or scan the directory
]

# Better way: scan the whole Documents folder for AGENTS.md
base_dir = r"c:\Users\Utilisateur\Documents"
count = 0

print(f"Syncing {master_agents_md} to all projects...")

for root, dirs, files in os.walk(base_dir):
    if "AGENTS.md" in files:
        target_path = os.path.join(root, "AGENTS.md")
        if os.path.abspath(target_path) != os.path.abspath(master_agents_md):
            shutil.copy2(master_agents_md, target_path)
            print(f"Synced: {target_path}")
            count += 1

print(f"Total synced: {count}")
