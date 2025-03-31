#!/bin/bash

# Make a backup of the repository
cp -r .git .git.bak

# Create a mirror of the repository
git clone --mirror . repo.git

# Use BFG to remove large files
java -jar bfg.jar --strip-blobs-bigger-than 10M repo.git

# Clean up the repository
cd repo.git
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Go back to the original repository
cd ..

# Get the new repository
git remote remove origin
git remote add origin ./repo.git
git fetch
git reset --hard origin/main

# Clean up
rm -rf repo.git

echo "Repository cleaned. Large files have been removed from history."
echo "Now you can push with: git push -f origin main:main"
