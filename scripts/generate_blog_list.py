#!/usr/bin/env python3
"""
Generate a JSON file with the list of blog posts from markdown files in the docs/blog directory.
This script should be run whenever a new blog post is added.
"""

import os
import json
import re
from datetime import datetime
import markdown
import glob

# Configuration
BLOG_DIR = "docs/blog"
OUTPUT_FILE = os.path.join(BLOG_DIR, "blog-list.json")
BLOG_URL_BASE = "blog"  # Relative to docs directory
EXCERPT_LENGTH = 200  # Characters for the excerpt

def extract_title_and_date(content, filename):
    """Extract title and date from markdown content"""
    title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
    title = title_match.group(1) if title_match else os.path.basename(filename).replace('.md', '')

    # Try to extract date from content
    date_match = re.search(r'\*Posted on ([^*]+)', content)
    if date_match:
        date_str = date_match.group(1).strip()
        try:
            # Try to parse the date
            date_obj = datetime.strptime(date_str, '%B %d, %Y')
            date = date_obj.strftime('%Y-%m-%d')
        except ValueError:
            # If date parsing fails, use file modification time
            date = datetime.fromtimestamp(os.path.getmtime(filename)).strftime('%Y-%m-%d')
    else:
        # Use file modification time as fallback
        date = datetime.fromtimestamp(os.path.getmtime(filename)).strftime('%Y-%m-%d')

    return title, date

def create_html_file(md_file, html_file, title):
    """Create an HTML file from a markdown file"""
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert markdown to HTML
    html_content = markdown.markdown(md_content, extensions=['fenced_code', 'codehilite'])

    # Create HTML file with proper layout
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-2Q74G78HEW"></script>
    <script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){{dataLayer.push(arguments);}}
    gtag('js', new Date());

    gtag('config', 'G-2Q74G78HEW');
    </script>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Neural DSL Blog</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/default.min.css">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f4f4f4; }}
        header {{ text-align: center; padding: 20px; background: #1f73b7; color: white; margin-bottom: 30px; }}
        h1 {{ margin: 0; }}
        .blog-container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .blog-title {{ color: #1f73b7; margin-top: 0; }}
        .blog-meta {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
        .blog-content {{ line-height: 1.6; }}
        .blog-content img {{ max-width: 100%; height: auto; }}
        .blog-content pre {{ background: #f8f8f8; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .blog-content code {{ background: #f8f8f8; padding: 2px 5px; border-radius: 3px; }}
        .back-link {{ margin-bottom: 20px; }}
        footer {{ text-align: center; padding: 20px; color: #666; margin-top: 30px; }}
    </style>
</head>
<body>
    <header>
        <h1>Neural DSL Blog</h1>
        <p>Latest news, updates, and tutorials</p>
    </header>

    <div class="blog-container">
        <div class="back-link">
            <a href="index.html">← Back to Blog</a>
        </div>

        <div class="blog-content">
            {html_content}
        </div>
    </div>

    <footer>
        <p>⭐ <a href="https://github.com/Lemniscate-world/Neural">Star us on GitHub</a> | Follow <a href="https://x.com/NLang4438">@NLang4438</a></p>
    </footer>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            // Apply syntax highlighting to code blocks
            document.querySelectorAll('pre code').forEach((block) => {{
                hljs.highlightBlock(block);
            }});
        }});
    </script>
</body>
</html>""")

def generate_blog_list():
    """Generate a JSON file with the list of blog posts"""
    blog_posts = []

    # Find all markdown files in the blog directory (excluding Dev.to posts)
    md_files = glob.glob(os.path.join(BLOG_DIR, "*.md"))
    md_files = [f for f in md_files if not os.path.basename(f).startswith("devto_")]

    # Print all found files for debugging
    print(f"Found {len(md_files)} markdown files:")
    for f in md_files:
        print(f"  - {f}")

    # Clear the existing blog-list.json file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('[]')

    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract title and date
            title, date = extract_title_and_date(content, md_file)

            # Create excerpt (first real paragraph without markdown formatting)
            text_content = re.sub(r'!\[.*?\]\(.*?\)', '', content)  # Remove images
            text_content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text_content)  # Replace links with text
            text_content = re.sub(r'#+ .*?\n', '', text_content)  # Remove headings
            text_content = re.sub(r'\*Posted on.*?\*', '', text_content)  # Remove date line

            # Find paragraphs (non-empty lines)
            paragraphs = [p.strip() for p in re.findall(r'(?:^|\n\n)([^\n]+)', text_content) if p.strip()]

            # Find the first real paragraph (not a heading, not metadata)
            for p in paragraphs:
                if p and not p.startswith('#') and not p.startswith('*Posted') and len(p) > 30:
                    excerpt = p.strip()
                    break
            else:
                excerpt = paragraphs[0] if paragraphs else ""

            if len(excerpt) > EXCERPT_LENGTH:
                excerpt = excerpt[:EXCERPT_LENGTH] + "..."

            # Generate HTML filename
            basename = os.path.basename(md_file).replace('.md', '')
            html_filename = f"{basename}.html"
            html_filepath = os.path.join(BLOG_DIR, html_filename)

            # Create HTML file
            create_html_file(md_file, html_filepath, title)

            # Add to blog posts list
            blog_posts.append({
                "title": title,
                "date": date,
                "excerpt": excerpt,
                "filename": os.path.basename(md_file),
                "url": html_filename
            })

            print(f"Processed: {md_file} -> {html_filepath}")

        except Exception as e:
            print(f"Error processing {md_file}: {e}")

    # Sort by date (newest first)
    blog_posts.sort(key=lambda x: x["date"], reverse=True)

    # Write to JSON file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(blog_posts, f, indent=2)

    print(f"Generated blog list with {len(blog_posts)} posts: {OUTPUT_FILE}")

if __name__ == "__main__":
    # Create blog directory if it doesn't exist
    os.makedirs(BLOG_DIR, exist_ok=True)
    generate_blog_list()
