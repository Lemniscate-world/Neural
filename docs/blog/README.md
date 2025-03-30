# Neural DSL Blog

This directory contains the blog posts for the Neural DSL website.

## Adding a New Blog Post

1. Create a new markdown file in this directory with a descriptive name, e.g., `website_v0.2.6_release.md`.
   - Use the `website_` prefix for posts that will appear on the website.
   - Use the `devto_` prefix for posts that are formatted for Dev.to.

2. Format your blog post with the following structure:
   ```markdown
   # Title of Your Blog Post

   ![Optional Image](../assets/images/your-image.png)

   *Posted on Month Day, Year by Your Name*

   First paragraph of your blog post...

   ## Section Heading

   Content of your section...

   ```

3. Run the blog generation script to create the HTML files and update the blog list:
   ```bash
   python scripts/generate_blog_list.py
   ```

4. The script will:
   - Generate an HTML version of your markdown file
   - Update the `blog-list.json` file with your new post
   - Your post will automatically appear on the website

## Blog Post Guidelines

- Keep titles concise and descriptive
- Include a clear date in the format `Month Day, Year` (e.g., `March 24, 2025`)
- For release announcements, emphasize one main feature plus other fixes
- Include code examples when relevant
- Use proper markdown formatting for headings, lists, and code blocks
- Keep images to a reasonable size (max width 900px recommended)

## Dev.to Posts

For posts that will also be published on Dev.to:

1. Create a file with the `devto_` prefix
2. Include the Dev.to frontmatter at the top:
   ```markdown
   ---
   title: "Your Title Here"
   published: true
   description: "Brief description of your post"
   tags: tag1, tag2, tag3
   cover_image: https://url-to-your-cover-image.png
   ---
   ```

3. These posts won't appear on the website but can be used as a reference for your Dev.to publications.
