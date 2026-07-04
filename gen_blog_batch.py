"""Quick HTML generator for blog posts."""
import re

def md_to_html(md_file, html_file, post_id, bug_id, title, gap):
    with open(md_file, encoding='utf-8') as f:
        md = f.read()
    
    # Simple MD→HTML conversion
    html_body = []
    in_code = False
    for line in md.split('\n'):
        s = line.strip()
        if s.startswith('```'):
            if in_code:
                html_body.append('</code></pre>')
                in_code = False
            else:
                html_body.append('<pre><code>')
                in_code = True
            continue
        if in_code:
            html_body.append(line)
            continue
        if s.startswith('# ') and not s.startswith('# POST'):
            html_body.append(f'<h2>{s[2:]}</h2>')
        elif s.startswith('## '):
            html_body.append(f'<h3>{s[3:]}</h3>')
        elif s.startswith('|') and '---' not in s:
            html_body.append(f'<p><code>{s}</code></p>')
        elif s.startswith('- '):
            html_body.append(f'<li>{s[2:]}</li>')
        elif s.startswith('> '):
            html_body.append(f'<blockquote>{s[2:]}</blockquote>')
        elif s.startswith('**') and ':**' in s:
            html_body.append(f'<p><strong>{s}</strong></p>')
        elif s and s[0].isalpha() or s.startswith('torch.') or s.startswith('http'):
            html_body.append(f'<p>{s}</p>')
        elif s:
            html_body.append(f'<p>{s}</p>')
    
    body = '\n'.join(html_body)
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{title} - NeuralDBG Blog</title>
<link rel="icon" href="../assets/neuraldbg-logo.svg" type="image/svg+xml">
<style>
:root{{--bg:#0d1117;--surface:#161b22;--text:#e6edf3;--muted:#8b949e;--accent:#58a6ff;--accent2:#a371f7;--success:#3fb950;--danger:#f85149;--border:#30363d}}
*{{box-sizing:border-box}}body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;background:var(--bg);color:var(--text);margin:0;line-height:1.7}}
.container{{max-width:760px;margin:0 auto;padding:0 1.5rem}}
header{{padding:2.5rem 0 1.5rem;border-bottom:1px solid var(--border)}}
.breadcrumb{{color:var(--muted);font-size:.9rem}}.breadcrumb a{{color:var(--accent);text-decoration:none}}
h1{{font-size:2.1rem;margin:.5rem 0 .75rem}}
h2{{font-size:1.5rem;margin:2.5rem 0 1rem;color:var(--accent2);border-bottom:1px solid var(--border);padding-bottom:.3rem}}
h3{{font-size:1.2rem;margin:1.8rem 0 .8rem}}
.meta{{color:var(--muted);font-size:.9rem;margin:.5rem 0 1.5rem}}
a{{color:var(--accent)}}pre{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1rem;overflow-x:auto;font-size:.9rem}}
code{{font-family:"SF Mono","Fira Code",monospace;font-size:.88em}}
pre code{{background:transparent;padding:0}}
table{{width:100%;border-collapse:collapse;margin:1.5rem 0}}
th,td{{padding:.6rem .8rem;border:1px solid var(--border);text-align:left}}
th{{background:var(--surface)}}tr:nth-child(even){{background:var(--surface)}}
.highlight{{background:var(--surface);border-left:3px solid var(--accent2);padding:1rem;margin:1rem 0;border-radius:0 8px 8px 0}}
blockquote{{border-left:3px solid var(--accent);padding:.5rem 1rem;margin:1rem 0;color:var(--muted)}}
footer{{margin-top:4rem;padding:2rem 0;border-top:1px solid var(--border);color:var(--muted);font-size:.85rem}}
</style>
</head>
<body>
<header>
<div class="container">
<div class="breadcrumb"><a href="index.html">NeuralDBG Blog</a> / {post_id}</div>
<h1>{title}</h1>
<div class="meta"><span>2026-07-04</span> · <span style="color:var(--accent2)">{bug_id}</span> · <span style="color:var(--success)">Gap: {gap}</span></div>
</div>
</header>
<article class="container">
{body}
<div class="highlight"><strong>Detected by <a href="https://github.com/LambdaSection/NeuralDBG">NeuralDBG</a></strong> - causal diagnostic engine for PyTorch training. <a href="index.html">All post-mortems</a></div>
</article>
<footer><div class="container"><p>MIT License · <a href="https://github.com/LambdaSection/NeuralDBG">GitHub</a></p></div></footer>
</body>
</html>'''
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'  {html_file}')


# Generate all 3
base = 'docs/blog'
posts = [
    (f'{base}/POST-006-svdvals-nan-swallowing.md', f'{base}/post-006-bug-006-postmortem.html',
     'POST-006', 'BUG-006', 'Silent NaN Swallowing: When svdvals Lies About Your Data', '+2 / PR #188053'),
    (f'{base}/POST-008-fnormalize-gradient-corruption.md', f'{base}/post-008-bug-008-postmortem.html',
     'POST-008', 'BUG-008', 'The Billion-Dollar Gradient: When F.normalize Silently Corrupts Your Weights', '+17 / PR #188066'),
    (f'{base}/POST-002-varlen-attn-nan.md', f'{base}/post-002-bug-002-postmortem.html',
     'POST-002', 'BUG-002', 'The Silent NaN Factory: varlen_attn and the Padding Problem', 'REAL FIX / PR #188933'),
]

for args in posts:
    md_to_html(*args)

print('Done.')
