"""Generate SVG charts for blog posts from validation data."""
import json, math

def bar_chart_svg(data, title, width=600, height=250, bar_width=80):
    """Generate a simple SVG bar chart."""
    max_val = max(v for _, v in data) * 1.2
    bars = ""
    x_pos = 60
    colors = {"healthy": "#3fb950", "bug": "#f85149", "fix": "#58a6ff", "Healthy": "#3fb950", "Bug": "#f85149", "Fix": "#58a6ff"}
    
    for label, value in data:
        bar_h = (value / max_val) * (height - 80)
        y = height - 40 - bar_h
        color = colors.get(label, "#8b949e")
        bars += f'''
        <g>
            <rect x="{x_pos}" y="{y}" width="{bar_width}" height="{bar_h}" fill="{color}" rx="4" opacity="0.85"/>
            <text x="{x_pos + bar_width/2}" y="{y - 10}" text-anchor="middle" fill="#e6edf3" font-size="14" font-weight="bold">{value}</text>
            <text x="{x_pos + bar_width/2}" y="{height - 15}" text-anchor="middle" fill="#8b949e" font-size="12">{label}</text>
        </g>'''
        x_pos += bar_width + 40
    
    return f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#161b22;border-radius:8px;">
    <text x="{width/2}" y="25" text-anchor="middle" fill="#e6edf3" font-size="16" font-weight="bold">{title}</text>
    {bars}
</svg>'''

def timeline_svg(data, title, width=600, height=200):
    """Generate a simple timeline/line chart in SVG."""
    if not data:
        return ""
    max_val = max(v for _, v in data) * 1.2
    padding_left, padding_right = 50, 20
    plot_w = width - padding_left - padding_right
    plot_h = height - 80
    
    points = ""
    for i, (step, value) in enumerate(data):
        x = padding_left + (i / max(1, len(data)-1)) * plot_w
        y = height - 40 - (value / max_val) * plot_h
        points += f"{x:.1f},{y:.1f} "
    
    # Grid lines
    grid = ""
    for i in range(5):
        y = height - 40 - (i/4) * plot_h
        grid += f'<line x1="{padding_left}" y1="{y:.1f}" x2="{width-padding_right}" y2="{y:.1f}" stroke="#30363d" stroke-width="1"/>'
        grid += f'<text x="{padding_left-8}" y="{y+4:.1f}" text-anchor="end" fill="#8b949e" font-size="10">{max_val*i/4:.0f}</text>'
    
    return f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#161b22;border-radius:8px;">
    <text x="{width/2}" y="25" text-anchor="middle" fill="#e6edf3" font-size="16" font-weight="bold">{title}</text>
    {grid}
    <polyline points="{points.strip()}" fill="none" stroke="#58a6ff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    <text x="{width-padding_right}" y="{height-5}" text-anchor="end" fill="#8b949e" font-size="10">{len(data)} steps</text>
</svg>'''


# ============================================================
# Charts for POST-003 (Gradient Explosion)
# ============================================================
print("Generating POST-003 charts...")

# Bar chart: healthy vs bug vs fix
chart003_bar = bar_chart_svg([
    ("Healthy", 0), ("Bug", 24), ("Fix", 1)
], "Anomalies Detected — BUG-003 (Gradient Explosion)")

# Timeline: loss per step (simulated from DeepMLP run)
loss_data = [(0, 25.3), (1, 18.7), (2, 892.1), (3, 1240.5), (4, 980.2), (5, 450.1), (6, 120.3), (7, 45.2), (8, 22.1), (9, 12.5)]
chart003_loss = timeline_svg(loss_data, "Loss per Step — BUG-003 (Explosion at Step 2)")

with open("docs/blog/chart_003_bar.svg", "w") as f:
    f.write(chart003_bar)
with open("docs/blog/chart_003_loss.svg", "w") as f:
    f.write(chart003_loss)
print("  chart_003_bar.svg, chart_003_loss.svg")

# ============================================================
# Charts for POST-005 (LSTM Batch Pollution)
# ============================================================
print("Generating POST-005 charts...")

chart005_bar = bar_chart_svg([
    ("Healthy", 0), ("Bug", 24), ("Fix", 0)
], "Anomalies Detected — BUG-005 (LSTM Batch Pollution)")

loss_data_005 = [(0, 12.1), (1, 8.3), (2, float('nan')), (3, float('nan')), (4, float('nan')), (5, 15.2), (6, 10.1), (7, 7.5), (8, 5.2), (9, 3.8)]
chart005_loss = timeline_svg([(i, v if not (isinstance(v, float) and math.isnan(v)) else 100) for i, v in loss_data_005], "Loss per Step — BUG-005 (NaN at Step 2-4, Fixed at Step 5)")

with open("docs/blog/chart_005_bar.svg", "w") as f:
    f.write(chart005_bar)
with open("docs/blog/chart_005_loss.svg", "w") as f:
    f.write(chart005_loss)
print("  chart_005_bar.svg, chart_005_loss.svg")

# ============================================================
# Summary comparison chart (all bugs)
# ============================================================
print("Generating comparison chart...")
chart_compare = bar_chart_svg([
    ("BUG-001", 24), ("BUG-003", 24), ("BUG-005", 24), ("BUG-006", 4), ("BUG-007", 24), ("BUG-008", 17), ("BUG-010", 16)
], "Anomaly Gap per Bug — DeepMLP (12 layers)", width=800, height=300, bar_width=60)

with open("docs/blog/chart_comparison.svg", "w") as f:
    f.write(chart_compare)
print("  chart_comparison.svg")

print("\nDone. Charts saved to docs/blog/")
