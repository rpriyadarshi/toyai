#!/usr/bin/env python3
"""
Generate proper SVG files for probability diagrams with actual graph rendering.
The diagram generator doesn't support graph components, so we create these manually.
"""

import math
from pathlib import Path

def create_bar_chart_svg(title, width, height, bars, x_label, y_label, note, output_path):
    """Create a bar chart SVG."""
    chart_width = width - 200
    chart_height = height - 150
    chart_x = 100
    chart_y = 50
    
    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <g transform="translate({chart_x}, {chart_y})">
    <text x="{chart_width/2}" y="-20" font-family="Arial" font-size="18" font-weight="bold" text-anchor="middle">{title}</text>
    <!-- Axes -->
    <line x1="0" y1="{chart_height}" x2="{chart_width}" y2="{chart_height}" stroke="#333" stroke-width="2"/>
    <line x1="0" y1="{chart_height}" x2="0" y2="0" stroke="#333" stroke-width="2"/>
    <!-- Axis labels -->
    <text x="{chart_width/2}" y="{chart_height + 35}" font-family="Arial" font-size="14" text-anchor="middle">{x_label}</text>
    <text x="-40" y="{chart_height/2}" font-family="Arial" font-size="14" text-anchor="middle" dominant-baseline="middle" transform="rotate(-90, -40, {chart_height/2})">{y_label}</text>
    <!-- Y-axis labels and grid -->
'''
    
    # Y-axis labels and grid lines
    max_val = max(b['value'] for b in bars) if bars else 1.0
    for i in range(6):
        y_val = i / 5.0
        y_pos = chart_height - (y_val / max_val) * chart_height
        label_y = chart_height - (y_val / max_val) * chart_height
        svg += f'    <line x1="0" y1="{y_pos}" x2="{chart_width}" y2="{y_pos}" stroke="#ccc" stroke-width="1" stroke-dasharray="2,2"/>\n'
        svg += f'    <text x="-10" y="{label_y + 5}" font-family="Arial" font-size="11">{y_val:.1f}</text>\n'
    
    # Bars
    bar_width = (chart_width - 20) / len(bars) - 20 if bars else 60
    for i, bar in enumerate(bars):
        bar_height = (bar['value'] / max_val) * chart_height
        bar_x = 20 + i * (bar_width + 20)
        bar_y = chart_height - bar_height
        svg += f'''    <!-- {bar['label']} -->
    <rect x="{bar_x}" y="{bar_y}" width="{bar_width}" height="{bar_height}" fill="{bar['color']}" opacity="0.8"/>
    <text x="{bar_x + bar_width/2}" y="{bar_y - 5}" font-family="Arial" font-size="12" text-anchor="middle">{bar['value']:.3f}</text>
    <text x="{bar_x + bar_width/2}" y="{chart_height + 20}" font-family="Arial" font-size="12" text-anchor="middle">{bar['label']}</text>
'''
    
    svg += f'''    <!-- Note -->
    <text x="{chart_width/2}" y="{chart_height + 55}" font-family="Arial" font-size="12" fill="#666" text-anchor="middle">{note}</text>
  </g>
</svg>'''
    
    Path(output_path).write_text(svg)
    print(f"Created {output_path}")

def create_step_function_svg(title, width, height, points, x_label, y_label, note, output_path):
    """Create a step function (CDF) SVG."""
    chart_width = width - 200
    chart_height = height - 150
    chart_x = 100
    chart_y = 50
    
    # Scale points to chart coordinates
    x_max = max(p[0] for p in points)
    y_max = max(p[1] for p in points)
    
    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <g transform="translate({chart_x}, {chart_y})">
    <text x="{chart_width/2}" y="-20" font-family="Arial" font-size="18" font-weight="bold" text-anchor="middle">{title}</text>
    <!-- Axes -->
    <line x1="0" y1="{chart_height}" x2="{chart_width}" y2="{chart_height}" stroke="#333" stroke-width="2"/>
    <line x1="0" y1="{chart_height}" x2="0" y2="0" stroke="#333" stroke-width="2"/>
    <!-- Axis labels -->
    <text x="{chart_width/2}" y="{chart_height + 35}" font-family="Arial" font-size="14" text-anchor="middle">{x_label}</text>
    <text x="-40" y="{chart_height/2}" font-family="Arial" font-size="14" text-anchor="middle" dominant-baseline="middle" transform="rotate(-90, -40, {chart_height/2})">{y_label}</text>
    <!-- Y-axis labels and grid -->
'''
    
    # Grid and labels
    for i in range(6):
        y_val = i / 5.0
        y_pos = chart_height - (y_val / y_max) * chart_height
        svg += f'    <line x1="0" y1="{y_pos}" x2="{chart_width}" y2="{y_pos}" stroke="#ccc" stroke-width="1" stroke-dasharray="2,2"/>\n'
        svg += f'    <text x="-10" y="{y_pos + 5}" font-family="Arial" font-size="11">{y_val:.2f}</text>\n'
    
    # Step function
    path_d = []
    for i, (x, y) in enumerate(points):
        x_scaled = (x / x_max) * chart_width
        y_scaled = chart_height - (y / y_max) * chart_height
        if i == 0:
            path_d.append(f"M {x_scaled} {y_scaled}")
        else:
            # Horizontal line to new x, then vertical to new y
            prev_x, prev_y = points[i-1]
            prev_x_scaled = (prev_x / x_max) * chart_width
            prev_y_scaled = chart_height - (prev_y / y_max) * chart_height
            path_d.append(f"H {x_scaled}")
            path_d.append(f"V {y_scaled}")
    
    svg += f'''    <!-- Step function -->
    <path d="{' '.join(path_d)}" stroke="#2563eb" stroke-width="3" fill="none"/>
'''
    
    # X-axis labels
    for x, y in points:
        x_scaled = (x / x_max) * chart_width
        svg += f'    <text x="{x_scaled}" y="{chart_height + 20}" font-family="Arial" font-size="11" text-anchor="middle">{x}</text>\n'
    
    svg += f'''    <!-- Note -->
    <text x="{chart_width/2}" y="{chart_height + 55}" font-family="Arial" font-size="12" fill="#666" text-anchor="middle">{note}</text>
  </g>
</svg>'''
    
    Path(output_path).write_text(svg)
    print(f"Created {output_path}")

# Create all diagrams
output_dir = Path("book/images/probability-statistics")
output_dir.mkdir(parents=True, exist_ok=True)

# PMF Biased Coin
create_bar_chart_svg(
    "Biased Coin PMF: P(Heads) = 0.7",
    500, 400,
    [
        {'label': 'Heads', 'value': 0.7, 'color': '#16a34a'},
        {'label': 'Tails', 'value': 0.3, 'color': '#dc2626'}
    ],
    "Outcome", "Probability",
    "P(Heads) = 0.7, P(Tails) = 0.3 | Sum = 1.0 ✓",
    output_dir / "probability-pmf-biased-coin.svg"
)

# CDF Fair Die
create_step_function_svg(
    "Fair Die CDF: P(X ≤ x)",
    600, 400,
    [(0, 0), (1, 0.167), (2, 0.333), (3, 0.5), (4, 0.667), (5, 0.833), (6, 1.0), (7, 1.0)],
    "x", "F(x) = P(X ≤ x)",
    "Step function: F(0)=0, F(1)=1/6, F(2)=2/6, ..., F(6)=1, F(7)=1",
    output_dir / "probability-cdf-fair-die.svg"
)

print("All diagrams generated!")
