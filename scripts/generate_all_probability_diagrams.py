#!/usr/bin/env python3
"""
Generate all probability diagram SVGs with proper graph rendering.
The diagram generator doesn't support graph components, so we create these manually.
"""

import math
from pathlib import Path

def create_bar_chart_svg(title, width, height, bars, x_label, y_label, note, output_path, y_max=None):
    """Create a bar chart SVG."""
    chart_width = width - 200
    chart_height = height - 150
    chart_x = 100
    chart_y = 50
    
    if y_max is None:
        y_max = max(b['value'] for b in bars) if bars else 1.0
    
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
    num_ticks = 6
    for i in range(num_ticks):
        y_val = (i / (num_ticks - 1)) * y_max
        y_pos = chart_height - (y_val / y_max) * chart_height
        svg += f'    <line x1="0" y1="{y_pos}" x2="{chart_width}" y2="{y_pos}" stroke="#ccc" stroke-width="1" stroke-dasharray="2,2"/>\n'
        if y_max <= 1.0:
            svg += f'    <text x="-15" y="{y_pos + 5}" font-family="Arial" font-size="11">{y_val:.2f}</text>\n'
        else:
            svg += f'    <text x="-20" y="{y_pos + 5}" font-family="Arial" font-size="11">{y_val:.1f}</text>\n'
    
    # Bars
    if bars:
        bar_width = (chart_width - 20) / len(bars) - 20 if len(bars) > 1 else chart_width - 40
        for i, bar in enumerate(bars):
            bar_height = (bar['value'] / y_max) * chart_height
            bar_x = 20 + i * (bar_width + 20) if len(bars) > 1 else 20
            bar_y = chart_height - bar_height
            # Place value label inside bar if tall enough, otherwise above (but not negative)
            if bar_height > 20:
                value_label_y = bar_y + bar_height / 2
            else:
                value_label_y = max(5, bar_y - 5)
            svg += f'''    <!-- {bar['label']} -->
    <rect x="{bar_x}" y="{bar_y}" width="{bar_width}" height="{bar_height}" fill="{bar['color']}" opacity="0.8"/>
    <text x="{bar_x + bar_width/2}" y="{value_label_y}" font-family="Arial" font-size="12" text-anchor="middle" fill="{'white' if bar_height > 20 else '#1e293b'}">{bar['value']:.3f}</text>
    <text x="{bar_x + bar_width/2}" y="{chart_height + 20}" font-family="Arial" font-size="12" text-anchor="middle">{bar['label']}</text>
'''
    
    svg += f'''    <!-- Note -->
    <text x="{chart_width/2}" y="{chart_height + 55}" font-family="Arial" font-size="12" fill="#666" text-anchor="middle">{note}</text>
  </g>
</svg>'''
    
    Path(output_path).write_text(svg)
    print(f"Created {output_path}")

def create_bar_chart_with_line_svg(title, width, height, bars, line_y, line_label, x_label, y_label, note, output_path, y_max=None):
    """Create a bar chart SVG with a horizontal reference line."""
    chart_width = width - 200
    chart_height = height - 150
    chart_x = 100
    chart_y = 50
    
    if y_max is None:
        y_max = max(b['value'] for b in bars) if bars else 1.0
    
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
    num_ticks = 6
    for i in range(num_ticks):
        y_val = (i / (num_ticks - 1)) * y_max
        y_pos = chart_height - (y_val / y_max) * chart_height
        svg += f'    <line x1="0" y1="{y_pos}" x2="{chart_width}" y2="{y_pos}" stroke="#ccc" stroke-width="1" stroke-dasharray="2,2"/>\n'
        svg += f'    <text x="-20" y="{y_pos + 5}" font-family="Arial" font-size="11">{y_val:.2f}</text>\n'
    
    # Reference line
    if line_y is not None:
        line_y_pos = chart_height - (line_y / y_max) * chart_height
        svg += f'''    <!-- Reference line -->
    <line x1="0" y1="{line_y_pos}" x2="{chart_width}" y2="{line_y_pos}" stroke="#dc2626" stroke-width="2" stroke-dasharray="4,4"/>
    <text x="{chart_width - 80}" y="{line_y_pos - 5}" font-family="Arial" font-size="12" fill="#dc2626" text-anchor="end">{line_label}</text>
'''
    
    # Bars
    if bars:
        bar_width = (chart_width - 20) / len(bars) - 20 if len(bars) > 1 else chart_width - 40
        for i, bar in enumerate(bars):
            bar_height = (bar['value'] / y_max) * chart_height
            bar_x = 20 + i * (bar_width + 20) if len(bars) > 1 else 20
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
            prev_x, prev_y = points[i-1]
            prev_x_scaled = (prev_x / x_max) * chart_width
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

def create_normal_distribution_svg(title, width, height, x_range, y_range, mean, std, note, output_path):
    """Create a normal distribution bell curve SVG."""
    chart_width = width - 200
    chart_height = height - 150
    chart_x = 100
    chart_y = 50
    
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # Generate bell curve points
    def normal_pdf(x, mu, sigma):
        return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    points = []
    num_points = 100
    for i in range(num_points + 1):
        x = x_min + (x_max - x_min) * i / num_points
        y = normal_pdf(x, mean, std)
        points.append((x, y))
    
    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <g transform="translate({chart_x}, {chart_y})">
    <text x="{chart_width/2}" y="-20" font-family="Arial" font-size="18" font-weight="bold" text-anchor="middle">{title}</text>
    <!-- Axes -->
    <line x1="0" y1="{chart_height}" x2="{chart_width}" y2="{chart_height}" stroke="#333" stroke-width="2"/>
    <line x1="0" y1="{chart_height}" x2="0" y2="0" stroke="#333" stroke-width="2"/>
    <!-- Axis labels -->
    <text x="{chart_width/2}" y="{chart_height + 35}" font-family="Arial" font-size="14" text-anchor="middle">x</text>
    <text x="-40" y="{chart_height/2}" font-family="Arial" font-size="14" text-anchor="middle" dominant-baseline="middle" transform="rotate(-90, -40, {chart_height/2})">f(x)</text>
    <!-- Y-axis labels and grid -->
'''
    
    # Grid and labels
    for i in range(6):
        y_val = y_min + (y_max - y_min) * i / 5.0
        y_pos = chart_height - ((y_val - y_min) / (y_max - y_min)) * chart_height
        svg += f'    <line x1="0" y1="{y_pos}" x2="{chart_width}" y2="{y_pos}" stroke="#ccc" stroke-width="1" stroke-dasharray="2,2"/>\n'
        svg += f'    <text x="-20" y="{y_pos + 5}" font-family="Arial" font-size="11">{y_val:.2f}</text>\n'
    
    # Shaded regions (68%, 95%, 99.7%)
    # 68% region (-1σ to +1σ)
    x_68_min = mean - std
    x_68_max = mean + std
    region_68_points = []
    for i in range(50):
        x = x_68_min + (x_68_max - x_68_min) * i / 49
        y = normal_pdf(x, mean, std)
        x_scaled = ((x - x_min) / (x_max - x_min)) * chart_width
        y_scaled = chart_height - ((y - y_min) / (y_max - y_min)) * chart_height
        region_68_points.append(f"{x_scaled},{y_scaled}")
    x_68_min_scaled = ((x_68_min - x_min) / (x_max - x_min)) * chart_width
    x_68_max_scaled = ((x_68_max - x_min) / (x_max - x_min)) * chart_width
    svg += f'''    <!-- 68% region -->
    <polygon points="{x_68_min_scaled},{chart_height} {' '.join(region_68_points)} {x_68_max_scaled},{chart_height}" fill="#16a34a" opacity="0.3"/>
'''
    
    # Bell curve
    path_d = "M "
    for x, y in points:
        x_scaled = ((x - x_min) / (x_max - x_min)) * chart_width
        y_scaled = chart_height - ((y - y_min) / (y_max - y_min)) * chart_height
        path_d += f"{x_scaled},{y_scaled} "
    
    svg += f'''    <!-- Bell curve -->
    <path d="{path_d}" stroke="#2563eb" stroke-width="3" fill="none"/>
'''
    
    # Mean line
    mean_x_scaled = ((mean - x_min) / (x_max - x_min)) * chart_width
    svg += f'''    <!-- Mean line -->
    <line x1="{mean_x_scaled}" y1="0" x2="{mean_x_scaled}" y2="{chart_height}" stroke="#dc2626" stroke-width="2" stroke-dasharray="4,4"/>
    <text x="{mean_x_scaled + 5}" y="15" font-family="Arial" font-size="12" fill="#dc2626">μ = {mean}</text>
'''
    
    svg += f'''    <!-- Note -->
    <text x="{chart_width/2}" y="{chart_height + 55}" font-family="Arial" font-size="12" fill="#666" text-anchor="middle">{note}</text>
  </g>
</svg>'''
    
    Path(output_path).write_text(svg)
    print(f"Created {output_path}")

def create_side_by_side_bar_charts_svg(title, width, height, left_title, left_bars, right_title, right_bars, note, output_path):
    """Create side-by-side bar charts SVG."""
    panel_width = (width - 100) / 2
    panel_height = height - 150
    left_x = 50
    right_x = left_x + panel_width + 50
    chart_y = 50
    
    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <text x="{width/2}" y="30" font-family="Arial" font-size="18" font-weight="bold" text-anchor="middle">{title}</text>
  
  <!-- Left panel -->
  <g transform="translate({left_x}, {chart_y})">
    <text x="{panel_width/2}" y="-20" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle">{left_title}</text>
    <rect x="0" y="0" width="{panel_width}" height="{panel_height}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2" rx="6"/>
    <text x="{panel_width/2}" y="25" font-family="Arial" font-size="12" fill="#64748b" text-anchor="middle">High Entropy</text>
'''
    
    # Left chart
    chart_w = panel_width - 80
    chart_h = panel_height - 80
    chart_x_offset = 40
    chart_y_offset = 50
    
    max_val = max(b['value'] for b in left_bars) if left_bars else 1.0
    svg += f'''    <!-- Left axes -->
    <line x1="{chart_x_offset}" y1="{chart_y_offset + chart_h}" x2="{chart_x_offset + chart_w}" y2="{chart_y_offset + chart_h}" stroke="#333" stroke-width="2"/>
    <line x1="{chart_x_offset}" y1="{chart_y_offset + chart_h}" x2="{chart_x_offset}" y2="{chart_y_offset}" stroke="#333" stroke-width="2"/>
'''
    
    # Grid
    for i in range(6):
        y_val = i / 5.0
        y_pos = chart_y_offset + chart_h - (y_val / max_val) * chart_h
        svg += f'    <line x1="{chart_x_offset}" y1="{y_pos}" x2="{chart_x_offset + chart_w}" y2="{y_pos}" stroke="#ccc" stroke-width="1" stroke-dasharray="2,2"/>\n'
    
    # Left bars
    if left_bars:
        bar_width = (chart_w - 20) / len(left_bars) - 10
        for i, bar in enumerate(left_bars):
            bar_height = (bar['value'] / max_val) * chart_h
            bar_x = chart_x_offset + 10 + i * (bar_width + 10)
            bar_y = chart_y_offset + chart_h - bar_height
            # Place value label inside bar if tall enough, otherwise above (but not negative)
            if bar_height > 18:
                value_label_y = bar_y + bar_height / 2
            else:
                value_label_y = max(chart_y_offset + 5, bar_y - 5)
            svg += f'''    <rect x="{bar_x}" y="{bar_y}" width="{bar_width}" height="{bar_height}" fill="{bar['color']}" opacity="0.8"/>
    <text x="{bar_x + bar_width/2}" y="{value_label_y}" font-family="Arial" font-size="11" text-anchor="middle" fill="{'white' if bar_height > 18 else '#1e293b'}">{bar['value']:.2f}</text>
    <text x="{bar_x + bar_width/2}" y="{chart_y_offset + chart_h + 15}" font-family="Arial" font-size="11" text-anchor="middle">{bar['label']}</text>
'''
    
    svg += f'''    <text x="{panel_width/2}" y="{panel_height - 10}" font-family="Arial" font-size="12" fill="#dc2626" text-anchor="middle" font-weight="bold">H(P) = 0.693</text>
  </g>
  
  <!-- Right panel -->
  <g transform="translate({right_x}, {chart_y})">
    <text x="{panel_width/2}" y="-20" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle">{right_title}</text>
    <rect x="0" y="0" width="{panel_width}" height="{panel_height}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2" rx="6"/>
    <text x="{panel_width/2}" y="25" font-family="Arial" font-size="12" fill="#64748b" text-anchor="middle">Low Entropy</text>
'''
    
    # Right chart
    max_val = max(b['value'] for b in right_bars) if right_bars else 1.0
    svg += f'''    <!-- Right axes -->
    <line x1="{chart_x_offset}" y1="{chart_y_offset + chart_h}" x2="{chart_x_offset + chart_w}" y2="{chart_y_offset + chart_h}" stroke="#333" stroke-width="2"/>
    <line x1="{chart_x_offset}" y1="{chart_y_offset + chart_h}" x2="{chart_x_offset}" y2="{chart_y_offset}" stroke="#333" stroke-width="2"/>
'''
    
    # Grid
    for i in range(6):
        y_val = i / 5.0
        y_pos = chart_y_offset + chart_h - (y_val / max_val) * chart_h
        svg += f'    <line x1="{chart_x_offset}" y1="{y_pos}" x2="{chart_x_offset + chart_w}" y2="{y_pos}" stroke="#ccc" stroke-width="1" stroke-dasharray="2,2"/>\n'
    
    # Right bars
    if right_bars:
        bar_width = (chart_w - 20) / len(right_bars) - 10
        for i, bar in enumerate(right_bars):
            bar_height = (bar['value'] / max_val) * chart_h
            bar_x = chart_x_offset + 10 + i * (bar_width + 10)
            bar_y = chart_y_offset + chart_h - bar_height
            # Place value label inside bar if tall enough, otherwise above (but not negative)
            if bar_height > 18:
                value_label_y = bar_y + bar_height / 2
            else:
                value_label_y = max(chart_y_offset + 5, bar_y - 5)
            svg += f'''    <rect x="{bar_x}" y="{bar_y}" width="{bar_width}" height="{bar_height}" fill="{bar['color']}" opacity="0.8"/>
    <text x="{bar_x + bar_width/2}" y="{value_label_y}" font-family="Arial" font-size="11" text-anchor="middle" fill="{'white' if bar_height > 18 else '#1e293b'}">{bar['value']:.2f}</text>
    <text x="{bar_x + bar_width/2}" y="{chart_y_offset + chart_h + 15}" font-family="Arial" font-size="11" text-anchor="middle">{bar['label']}</text>
'''
    
    svg += f'''    <text x="{panel_width/2}" y="{panel_height - 10}" font-family="Arial" font-size="12" fill="#dc2626" text-anchor="middle" font-weight="bold">H(P) = 0.056</text>
  </g>
  
  <!-- Note -->
  <text x="{width/2}" y="{height - 20}" font-family="Arial" font-size="12" fill="#666" text-anchor="middle">{note}</text>
</svg>'''
    
    Path(output_path).write_text(svg)
    print(f"Created {output_path}")

def create_cross_entropy_svg(title, width, height, left_title, left_bars, right_title, right_bars, cross_entropy_val, note, output_path):
    """Create cross-entropy comparison SVG."""
    panel_width = (width - 100) / 2
    panel_height = height - 150
    left_x = 50
    right_x = left_x + panel_width + 50
    chart_y = 50
    
    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <text x="{width/2}" y="30" font-family="Arial" font-size="18" font-weight="bold" text-anchor="middle">{title}</text>
  
  <!-- Left panel -->
  <g transform="translate({left_x}, {chart_y})">
    <text x="{panel_width/2}" y="-20" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle">{left_title}</text>
    <rect x="0" y="0" width="{panel_width}" height="{panel_height}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2" rx="6"/>
    <text x="{panel_width/2}" y="25" font-family="Arial" font-size="12" fill="#64748b" text-anchor="middle">One-Hot: [0, 0, 1, 0]</text>
'''
    
    # Left chart
    chart_w = panel_width - 80
    chart_h = panel_height - 80
    chart_x_offset = 40
    chart_y_offset = 50
    
    max_val = 1.0
    svg += f'''    <!-- Left axes -->
    <line x1="{chart_x_offset}" y1="{chart_y_offset + chart_h}" x2="{chart_x_offset + chart_w}" y2="{chart_y_offset + chart_h}" stroke="#333" stroke-width="2"/>
    <line x1="{chart_x_offset}" y1="{chart_y_offset + chart_h}" x2="{chart_x_offset}" y2="{chart_y_offset}" stroke="#333" stroke-width="2"/>
    <text x="{chart_x_offset + chart_w/2}" y="{chart_y_offset + chart_h + 25}" font-family="Arial" font-size="12" text-anchor="middle">Class</text>
    <text x="{chart_x_offset - 25}" y="{chart_y_offset + chart_h/2}" font-family="Arial" font-size="12" text-anchor="middle" dominant-baseline="middle" transform="rotate(-90, {chart_x_offset - 25}, {chart_y_offset + chart_h/2})">Probability</text>
'''
    
    # Grid
    for i in range(6):
        y_val = i / 5.0
        y_pos = chart_y_offset + chart_h - (y_val / max_val) * chart_h
        svg += f'    <line x1="{chart_x_offset}" y1="{y_pos}" x2="{chart_x_offset + chart_w}" y2="{y_pos}" stroke="#ccc" stroke-width="1" stroke-dasharray="2,2"/>\n'
    
    # Left bars
    if left_bars:
        bar_width = (chart_w - 20) / len(left_bars) - 5
        for i, bar in enumerate(left_bars):
            bar_height = (bar['value'] / max_val) * chart_h
            bar_x = chart_x_offset + 10 + i * (bar_width + 5)
            bar_y = chart_y_offset + chart_h - bar_height
            # Place value label inside bar if tall enough, otherwise above (but not negative)
            if bar_height > 18:
                value_label_y = bar_y + bar_height / 2
            else:
                value_label_y = max(chart_y_offset + 5, bar_y - 5)
            svg += f'''    <rect x="{bar_x}" y="{bar_y}" width="{bar_width}" height="{bar_height}" fill="{bar['color']}" opacity="0.8"/>
    <text x="{bar_x + bar_width/2}" y="{value_label_y}" font-family="Arial" font-size="11" text-anchor="middle" fill="{'white' if bar_height > 18 else '#1e293b'}">{bar['value']:.1f}</text>
    <text x="{bar_x + bar_width/2}" y="{chart_y_offset + chart_h + 15}" font-family="Arial" font-size="11" text-anchor="middle">{bar['label']}</text>
'''
    
    svg += f'''  </g>
  
  <!-- Right panel -->
  <g transform="translate({right_x}, {chart_y})">
    <text x="{panel_width/2}" y="-20" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle">{right_title}</text>
    <rect x="0" y="0" width="{panel_width}" height="{panel_height}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2" rx="6"/>
    <text x="{panel_width/2}" y="25" font-family="Arial" font-size="12" fill="#64748b" text-anchor="middle">[0.1, 0.2, 0.6, 0.1]</text>
'''
    
    # Right chart
    svg += f'''    <!-- Right axes -->
    <line x1="{chart_x_offset}" y1="{chart_y_offset + chart_h}" x2="{chart_x_offset + chart_w}" y2="{chart_y_offset + chart_h}" stroke="#333" stroke-width="2"/>
    <line x1="{chart_x_offset}" y1="{chart_y_offset + chart_h}" x2="{chart_x_offset}" y2="{chart_y_offset}" stroke="#333" stroke-width="2"/>
    <text x="{chart_x_offset + chart_w/2}" y="{chart_y_offset + chart_h + 25}" font-family="Arial" font-size="12" text-anchor="middle">Class</text>
    <text x="{chart_x_offset - 25}" y="{chart_y_offset + chart_h/2}" font-family="Arial" font-size="12" text-anchor="middle" dominant-baseline="middle" transform="rotate(-90, {chart_x_offset - 25}, {chart_y_offset + chart_h/2})">Probability</text>
'''
    
    # Grid
    for i in range(6):
        y_val = i / 5.0
        y_pos = chart_y_offset + chart_h - (y_val / max_val) * chart_h
        svg += f'    <line x1="{chart_x_offset}" y1="{y_pos}" x2="{chart_x_offset + chart_w}" y2="{y_pos}" stroke="#ccc" stroke-width="1" stroke-dasharray="2,2"/>\n'
    
    # Right bars
    if right_bars:
        bar_width = (chart_w - 20) / len(right_bars) - 5
        for i, bar in enumerate(right_bars):
            bar_height = (bar['value'] / max_val) * chart_h
            bar_x = chart_x_offset + 10 + i * (bar_width + 5)
            bar_y = chart_y_offset + chart_h - bar_height
            # Place value label inside bar if tall enough, otherwise above (but not negative)
            if bar_height > 18:
                value_label_y = bar_y + bar_height / 2
            else:
                value_label_y = max(chart_y_offset + 5, bar_y - 5)
            svg += f'''    <rect x="{bar_x}" y="{bar_y}" width="{bar_width}" height="{bar_height}" fill="{bar['color']}" opacity="0.8"/>
    <text x="{bar_x + bar_width/2}" y="{value_label_y}" font-family="Arial" font-size="11" text-anchor="middle" fill="{'white' if bar_height > 18 else '#1e293b'}">{bar['value']:.2f}</text>
    <text x="{bar_x + bar_width/2}" y="{chart_y_offset + chart_h + 15}" font-family="Arial" font-size="11" text-anchor="middle">{bar['label']}</text>
'''
    
    svg += f'''  </g>
  
  <!-- Cross-entropy label -->
  <text x="{width/2}" y="{chart_y + panel_height + 20}" font-family="Arial" font-size="14" fill="#dc2626" text-anchor="middle" font-weight="bold">H(P, Q) = {cross_entropy_val:.3f} (Cross-Entropy Loss)</text>
  
  <!-- Note -->
  <text x="{width/2}" y="{height - 20}" font-family="Arial" font-size="12" fill="#666" text-anchor="middle">{note}</text>
</svg>'''
    
    Path(output_path).write_text(svg)
    print(f"Created {output_path}")

def create_zscore_svg(title, width, height, note, output_path):
    """Create z-score normalization before/after SVG."""
    panel_width = width - 100
    panel_height = (height - 200) / 2
    panel_x = 50
    top_y = 50
    bottom_y = top_y + panel_height + 100
    
    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <text x="{width/2}" y="30" font-family="Arial" font-size="18" font-weight="bold" text-anchor="middle">{title}</text>
  
  <!-- Top panel: Original -->
  <g transform="translate({panel_x}, {top_y})">
    <rect x="0" y="0" width="{panel_width}" height="{panel_height}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2" rx="6"/>
    <text x="{panel_width/2}" y="25" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle">Original Distribution</text>
    <text x="{panel_width/2}" y="45" font-family="Arial" font-size="12" fill="#64748b" text-anchor="middle">Mean μ, Standard Deviation σ</text>
    
    <!-- Chart -->
    <g transform="translate(50, 70)">
      <line x1="0" y1="{panel_height - 100}" x2="{panel_width - 100}" y2="{panel_height - 100}" stroke="#333" stroke-width="2"/>
      <line x1="0" y1="{panel_height - 100}" x2="0" y2="0" stroke="#333" stroke-width="2"/>
      <text x="{(panel_width - 100)/2}" y="{panel_height - 70}" font-family="Arial" font-size="12" text-anchor="middle">Value</text>
      <text x="-25" y="{(panel_height - 100)/2}" font-family="Arial" font-size="12" text-anchor="middle" dominant-baseline="middle" transform="rotate(-90, -25, {(panel_height - 100)/2})">Frequency</text>
      
      <!-- Distribution curve (simplified bell-like) -->
      <path d="M 50,{panel_height - 100} Q 150,{panel_height - 150} 250,{panel_height - 120} T 450,{panel_height - 100}" 
            stroke="#2563eb" stroke-width="2" fill="none"/>
      
      <!-- Mean line -->
      <line x1="250" y1="0" x2="250" y2="{panel_height - 100}" stroke="#dc2626" stroke-width="2" stroke-dasharray="4,4"/>
      <text x="255" y="15" font-family="Arial" font-size="11" fill="#dc2626">μ</text>
    </g>
  </g>
  
  <!-- Arrow -->
  <g transform="translate({width/2 - 50}, {top_y + panel_height + 20})">
    <text x="50" y="15" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle">z = (x - μ) / σ</text>
    <path d="M 50,25 L 50,35" stroke="#2563eb" stroke-width="2" marker-end="url(#arrow-down)"/>
  </g>
  <defs>
    <marker id="arrow-down" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
      <polygon points="0,0 8,8 0,8" fill="#2563eb"/>
    </marker>
  </defs>
  
  <!-- Bottom panel: Normalized -->
  <g transform="translate({panel_x}, {bottom_y})">
    <rect x="0" y="0" width="{panel_width}" height="{panel_height}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2" rx="6"/>
    <text x="{panel_width/2}" y="25" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle">Normalized Distribution</text>
    <text x="{panel_width/2}" y="45" font-family="Arial" font-size="12" fill="#64748b" text-anchor="middle">Mean 0, Standard Deviation 1</text>
    
    <!-- Chart -->
    <g transform="translate(50, 70)">
      <line x1="0" y1="{panel_height - 100}" x2="{panel_width - 100}" y2="{panel_height - 100}" stroke="#333" stroke-width="2"/>
      <line x1="0" y1="{panel_height - 100}" x2="0" y2="0" stroke="#333" stroke-width="2"/>
      <text x="{(panel_width - 100)/2}" y="{panel_height - 70}" font-family="Arial" font-size="12" text-anchor="middle">Z-Score</text>
      <text x="-25" y="{(panel_height - 100)/2}" font-family="Arial" font-size="12" text-anchor="middle" dominant-baseline="middle" transform="rotate(-90, -25, {(panel_height - 100)/2})">Frequency</text>
      
      <!-- Distribution curve centered at 0 -->
      <path d="M 50,{panel_height - 100} Q 150,{panel_height - 150} 250,{panel_height - 120} T 450,{panel_height - 100}" 
            stroke="#16a34a" stroke-width="2" fill="none"/>
      
      <!-- Mean line at 0 -->
      <line x1="250" y1="0" x2="250" y2="{panel_height - 100}" stroke="#dc2626" stroke-width="2" stroke-dasharray="4,4"/>
      <text x="255" y="15" font-family="Arial" font-size="11" fill="#dc2626">μ = 0</text>
    </g>
  </g>
  
  <!-- Note -->
  <text x="{width/2}" y="{height - 20}" font-family="Arial" font-size="12" fill="#666" text-anchor="middle">{note}</text>
</svg>'''
    
    Path(output_path).write_text(svg)
    print(f"Created {output_path}")

def create_sample_space_svg(title, width, height, note, output_path):
    """Create sample space and events visualization."""
    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <g transform="translate(50, 50)">
    <text x="{width/2 - 100}" y="-20" font-family="Arial" font-size="18" font-weight="bold" text-anchor="middle">{title}</text>
    
    <!-- Sample space box -->
    <rect x="0" y="0" width="{width - 100}" height="{height - 150}" fill="#f8fafc" stroke="#64748b" stroke-width="2" stroke-dasharray="4,4" rx="4"/>
    <text x="{(width - 100)/2}" y="20" font-family="Arial" font-size="14" font-weight="bold" fill="#1e293b">Sample Space Ω = {1, 2, 3, 4, 5, 6}</text>
    
    <!-- Outcomes -->
    <g transform="translate(50, 70)">
      <rect x="0" y="0" width="60" height="50" fill="#e1f5ff" stroke="#2563eb" stroke-width="2" rx="6"/>
      <text x="30" y="30" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle" fill="#1e293b">1</text>
      
      <rect x="100" y="0" width="60" height="50" fill="#c8e6c9" stroke="#16a34a" stroke-width="2" rx="6"/>
      <text x="130" y="30" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle" fill="#1e293b">2</text>
      <text x="130" y="45" font-family="Arial" font-size="11" text-anchor="middle" fill="#16a34a">E</text>
      
      <rect x="200" y="0" width="60" height="50" fill="#e1f5ff" stroke="#2563eb" stroke-width="2" rx="6"/>
      <text x="230" y="30" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle" fill="#1e293b">3</text>
      
      <rect x="300" y="0" width="60" height="50" fill="#c8e6c9" stroke="#16a34a" stroke-width="2" rx="6"/>
      <text x="330" y="30" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle" fill="#1e293b">4</text>
      <text x="330" y="45" font-family="Arial" font-size="11" text-anchor="middle" fill="#16a34a">E</text>
      
      <rect x="400" y="0" width="60" height="50" fill="#fff4e1" stroke="#f59e0b" stroke-width="2" rx="6"/>
      <text x="430" y="30" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle" fill="#1e293b">5</text>
      <text x="430" y="45" font-family="Arial" font-size="11" text-anchor="middle" fill="#f59e0b">F</text>
      
      <rect x="500" y="0" width="60" height="50" fill="#f3e5f5" stroke="#9333ea" stroke-width="2" rx="6"/>
      <text x="530" y="30" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle" fill="#1e293b">6</text>
      <text x="530" y="45" font-family="Arial" font-size="11" text-anchor="middle" fill="#9333ea">E, F</text>
    </g>
    
    <!-- Labels -->
    <g transform="translate(50, 170)">
      <rect x="0" y="0" width="200" height="50" fill="#c8e6c9" stroke="#16a34a" stroke-width="2" rx="6"/>
      <text x="100" y="25" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="#1e293b">Event E = {2, 4, 6}</text>
      <text x="100" y="40" font-family="Arial" font-size="11" text-anchor="middle" fill="#64748b">Even numbers</text>
      
      <rect x="300" y="0" width="200" height="50" fill="#fff4e1" stroke="#f59e0b" stroke-width="2" rx="6"/>
      <text x="400" y="25" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="#1e293b">Event F = {5, 6}</text>
      <text x="400" y="40" font-family="Arial" font-size="11" text-anchor="middle" fill="#64748b">Greater than 4</text>
    </g>
    
    <g transform="translate(200, 250)">
      <rect x="0" y="0" width="200" height="50" fill="#f3e5f5" stroke="#9333ea" stroke-width="2" rx="6"/>
      <text x="100" y="25" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="#1e293b">E ∩ F = {6}</text>
      <text x="100" y="40" font-family="Arial" font-size="11" text-anchor="middle" fill="#64748b">Intersection</text>
    </g>
    
    <!-- Note -->
    <text x="{(width - 100)/2}" y="{height - 100}" font-family="Arial" font-size="12" fill="#666" text-anchor="middle">{note}</text>
  </g>
</svg>'''
    
    Path(output_path).write_text(svg)
    print(f"Created {output_path}")

def create_conditional_svg(title, width, height, note, output_path):
    """Create conditional probability visualization."""
    panel_width = (width - 100) / 2
    panel_height = height - 150
    left_x = 50
    right_x = left_x + panel_width + 50
    chart_y = 50
    
    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <text x="{width/2}" y="30" font-family="Arial" font-size="18" font-weight="bold" text-anchor="middle">{title}</text>
  
  <!-- Left panel -->
  <g transform="translate({left_x}, {chart_y})">
    <rect x="0" y="0" width="{panel_width}" height="{panel_height}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2" rx="6"/>
    <text x="{panel_width/2}" y="25" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle">Full Sample Space</text>
    <text x="{panel_width/2}" y="45" font-family="Arial" font-size="12" fill="#64748b" text-anchor="middle">Ω = {1, 2, 3, 4, 5, 6}</text>
    
    <!-- Events -->
    <g transform="translate(50, 100)">
      <rect x="0" y="0" width="80" height="50" fill="#c8e6c9" stroke="#16a34a" stroke-width="2" rx="6"/>
      <text x="40" y="30" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle">A = {2, 4, 6}</text>
      <text x="40" y="45" font-family="Arial" font-size="10" text-anchor="middle">Even</text>
      
      <rect x="100" y="0" width="80" height="50" fill="#fff4e1" stroke="#f59e0b" stroke-width="2" rx="6"/>
      <text x="140" y="30" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle">B = {4, 5, 6}</text>
      <text x="140" y="45" font-family="Arial" font-size="10" text-anchor="middle">> 3</text>
      
      <rect x="50" y="70" width="80" height="50" fill="#f3e5f5" stroke="#9333ea" stroke-width="2" rx="6"/>
      <text x="90" y="95" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle">A ∩ B = {4, 6}</text>
      <text x="90" y="110" font-family="Arial" font-size="10" text-anchor="middle">Intersection</text>
    </g>
  </g>
  
  <!-- Arrow -->
  <path d="M {left_x + panel_width + 10},{chart_y + panel_height/2} L {right_x - 10},{chart_y + panel_height/2}" 
        stroke="#2563eb" stroke-width="2" marker-end="url(#arrow-right)"/>
  <defs>
    <marker id="arrow-right" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <polygon points="0,0 8,3 0,6" fill="#2563eb"/>
    </marker>
  </defs>
  
  <!-- Right panel -->
  <g transform="translate({right_x}, {chart_y})">
    <rect x="0" y="0" width="{panel_width}" height="{panel_height}" fill="#fff4e1" stroke="#f59e0b" stroke-width="2" rx="6" opacity="0.3"/>
    <rect x="0" y="0" width="{panel_width}" height="{panel_height}" fill="none" stroke="#f59e0b" stroke-width="2" rx="6"/>
    <text x="{panel_width/2}" y="25" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle">Conditioned on B</text>
    <text x="{panel_width/2}" y="45" font-family="Arial" font-size="12" fill="#64748b" text-anchor="middle">Restricted to B = {4, 5, 6}</text>
    
    <!-- Restricted events -->
    <g transform="translate(50, 100)">
      <rect x="50" y="0" width="80" height="50" fill="#fff4e1" stroke="#f59e0b" stroke-width="2" rx="6"/>
      <text x="90" y="30" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle">B = {4, 5, 6}</text>
      <text x="90" y="45" font-family="Arial" font-size="10" text-anchor="middle">Shaded</text>
      
      <rect x="50" y="70" width="80" height="50" fill="#f3e5f5" stroke="#9333ea" stroke-width="2" rx="6"/>
      <text x="90" y="95" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle">A|B = {4, 6}</text>
      <text x="90" y="110" font-family="Arial" font-size="10" text-anchor="middle">P(A|B) = 2/3</text>
    </g>
  </g>
  
  <!-- Note -->
  <text x="{width/2}" y="{height - 20}" font-family="Arial" font-size="12" fill="#666" text-anchor="middle">{note}</text>
</svg>'''
    
    Path(output_path).write_text(svg)
    print(f"Created {output_path}")

# Create all diagrams
output_dir = Path("book/images/probability-statistics")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Sample Space and Events
create_sample_space_svg(
    "Sample Space and Events: Die Roll Example",
    700, 400,
    "Ω = {1, 2, 3, 4, 5, 6} | E = {2, 4, 6} (even) | F = {5, 6} (>4) | E ∩ F = {6}",
    output_dir / "probability-sample-space-events.svg"
)

# 2. Conditional Probability
create_conditional_svg(
    "Conditional Probability: P(A|B) = P(A ∩ B) / P(B)",
    800, 400,
    "Left: Full space with A and B | Right: Restricted to B, showing P(A|B) = 2/3",
    output_dir / "probability-conditional.svg"
)

# 3. PMF Fair Die
create_bar_chart_svg(
    "Fair Die PMF: Uniform Distribution",
    600, 400,
    [
        {'label': '1', 'value': 0.167, 'color': '#2563eb'},
        {'label': '2', 'value': 0.167, 'color': '#2563eb'},
        {'label': '3', 'value': 0.167, 'color': '#2563eb'},
        {'label': '4', 'value': 0.167, 'color': '#2563eb'},
        {'label': '5', 'value': 0.167, 'color': '#2563eb'},
        {'label': '6', 'value': 0.167, 'color': '#2563eb'}
    ],
    "Outcome", "Probability",
    "P(X = i) = 1/6 ≈ 0.167 for all i ∈ {1, 2, 3, 4, 5, 6}",
    output_dir / "probability-pmf-fair-die.svg"
)

# 4. PMF Biased Coin
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

# 5. CDF Fair Die
create_step_function_svg(
    "Fair Die CDF: P(X ≤ x)",
    600, 400,
    [(0, 0), (1, 0.167), (2, 0.333), (3, 0.5), (4, 0.667), (5, 0.833), (6, 1.0), (7, 1.0)],
    "x", "F(x) = P(X ≤ x)",
    "Step function: F(0)=0, F(1)=1/6, F(2)=2/6, ..., F(6)=1, F(7)=1",
    output_dir / "probability-cdf-fair-die.svg"
)

# 6. Expected Value
create_bar_chart_with_line_svg(
    "Expected Value = 3.5: Weighted Average of Outcomes",
    600, 400,
    [
        {'label': '1', 'value': 0.167, 'color': '#2563eb'},
        {'label': '2', 'value': 0.333, 'color': '#2563eb'},
        {'label': '3', 'value': 0.5, 'color': '#2563eb'},
        {'label': '4', 'value': 0.667, 'color': '#2563eb'},
        {'label': '5', 'value': 0.833, 'color': '#2563eb'},
        {'label': '6', 'value': 1.0, 'color': '#2563eb'}
    ],
    0.583, "Mean = 3.5",
    "Outcome", "Weighted Contribution",
    "E[X] = Σ x_i · P(x_i) = 1·(1/6) + 2·(1/6) + ... + 6·(1/6) = 21/6 = 3.5",
    output_dir / "probability-expected-value.svg",
    y_max=1.2
)

# 7. Variance
create_bar_chart_with_line_svg(
    "Variance: Average Squared Deviation from Mean",
    600, 400,
    [
        {'label': '1', 'value': 6.25, 'color': '#dc2626'},
        {'label': '2', 'value': 2.25, 'color': '#f59e0b'},
        {'label': '3', 'value': 0.25, 'color': '#16a34a'},
        {'label': '4', 'value': 0.25, 'color': '#16a34a'},
        {'label': '5', 'value': 2.25, 'color': '#f59e0b'},
        {'label': '6', 'value': 6.25, 'color': '#dc2626'}
    ],
    0, "Mean = 3.5",
    "Outcome", "Squared Deviation",
    "Var(X) = Σ (x_i - μ)² · P(x_i) = 2.92 | Mean μ = 3.5",
    output_dir / "probability-variance.svg",
    y_max=7.0
)

# 8. Normal Distribution
create_normal_distribution_svg(
    "Normal Distribution: μ=0, σ=1 with 68-95-99.7 Rule",
    700, 500,
    (-4, 4), (0, 0.5),
    0, 1,
    "68% within 1σ (green) | 95% within 2σ (orange) | 99.7% within 3σ | Mean μ = 0",
    output_dir / "probability-normal-distribution.svg"
)

# 9. Entropy Comparison
create_side_by_side_bar_charts_svg(
    "Entropy: High Uncertainty vs Low Uncertainty",
    800, 400,
    "Fair Coin",
    [
        {'label': 'Heads', 'value': 0.5, 'color': '#2563eb'},
        {'label': 'Tails', 'value': 0.5, 'color': '#2563eb'}
    ],
    "Biased Coin",
    [
        {'label': 'Heads', 'value': 0.99, 'color': '#16a34a'},
        {'label': 'Tails', 'value': 0.01, 'color': '#dc2626'}
    ],
    "Left: Uniform distribution → High entropy (0.693) | Right: Concentrated distribution → Low entropy (0.056)",
    output_dir / "probability-entropy-comparison.svg"
)

# 10. Cross-Entropy
create_cross_entropy_svg(
    "Cross-Entropy: Comparing True vs Predicted Distribution",
    800, 400,
    "True Distribution",
    [
        {'label': 'A', 'value': 0, 'color': '#dc2626'},
        {'label': 'B', 'value': 0, 'color': '#dc2626'},
        {'label': 'C', 'value': 1, 'color': '#16a34a'},
        {'label': 'D', 'value': 0, 'color': '#dc2626'}
    ],
    "Predicted Distribution",
    [
        {'label': 'A', 'value': 0.1, 'color': '#f59e0b'},
        {'label': 'B', 'value': 0.2, 'color': '#f59e0b'},
        {'label': 'C', 'value': 0.6, 'color': '#16a34a'},
        {'label': 'D', 'value': 0.1, 'color': '#f59e0b'}
    ],
    0.511,
    "Left: True one-hot [0,0,1,0] | Right: Predicted [0.1,0.2,0.6,0.1] | H(P,Q) = -log(0.6) ≈ 0.511",
    output_dir / "probability-cross-entropy.svg"
)

# 11. Z-Score Normalization
create_zscore_svg(
    "Z-Score Normalization: Transform to Mean 0, Std 1",
    700, 500,
    "Top: Original data (μ, σ) | Bottom: Normalized (μ=0, σ=1) | All z-scores have mean 0, std 1",
    output_dir / "probability-zscore-normalization.svg"
)

print("\nAll 11 probability diagrams generated with proper graph rendering!")
