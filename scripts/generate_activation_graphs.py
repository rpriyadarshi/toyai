#!/usr/bin/env python3
"""
Generate SVG graphs showing multiple straight lines before and after applying activation functions.
"""

import math
import os

def generate_svg(filename, title, lines_data, x_range=(-3, 3), y_range=(-3, 5), width=500, height=350):
    """
    Generate an SVG graph with multiple lines.
    lines_data: list of dicts with 'points', 'color', 'label' keys
    """
    padding = 60
    plot_width = width - 2 * padding
    plot_height = height - 2 * padding
    
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    def x_to_svg(x):
        return padding + (x - x_min) / (x_max - x_min) * plot_width
    
    def y_to_svg(y):
        return height - padding - (y - y_min) / (y_max - y_min) * plot_height
    
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <style>
    .axis {{ stroke: #666; stroke-width: 1; }}
    .grid {{ stroke: #ddd; stroke-width: 0.5; }}
    .title {{ font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; fill: #1e293b; }}
    .label {{ font-family: Arial, sans-serif; font-size: 11px; fill: #64748b; }}
    .legend {{ font-family: Arial, sans-serif; font-size: 11px; }}
  </style>
  
  <!-- Title -->
  <text x="{width/2}" y="25" text-anchor="middle" class="title">{title}</text>
  
  <!-- Grid lines -->
  <g class="grid">
'''
    
    # Vertical grid lines
    for i in range(int(x_min), int(x_max) + 1):
        if i != 0:
            x = x_to_svg(i)
            svg += f'    <line x1="{x}" y1="{padding}" x2="{x}" y2="{height - padding}"/>\n'
    
    # Horizontal grid lines
    for i in range(int(y_min), int(y_max) + 1):
        if i != 0:
            y = y_to_svg(i)
            svg += f'    <line x1="{padding}" y1="{y}" x2="{width - padding}" y2="{y}"/>\n'
    
    svg += '  </g>\n'
    
    # Axes
    svg += '  <g class="axis">\n'
    # X-axis
    x_axis_y = y_to_svg(0)
    svg += f'    <line x1="{padding}" y1="{x_axis_y}" x2="{width - padding}" y2="{x_axis_y}"/>\n'
    # Y-axis
    y_axis_x = x_to_svg(0)
    svg += f'    <line x1="{y_axis_x}" y1="{padding}" x2="{y_axis_x}" y2="{height - padding}"/>\n'
    svg += '  </g>\n'
    
    # Axis labels
    svg += '  <g class="label">\n'
    # X-axis labels
    for i in range(int(x_min), int(x_max) + 1):
        if i != 0:
            x = x_to_svg(i)
            svg += f'    <text x="{x}" y="{x_axis_y + 20}" text-anchor="middle">{i}</text>\n'
    # Y-axis labels
    for i in range(int(y_min), int(y_max) + 1):
        if i != 0:
            y = y_to_svg(i)
            svg += f'    <text x="{y_axis_x - 10}" y="{y + 4}" text-anchor="end">{i}</text>\n'
    svg += '  </g>\n'
    
    # Plot the lines
    for line_data in lines_data:
        points = line_data['points']
        color = line_data['color']
        if points:
            path_data = f'M {x_to_svg(points[0][0])},{y_to_svg(points[0][1])}'
            for x, y in points[1:]:
                path_data += f' L {x_to_svg(x)},{y_to_svg(y)}'
            svg += f'  <path d="{path_data}" stroke="{color}" stroke-width="2" fill="none"/>\n'
    
    # Legend
    legend_x = width - padding - 10
    legend_y = padding + 10
    svg += f'  <g class="legend">\n'
    for i, line_data in enumerate(lines_data):
        y_pos = legend_y + i * 20
        color = line_data['color']
        label = line_data['label']
        svg += f'    <line x1="{legend_x - 50}" y1="{y_pos}" x2="{legend_x - 30}" y2="{y_pos}" stroke="{color}" stroke-width="2"/>\n'
        svg += f'    <text x="{legend_x - 25}" y="{y_pos + 4}" fill="#1e293b">{label}</text>\n'
    svg += '  </g>\n'
    
    svg += '</svg>\n'
    
    with open(filename, 'w') as f:
        f.write(svg)
    print(f"Generated: {filename}")

# Define multiple linear functions
linear_functions = [
    {'m': 2, 'c': 1, 'label': 'y = 2x + 1', 'color': '#2563eb'},      # Blue
    {'m': -1, 'c': 2, 'label': 'y = -x + 2', 'color': '#dc2626'},    # Red
    {'m': 0.5, 'c': -1, 'label': 'y = 0.5x - 1', 'color': '#16a34a'}, # Green
    {'m': -1.5, 'c': 0.5, 'label': 'y = -1.5x + 0.5', 'color': '#9333ea'}, # Purple
]

# Generate x values
x_values = [x / 10.0 for x in range(-30, 31)]  # -3 to 3 in steps of 0.1

# Create output directory
os.makedirs('/home/rohit/src/toyai-1/book/images/activation-examples', exist_ok=True)

# Generate linear functions
linear_lines = []
for func in linear_functions:
    points = [(x, func['m'] * x + func['c']) for x in x_values]
    linear_lines.append({
        'points': points,
        'color': func['color'],
        'label': func['label']
    })

# Generate ReLU applied
relu_lines = []
for func in linear_functions:
    points = [(x, max(0, func['m'] * x + func['c'])) for x in x_values]
    relu_lines.append({
        'points': points,
        'color': func['color'],
        'label': func['label']
    })

# Generate Sigmoid applied
sigmoid_lines = []
for func in linear_functions:
    points = [(x, 1 / (1 + math.exp(-(func['m'] * x + func['c'])))) for x in x_values]
    sigmoid_lines.append({
        'points': points,
        'color': func['color'],
        'label': func['label']
    })

# Generate Tanh applied
tanh_lines = []
for func in linear_functions:
    points = [(x, math.tanh(func['m'] * x + func['c'])) for x in x_values]
    tanh_lines.append({
        'points': points,
        'color': func['color'],
        'label': func['label']
    })

# Generate SVGs
generate_svg(
    '/home/rohit/src/toyai-1/book/images/activation-examples/linear-function.svg',
    'Linear Functions',
    linear_lines,
    x_range=(-3, 3),
    y_range=(-3, 5)
)

generate_svg(
    '/home/rohit/src/toyai-1/book/images/activation-examples/relu-applied.svg',
    'After ReLU: max(0, y)',
    relu_lines,
    x_range=(-3, 3),
    y_range=(-1, 5)
)

generate_svg(
    '/home/rohit/src/toyai-1/book/images/activation-examples/sigmoid-applied.svg',
    'After Sigmoid: σ(y)',
    sigmoid_lines,
    x_range=(-3, 3),
    y_range=(0, 1)
)

generate_svg(
    '/home/rohit/src/toyai-1/book/images/activation-examples/tanh-applied.svg',
    'After Tanh: tanh(y)',
    tanh_lines,
    x_range=(-3, 3),
    y_range=(-1, 1)
)

print("\nAll graphs generated successfully!")
