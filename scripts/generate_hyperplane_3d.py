#!/usr/bin/env python3
"""
Generate 3D Hyperplane Diagram - Clean SVG Output

Creates a professional 3D diagram showing how a decision plane (hyperplane)
divides 3D space into positive and negative regions. Uses matplotlib's exact
projection to match the interactive viewer, then outputs clean SVG.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

def get_matplotlib_projection(elev=20, azim=30):
    """
    Get matplotlib's projection function for given view angles.
    This ensures the SVG matches the interactive viewer exactly.
    
    Args:
        elev: Elevation angle in degrees
        azim: Azimuth angle in degrees
    
    Returns:
        projection function that takes (x, y, z) and returns (x_2d, y_2d)
    """
    # Create a temporary figure to get matplotlib's projection
    fig = plt.figure(figsize=(1, 1))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)
    
    def project(x, y, z):
        """Project 3D coordinates using matplotlib's exact projection."""
        # Convert to numpy arrays if needed
        x_arr = np.array(x) if not isinstance(x, np.ndarray) else x
        y_arr = np.array(y) if not isinstance(y, np.ndarray) else y
        z_arr = np.array(z) if not isinstance(z, np.ndarray) else z
        
        # Get matplotlib's projection
        x_2d, y_2d = ax.transData.transform(np.column_stack([x_arr, y_arr, z_arr])).T
        return x_2d, y_2d
    
    # Close the temporary figure
    plt.close(fig)
    
    return project

def generate_hyperplane_3d(output_path):
    """
    Generate 3D hyperplane diagram showing decision plane dividing space.
    Outputs clean, human-readable SVG.
    
    Args:
        output_path: Path to output SVG file
    """
    width = 550
    height = 480
    padding = 60
    
    # View parameters - matching interactive viewer exactly
    elev = 20
    azim = 30
    
    # Use matplotlib to get exact projection matching interactive viewer
    # Create a temporary figure to establish the coordinate system
    fig_temp = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax_temp = fig_temp.add_subplot(111, projection='3d')
    ax_temp.set_xlim([-1.5, 1.5])
    ax_temp.set_ylim([-1.5, 1.5])
    ax_temp.set_zlim([-1.5, 1.5])
    ax_temp.view_init(elev=elev, azim=azim)
    
    # Force matplotlib to compute the projection
    plt.draw()
    fig_temp.canvas.draw()
    
    # Get the transform pipeline
    from mpl_toolkits.mplot3d import proj3d
    
    # Test projection to get proper scaling
    test_points = [
        (-1.2, -1.2, -1.2),
        (1.2, 1.2, 1.2),
        (0, 0, 0),
        (-1.2, 1.2, 0),
        (1.2, -1.2, 0)
    ]
    
    # Get projected coordinates in axes data space
    test_proj_2d = []
    for x, y, z in test_points:
        x_2d, y_2d, z_2d = proj3d.proj_transform(x, y, z, ax_temp.get_proj())
        # Transform from axes data coordinates to display coordinates
        display = ax_temp.transData.transform([(x_2d, y_2d)])[0]
        test_proj_2d.append(display)
    
    # Get figure bbox in display coordinates
    fig_bbox = fig_temp.get_window_extent()
    
    # Convert display coordinates to SVG coordinates (raw, no scaling)
    x_coords_raw = [p[0] / fig_bbox.width * width for p in test_proj_2d]
    y_coords_raw = [(1 - p[1] / fig_bbox.height) * height for p in test_proj_2d]
    
    x_min, x_max = min(x_coords_raw), max(x_coords_raw)
    y_min, y_max = min(y_coords_raw), max(y_coords_raw)
    
    # Calculate scale to fit in SVG with padding
    padding = 60
    title_space = 60
    available_width = width - 2 * padding
    available_height = height - 2 * padding - title_space
    
    range_x = x_max - x_min
    range_y = y_max - y_min
    
    scale_x = available_width / range_x if range_x > 0 else 1
    scale_y = available_height / range_y if range_y > 0 else 1
    scale = min(scale_x, scale_y) * 0.95  # Slight margin
    
    # Calculate center of projected points (before scaling)
    center_x_raw = (x_min + x_max) / 2
    center_y_raw = (y_min + y_max) / 2
    
    # Target center in SVG (accounting for title)
    target_x = width / 2
    target_y = (height - title_space) / 2 + title_space
    
    # Calculate offset: target - (scaled center)
    offset_x = target_x - center_x_raw * scale
    offset_y = target_y - center_y_raw * scale
    
    # Define to_svg function with proper transformation
    def to_svg(x_3d, y_3d, z_3d):
        """Convert 3D coordinates to SVG coordinates using matplotlib's exact projection."""
        x_2d, y_2d, z_2d = proj3d.proj_transform(x_3d, y_3d, z_3d, ax_temp.get_proj())
        
        # Transform from axes data coordinates to display coordinates
        display = ax_temp.transData.transform([(x_2d, y_2d)])[0]
        
        # Convert display coordinates to SVG coordinates (raw)
        svg_x_raw = display[0] / fig_bbox.width * width
        svg_y_raw = (1 - display[1] / fig_bbox.height) * height
        
        # Apply scaling: first center around origin, scale, then translate
        svg_x = (svg_x_raw - center_x_raw) * scale + target_x
        svg_y = (svg_y_raw - center_y_raw) * scale + target_y
        
        return svg_x, svg_y
    
    # Start building SVG
    svg_lines = []
    svg_lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    svg_lines.append(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">')
    svg_lines.append('  <defs>')
    svg_lines.append('    <marker id="axisArrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse">')
    svg_lines.append('      <path d="M 0,0 L 8,4 L 0,8 Z" fill="#1f2937"/>')
    svg_lines.append('    </marker>')
    svg_lines.append('    <marker id="arrow-green" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto" markerUnits="userSpaceOnUse">')
    svg_lines.append('      <path d="M 0,0 L 6,3 L 0,6 Z" fill="#16a34a"/>')
    svg_lines.append('    </marker>')
    svg_lines.append('    <marker id="arrow-red" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto" markerUnits="userSpaceOnUse">')
    svg_lines.append('      <path d="M 0,0 L 6,3 L 0,6 Z" fill="#dc2626"/>')
    svg_lines.append('    </marker>')
    # Gradient for positive region (green fading to white)
    svg_lines.append('    <linearGradient id="greenGradient" x1="0%" y1="0%" x2="0%" y2="100%">')
    svg_lines.append('      <stop offset="0%" style="stop-color:#16a34a;stop-opacity:0.4" />')
    svg_lines.append('      <stop offset="100%" style="stop-color:#ffffff;stop-opacity:0" />')
    svg_lines.append('    </linearGradient>')
    # Gradient for negative region (red fading to white)
    svg_lines.append('    <linearGradient id="redGradient" x1="0%" y1="0%" x2="0%" y2="100%">')
    svg_lines.append('      <stop offset="0%" style="stop-color:#dc2626;stop-opacity:0.4" />')
    svg_lines.append('      <stop offset="100%" style="stop-color:#ffffff;stop-opacity:0" />')
    svg_lines.append('    </linearGradient>')
    svg_lines.append('  </defs>')
    
    # Title
    svg_lines.append(f'  <text x="{width/2}" y="30" font-family="Arial, sans-serif" font-size="17" font-weight="bold" text-anchor="middle" fill="#1f2937">3D Input (d=3)</text>')
    svg_lines.append(f'  <text x="{width/2}" y="52" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#6b7280">w₁x₁ + w₂x₂ + w₃x₃ + b = 0</text>')
    
    # Draw axes
    # x₁-axis: from origin along x direction
    x1_start_x, x1_start_y = to_svg(-1, 0, 0)
    x1_end_x, x1_end_y = to_svg(1, 0, 0)
    svg_lines.append(f'  <line x1="{x1_start_x}" y1="{x1_start_y}" x2="{x1_end_x}" y2="{x1_end_y}" stroke="#1f2937" stroke-width="2.5" marker-end="url(#axisArrow)"/>')
    x1_label_x, x1_label_y = to_svg(1.2, -0.15, 0)
    svg_lines.append(f'  <text x="{x1_label_x}" y="{x1_label_y}" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#1f2937">x₁</text>')
    
    # x₂-axis: from origin along y direction
    x2_start_x, x2_start_y = to_svg(0, -1, 0)
    x2_end_x, x2_end_y = to_svg(0, 1, 0)
    svg_lines.append(f'  <line x1="{x2_start_x}" y1="{x2_start_y}" x2="{x2_end_x}" y2="{x2_end_y}" stroke="#1f2937" stroke-width="2.5" marker-end="url(#axisArrow)"/>')
    x2_label_x, x2_label_y = to_svg(-0.15, 1.2, 0)
    svg_lines.append(f'  <text x="{x2_label_x}" y="{x2_label_y}" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#1f2937">x₂</text>')
    
    # x₃-axis: from origin along z direction
    x3_start_x, x3_start_y = to_svg(0, 0, -1)
    x3_end_x, x3_end_y = to_svg(0, 0, 1)
    svg_lines.append(f'  <line x1="{x3_start_x}" y1="{x3_start_y}" x2="{x3_end_x}" y2="{x3_end_y}" stroke="#1f2937" stroke-width="2.5" marker-end="url(#axisArrow)"/>')
    x3_label_x, x3_label_y = to_svg(0.15, 0, 1.2)
    svg_lines.append(f'  <text x="{x3_label_x}" y="{x3_label_y}" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#1f2937">x₃</text>')
    
    # Origin - repositioned to avoid overlap
    origin_x, origin_y = to_svg(0, 0, 0)
    svg_lines.append(f'  <circle cx="{origin_x}" cy="{origin_y}" r="4" fill="#1f2937"/>')
    origin_label_x, origin_label_y = to_svg(0.25, -0.25, -0.25)
    svg_lines.append(f'  <text x="{origin_label_x}" y="{origin_label_y}" font-family="Arial, sans-serif" font-size="10" fill="#6b7280" font-weight="500">(0,0,0)</text>')
    
    # Decision plane: plane at z=0, extending along x₁ and x₂
    # Define plane corners
    plane_corners = [
        (-1, -1, 0),  # back-left
        (1, -1, 0),   # back-right
        (1, 1, 0),    # front-right
        (-1, 1, 0)    # front-left
    ]
    
    # Project corners to 2D
    plane_points = [to_svg(x, y, z) for x, y, z in plane_corners]
    points_str = ' '.join([f'{x},{y}' for x, y in plane_points])
    
    # Draw plane (semi-transparent blue)
    svg_lines.append(f'  <polygon points="{points_str}" fill="#2563eb" opacity="0.25" stroke="#2563eb" stroke-width="2.5"/>')
    
    # Grid lines on plane (optional, for depth)
    for i in range(-1, 2):
        if i != 0:
            grid_start = to_svg(i, -1, 0)
            grid_end = to_svg(i, 1, 0)
            svg_lines.append(f'  <line x1="{grid_start[0]}" y1="{grid_start[1]}" x2="{grid_end[0]}" y2="{grid_end[1]}" stroke="#3b82f6" stroke-width="1" opacity="0.4"/>')
            grid_start2 = to_svg(-1, i, 0)
            grid_end2 = to_svg(1, i, 0)
            svg_lines.append(f'  <line x1="{grid_start2[0]}" y1="{grid_start2[1]}" x2="{grid_end2[0]}" y2="{grid_end2[1]}" stroke="#3b82f6" stroke-width="1" opacity="0.4"/>')
    
    # Plane label - offset to avoid overlap with origin
    plane_label_x, plane_label_y = to_svg(0.3, -0.3, 0.3)
    svg_lines.append(f'  <text x="{plane_label_x}" y="{plane_label_y}" font-family="Arial, sans-serif" font-size="13" font-weight="600" fill="#2563eb" text-anchor="middle">Decision Plane</text>')
    
    # Positive region: entire half-space above plane (x₃ > 0)
    # Draw gradient cubes that fade with distance from plane
    # Create multiple layers at different z-levels
    # Very transparent to avoid obstructing the decision plane
    z_levels_pos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i, z_level in enumerate(z_levels_pos):
        # Opacity decreases with distance from plane - very transparent
        opacity = max(0.01, 0.06 * (1.0 - i / len(z_levels_pos)))
        # Draw cube face at this z-level (entire x₁-x₂ plane)
        pos_corners = [
            (-1.2, -1.2, z_level),  # back-left
            (1.2, -1.2, z_level),    # back-right
            (1.2, 1.2, z_level),     # front-right
            (-1.2, 1.2, z_level)     # front-left
        ]
        pos_points = [to_svg(x, y, z) for x, y, z in pos_corners]
        pos_points_str = ' '.join([f'{x},{y}' for x, y in pos_points])
        svg_lines.append(f'  <polygon points="{pos_points_str}" fill="#16a34a" opacity="{opacity}" stroke="none"/>')
    
    # Arrow to positive region - repositioned to avoid overlap
    arrow_start = to_svg(1.0, -1.0, 0.3)
    arrow_end = to_svg(1.0, -1.0, 0.8)
    svg_lines.append(f'  <path d="M {arrow_start[0]},{arrow_start[1]} L {arrow_end[0]},{arrow_end[1]}" stroke="#16a34a" stroke-width="2.2" fill="none" marker-end="url(#arrow-green)" stroke-linecap="round"/>')
    label_pos = to_svg(1.2, -1.2, 1.0)
    svg_lines.append(f'  <text x="{label_pos[0]}" y="{label_pos[1]}" font-family="Arial, sans-serif" font-size="11" font-weight="600" fill="#16a34a">w·x + b &gt; 0</text>')
    svg_lines.append(f'  <text x="{label_pos[0]}" y="{label_pos[1] + 14}" font-family="Arial, sans-serif" font-size="9" fill="#16a34a" font-style="italic">(all x₃ &gt; 0)</text>')
    
    # Negative region: entire half-space below plane (x₃ < 0)
    # Draw gradient cubes that fade with distance from plane
    # Very transparent to avoid obstructing the decision plane
    z_levels_neg = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0]
    for i, z_level in enumerate(z_levels_neg):
        # Opacity decreases with distance from plane - very transparent
        opacity = max(0.01, 0.06 * (1.0 - i / len(z_levels_neg)))
        # Draw cube face at this z-level (entire x₁-x₂ plane)
        neg_corners = [
            (-1.2, -1.2, z_level),  # back-left
            (1.2, -1.2, z_level),    # back-right
            (1.2, 1.2, z_level),     # front-right
            (-1.2, 1.2, z_level)     # front-left
        ]
        neg_points = [to_svg(x, y, z) for x, y, z in neg_corners]
        neg_points_str = ' '.join([f'{x},{y}' for x, y in neg_points])
        svg_lines.append(f'  <polygon points="{neg_points_str}" fill="#dc2626" opacity="{opacity}" stroke="none"/>')
    
    # Arrow to negative region - repositioned to avoid overlap
    arrow_start_neg = to_svg(-1.0, 1.0, -0.3)
    arrow_end_neg = to_svg(-1.0, 1.0, -0.8)
    svg_lines.append(f'  <path d="M {arrow_start_neg[0]},{arrow_start_neg[1]} L {arrow_end_neg[0]},{arrow_end_neg[1]}" stroke="#dc2626" stroke-width="2.2" fill="none" marker-end="url(#arrow-red)" stroke-linecap="round"/>')
    label_neg = to_svg(-1.2, 1.2, -1.0)
    svg_lines.append(f'  <text x="{label_neg[0]}" y="{label_neg[1]}" font-family="Arial, sans-serif" font-size="11" font-weight="600" fill="#dc2626" text-anchor="end">w·x + b &lt; 0</text>')
    svg_lines.append(f'  <text x="{label_neg[0]}" y="{label_neg[1] + 14}" font-family="Arial, sans-serif" font-size="9" fill="#dc2626" font-style="italic" text-anchor="end">(all x₃ &lt; 0)</text>')
    
    # Arrow to negative region
    arrow_start_neg = to_svg(-0.8, -0.8, -0.3)
    arrow_end_neg = to_svg(-0.8, -0.8, -0.8)
    svg_lines.append(f'  <path d="M {arrow_start_neg[0]},{arrow_start_neg[1]} L {arrow_end_neg[0]},{arrow_end_neg[1]}" stroke="#dc2626" stroke-width="2.2" fill="none" marker-end="url(#arrow-red)" stroke-linecap="round"/>')
    label_neg = to_svg(-1.0, -1.0, -1.0)
    svg_lines.append(f'  <text x="{label_neg[0]}" y="{label_neg[1]}" font-family="Arial, sans-serif" font-size="12" font-weight="600" fill="#dc2626">w·x + b &lt; 0</text>')
    svg_lines.append(f'  <text x="{label_neg[0]}" y="{label_neg[1] + 15}" font-family="Arial, sans-serif" font-size="10" fill="#dc2626" font-style="italic">(all x₃ &lt; 0)</text>')
    
    # Helper text
    helper_x, helper_y = to_svg(0, -0.8, -0.8)
    svg_lines.append(f'  <text x="{helper_x}" y="{helper_y}" font-family="Arial, sans-serif" font-size="10" fill="#9ca3af" text-anchor="middle" font-style="italic">Plane divides 3D space</text>')
    
    svg_lines.append('</svg>')
    
    # Close temporary figure
    plt.close(fig_temp)
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(svg_lines))
    
    print(f"Generated 3D hyperplane diagram: {output_path}")

if __name__ == "__main__":
    # Determine output path
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_path = project_root / "book" / "images" / "other" / "hyperplane-3d.svg"
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate diagram
    generate_hyperplane_3d(output_path)
    print("Done!")

