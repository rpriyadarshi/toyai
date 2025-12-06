#!/usr/bin/env python3
"""
Algorithmically fix connector path coordinates to match Inkscape's rendering.

This script:
1. Parses SVG to find all Inkscape connectors
2. For each connector, finds the start and end groups
3. Calculates bounding boxes of those groups (accounting for transforms)
4. Computes center-to-center line
5. Finds where line intersects box boundaries
6. Updates the `d` attribute with correct coordinates

This ensures SVG viewers render the same as Inkscape, while preserving connector attributes.

Usage:
    python3 scripts/fix-connector-paths.py <file1.svg> [file2.svg] ...
"""

import re
import sys
import xml.etree.ElementTree as ET
from typing import Dict, Tuple, Optional

def parse_transform(transform_str: str) -> Tuple[float, float]:
    """Parse translate(x, y) transform and return (x, y)."""
    match = re.search(r'translate\(([-\d.]+)\s*,\s*([-\d.]+)\)', transform_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    return 0.0, 0.0

def get_group_bbox(group_elem) -> Optional[Dict]:
    """Calculate bounding box of a group, accounting for transform."""
    # Find the rect element within the group
    rect = group_elem.find('.//{http://www.w3.org/2000/svg}rect')
    if rect is None:
        return None
    
    # Get transform
    transform = group_elem.get('transform', '')
    tx, ty = parse_transform(transform)
    
    # Get rect dimensions
    x = float(rect.get('x', '0'))
    y = float(rect.get('y', '0'))
    width = float(rect.get('width', '0'))
    height = float(rect.get('height', '0'))
    
    # Calculate absolute position
    abs_x = tx + x
    abs_y = ty + y
    
    # Calculate center
    center_x = abs_x + width / 2
    center_y = abs_y + height / 2
    
    return {
        'left': abs_x,
        'right': abs_x + width,
        'top': abs_y,
        'bottom': abs_y + height,
        'center_x': center_x,
        'center_y': center_y,
        'width': width,
        'height': height
    }

def line_box_intersection(x1: float, y1: float, x2: float, y2: float, 
                          bbox: Dict) -> Tuple[float, float]:
    """
    Find where a line from (x1,y1) to (x2,y2) intersects a bounding box.
    Returns the intersection point on the box boundary.
    """
    cx, cy = bbox['center_x'], bbox['center_y']
    left, right = bbox['left'], bbox['right']
    top, bottom = bbox['top'], bbox['bottom']
    
    # Calculate line direction
    dx = x2 - x1
    dy = y2 - y1
    
    # If line is vertical or horizontal, handle separately
    if abs(dx) < 0.001:
        # Vertical line
        if y1 < cy:
            return x1, top  # Exit at top
        else:
            return x1, bottom  # Exit at bottom
    
    if abs(dy) < 0.001:
        # Horizontal line
        if x1 < cx:
            return left, y1  # Exit at left
        else:
            return right, y1  # Exit at right
    
    # Calculate slope
    slope = dy / dx
    
    # Determine which edge the line exits/enters
    # Check all four edges and find the first intersection
    
    # Top edge (y = top)
    if (y1 < top and y2 >= top) or (y1 >= top and y2 < top):
        x_top = x1 + (top - y1) / slope if slope != 0 else x1
        if left <= x_top <= right:
            return x_top, top
    
    # Bottom edge (y = bottom)
    if (y1 < bottom and y2 >= bottom) or (y1 >= bottom and y2 < bottom):
        x_bottom = x1 + (bottom - y1) / slope if slope != 0 else x1
        if left <= x_bottom <= right:
            return x_bottom, bottom
    
    # Left edge (x = left)
    if (x1 < left and x2 >= left) or (x1 >= left and x2 < left):
        y_left = y1 + (left - x1) * slope
        if top <= y_left <= bottom:
            return left, y_left
    
    # Right edge (x = right)
    if (x1 < right and x2 >= right) or (x1 >= right and x2 < right):
        y_right = y1 + (right - x1) * slope
        if top <= y_right <= bottom:
            return right, y_right
    
    # Fallback: return closest edge point
    if x1 < cx:
        return left, cy
    else:
        return right, cy

def fix_connector_paths(svg_content: str) -> str:
    """Fix all connector paths in SVG content."""
    # Parse XML
    root = ET.fromstring(svg_content)
    
    # Build namespace map
    ns = {'svg': 'http://www.w3.org/2000/svg', 
          'inkscape': 'http://www.inkscape.org/namespaces/inkscape'}
    
    # Build group ID to bbox map
    groups = {}
    for group in root.iterfind('.//svg:g', ns):
        group_id = group.get('id')
        if group_id:
            bbox = get_group_bbox(group)
            if bbox:
                groups[group_id] = bbox
    
    # Find all connector paths
    for path in root.iterfind('.//svg:path', ns):
        connector_type = path.get('{http://www.inkscape.org/namespaces/inkscape}connector-type')
        if not connector_type:
            continue
        
        # Get connection references
        conn_start = path.get('{http://www.inkscape.org/namespaces/inkscape}connection-start', '')
        conn_end = path.get('{http://www.inkscape.org/namespaces/inkscape}connection-end', '')
        
        # Extract group IDs (remove # prefix)
        start_id = conn_start.lstrip('#') if conn_start else None
        end_id = conn_end.lstrip('#') if conn_end else None
        
        if not start_id or not end_id:
            continue
        
        start_bbox = groups.get(start_id)
        end_bbox = groups.get(end_id)
        
        if not start_bbox or not end_bbox:
            continue
        
        # Calculate center-to-center line
        x1, y1 = start_bbox['center_x'], start_bbox['center_y']
        x2, y2 = end_bbox['center_x'], end_bbox['center_y']
        
        # Find intersection points
        exit_point = line_box_intersection(x1, y1, x2, y2, start_bbox)
        enter_point = line_box_intersection(x2, y2, x1, y1, end_bbox)
        
        # Update d attribute
        new_d = f"M {exit_point[0]},{exit_point[1]} {enter_point[0]},{enter_point[1]}"
        path.set('d', new_d)
    
    # Convert back to string
    return ET.tostring(root, encoding='unicode')

def fix_file(filepath: str):
    """Fix connector paths in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix paths
        fixed = fix_connector_paths(content)
        
        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed)
        
        print(f"Fixed: {filepath}")
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    for filepath in sys.argv[1:]:
        fix_file(filepath)

