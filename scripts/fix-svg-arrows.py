#!/usr/bin/env python3
"""
Consolidated script to fix SVG arrows and connectors.

This script can:
1. Convert static arrow paths to Inkscape connectors
2. Fix connector path coordinates to match Inkscape's rendering
3. Do both operations in sequence

Usage:
    # Convert arrows to connectors
    python3 scripts/fix-svg-arrows.py --to-connectors <file1.svg> [file2.svg] ...
    
    # Fix connector path coordinates
    python3 scripts/fix-svg-arrows.py --fix-paths <file1.svg> [file2.svg] ...
    
    # Do both (recommended)
    python3 scripts/fix-svg-arrows.py --all <file1.svg> [file2.svg] ...
    
    # Fix all files in book/images/
    python3 scripts/fix-svg-arrows.py --all book/images/*.svg
"""

import re
import sys
import xml.etree.ElementTree as ET
from typing import Dict, Tuple, Optional

# ============================================================================
# Part 1: Convert static arrows to connectors
# ============================================================================

def add_inkscape_namespace(content: str) -> str:
    """Add inkscape namespace if missing."""
    if 'xmlns:inkscape' not in content:
        content = content.replace('<svg ', '<svg xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" ', 1)
    return content

def ensure_group_ids(content: str) -> str:
    """Ensure all groups with transforms have IDs."""
    group_counter = 1
    
    def add_id_to_group(match):
        nonlocal group_counter
        full = match.group(0)
        if 'id=' in full:
            return full
        new_id = f'g{group_counter}'
        group_counter += 1
        return full.replace('<g ', f'<g id="{new_id}" ', 1)
    
    return re.sub(r'<g\s+transform="translate\([^)]+\)"[^>]*(?<!id=)[^>]*>', add_id_to_group, content)

def extract_groups(content: str) -> list:
    """Extract all groups with positions and dimensions."""
    groups = []
    for match in re.finditer(r'<g[^>]*id="([^"]+)"[^>]*transform="translate\(([^)]+)\)"', content):
        gid = match.group(1)
        coords_str = match.group(2)
        coords = [float(c.strip()) for c in coords_str.split(',')]
        x, y = coords[0], coords[1] if len(coords) > 1 else 0.0
        
        # Find rect in this group to get dimensions
        group_end_pos = content.find('</g>', match.end())
        if group_end_pos > 0:
            group_content = content[match.start():group_end_pos]
            rect_match = re.search(r'<rect[^>]*width="([^"]+)"[^>]*height="([^"]+)"', group_content)
            if rect_match:
                w, h = float(rect_match.group(1)), float(rect_match.group(2))
                groups.append({
                    'id': gid,
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'cx': x + w/2, 'cy': y + h/2,
                    'right': x + w, 'left': x,
                    'top': y, 'bottom': y + h
                })
    return groups

def convert_to_connectors(content: str) -> str:
    """Convert static arrow paths to Inkscape connectors."""
    content = add_inkscape_namespace(content)
    content = ensure_group_ids(content)
    groups = extract_groups(content)
    
    def convert_path(match):
        full = match.group(0)
        if 'inkscape:connector-type' in full:
            return full
        
        d_match = re.search(r'd="([^"]+)"', full)
        if not d_match:
            return full
        
        coords = re.findall(r'[ML]\s+([-\d.]+)\s+([-\d.]+)', d_match.group(1))
        if len(coords) < 2:
            return full
        
        sx, sy = float(coords[0][0]), float(coords[0][1])
        ex, ey = float(coords[-1][0]), float(coords[-1][1])
        
        def closest(x, y):
            best = None
            best_dist = float('inf')
            for g in groups:
                d1 = abs(x - g['cx']) + abs(y - g['cy'])
                d2 = abs(x - g['right']) + abs(y - g['cy'])
                d3 = abs(x - g['left']) + abs(y - g['cy'])
                d4 = abs(x - g['cx']) + abs(y - g['top'])
                d5 = abs(x - g['cx']) + abs(y - g['bottom'])
                d = min(d1, d2, d3, d4, d5)
                if d < best_dist:
                    best_dist = d
                    best = g
            return best
        
        sg = closest(sx, sy)
        eg = closest(ex, ey)
        
        if not sg or not eg:
            return full
        
        attrs = f' inkscape:connector-type="polyline" inkscape:connector-curvature="0" inkscape:connection-start="#{sg["id"]}" inkscape:connection-end="#{eg["id"]}"'
        
        if '/>' in full:
            return full.replace('/>', attrs + ' />')
        elif full.rstrip().endswith('>'):
            return full.rstrip()[:-1] + attrs + '>\n' if full.endswith('\n') else full[:-1] + attrs + '>'
        return full
    
    pattern = r'<path[^>]*(?:class="[^"]*arrow[^"]*"|marker-end="[^"]*")[^>]*?/?>'
    content = re.sub(pattern, convert_path, content)
    
    return content

# ============================================================================
# Part 2: Fix connector path coordinates
# ============================================================================

def parse_transform(transform_str: str) -> Tuple[float, float]:
    """Parse translate(x, y) transform and return (x, y)."""
    match = re.search(r'translate\(([-\d.]+)\s*,\s*([-\d.]+)\)', transform_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    return 0.0, 0.0

def get_group_bbox(group_elem) -> Optional[Dict]:
    """Calculate bounding box of a group, accounting for transform."""
    rect = group_elem.find('.//{http://www.w3.org/2000/svg}rect')
    if rect is None:
        return None
    
    transform = group_elem.get('transform', '')
    tx, ty = parse_transform(transform)
    
    x = float(rect.get('x', '0'))
    y = float(rect.get('y', '0'))
    width = float(rect.get('width', '0'))
    height = float(rect.get('height', '0'))
    
    abs_x = tx + x
    abs_y = ty + y
    
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
    """Find where a line from (x1,y1) to (x2,y2) intersects a bounding box."""
    cx, cy = bbox['center_x'], bbox['center_y']
    left, right = bbox['left'], bbox['right']
    top, bottom = bbox['top'], bbox['bottom']
    
    dx = x2 - x1
    dy = y2 - y1
    
    # Handle vertical/horizontal lines
    if abs(dx) < 0.001:
        return (x1, top) if y1 < cy else (x1, bottom)
    
    if abs(dy) < 0.001:
        return (left, y1) if x1 < cx else (right, y1)
    
    slope = dy / dx
    
    # Check all four edges
    intersections = []
    
    # Top edge
    if (y1 < top and y2 >= top) or (y1 >= top and y2 < top):
        x_top = x1 + (top - y1) / slope
        if left <= x_top <= right:
            intersections.append((x_top, top, abs(y1 - top)))
    
    # Bottom edge
    if (y1 < bottom and y2 >= bottom) or (y1 >= bottom and y2 < bottom):
        x_bottom = x1 + (bottom - y1) / slope
        if left <= x_bottom <= right:
            intersections.append((x_bottom, bottom, abs(y1 - bottom)))
    
    # Left edge
    if (x1 < left and x2 >= left) or (x1 >= left and x2 < left):
        y_left = y1 + (left - x1) * slope
        if top <= y_left <= bottom:
            intersections.append((left, y_left, abs(x1 - left)))
    
    # Right edge
    if (x1 < right and x2 >= right) or (x1 >= right and x2 < right):
        y_right = y1 + (right - x1) * slope
        if top <= y_right <= bottom:
            intersections.append((right, y_right, abs(x1 - right)))
    
    if intersections:
        # Return closest intersection
        intersections.sort(key=lambda p: p[2])
        return intersections[0][0], intersections[0][1]
    
    # Fallback
    if x1 < cx:
        return left, cy
    else:
        return right, cy

def fix_connector_paths(svg_content: str) -> str:
    """Fix all connector paths in SVG content."""
    root = ET.fromstring(svg_content)
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
        
        conn_start = path.get('{http://www.inkscape.org/namespaces/inkscape}connection-start', '')
        conn_end = path.get('{http://www.inkscape.org/namespaces/inkscape}connection-end', '')
        
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
        new_d = f"M {exit_point[0]:.5f},{exit_point[1]:.5f} {enter_point[0]:.5f},{enter_point[1]:.5f}"
        path.set('d', new_d)
    
    return ET.tostring(root, encoding='unicode')

# ============================================================================
# Main
# ============================================================================

def fix_file(filepath: str, to_connectors: bool = False, fix_paths: bool = False):
    """Fix SVG file based on options."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        if to_connectors:
            content = convert_to_connectors(content)
        
        if fix_paths:
            content = fix_connector_paths(content)
        
        # Only write if content changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {filepath}")
        else:
            print(f"No changes: {filepath}")
            
    except Exception as e:
        print(f"Error fixing {filepath}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    
    mode = sys.argv[1]
    files = sys.argv[2:]
    
    if mode == '--to-connectors':
        for filepath in files:
            fix_file(filepath, to_connectors=True, fix_paths=False)
    elif mode == '--fix-paths':
        for filepath in files:
            fix_file(filepath, to_connectors=False, fix_paths=True)
    elif mode == '--all':
        for filepath in files:
            fix_file(filepath, to_connectors=True, fix_paths=True)
    else:
        print(f"Unknown mode: {mode}")
        print(__doc__)
        sys.exit(1)

