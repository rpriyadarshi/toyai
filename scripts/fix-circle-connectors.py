#!/usr/bin/env python3
"""
Fix connector paths for circles (neurons) in SVG files.
Calculates correct intersection points where lines from center-to-center
intersect circle boundaries.
"""

import sys
import re
import math
import xml.etree.ElementTree as ET

def circle_line_intersection(cx, cy, r, x1, y1, x2, y2):
    """Find where line from (x1,y1) to (x2,y2) intersects circle at (cx,cy) with radius r."""
    dx = x2 - x1
    dy = y2 - y1
    
    if abs(dx) < 0.001 and abs(dy) < 0.001:
        return None
    
    fx = x1 - cx
    fy = y1 - cy
    
    a = dx*dx + dy*dy
    b = 2*(fx*dx + fy*dy)
    c = fx*fx + fy*fy - r*r
    
    discriminant = b*b - 4*a*c
    if discriminant < 0:
        return None
    
    t1 = (-b - math.sqrt(discriminant)) / (2*a)
    t2 = (-b + math.sqrt(discriminant)) / (2*a)
    
    # For exit point (from source), use t closest to 0 (start of line)
    # For entry point (to destination), use t closest to 1 (end of line)
    # Actually, we want the intersection in the direction from center1 to center2
    # So for source circle, we want the point further from center1 (larger t)
    # For dest circle, we want the point closer to center2 (smaller t from center1's perspective)
    
    # Use the intersection that's in the direction from (x1,y1) toward (x2,y2)
    if t1 >= 0 and t1 <= 1:
        t = t1
    elif t2 >= 0 and t2 <= 1:
        t = t2
    else:
        # Line doesn't intersect in the segment, use the closer one
        t = t1 if abs(t1) < abs(t2) else t2
    
    ix = x1 + t*dx
    iy = y1 + t*dy
    return (ix, iy)

def parse_transform(transform_str):
    """Parse translate(x, y) transform."""
    if not transform_str:
        return 0.0, 0.0
    match = re.search(r'translate\(([-\d.]+)\s*,\s*([-\d.]+)\)', transform_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    return 0.0, 0.0

def build_parent_map(root):
    """Build a map of element -> parent."""
    parent_map = {}
    for parent in root.iter():
        for child in parent:
            parent_map[child] = parent
    return parent_map

def get_circle_position(group_elem, root, parent_map):
    """Get absolute position of circle in a group, accounting for all parent transforms."""
    # Find circle in group
    circle = group_elem.find('.//{http://www.w3.org/2000/svg}circle')
    if circle is None:
        return None
    
    cx = float(circle.get('cx', '0'))
    cy = float(circle.get('cy', '0'))
    r = float(circle.get('r', '0'))
    
    # Accumulate transforms from root to this group
    x, y = cx, cy
    elem = group_elem
    visited = set()
    while elem is not None and elem != root and elem not in visited:
        visited.add(elem)
        transform = elem.get('transform', '')
        tx, ty = parse_transform(transform)
        x += tx
        y += ty
        elem = parent_map.get(elem)
    
    return (x, y, r)

def fix_file(filepath):
    """Fix connector paths in an SVG file."""
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    # Register default namespace
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    ET.register_namespace('inkscape', 'http://www.inkscape.org/namespaces/inkscape')
    
    # Build parent map
    parent_map = build_parent_map(root)
    
    # Find all connector paths
    paths = root.findall('.//{http://www.w3.org/2000/svg}path')
    paths = [p for p in paths if 'connector-type' in p.attrib or 
             any(k.endswith('connector-type') for k in p.attrib.keys())]
    
    fixed_count = 0
    for path in paths:
        # Get connection IDs (handle both with and without namespace prefix)
        start_id = None
        end_id = None
        for key, value in path.attrib.items():
            if 'connection-start' in key:
                start_id = value.lstrip('#')
            elif 'connection-end' in key:
                end_id = value.lstrip('#')
        
        if not start_id or not end_id:
            continue
        
        # Find the groups
        start_group = None
        end_group = None
        for elem in root.iter():
            if elem.get('id') == start_id:
                start_group = elem
            if elem.get('id') == end_id:
                end_group = elem
        
        if start_group is None or end_group is None:
            continue
        
        start_pos = get_circle_position(start_group, root, parent_map)
        end_pos = get_circle_position(end_group, root, parent_map)
        
        if start_pos and end_pos:
            cx1, cy1, r1 = start_pos
            cx2, cy2, r2 = end_pos
            
            # Calculate intersection points
            # For exit from source circle: line from center1 to center2
            p1 = circle_line_intersection(cx1, cy1, r1, cx1, cy1, cx2, cy2)
            # For entry to dest circle: line from center1 to center2  
            p2 = circle_line_intersection(cx2, cy2, r2, cx1, cy1, cx2, cy2)
            
            if p1 and p2:
                new_d = f"M {p1[0]:.1f} {p1[1]:.1f} L {p2[0]:.1f} {p2[1]:.1f}"
                path.set('d', new_d)
                fixed_count += 1
    
    if fixed_count > 0:
        # Write without namespace prefixes
        ET.register_namespace('', 'http://www.w3.org/2000/svg')
        ET.register_namespace('inkscape', 'http://www.inkscape.org/namespaces/inkscape')
        tree.write(filepath, encoding='utf-8', xml_declaration=True, method='xml')
        print(f"Fixed {fixed_count} connector paths in {filepath}")
    else:
        print(f"No connector paths found or fixed in {filepath}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 fix-circle-connectors.py <file1.svg> [file2.svg] ...")
        sys.exit(1)
    
    for filepath in sys.argv[1:]:
        fix_file(filepath)

