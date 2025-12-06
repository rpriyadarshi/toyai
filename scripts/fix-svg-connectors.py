#!/usr/bin/env python3
"""
Utility script to convert arrow paths in SVG files to Inkscape connectors.

This script:
1. Adds IDs to box groups that don't have them
2. Converts all arrow paths (with arrow classes or marker-end) to Inkscape connectors
3. Matches arrow start/end points to box groups
4. Adds required connector attributes (inkscape:connector-type, connection-start, connection-end)

Usage:
    python3 scripts/fix-svg-connectors.py <file1.svg> [file2.svg] ...
    
Or to fix all files in book/images/:
    python3 scripts/fix-svg-connectors.py book/images/*.svg
"""
import re
import sys

def fix_svg_arrows_simple(filepath):
    """Simple approach: add connector attributes to all arrow paths"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Add inkscape namespace if missing
    if 'xmlns:inkscape' not in content:
        content = content.replace('<svg ', '<svg xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" ', 1)
    
    # First, ensure all groups have IDs
    group_counter = 1
    
    def add_id_to_group(match):
        nonlocal group_counter
        full = match.group(0)
        if 'id=' in full:
            return full
        # Add ID
        new_id = f'g{group_counter}'
        group_counter += 1
        return full.replace('<g ', f'<g id="{new_id}" ', 1)
    
    content = re.sub(r'<g\s+transform="translate\([^)]+\)"[^>]*(?<!id=)[^>]*>', add_id_to_group, content)
    
    # Extract all groups with positions
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
    
    # Now convert paths
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
    
    # Match paths with arrow class or marker-end
    pattern = r'<path[^>]*(?:class="[^"]*arrow[^"]*"|marker-end="[^"]*")[^>]*?/?>'
    content = re.sub(pattern, convert_path, content)
    
    return content

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    files = sys.argv[1:]
    
    for filepath in files:
        try:
            fixed = fix_svg_arrows_simple(filepath)
            with open(filepath, 'w') as f:
                f.write(fixed)
            print(f"Fixed: {filepath}")
        except Exception as e:
            print(f"Error fixing {filepath}: {e}")
