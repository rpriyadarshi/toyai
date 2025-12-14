#!/usr/bin/env python3
"""
Calculate bounding box for SVG and adjust canvas dimensions.
"""
import xml.etree.ElementTree as ET
import re
import sys

def parse_transform(transform_str):
    """Parse transform string and return (tx, ty) translation."""
    if not transform_str:
        return (0, 0)
    match = re.search(r'translate\(([^,]+),\s*([^)]+)\)', transform_str)
    if match:
        return (float(match.group(1)), float(match.group(2)))
    return (0, 0)

def get_text_bounds(text_elem, font_size, font_family):
    """Estimate text bounds. Returns (width, height)."""
    text = text_elem.text or ""
    # Rough estimates based on font size
    # Arial/Courier: ~0.6 * font_size per character width
    # Height: ~1.2 * font_size
    char_width = 0.6 * font_size
    width = len(text) * char_width
    height = 1.2 * font_size
    return (width, height)

def get_absolute_coords(elem, parent_tx=0, parent_ty=0):
    """Get absolute coordinates accounting for transforms."""
    transform = elem.get('transform', '')
    tx, ty = parse_transform(transform)
    abs_tx = parent_tx + tx
    abs_ty = parent_ty + ty
    return (abs_tx, abs_ty)

def parse_path_d(d_str):
    """Extract all coordinates from path d attribute."""
    coords = []
    # Match M and L commands with coordinates
    pattern = r'[ML]\s+([\d.]+)\s+([\d.]+)'
    matches = re.findall(pattern, d_str)
    for match in matches:
        coords.append((float(match[0]), float(match[1])))
    return coords

def calculate_bounds(svg_file):
    """Calculate bounding box of all elements in SVG."""
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    
    def process_element(elem, parent_tx=0, parent_ty=0):
        nonlocal min_x, min_y, max_x, max_y
        
        abs_tx, abs_ty = get_absolute_coords(elem, parent_tx, parent_ty)
        
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        
        if tag == 'rect':
            x = float(elem.get('x', 0))
            y = float(elem.get('y', 0))
            w = float(elem.get('width', 0))
            h = float(elem.get('height', 0))
            abs_x = abs_tx + x
            abs_y = abs_ty + y
            min_x = min(min_x, abs_x)
            min_y = min(min_y, abs_y)
            max_x = max(max_x, abs_x + w)
            max_y = max(max_y, abs_y + h)
        
        elif tag == 'circle':
            cx = float(elem.get('cx', 0))
            cy = float(elem.get('cy', 0))
            r = float(elem.get('r', 0))
            abs_cx = abs_tx + cx
            abs_cy = abs_ty + cy
            min_x = min(min_x, abs_cx - r)
            min_y = min(min_y, abs_cy - r)
            max_x = max(max_x, abs_cx + r)
            max_y = max(max_y, abs_cy + r)
        
        elif tag == 'text':
            x = float(elem.get('x', 0))
            y = float(elem.get('y', 0))
            text_anchor = elem.get('text-anchor', 'start')
            
            # Get font size from class or style
            font_size = 12  # default
            class_name = elem.get('class', '')
            if 'neuron-index' in class_name:
                font_size = 9
            elif 'neuron-value' in class_name:
                font_size = 11
            elif 'layer-label' in class_name:
                font_size = 14
            elif 'bias-label' in class_name:
                font_size = 10
            
            text = elem.text or ""
            if text:
                # Estimate text width
                char_width = 0.6 * font_size
                text_width = len(text) * char_width
                text_height = 1.2 * font_size
                
                abs_x = abs_tx + x
                abs_y = abs_ty + y
                
                if text_anchor == 'middle':
                    text_left = abs_x - text_width / 2
                    text_right = abs_x + text_width / 2
                elif text_anchor == 'end':
                    text_left = abs_x - text_width
                    text_right = abs_x
                else:  # start
                    text_left = abs_x
                    text_right = abs_x + text_width
                
                text_top = abs_y - text_height
                text_bottom = abs_y
                
                min_x = min(min_x, text_left)
                min_y = min(min_y, text_top)
                max_x = max(max_x, text_right)
                max_y = max(max_y, text_bottom)
        
        elif tag == 'path':
            d = elem.get('d', '')
            if d:
                coords = parse_path_d(d)
                for cx, cy in coords:
                    # Path coordinates are already absolute
                    min_x = min(min_x, cx)
                    min_y = min(min_y, cy)
                    max_x = max(max_x, cx)
                    max_y = max(max_y, cy)
        
        # Process children
        for child in elem:
            process_element(child, abs_tx, abs_ty)
    
    # Process all elements
    for child in root:
        process_element(child)
    
    return (min_x, min_y, max_x, max_y)

def main():
    if len(sys.argv) != 2:
        print("Usage: calculate-bounding-box.py <svg_file>")
        sys.exit(1)
    
    svg_file = sys.argv[1]
    min_x, min_y, max_x, max_y = calculate_bounds(svg_file)
    
    print(f"Bounding box: ({min_x:.1f}, {min_y:.1f}) to ({max_x:.1f}, {max_y:.1f})")
    print(f"Content width: {max_x - min_x:.1f}")
    print(f"Content height: {max_y - min_y:.1f}")
    
    # Add 20px padding
    padding = 20
    padded_min_x = min_x - padding
    padded_min_y = min_y - padding
    padded_max_x = max_x + padding
    padded_max_y = max_y + padding
    
    canvas_width = padded_max_x - padded_min_x
    canvas_height = padded_max_y - padded_min_y
    
    print(f"\nWith 20px padding:")
    print(f"Canvas width: {canvas_width:.1f}")
    print(f"Canvas height: {canvas_height:.1f}")
    print(f"Content offset: ({-padded_min_x:.1f}, {-padded_min_y:.1f})")

if __name__ == '__main__':
    main()

