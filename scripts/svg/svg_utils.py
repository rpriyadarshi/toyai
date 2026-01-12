#!/usr/bin/env python3
"""
Shared SVG utility functions for content bounds calculation.
"""

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple


def parse_transform(transform_str: Optional[str]) -> Tuple[float, float, float, float, float, float]:
    """
    Parse SVG transform string into transformation matrix.
    Returns (a, b, c, d, e, f) representing matrix:
    [a c e]
    [b d f]
    [0 0 1]
    """
    if not transform_str:
        return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    
    # Start with identity matrix
    matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    
    # Parse transform functions
    # translate(tx, ty)
    translate_match = re.search(r'translate\(([^,)]+)(?:,\s*([^)]+))?\)', transform_str)
    if translate_match:
        tx = float(translate_match.group(1))
        ty = float(translate_match.group(2)) if translate_match.group(2) else 0.0
        matrix[4] += tx
        matrix[5] += ty
    
    # scale(sx, sy) or scale(s)
    scale_match = re.search(r'scale\(([^,)]+)(?:,\s*([^)]+))?\)', transform_str)
    if scale_match:
        sx = float(scale_match.group(1))
        sy = float(scale_match.group(2)) if scale_match.group(2) else sx
        matrix[0] *= sx
        matrix[3] *= sy
    
    # rotate(angle, cx, cy) or rotate(angle)
    rotate_match = re.search(r'rotate\(([^,)]+)(?:,\s*([^,)]+)(?:,\s*([^)]+))?)?\)', transform_str)
    if rotate_match:
        angle = math.radians(float(rotate_match.group(1)))
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        cx = float(rotate_match.group(2)) if rotate_match.group(2) else 0.0
        cy = float(rotate_match.group(3)) if rotate_match.group(3) else 0.0
        
        # Rotation around point (cx, cy)
        # Translate to origin, rotate, translate back
        new_matrix = [1.0, 0.0, 0.0, 1.0, -cx, -cy]
        new_matrix = multiply_matrix(new_matrix, [cos_a, sin_a, -sin_a, cos_a, 0.0, 0.0])
        new_matrix = multiply_matrix(new_matrix, [1.0, 0.0, 0.0, 1.0, cx, cy])
        matrix = multiply_matrix(matrix, new_matrix)
    
    # matrix(a, b, c, d, e, f)
    matrix_match = re.search(r'matrix\(([^,)]+),\s*([^,)]+),\s*([^,)]+),\s*([^,)]+),\s*([^,)]+),\s*([^)]+)\)', transform_str)
    if matrix_match:
        a = float(matrix_match.group(1))
        b = float(matrix_match.group(2))
        c = float(matrix_match.group(3))
        d = float(matrix_match.group(4))
        e = float(matrix_match.group(5))
        f = float(matrix_match.group(6))
        matrix = multiply_matrix(matrix, [a, b, c, d, e, f])
    
    return tuple(matrix)


def multiply_matrix(m1: list, m2: list) -> list:
    """Multiply two 2D transformation matrices"""
    # m1 and m2 are [a, b, c, d, e, f]
    # Result: m1 * m2
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    
    return [
        a1 * a2 + c1 * b2,      # a
        b1 * a2 + d1 * b2,      # b
        a1 * c2 + c1 * d2,      # c
        b1 * c2 + d1 * d2,      # d
        a1 * e2 + c1 * f2 + e1, # e
        b1 * e2 + d1 * f2 + f1  # f
    ]


def apply_transform(x: float, y: float, matrix: Tuple[float, float, float, float, float, float]) -> Tuple[float, float]:
    """Apply transformation matrix to a point"""
    a, b, c, d, e, f = matrix
    new_x = a * x + c * y + e
    new_y = b * x + d * y + f
    return (new_x, new_y)


def get_element_bounds(element: ET.Element, transform_matrix: Tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)) -> Optional[Tuple[float, float, float, float]]:
    """
    Calculate bounding box for an SVG element.
    Returns (min_x, min_y, max_x, max_y) or None if element has no bounds.
    """
    tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag
    
    # Get element's transform and combine with parent transform
    transform_str = element.get('transform', '')
    element_transform = parse_transform(transform_str)
    combined_transform = tuple(multiply_matrix(list(transform_matrix), list(element_transform)))
    
    bounds = None
    
    if tag == 'rect':
        x = float(element.get('x', 0))
        y = float(element.get('y', 0))
        width = float(element.get('width', 0))
        height = float(element.get('height', 0))
        
        # Get all four corners
        corners = [
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height)
        ]
        
        # Transform all corners
        transformed_corners = [apply_transform(cx, cy, combined_transform) for cx, cy in corners]
        
        if transformed_corners:
            xs = [p[0] for p in transformed_corners]
            ys = [p[1] for p in transformed_corners]
            bounds = (min(xs), min(ys), max(xs), max(ys))
    
    elif tag == 'circle':
        cx = float(element.get('cx', 0))
        cy = float(element.get('cy', 0))
        r = float(element.get('r', 0))
        
        # Get bounding box of circle (before transform)
        corners = [
            (cx - r, cy - r),
            (cx + r, cy - r),
            (cx + r, cy + r),
            (cx - r, cy + r)
        ]
        
        transformed_corners = [apply_transform(cx, cy, combined_transform) for cx, cy in corners]
        
        if transformed_corners:
            xs = [p[0] for p in transformed_corners]
            ys = [p[1] for p in transformed_corners]
            bounds = (min(xs), min(ys), max(xs), max(ys))
    
    elif tag == 'ellipse':
        cx = float(element.get('cx', 0))
        cy = float(element.get('cy', 0))
        rx = float(element.get('rx', 0))
        ry = float(element.get('ry', 0))
        
        # Get bounding box of ellipse
        corners = [
            (cx - rx, cy - ry),
            (cx + rx, cy - ry),
            (cx + rx, cy + ry),
            (cx - rx, cy + ry)
        ]
        
        transformed_corners = [apply_transform(cx, cy, combined_transform) for cx, cy in corners]
        
        if transformed_corners:
            xs = [p[0] for p in transformed_corners]
            ys = [p[1] for p in transformed_corners]
            bounds = (min(xs), min(ys), max(xs), max(ys))
    
    elif tag == 'line':
        x1 = float(element.get('x1', 0))
        y1 = float(element.get('y1', 0))
        x2 = float(element.get('x2', 0))
        y2 = float(element.get('y2', 0))
        
        p1 = apply_transform(x1, y1, combined_transform)
        p2 = apply_transform(x2, y2, combined_transform)
        
        bounds = (min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1]))
    
    elif tag == 'path':
        d = element.get('d', '')
        if not d:
            return None
        
        # Parse path data to get all points
        points = []
        current_x, current_y = 0.0, 0.0
        start_x, start_y = 0.0, 0.0
        
        # Split path into commands
        commands = re.findall(r'[MLHVCSQTAZ][^MLHVCSQTAZ]*', d, re.IGNORECASE)
        
        for cmd_str in commands:
            if not cmd_str:
                continue
            
            cmd = cmd_str[0].upper()
            coords_str = cmd_str[1:].strip()
            
            if cmd == 'M':  # Move to
                coords = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', coords_str)
                if len(coords) >= 2:
                    current_x = float(coords[0])
                    current_y = float(coords[1])
                    start_x, start_y = current_x, current_y
                    points.append((current_x, current_y))
                    # Handle multiple coordinates (implicit L commands)
                    for i in range(2, len(coords), 2):
                        if i + 1 < len(coords):
                            current_x = float(coords[i])
                            current_y = float(coords[i + 1])
                            points.append((current_x, current_y))
            
            elif cmd == 'L':  # Line to
                coords = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', coords_str)
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        current_x = float(coords[i])
                        current_y = float(coords[i + 1])
                        points.append((current_x, current_y))
            
            elif cmd == 'H':  # Horizontal line
                coords = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', coords_str)
                for x in coords:
                    current_x = float(x)
                    points.append((current_x, current_y))
            
            elif cmd == 'V':  # Vertical line
                coords = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', coords_str)
                for y in coords:
                    current_y = float(y)
                    points.append((current_x, current_y))
            
            elif cmd == 'C':  # Cubic Bezier - approximate with control points
                coords = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', coords_str)
                for i in range(0, len(coords), 6):
                    if i + 5 < len(coords):
                        # Include control points and end point
                        points.append((float(coords[i]), float(coords[i + 1])))  # cp1
                        points.append((float(coords[i + 2]), float(coords[i + 3])))  # cp2
                        current_x = float(coords[i + 4])
                        current_y = float(coords[i + 5])
                        points.append((current_x, current_y))
            
            elif cmd == 'S':  # Smooth cubic Bezier
                coords = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', coords_str)
                for i in range(0, len(coords), 4):
                    if i + 3 < len(coords):
                        # Approximate with control point and end point
                        points.append((float(coords[i]), float(coords[i + 1])))  # cp2
                        current_x = float(coords[i + 2])
                        current_y = float(coords[i + 3])
                        points.append((current_x, current_y))
            
            elif cmd == 'Q':  # Quadratic Bezier
                coords = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', coords_str)
                for i in range(0, len(coords), 4):
                    if i + 3 < len(coords):
                        points.append((float(coords[i]), float(coords[i + 1])))  # cp
                        current_x = float(coords[i + 2])
                        current_y = float(coords[i + 3])
                        points.append((current_x, current_y))
            
            elif cmd == 'T':  # Smooth quadratic Bezier
                coords = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', coords_str)
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        current_x = float(coords[i])
                        current_y = float(coords[i + 1])
                        points.append((current_x, current_y))
            
            elif cmd == 'A':  # Arc - approximate with start and end points
                coords = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', coords_str)
                for i in range(0, len(coords), 7):
                    if i + 6 < len(coords):
                        rx = float(coords[i])
                        ry = float(coords[i + 1])
                        # Include bounding box of arc
                        current_x = float(coords[i + 5])
                        current_y = float(coords[i + 6])
                        points.append((current_x - rx, current_y - ry))
                        points.append((current_x + rx, current_y + ry))
                        points.append((current_x, current_y))
            
            elif cmd == 'Z':  # Close path
                if points:
                    points.append((start_x, start_y))
        
        if not points:
            return None
        
        # Transform all points
        transformed_points = [apply_transform(px, py, combined_transform) for px, py in points]
        
        if transformed_points:
            xs = [p[0] for p in transformed_points]
            ys = [p[1] for p in transformed_points]
            bounds = (min(xs), min(ys), max(xs), max(ys))
    
    elif tag == 'polygon' or tag == 'polyline':
        points_str = element.get('points', '')
        if not points_str:
            return None
        
        # Parse points: "x1,y1 x2,y2 ..." or "x1,y1, x2,y2, ..."
        coords = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', points_str)
        points = []
        for i in range(0, len(coords), 2):
            if i + 1 < len(coords):
                points.append((float(coords[i]), float(coords[i + 1])))
        
        if not points:
            return None
        
        transformed_points = [apply_transform(px, py, combined_transform) for px, py in points]
        
        if transformed_points:
            xs = [p[0] for p in transformed_points]
            ys = [p[1] for p in transformed_points]
            bounds = (min(xs), min(ys), max(xs), max(ys))
    
    elif tag == 'text':
        x = float(element.get('x', 0))
        y = float(element.get('y', 0))
        
        # Estimate text bounds based on font-size
        font_size = float(element.get('font-size', 12))
        text_content = (element.text or '').strip()
        
        # Rough estimate: assume average character width is 0.6 * font_size
        # Text height: baseline is at y, text extends above and slightly below
        estimated_width = len(text_content) * font_size * 0.6 if text_content else font_size
        
        # Account for text-anchor (horizontal alignment)
        text_anchor = element.get('text-anchor', 'start')
        if text_anchor == 'middle':
            x -= estimated_width / 2
        elif text_anchor == 'end':
            x -= estimated_width
        
        # Account for dominant-baseline (vertical alignment)
        # In SVG, y position is typically the baseline
        # Text extends ~0.7*font_size above baseline (for capitals/ascenders)
        # and ~0.3*font_size below baseline (for descenders)
        dominant_baseline = element.get('dominant-baseline', 'auto')
        if dominant_baseline == 'middle':
            # Text centered vertically on y
            y_top = y - font_size * 0.5
            y_bottom = y + font_size * 0.5
        elif dominant_baseline == 'hanging':
            # Top of text at y (like hanging from a line)
            y_top = y
            y_bottom = y + font_size
        else:
            # Default: baseline at y
            # Most text (capitals, numbers) extends ~0.6*font_size above baseline
            # Some text (g, p, q, y) extends ~0.2*font_size below baseline
            # Use conservative estimate
            y_top = y - font_size * 0.6
            y_bottom = y + font_size * 0.2
        
        # Check if this text has a rotation transform
        # If it does, we need to be more careful about bounds
        element_transform_str = element.get('transform', '')
        has_rotation = 'rotate' in element_transform_str.lower()
        
        # For rotated text, use a more conservative bounding box
        # Rotated text can extend in unexpected directions
        if has_rotation:
            # For rotated text, use a square bounding box centered on (x, y)
            # with size based on text length and font size
            text_size = max(estimated_width, font_size)
            half_size = text_size * 0.5
            corners = [
                (x - half_size, y - half_size),
                (x + half_size, y - half_size),
                (x + half_size, y + half_size),
                (x - half_size, y + half_size)
            ]
        else:
            corners = [
                (x, y_top),
                (x + estimated_width, y_top),
                (x + estimated_width, y_bottom),
                (x, y_bottom)
            ]
        
        transformed_corners = [apply_transform(cx, cy, combined_transform) for cx, cy in corners]
        
        if transformed_corners:
            xs = [p[0] for p in transformed_corners]
            ys = [p[1] for p in transformed_corners]
            bounds = (min(xs), min(ys), max(xs), max(ys))
    
    elif tag == 'g':
        # Groups don't have bounds themselves, but we'll recurse into children
        return None
    
    return bounds


def calculate_content_bounds_inkscape(svg_path: Path) -> Optional[Tuple[float, float, float, float]]:
    """
    Calculate bounds using Inkscape's --query-all to get actual bounding boxes.
    Iterates over ALL objects, gets bounding box for each, then runs min/max.
    
    Returns (min_x, min_y, max_x, max_y) or None if failed.
    """
    import subprocess
    
    result = subprocess.run([
        'inkscape',
        '--query-all',
        str(svg_path)
    ], capture_output=True, text=True)
    
    if not result.stdout.strip():
        return None
    
    # Parse output: id, x, y, width, height (comma-separated)
    all_x = []
    all_y = []
    all_max_x = []
    all_max_y = []
    
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 5:
            try:
                obj_id = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                
                # (x, y) is top-left corner, so bounds are:
                all_x.append(x)
                all_y.append(y)
                all_max_x.append(x + w)
                all_max_y.append(y + h)
            except (ValueError, IndexError):
                continue
    
    if not all_x:
        return None
    
    # Run min/max to get overall bounding box
    min_x = min(all_x)
    min_y = min(all_y)
    max_x = max(all_max_x)
    max_y = max(all_max_y)
    
    return (min_x, min_y, max_x, max_y)


def calculate_content_bounds(svg_root: ET.Element, transform_matrix: Tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)) -> Optional[Tuple[float, float, float, float]]:
    """
    Traverse SVG tree and calculate bounding box of all visible elements.
    Returns (min_x, min_y, max_x, max_y) or None if no content found.
    """
    all_bounds = []
    
    # Get element's transform
    transform_str = svg_root.get('transform', '')
    element_transform = parse_transform(transform_str)
    combined_transform = tuple(multiply_matrix(list(transform_matrix), list(element_transform)))
    
    # Check if this element itself has bounds
    element_bounds = get_element_bounds(svg_root, transform_matrix)
    if element_bounds:
        all_bounds.append(element_bounds)
    
    # Recurse into children
    for child in svg_root:
        child_bounds = calculate_content_bounds(child, combined_transform)
        if child_bounds:
            all_bounds.append(child_bounds)
    
    if not all_bounds:
        return None
    
    # Combine all bounds
    min_x = min(b[0] for b in all_bounds)
    min_y = min(b[1] for b in all_bounds)
    max_x = max(b[2] for b in all_bounds)
    max_y = max(b[3] for b in all_bounds)
    
    return (min_x, min_y, max_x, max_y)
