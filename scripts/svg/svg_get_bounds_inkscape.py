#!/usr/bin/env python3
"""
Get actual SVG bounds using Inkscape's query capabilities.

This uses Inkscape to get the REAL bounding boxes of all objects,
including accurate text measurements based on actual font metrics.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


def get_actual_bounds(svg_path: Path) -> Optional[Tuple[float, float, float, float]]:
    """
    Get actual bounding box using Inkscape's --query-all.
    Returns (min_x, min_y, max_x, max_y) or None if failed.
    """
    result = subprocess.run([
        'inkscape',
        '--query-all',
        str(svg_path)
    ], capture_output=True, text=True)
    
    # Inkscape may return non-zero but still have output
    if not result.stdout.strip():
        if result.stderr:
            print(f"Inkscape error: {result.stderr}", file=sys.stderr)
        return None
    
    # Parse output: id, x, y, width, height (comma-separated)
    all_x = []
    all_y = []
    all_max_x = []
    all_max_y = []
    
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
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
                # min_x = x, min_y = y
                # max_x = x + w, max_y = y + h
                all_x.append(x)
                all_y.append(y)
                all_max_x.append(x + w)
                all_max_y.append(y + h)
            except (ValueError, IndexError):
                continue
    
    if not all_x:
        return None
    
    min_x = min(all_x)
    min_y = min(all_y)
    max_x = max(all_max_x)
    max_y = max(all_max_y)
    
    return (min_x, min_y, max_x, max_y)


def main():
    """CLI for testing"""
    if len(sys.argv) < 2:
        print("Usage: svg_get_bounds_inkscape.py <svg_file>")
        sys.exit(1)
    
    svg_path = Path(sys.argv[1])
    if not svg_path.exists():
        print(f"Error: File not found: {svg_path}")
        sys.exit(1)
    
    bounds = get_actual_bounds(svg_path)
    if bounds:
        min_x, min_y, max_x, max_y = bounds
        width = max_x - min_x
        height = max_y - min_y
        print(f"Bounds: min_x={min_x:.2f}, min_y={min_y:.2f}, max_x={max_x:.2f}, max_y={max_y:.2f}")
        print(f"Dimensions: {width:.2f} x {height:.2f}")
    else:
        print("Failed to get bounds")
        sys.exit(1)


if __name__ == '__main__':
    main()
