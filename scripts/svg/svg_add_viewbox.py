#!/usr/bin/env python3
"""
SVG Add ViewBox

Adds viewBox attribute to SVG based on current width/height.
Sets viewBox to "0 0 width height" (industry best practice).
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple


def add_viewbox(svg_path: Path) -> Tuple[bool, str]:
    """
    Add viewBox attribute to SVG based on width/height.
    Sets viewBox to "0 0 width height".
    
    Returns:
        (success, error_message)
    """
    if not svg_path.exists():
        return False, f"SVG file not found: {svg_path}"
    
    try:
        # Parse SVG
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Remove namespace for easier processing
        if root.tag.startswith('{'):
            # Extract namespace
            namespace = root.tag[1:].split('}')[0]
            ET.register_namespace('', namespace)
        
        # Get current width and height
        width_str = root.get('width', '')
        height_str = root.get('height', '')
        
        if not width_str or not height_str:
            return False, "SVG must have width and height attributes"
        
        # Parse width/height (handle units)
        width_match = re.match(r'([\d.]+)', width_str)
        height_match = re.match(r'([\d.]+)', height_str)
        
        if not width_match or not height_match:
            return False, "Could not parse width/height values"
        
        width = float(width_match.group(1))
        height = float(height_match.group(1))
        
        # Round to 2 decimal places
        width = round(width, 2)
        height = round(height, 2)
        
        # Add or update viewBox (starting at 0,0)
        viewbox_str = f"0 0 {width} {height}"
        root.set('viewBox', viewbox_str)
        
        # Write back to file
        tree.write(svg_path, encoding='utf-8', xml_declaration=True)
        
        return True, f"Added viewBox: '{viewbox_str}'"
    
    except ET.ParseError as e:
        return False, f"Failed to parse SVG: {e}"
    except Exception as e:
        return False, f"Error processing SVG: {e}"


def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(
        description='Add viewBox attribute to SVG based on width/height. Sets viewBox to "0 0 width height" (industry best practice).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python svg_add_viewbox.py diagram.svg
        """
    )
    
    parser.add_argument('svg_file', type=Path, help='SVG file to process')
    
    args = parser.parse_args()
    
    success, message = add_viewbox(args.svg_file)
    
    if success:
        print(f"✓ {message}")
        sys.exit(0)
    else:
        print(f"✗ Error: {message}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
