#!/usr/bin/env python3
"""
SVG Add Padding

Adds padding to SVG dimensions by expanding width/height.
If viewBox exists, it is also expanded.
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple


def add_padding(svg_path: Path, padding: float) -> Tuple[bool, str]:
    """
    Add padding to SVG by expanding width/height and viewBox.
    
    Args:
        svg_path: Path to SVG file
        padding: Padding in pixels to add on all sides
    
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
        
        current_width = float(width_match.group(1))
        current_height = float(height_match.group(1))
        
        # Add padding (2x because padding is on both sides)
        new_width = current_width + (padding * 2)
        new_height = current_height + (padding * 2)
        
        # Round to 2 decimal places
        new_width = round(new_width, 2)
        new_height = round(new_height, 2)
        
        # Update width and height
        root.set('width', str(new_width))
        root.set('height', str(new_height))
        
        # Update viewBox if it exists
        viewbox_str = root.get('viewBox', '')
        if viewbox_str:
            # Parse viewBox: "minX minY width height"
            viewbox_parts = viewbox_str.strip().split()
            if len(viewbox_parts) == 4:
                min_x = float(viewbox_parts[0])
                min_y = float(viewbox_parts[1])
                viewbox_width = float(viewbox_parts[2])
                viewbox_height = float(viewbox_parts[3])
                
                # Expand viewBox (subtract padding from min, add to dimensions)
                new_min_x = round(min_x - padding, 2)
                new_min_y = round(min_y - padding, 2)
                new_viewbox_width = round(viewbox_width + (padding * 2), 2)
                new_viewbox_height = round(viewbox_height + (padding * 2), 2)
                
                root.set('viewBox', f"{new_min_x} {new_min_y} {new_viewbox_width} {new_viewbox_height}")
        
        # Adjust transforms to account for padding
        # When padding is added, content needs to be shifted by padding amount
        # Find all groups with transforms and adjust them
        for g in root.findall('.//{http://www.w3.org/2000/svg}g'):
            transform = g.get('transform', '')
            if transform:
                # Parse existing transform to add padding offset
                # We need to add translate(padding, padding) to the transform
                # Check if it starts with translate
                if transform.startswith('translate'):
                    # Parse first translate values
                    match = re.match(r'translate\(([^,)]+)(?:,\s*([^)]+))?\)', transform)
                    if match:
                        tx = float(match.group(1))
                        ty = float(match.group(2)) if match.group(2) else 0.0
                        # Add padding to the translate
                        new_tx = round(tx + padding, 2)
                        new_ty = round(ty + padding, 2)
                        # Replace the translate part
                        new_transform = f'translate({new_tx}, {new_ty})'
                        # If there's more to the transform, preserve it
                        rest = transform[match.end():].strip()
                        if rest:
                            new_transform = f'{new_transform} {rest}'
                        g.set('transform', new_transform)
                else:
                    # Prepend translate(padding, padding)
                    new_transform = f'translate({padding}, {padding}) {transform}'
                    g.set('transform', new_transform)
        
        # Write back to file
        tree.write(svg_path, encoding='utf-8', xml_declaration=True)
        
        return True, f"Added {padding}px padding: {current_width}x{current_height} -> {new_width}x{new_height}"
    
    except ET.ParseError as e:
        return False, f"Failed to parse SVG: {e}"
    except Exception as e:
        return False, f"Error processing SVG: {e}"


def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(
        description='Add padding to SVG dimensions. Expands width/height and viewBox (if present).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python svg_add_padding.py diagram.svg --padding 10
        """
    )
    
    parser.add_argument('svg_file', type=Path, help='SVG file to process')
    parser.add_argument('--padding', type=float, required=True,
                       help='Padding in pixels to add on all sides')
    
    args = parser.parse_args()
    
    success, message = add_padding(args.svg_file, args.padding)
    
    if success:
        print(f"✓ {message}")
        sys.exit(0)
    else:
        print(f"✗ Error: {message}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
