#!/usr/bin/env python3
"""
SVG Resize to Content

Removes viewBox and resizes SVG to tightly fit content.
Sets width/height to content bounds starting at origin (0,0).
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

# Import from same directory
sys.path.insert(0, str(Path(__file__).parent))
from svg_utils import calculate_content_bounds_inkscape


def resize_to_content(svg_path: Path) -> Tuple[bool, str]:
    """
    Remove viewBox and resize SVG to fit content.
    Content is repositioned to start at (0,0).
    
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
        
        # Calculate content bounds using Inkscape: iterate over ALL objects,
        # get bounding box for each, then run min/max
        bounds = calculate_content_bounds_inkscape(svg_path)
        
        if bounds is None:
            return False, "No visible content found in SVG or Inkscape query failed"
        
        min_x, min_y, max_x, max_y = bounds
        
        # Calculate dimensions
        width = max_x - min_x
        height = max_y - min_y
        
        # Ensure positive dimensions
        if width <= 0 or height <= 0:
            return False, f"Invalid bounds: width={width}, height={height}"
        
        # Round to 2 decimal places
        width = round(width, 2)
        height = round(height, 2)
        
        # Remove viewBox if it exists
        if 'viewBox' in root.attrib:
            del root.attrib['viewBox']
        
        # Update width and height
        root.set('width', str(width))
        root.set('height', str(height))
        
        # Translate all content to start at (0,0)
        # We need to apply a transform to shift content by (-min_x, -min_y)
        # Check if root has children
        if len(root) > 0:
            # Check if there's already a single group child - if so, update its transform
            children = list(root)
            if len(children) == 1 and children[0].tag.split('}')[-1] == 'g':
                # Update existing group's transform
                existing_transform = children[0].get('transform', '')
                translate_x = round(-min_x, 2)
                translate_y = round(-min_y, 2)
                if existing_transform:
                    # Combine transforms
                    new_transform = f'translate({translate_x}, {translate_y}) {existing_transform}'
                else:
                    new_transform = f'translate({translate_x}, {translate_y})'
                children[0].set('transform', new_transform)
            else:
                # Create a wrapper group with translate transform
                wrapper = ET.Element('g')
                translate_x = round(-min_x, 2)
                translate_y = round(-min_y, 2)
                wrapper.set('transform', f'translate({translate_x}, {translate_y})')
                
                # Move all children to wrapper
                for child in children:
                    root.remove(child)
                    wrapper.append(child)
                
                # Add wrapper to root
                root.append(wrapper)
        
        # Write back to file
        tree.write(svg_path, encoding='utf-8', xml_declaration=True)
        
        return True, f"Resized to {width}x{height}, removed viewBox, content repositioned to (0,0)"
    
    except ET.ParseError as e:
        return False, f"Failed to parse SVG: {e}"
    except Exception as e:
        return False, f"Error processing SVG: {e}"


def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(
        description='Remove viewBox and resize SVG to fit content. Content is repositioned to start at (0,0).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python svg_resize_to_content.py diagram.svg
        """
    )
    
    parser.add_argument('svg_file', type=Path, help='SVG file to process')
    
    args = parser.parse_args()
    
    success, message = resize_to_content(args.svg_file)
    
    if success:
        print(f"✓ {message}")
        sys.exit(0)
    else:
        print(f"✗ Error: {message}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
