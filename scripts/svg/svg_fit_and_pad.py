#!/usr/bin/env python3
"""
SVG Fit and Pad

Uses Inkscape to:
1. Resize SVG to fit content (--export-area-drawing)
2. Add padding/margin (--export-margin)
3. Add viewBox attribute

This consolidates resize-to-content, add-padding, and add-viewbox operations.
"""

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple


def fit_and_pad(svg_path: Path, padding: float = 20.0) -> Tuple[bool, str]:
    """
    Resize SVG to content, add padding, and add viewBox using Inkscape.
    
    Args:
        svg_path: Path to SVG file
        padding: Padding in pixels to add on all sides (default: 20.0)
    
    Returns:
        (success, error_message)
    """
    if not svg_path.exists():
        return False, f"SVG file not found: {svg_path}"
    
    try:
        # Step 1 & 2: Use Inkscape to resize to content and add padding
        # --export-area-drawing: Export only the drawing area (resize to content)
        # --export-margin: Add margin around export area (must be integer)
        # --export-type=svg: Export as SVG
        # --export-overwrite: Overwrite the input file
        padding_int = int(round(padding))
        result = subprocess.run([
            'inkscape',
            '--export-area-drawing',
            '--export-margin', str(padding_int),
            '--export-type', 'svg',
            '--export-filename', str(svg_path),
            '--export-overwrite',
            '--batch-process',
            str(svg_path)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return False, f"Inkscape failed: {error_msg}"
        
        # Step 3: Add viewBox attribute
        # Parse the SVG that Inkscape just created
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Remove namespace for easier processing
        if root.tag.startswith('{'):
            namespace = root.tag[1:].split('}')[0]
            ET.register_namespace('', namespace)
        
        # Get width and height
        width_str = root.get('width', '')
        height_str = root.get('height', '')
        
        if not width_str or not height_str:
            return False, "SVG must have width and height attributes after Inkscape processing"
        
        # Parse width/height (handle units)
        import re
        width_match = re.match(r'([\d.]+)', width_str)
        height_match = re.match(r'([\d.]+)', height_str)
        
        if not width_match or not height_match:
            return False, "Could not parse width/height values"
        
        width = float(width_match.group(1))
        height = float(height_match.group(1))
        
        # Round to 2 decimal places for consistency
        width = round(width, 2)
        height = round(height, 2)
        
        # Add or update viewBox (always starting at 0,0)
        # When Inkscape exports with --export-area-drawing and --export-margin,
        # it should position content starting at (0,0) with padding around it.
        # The viewBox should always be "0 0 width height" to match the SVG dimensions.
        viewbox_str = f"0 0 {width} {height}"
        root.set('viewBox', viewbox_str)
        
        # Write back to file
        tree.write(svg_path, encoding='utf-8', xml_declaration=True)
        
        return True, f"Resized to content, added {padding}px padding, added viewBox: {width}x{height}"
    
    except FileNotFoundError:
        return False, "Inkscape not found. Please install Inkscape to use this feature."
    except ET.ParseError as e:
        return False, f"Failed to parse SVG: {e}"
    except Exception as e:
        return False, f"Error processing SVG: {e}"


def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(
        description='Resize SVG to content, add padding, and add viewBox using Inkscape. This consolidates resize-to-content, add-padding, and add-viewbox operations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python svg_fit_and_pad.py diagram.svg
  python svg_fit_and_pad.py diagram.svg --padding 10
        """
    )
    
    parser.add_argument('svg_file', type=Path, help='SVG file to process')
    parser.add_argument('--padding', type=float, default=20.0,
                       help='Padding in pixels to add on all sides (default: 20.0)')
    
    args = parser.parse_args()
    
    success, message = fit_and_pad(args.svg_file, args.padding)
    
    if success:
        print(f"✓ {message}")
        sys.exit(0)
    else:
        print(f"✗ Error: {message}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
