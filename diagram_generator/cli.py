#!/usr/bin/env python3
"""
SVG Standardization Tool - Main entry point for diagram standardization.

This tool provides commands for:
- Generating SVG from JSON diagram definitions
- Standardizing existing SVGs
- Validating diagram compliance
"""

import argparse
import json
import sys
from pathlib import Path

from diagram_generator.core.generator import SVGGenerator
from diagram_generator.core.diagram import SVGDataset


def generate_command(args):
    """Generate SVG from JSON diagram definition."""
    diagram_path = Path(args.diagram_json)
    output_path = Path(args.output) if args.output else diagram_path.with_suffix('.svg')
    
    if not diagram_path.exists():
        print(f"Error: Diagram file not found: {diagram_path}")
        return 1
    
    try:
        # Load diagram JSON
        with open(diagram_path, 'r') as f:
            diagram_json = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in diagram file: {e}")
        return 1
    except Exception as e:
        print(f"Error: Failed to read diagram file: {e}")
        return 1
    
    try:
        # Generate SVG
        generator = SVGGenerator()
        svg_xml = generator.generate(diagram_json, diagram_path=diagram_path)
    except Exception as e:
        print(f"Error: Failed to generate SVG: {e}")
        return 1
    
    try:
        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(svg_xml)
        print(f"Generated SVG: {output_path}")
        return 0
    except Exception as e:
        print(f"Error: Failed to write SVG file: {e}")
        return 1


def validate_command(args):
    """Validate diagram JSON structure."""
    diagram_path = Path(args.diagram_json)
    
    try:
        with open(diagram_path, 'r') as f:
            diagram_json = json.load(f)
        
        # Basic validation
        required_keys = ["metadata", "components", "connections", "labels"]
        missing_keys = [key for key in required_keys if key not in diagram_json]
        
        if missing_keys:
            print(f"Error: Missing required keys: {missing_keys}")
            return 1
        
        # Validate metadata
        metadata = diagram_json["metadata"]
        if "width" not in metadata or "height" not in metadata:
            print("Error: Metadata must contain width and height")
            return 1
        
        # Validate components
        for i, comp in enumerate(diagram_json.get("components", [])):
            if "id" not in comp:
                print(f"Error: Component {i} missing 'id'")
                return 1
            if "template" not in comp:
                print(f"Error: Component {i} missing 'template'")
                return 1
        
        print("✓ Diagram JSON is valid")
        return 0
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="SVG Diagram Standardization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate SVG from JSON")
    gen_parser.add_argument("diagram_json", help="Path to diagram JSON file")
    gen_parser.add_argument("-o", "--output", help="Output SVG path (default: same as input with .svg extension)")
    
    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate diagram JSON")
    val_parser.add_argument("diagram_json", help="Path to diagram JSON file")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "generate":
        sys.exit(generate_command(args))
    elif args.command == "validate":
        sys.exit(validate_command(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

