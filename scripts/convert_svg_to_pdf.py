#!/usr/bin/env python3
"""
SVG to PDF Converter

Converts SVG diagrams to high-quality PDF format for LaTeX inclusion.
Supports multiple conversion backends with fallback options.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple
import multiprocessing
from functools import partial


def find_converter() -> Optional[str]:
    """Find available SVG to PDF converter (prefers Inkscape for best quality)"""
    # Try Inkscape first (best quality, handles Inkscape connectors properly)
    try:
        result = subprocess.run(['inkscape', '--version'], 
                              capture_output=True, check=True)
        return 'inkscape'
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Try rsvg-convert (good quality, lighter than Inkscape)
    try:
        result = subprocess.run(['rsvg-convert', '--version'], 
                              capture_output=True, check=True)
        return 'rsvg-convert'
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Try cairosvg (Python library, fallback)
    try:
        import cairosvg
        return 'cairosvg'
    except ImportError:
        pass
    
    return None


def convert_with_inkscape(svg_path: Path, pdf_path: Path) -> Tuple[bool, str]:
    """Convert SVG to PDF using Inkscape with optimal settings for diagrams"""
    try:
        result = subprocess.run([
            'inkscape',
            '--export-filename', str(pdf_path),
            '--export-type', 'pdf',
            '--export-area-drawing',  # Export only the drawing area, not the full page
            '--export-text-to-path',  # Convert text to paths for better compatibility
            str(svg_path)
        ], capture_output=True, text=True, check=True)
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, f"Inkscape error: {e.stderr}"
    except FileNotFoundError:
        return False, "Inkscape not found"


def convert_with_rsvg(svg_path: Path, pdf_path: Path) -> Tuple[bool, str]:
    """Convert SVG to PDF using rsvg-convert"""
    try:
        result = subprocess.run([
            'rsvg-convert',
            '-f', 'pdf',
            '-o', str(pdf_path),
            str(svg_path)
        ], capture_output=True, text=True, check=True)
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, f"rsvg-convert error: {e.stderr}"
    except FileNotFoundError:
        return False, "rsvg-convert not found"


def convert_with_cairosvg(svg_path: Path, pdf_path: Path) -> Tuple[bool, str]:
    """Convert SVG to PDF using cairosvg (Python library)"""
    try:
        import cairosvg
        cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
        return True, ""
    except ImportError:
        return False, "cairosvg not installed (pip install cairosvg)"
    except Exception as e:
        return False, f"cairosvg error: {str(e)}"


def convert_svg_to_pdf(svg_path: Path, pdf_path: Path, 
                      converter: Optional[str] = None) -> Tuple[bool, str]:
    """
    Convert SVG to PDF using best available converter
    
    Args:
        svg_path: Path to input SVG file
        pdf_path: Path to output PDF file
        converter: Force specific converter ('inkscape', 'rsvg-convert', 'cairosvg')
    
    Returns:
        (success, error_message)
    """
    if not svg_path.exists():
        return False, f"SVG file not found: {svg_path}"
    
    # Ensure output directory exists
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine converter
    if converter is None:
        converter = find_converter()
        if converter is None:
            return False, "No SVG to PDF converter found. Install inkscape, rsvg-convert, or cairosvg"
    
    # Convert based on available converter
    if converter == 'inkscape':
        return convert_with_inkscape(svg_path, pdf_path)
    elif converter == 'rsvg-convert':
        return convert_with_rsvg(svg_path, pdf_path)
    elif converter == 'cairosvg':
        return convert_with_cairosvg(svg_path, pdf_path)
    else:
        return False, f"Unknown converter: {converter}"


def convert_single_diagram(args: Tuple[Path, Path, Optional[str]]) -> Tuple[Path, bool, str]:
    """Convert single diagram (for parallel processing)"""
    svg_path, pdf_path, converter = args
    success, error = convert_svg_to_pdf(svg_path, pdf_path, converter)
    return (svg_path, success, error)


def convert_diagrams_parallel(svg_files: list, output_dir: Path, 
                              converter: Optional[str] = None,
                              max_workers: Optional[int] = None) -> dict:
    """
    Convert multiple SVG files to PDF in parallel
    
    Args:
        svg_files: List of SVG file paths
        output_dir: Directory for output PDF files
        converter: Force specific converter
        max_workers: Maximum parallel workers (None = CPU count)
    
    Returns:
        Dictionary mapping svg_path -> (success, error_message)
    """
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    # Prepare conversion tasks
    tasks = []
    for svg_path in svg_files:
        # Preserve directory structure in output
        rel_path = svg_path.relative_to(svg_path.parents[1])  # Relative to book/images
        pdf_path = output_dir / rel_path.with_suffix('.pdf')
        tasks.append((svg_path, pdf_path, converter))
    
    # Convert in parallel
    results = {}
    with multiprocessing.Pool(processes=max_workers) as pool:
        for svg_path, success, error in pool.map(convert_single_diagram, tasks):
            results[svg_path] = (success, error)
    
    return results


def find_all_svg_files(images_dir: Path) -> list:
    """Find all SVG files in images directory"""
    svg_files = []
    for svg_path in images_dir.rglob("*.svg"):
        svg_files.append(svg_path)
    return sorted(svg_files)


def main():
    """CLI interface for SVG to PDF conversion"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert SVG diagrams to PDF')
    parser.add_argument('svg_file', nargs='?', type=Path,
                       help='SVG file to convert (or directory for batch)')
    parser.add_argument('-o', '--output', type=Path,
                       help='Output PDF file (or directory for batch)')
    parser.add_argument('--converter', choices=['inkscape', 'rsvg-convert', 'cairosvg'],
                       help='Force specific converter')
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel processing for batch conversion')
    parser.add_argument('--max-workers', type=int,
                       help='Maximum parallel workers')
    
    args = parser.parse_args()
    
    if args.svg_file is None:
        # Default: convert all SVGs in book/images
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        images_dir = project_root / "book" / "images"
        output_dir = project_root / "output" / "diagrams"
        
        if not images_dir.exists():
            print(f"Error: Images directory not found: {images_dir}")
            sys.exit(1)
        
        svg_files = find_all_svg_files(images_dir)
        if not svg_files:
            print("No SVG files found")
            sys.exit(0)
        
        print(f"Found {len(svg_files)} SVG files")
        
        if args.parallel or len(svg_files) > 1:
            results = convert_diagrams_parallel(svg_files, output_dir, 
                                                args.converter, args.max_workers)
            success_count = sum(1 for success, _ in results.values() if success)
            print(f"\nConverted {success_count}/{len(svg_files)} diagrams")
            
            # Report errors
            for svg_path, (success, error) in results.items():
                if not success:
                    print(f"  ✗ {svg_path.name}: {error}")
        else:
            # Single file
            pdf_path = output_dir / args.svg_file.stem / ".pdf"
            success, error = convert_svg_to_pdf(args.svg_file, pdf_path, args.converter)
            if success:
                print(f"✓ Converted: {pdf_path}")
            else:
                print(f"✗ Error: {error}")
                sys.exit(1)
    else:
        # Single file conversion
        if args.output is None:
            args.output = args.svg_file.with_suffix('.pdf')
        
        success, error = convert_svg_to_pdf(args.svg_file, args.output, args.converter)
        if success:
            print(f"✓ Converted: {args.output}")
        else:
            print(f"✗ Error: {error}")
            sys.exit(1)


if __name__ == '__main__':
    main()

