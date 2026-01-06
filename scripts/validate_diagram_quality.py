#!/usr/bin/env python3
"""
Diagram Quality Validator

Validates that SVG diagrams convert properly and maintain quality
in PDF/PNG formats.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class DiagramQualityValidator:
    """Validates diagram conversion quality"""
    
    def __init__(self, images_dir: Path, output_diagrams_dir: Path):
        self.images_dir = images_dir
        self.output_diagrams_dir = output_diagrams_dir
    
    def find_all_svgs(self) -> List[Path]:
        """Find all SVG files"""
        svg_files = []
        for svg_path in self.images_dir.rglob("*.svg"):
            svg_files.append(svg_path)
        return sorted(svg_files)
    
    def check_pdf_exists(self, svg_path: Path) -> Tuple[bool, Optional[Path]]:
        """Check if PDF version exists"""
        rel_path = svg_path.relative_to(self.images_dir)
        pdf_path = self.output_diagrams_dir / rel_path.with_suffix('.pdf')
        
        if pdf_path.exists():
            return True, pdf_path
        return False, None
    
    def check_png_exists(self, svg_path: Path) -> Tuple[bool, Optional[Path]]:
        """Check if PNG version exists (for Word)"""
        rel_path = svg_path.relative_to(self.images_dir)
        png_path = self.output_diagrams_dir.parent / "word" / "images" / rel_path.with_suffix('.png')
        
        if png_path.exists():
            return True, png_path
        return False, None
    
    def validate_pdf_quality(self, pdf_path: Path) -> Tuple[bool, str]:
        """Validate PDF quality (basic checks)"""
        if not pdf_path.exists():
            return False, "PDF file not found"
        
        # Check file size (should be reasonable)
        size = pdf_path.stat().st_size
        if size < 100:  # Too small, probably empty
            return False, f"PDF file too small ({size} bytes)"
        
        if size > 50 * 1024 * 1024:  # > 50MB, probably too large
            return False, f"PDF file too large ({size / 1024 / 1024:.1f} MB)"
        
        return True, "OK"
    
    def validate_png_quality(self, png_path: Path, min_dpi: int = 300) -> Tuple[bool, str]:
        """Validate PNG quality"""
        if not png_path.exists():
            return False, "PNG file not found"
        
        if not HAS_PIL:
            # Basic size check
            size = png_path.stat().st_size
            if size < 1000:
                return False, f"PNG file too small ({size} bytes)"
            return True, "OK (PIL not available for detailed check)"
        
        try:
            img = Image.open(png_path)
            width, height = img.size
            
            # Estimate DPI (assuming letter size: 8.5x11 inches)
            # This is approximate
            estimated_dpi = min(width / 8.5, height / 11)
            
            if estimated_dpi < min_dpi:
                return False, f"Estimated DPI too low: {estimated_dpi:.0f} (min: {min_dpi})"
            
            # Check format
            if img.format != 'PNG':
                return False, f"Wrong format: {img.format} (expected PNG)"
            
            return True, f"OK ({width}x{height}, ~{estimated_dpi:.0f} DPI)"
        except Exception as e:
            return False, f"Error reading PNG: {str(e)}"
    
    def validate_all(self, verbose: bool = False) -> Dict:
        """Validate all diagrams"""
        svg_files = self.find_all_svgs()
        results = {
            "total": len(svg_files),
            "pdf_ok": 0,
            "pdf_missing": 0,
            "pdf_errors": [],
            "png_ok": 0,
            "png_missing": 0,
            "png_errors": [],
            "errors": []
        }
        
        for svg_path in svg_files:
            if verbose:
                print(f"Checking {svg_path.name}...")
            
            # Check PDF
            pdf_exists, pdf_path = self.check_pdf_exists(svg_path)
            if pdf_exists:
                pdf_ok, pdf_msg = self.validate_pdf_quality(pdf_path)
                if pdf_ok:
                    results["pdf_ok"] += 1
                else:
                    results["pdf_errors"].append((svg_path.name, pdf_msg))
            else:
                results["pdf_missing"] += 1
                results["errors"].append(f"{svg_path.name}: PDF not found")
            
            # Check PNG (for Word)
            png_exists, png_path = self.check_png_exists(svg_path)
            if png_exists:
                png_ok, png_msg = self.validate_png_quality(png_path)
                if png_ok:
                    results["png_ok"] += 1
                else:
                    results["png_errors"].append((svg_path.name, png_msg))
            else:
                results["png_missing"] += 1
        
        return results
    
    def print_report(self, results: Dict):
        """Print validation report"""
        print("\n" + "=" * 70)
        print("Diagram Quality Validation Report")
        print("=" * 70)
        print(f"\nTotal SVG diagrams: {results['total']}")
        
        print(f"\nPDF Validation:")
        print(f"  ✓ Valid: {results['pdf_ok']}")
        print(f"  ✗ Missing: {results['pdf_missing']}")
        if results['pdf_errors']:
            print(f"  ✗ Errors: {len(results['pdf_errors'])}")
            for name, error in results['pdf_errors']:
                print(f"    - {name}: {error}")
        
        print(f"\nPNG Validation (for Word):")
        print(f"  ✓ Valid: {results['png_ok']}")
        print(f"  ✗ Missing: {results['png_missing']}")
        if results['png_errors']:
            print(f"  ✗ Errors: {len(results['png_errors'])}")
            for name, error in results['png_errors']:
                print(f"    - {name}: {error}")
        
        if results['errors']:
            print(f"\nOther Errors:")
            for error in results['errors']:
                print(f"  - {error}")
        
        print("\n" + "=" * 70)
        
        # Return exit code
        if results['pdf_missing'] > 0 or results['pdf_errors'] or \
           results['png_missing'] > 0 or results['png_errors']:
            return 1
        return 0


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate diagram conversion quality')
    parser.add_argument('--images-dir', type=Path, default=Path("book/images"),
                       help='Directory containing SVG images')
    parser.add_argument('--output-dir', type=Path, default=Path("output"),
                       help='Output directory with converted diagrams')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    validator = DiagramQualityValidator(args.images_dir, args.output_dir / "diagrams")
    results = validator.validate_all(args.verbose)
    exit_code = validator.print_report(results)
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()

