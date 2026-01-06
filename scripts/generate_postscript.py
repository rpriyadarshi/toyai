#!/usr/bin/env python3
"""
PostScript Generator

Converts PDF to PostScript format for print workflows.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple
from build_cache import BuildCache


class PostScriptGenerator:
    """Generates PostScript from PDF"""
    
    def __init__(self, pdf_file: Path, output_ps: Path, cache: BuildCache, 
                 force: bool = False):
        self.pdf_file = pdf_file
        self.output_ps = output_ps
        self.cache = cache
        self.force = force
    
    def find_converter(self) -> Optional[str]:
        """Find available PDF to PostScript converter"""
        # Try pdftops first (best quality)
        try:
            subprocess.run(['pdftops', '-v'], 
                         capture_output=True, check=True)
            return 'pdftops'
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Try pdf2ps (alternative)
        try:
            subprocess.run(['pdf2ps', '-v'], 
                         capture_output=True, check=True)
            return 'pdf2ps'
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        return None
    
    def convert_with_pdftops(self) -> Tuple[bool, str]:
        """Convert PDF to PostScript using pdftops"""
        try:
            result = subprocess.run([
                'pdftops',
                str(self.pdf_file),
                str(self.output_ps)
            ], capture_output=True, text=True, check=True)
            return True, ""
        except subprocess.CalledProcessError as e:
            return False, f"pdftops error: {e.stderr}"
        except FileNotFoundError:
            return False, "pdftops not found"
    
    def convert_with_pdf2ps(self) -> Tuple[bool, str]:
        """Convert PDF to PostScript using pdf2ps"""
        try:
            result = subprocess.run([
                'pdf2ps',
                str(self.pdf_file),
                str(self.output_ps)
            ], capture_output=True, text=True, check=True)
            return True, ""
        except subprocess.CalledProcessError as e:
            return False, f"pdf2ps error: {e.stderr}"
        except FileNotFoundError:
            return False, "pdf2ps not found"
    
    def generate(self) -> Tuple[bool, str]:
        """Generate PostScript from PDF"""
        if not self.pdf_file.exists():
            return False, f"PDF file not found: {self.pdf_file}"
        
        # Check if needs conversion
        if not self.force:
            source_deps = [str(self.pdf_file)]
            if not self.cache.is_output_changed(self.output_ps, source_deps):
                return True, "PostScript up-to-date"
        
        # Ensure output directory exists
        self.output_ps.parent.mkdir(parents=True, exist_ok=True)
        
        # Find converter
        converter = self.find_converter()
        if converter is None:
            return False, "No PDF to PostScript converter found. Install poppler-utils or ghostscript"
        
        # Convert
        if converter == 'pdftops':
            success, error = self.convert_with_pdftops()
        elif converter == 'pdf2ps':
            success, error = self.convert_with_pdf2ps()
        else:
            return False, f"Unknown converter: {converter}"
        
        if success:
            # Update cache
            source_deps = [str(self.pdf_file)]
            self.cache.update_output_cache(self.output_ps, source_deps)
        
        return success, error


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate PostScript from PDF')
    parser.add_argument('pdf_file', type=Path,
                       help='Input PDF file')
    parser.add_argument('-o', '--output', type=Path,
                       help='Output PostScript file')
    parser.add_argument('--force', action='store_true',
                       help='Force reconversion')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.pdf_file.with_suffix('.ps')
    
    # Initialize cache
    cache_dir = Path(".build")
    cache = BuildCache(cache_dir)
    
    # Generate PostScript
    generator = PostScriptGenerator(args.pdf_file, args.output, cache, args.force)
    success, error = generator.generate()
    
    if success:
        print(f"✓ PostScript generated: {args.output}")
        cache.save_cache()
    else:
        print(f"✗ Error: {error}")
        sys.exit(1)


if __name__ == '__main__':
    main()

