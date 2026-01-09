#!/usr/bin/env python3
"""
Main Build Script - Compiler-Like Book Builder

Unified entry point for building the book in multiple formats with
incremental builds, dependency tracking, and parallel processing.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Optional

# Add scripts directory to path for imports
_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from build_cache import BuildCache
from build_manifest import BuildManifest
from build_latex_book import LaTeXBookBuilder
from generate_pdf import PDFGenerator
from generate_postscript import PostScriptGenerator
from generate_word import WordGenerator
from convert_svg_to_pdf import convert_diagrams_parallel
from validate_diagram_quality import DiagramQualityValidator


class BookBuilder:
    """Main book builder with compiler-like features"""
    
    def __init__(self, book_dir: Path, output_dir: Path, 
                 cache_dir: Path = Path(".build"),
                 force: bool = False, verbose: bool = False, quiet: bool = False,
                 sage_template: bool = True):
        self.book_dir = book_dir
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.force = force
        self.verbose = verbose
        self.quiet = quiet
        self.sage_template = sage_template
        
        # Initialize cache and manifest
        self.cache = BuildCache(cache_dir)
        self.manifest = BuildManifest(self.cache)
        
        # Track build statistics
        self.stats = {
            "diagrams_converted": 0,
            "diagrams_skipped": 0,
            "chapters_processed": 0,
            "chapters_skipped": 0,
            "errors": []
        }
    
    def log(self, message: str, level: str = "info"):
        """Log message with appropriate level"""
        if self.quiet and level != "error":
            return
        
        prefix = {
            "info": "",
            "success": "✓",
            "warning": "⚠",
            "error": "✗",
            "progress": "[",
        }
        
        if level == "progress":
            print(message, end="", flush=True)
        else:
            if prefix[level]:
                print(f"{prefix[level]} {message}")
            else:
                print(message)
    
    def clean(self):
        """Clean all build artifacts"""
        import shutil
        
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.log("Removed .build/ directory", "success")
        
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            self.log("Removed output/ directory", "success")
        
        self.log("Clean complete", "success")
    
    def build_diagrams(self) -> bool:
        """Convert all SVG diagrams to PDF (incremental)"""
        images_dir = self.book_dir / "images"
        diagrams_dir = self.output_dir / "diagrams"
        
        if not images_dir.exists():
            self.log(f"Images directory not found: {images_dir}", "error")
            return False
        
        # Find all SVG files
        svg_files = []
        for svg_path in images_dir.rglob("*.svg"):
            svg_files.append(svg_path)
        svg_files = sorted(svg_files)
        if not svg_files:
            self.log("No SVG files found", "warning")
            return True
        
        # Determine which need conversion
        diagrams_to_convert = []
        for svg_path in svg_files:
            if self.force or self.cache.is_diagram_changed(svg_path):
                diagrams_to_convert.append(svg_path)
            else:
                self.stats["diagrams_skipped"] += 1
        
        if not diagrams_to_convert:
            self.log("All diagrams up-to-date", "success")
            return True
        
        self.log(f"Converting {len(diagrams_to_convert)} diagram(s)...", "info")
        
        # Convert in parallel
        from convert_svg_to_pdf import convert_svg_to_pdf, find_converter
        converter = find_converter()
        
        if converter is None:
            self.log("No SVG to PDF converter found. Install inkscape, rsvg-convert, or cairosvg", "error")
            return False
        
        success_count = 0
        for i, svg_path in enumerate(diagrams_to_convert, 1):
            if not self.quiet:
                self.log(f"[{i}/{len(diagrams_to_convert)}] {svg_path.name}...", "progress")
            
            # Determine output path
            rel_path = svg_path.relative_to(images_dir)
            pdf_path = diagrams_dir / rel_path.with_suffix('.pdf')
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert
            success, error = convert_svg_to_pdf(svg_path, pdf_path, converter)
            
            if success:
                self.cache.update_diagram_cache(svg_path, pdf_path)
                self.manifest.update_manifest_from_conversion(svg_path, pdf_path)
                success_count += 1
                self.stats["diagrams_converted"] += 1
                if not self.quiet:
                    print(" ✓")
            else:
                self.log(f"Failed to convert {svg_path.name}: {error}", "error")
                self.stats["errors"].append(f"Diagram {svg_path.name}: {error}")
                if not self.quiet:
                    print(" ✗")
        
        if success_count == len(diagrams_to_convert):
            self.log(f"All {success_count} diagram(s) converted successfully", "success")
            return True
        else:
            self.log(f"Converted {success_count}/{len(diagrams_to_convert)} diagram(s)", "warning")
            return success_count > 0
    
    def build_latex(self) -> Optional[Path]:
        """Build LaTeX source from Markdown chapters"""
        self.log("Building LaTeX source...", "info")
        
        builder = LaTeXBookBuilder(self.book_dir, self.output_dir, 
                                  self.cache, self.manifest, self.force, self.sage_template)
        main_tex = builder.build_complete_latex()
        
        if main_tex:
            self.log(f"LaTeX source built: {main_tex}", "success")
            return main_tex
        else:
            self.log("Failed to build LaTeX source", "error")
            return None
    
    def build_pdf(self, latex_file: Optional[Path] = None) -> Optional[Path]:
        """Build PDF from LaTeX source"""
        if latex_file is None:
            latex_file = self.output_dir / "latex" / "book.tex"
        
        if not latex_file.exists():
            self.log(f"LaTeX file not found: {latex_file}", "error")
            return None
        
        self.log("Generating PDF...", "info")
        
        output_pdf = self.output_dir / "pdf" / "book.pdf"
        generator = PDFGenerator(latex_file, output_pdf, self.cache, self.force)
        success, error = generator.generate()
        
        if success:
            if error and "up-to-date" in error:
                self.log("PDF up-to-date", "success")
            else:
                self.log(f"PDF generated: {output_pdf}", "success")
            return output_pdf
        else:
            self.log(f"Failed to generate PDF: {error}", "error")
            return None
    
    def build_postscript(self, pdf_file: Optional[Path] = None) -> Optional[Path]:
        """Build PostScript from PDF"""
        if pdf_file is None:
            pdf_file = self.output_dir / "pdf" / "book.pdf"
        
        if not pdf_file.exists():
            self.log(f"PDF file not found: {pdf_file}", "error")
            return None
        
        self.log("Generating PostScript...", "info")
        
        output_ps = self.output_dir / "postscript" / "book.ps"
        generator = PostScriptGenerator(pdf_file, output_ps, self.cache, self.force)
        success, error = generator.generate()
        
        if success:
            if error and "up-to-date" in error:
                self.log("PostScript up-to-date", "success")
            else:
                self.log(f"PostScript generated: {output_ps}", "success")
            return output_ps
        else:
            self.log(f"Failed to generate PostScript: {error}", "error")
            return None
    
    def build_word(self) -> Optional[Path]:
        """Build Word document"""
        self.log("Generating Word document...", "info")
        
        generator = WordGenerator(self.book_dir, self.output_dir, 
                                 self.cache, self.manifest, self.force)
        output_docx = generator.build_word_document()
        
        if output_docx:
            self.log(f"Word document generated: {output_docx}", "success")
            return output_docx
        else:
            self.log("Failed to generate Word document", "error")
            return None
    
    def build_all(self):
        """Build all formats"""
        start_time = time.time()
        
        self.log("=" * 70, "info")
        self.log("Building Complete Book", "info")
        self.log("=" * 70, "info")
        self.log("", "info")
        
        # Build diagrams first (needed by all formats)
        if not self.build_diagrams():
            self.log("Diagram conversion failed, continuing anyway...", "warning")
        
        # Build LaTeX
        latex_file = self.build_latex()
        if not latex_file:
            self.log("LaTeX build failed", "error")
            return False
        
        # Build PDF
        pdf_file = self.build_pdf(latex_file)
        
        # Build PostScript (if PDF succeeded)
        if pdf_file:
            self.build_postscript(pdf_file)
        
        # Build Word
        self.build_word()
        
        # Save cache
        self.cache.save_cache()
        
        # Print summary
        elapsed = time.time() - start_time
        self.log("", "info")
        self.log("=" * 70, "info")
        self.log("Build Summary", "info")
        self.log("=" * 70, "info")
        self.log(f"Diagrams: {self.stats['diagrams_converted']} converted, "
                f"{self.stats['diagrams_skipped']} skipped", "info")
        self.log(f"Chapters: {self.stats['chapters_processed']} processed, "
                f"{self.stats['chapters_skipped']} skipped", "info")
        if self.stats['errors']:
            self.log(f"Errors: {len(self.stats['errors'])}", "error")
        self.log(f"Build time: {elapsed:.1f}s", "info")
        self.log("=" * 70, "info")
        
        return len(self.stats['errors']) == 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Build book in multiple formats (compiler-like build system)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/build_book.py --all              # Build all formats
  python3 scripts/build_book.py --latex --pdf      # Build LaTeX and PDF
  python3 scripts/build_book.py --pdf --force      # Force rebuild PDF
  python3 scripts/build_book.py --clean            # Clean build artifacts
        """
    )
    
    # Build targets
    parser.add_argument('--latex', action='store_true',
                       help='Generate LaTeX source')
    parser.add_argument('--pdf', action='store_true',
                       help='Generate PDF (requires LaTeX)')
    parser.add_argument('--ps', action='store_true',
                       help='Generate PostScript (requires PDF)')
    parser.add_argument('--word', action='store_true',
                       help='Generate Word document')
    parser.add_argument('--all', action='store_true',
                       help='Generate all formats')
    
    # Build options
    parser.add_argument('--force', action='store_true',
                       help='Force rebuild everything (ignore cache)')
    parser.add_argument('--clean', action='store_true',
                       help='Clean all build artifacts')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quiet', action='store_true',
                       help='Quiet mode (errors only)')
    parser.add_argument('--standard', action='store_true',
                       help='Use standard template instead of SAGE (SAGE is default)')
    # SAGE is default, so we track it as "not standard"
    parser.set_defaults(sage=True)
    
    # Paths
    parser.add_argument('--book-dir', type=Path, default=Path("book"),
                       help='Directory containing Markdown chapters')
    parser.add_argument('--output-dir', type=Path, default=Path("output"),
                       help='Output directory')
    parser.add_argument('--cache-dir', type=Path, default=Path(".build"),
                       help='Build cache directory')
    
    args = parser.parse_args()
    
    # Determine template: SAGE is default unless --standard is specified
    use_sage = not getattr(args, 'standard', False)
    
    # Initialize builder
    builder = BookBuilder(
        args.book_dir,
        args.output_dir,
        args.cache_dir,
        args.force,
        args.verbose,
        args.quiet,
        use_sage
    )
    
    # Handle clean
    if args.clean:
        builder.clean()
        return 0
    
    # Determine build targets
    if args.all:
        success = builder.build_all()
    else:
        success = True
        
        # Build diagrams if needed
        if args.latex or args.pdf or args.ps or args.word:
            if not builder.build_diagrams():
                success = False
        
        # Build LaTeX
        if args.latex or args.pdf or args.ps:
            latex_file = builder.build_latex()
            if not latex_file:
                success = False
        
        # Build PDF
        if args.pdf or args.ps:
            pdf_file = builder.build_pdf()
            if not pdf_file:
                success = False
        
        # Build PostScript
        if args.ps:
            if not builder.build_postscript():
                success = False
        
        # Build Word
        if args.word:
            if not builder.build_word():
                success = False
        
        # Save cache
        builder.cache.save_cache()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
