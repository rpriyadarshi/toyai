#!/usr/bin/env python3
"""
PDF Generator

Compiles LaTeX source to high-quality PDF with multiple passes
for proper cross-references and table of contents.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple
from build_cache import BuildCache


class PDFGenerator:
    """Generates PDF from LaTeX source"""
    
    def __init__(self, latex_file: Path, output_pdf: Path, cache: BuildCache, 
                 force: bool = False):
        self.latex_file = latex_file
        self.output_pdf = output_pdf
        self.cache = cache
        self.force = force
        self.work_dir = latex_file.parent
    
    def find_latex_engine(self) -> Optional[str]:
        """Find available LaTeX engine"""
        # Try XeLaTeX first (better Unicode support)
        try:
            subprocess.run(['xelatex', '--version'], 
                         capture_output=True, check=True)
            return 'xelatex'
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Try pdfLaTeX
        try:
            subprocess.run(['pdflatex', '--version'], 
                         capture_output=True, check=True)
            return 'pdflatex'
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        return None
    
    def compile_latex(self, engine: str, passes: int = 3) -> Tuple[bool, str]:
        """
        Compile LaTeX to PDF with multiple passes
        
        Args:
            engine: LaTeX engine ('xelatex' or 'pdflatex')
            passes: Number of compilation passes (default 3 for TOC/cross-refs)
        
        Returns:
            (success, error_message)
        """
        if not self.latex_file.exists():
            return False, f"LaTeX file not found: {self.latex_file}"
        
        # Ensure output directory exists
        self.output_pdf.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if needs compilation
        if not self.force:
            source_deps = [str(self.latex_file)]
            if not self.cache.is_output_changed(self.output_pdf, source_deps):
                return True, "PDF up-to-date"
        
        # Compile with multiple passes
        for pass_num in range(1, passes + 1):
            try:
                result = subprocess.run([
                    engine,
                    '-interaction=nonstopmode',
                    '-output-directory', str(self.work_dir),
                    str(self.latex_file.name)
                ], cwd=str(self.work_dir), capture_output=True, text=True, check=False)
                
                if result.returncode != 0:
                    # Check for fatal errors (warnings are OK)
                    error_output = result.stderr or result.stdout
                    # Only fail on fatal errors, not warnings
                    if "Fatal" in error_output or ("Error" in error_output and "Output written" not in error_output):
                        return False, f"LaTeX compilation error (pass {pass_num}):\n{error_output[-1000:]}"
            except FileNotFoundError:
                return False, f"LaTeX engine not found: {engine}"
            except Exception as e:
                return False, f"Compilation error: {str(e)}"
        
        # Check if PDF was created and copy to output location
        pdf_in_work = self.work_dir / f"{self.latex_file.stem}.pdf"
        if pdf_in_work.exists():
            import shutil
            # Always copy (don't move) so we keep the PDF in work directory for LaTeX Workshop
            shutil.copy2(str(pdf_in_work), str(self.output_pdf))
        elif self.output_pdf.exists():
            # PDF exists but wasn't regenerated - might be an error
            pass  # Keep existing PDF
        else:
            return False, "PDF file was not generated"
        
        # Update cache
        source_deps = [str(self.latex_file)]
        self.cache.update_output_cache(self.output_pdf, source_deps)
        
        return True, ""
    
    def generate(self) -> Tuple[bool, str]:
        """Generate PDF from LaTeX source"""
        engine = self.find_latex_engine()
        if engine is None:
            return False, "No LaTeX engine found. Install texlive-xetex or texlive-latex-base"
        
        return self.compile_latex(engine)


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate PDF from LaTeX source')
    parser.add_argument('latex_file', type=Path,
                       help='LaTeX source file')
    parser.add_argument('-o', '--output', type=Path,
                       help='Output PDF file')
    parser.add_argument('--force', action='store_true',
                       help='Force recompilation')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.latex_file.with_suffix('.pdf')
    
    # Initialize cache
    cache_dir = Path(".build")
    cache = BuildCache(cache_dir)
    
    # Generate PDF
    generator = PDFGenerator(args.latex_file, args.output, cache, args.force)
    success, error = generator.generate()
    
    if success:
        print(f"✓ PDF generated: {args.output}")
        cache.save_cache()
    else:
        print(f"✗ Error: {error}")
        sys.exit(1)


if __name__ == '__main__':
    main()

