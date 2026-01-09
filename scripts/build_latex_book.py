#!/usr/bin/env python3
"""
LaTeX Book Builder

Converts Markdown chapters to LaTeX format with proper structure,
diagram handling, and incremental build support.
"""

import subprocess
import sys
import re
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from build_cache import BuildCache
from build_manifest import BuildManifest
from convert_svg_to_pdf import convert_diagrams_parallel, find_converter


class LaTeXBookBuilder:
    """Builds LaTeX book from Markdown chapters"""
    
    def __init__(self, book_dir: Path, output_dir: Path, cache: BuildCache, 
                 manifest: BuildManifest, force: bool = False, sage_template: bool = True):
        self.book_dir = book_dir
        self.output_dir = output_dir
        self.latex_dir = output_dir / "latex"
        self.diagrams_dir = output_dir / "diagrams"
        self.cache = cache
        self.manifest = manifest
        self.force = force
        self.sage_template = sage_template
        
        # Ensure directories exist
        self.latex_dir.mkdir(parents=True, exist_ok=True)
        self.diagrams_dir.mkdir(parents=True, exist_ok=True)
        (self.latex_dir / "chapters").mkdir(exist_ok=True)
    
    def get_front_matter_order(self) -> List[str]:
        """Get ordered list of front matter files"""
        return [
            "00a-preface.md",
            "00b-toc.md",  # Table of Contents (for markdown navigation)
        ]
    
    def get_main_matter_order(self) -> List[str]:
        """Get ordered list of main matter chapter files"""
        return [
            # Introduction
            "00c-introduction.md",
            # Part I: Foundations
            "01-neural-networks-perceptron.md",
            "02-multilayer-networks-architecture.md",
            "03-learning-algorithms.md",
            "04-training-neural-networks.md",
            "05-matrix-core.md",
            "06-embeddings.md",
            "07-attention-intuition.md",
            "08-why-transformers.md",
            
            # Part II: Examples
            "09-example1-forward-pass.md",
            "10-example2-single-step.md",
            "11-example3-full-backprop.md",
            "12-example4-multiple-patterns.md",
            "13-example5-feedforward.md",
            "14-example6-complete.md",
            "15-example7-character-recognition.md",
            
            # Appendices
            "appendix-a-matrix-calculus.md",
            "appendix-b-terminology-reference.md",
            "appendix-c-hand-calculation-tips.md",
            "appendix-d-common-mistakes.md",
        ]
    
    def get_back_matter_order(self) -> List[str]:
        """Get ordered list of back matter files"""
        return [
            "conclusion.md",
        ]
    
    def get_chapter_order(self) -> List[str]:
        """Get ordered list of all chapter files (for backward compatibility)"""
        return (self.get_front_matter_order() + 
                self.get_main_matter_order() + 
                self.get_back_matter_order())
    
    def convert_diagrams_for_chapter(self, chapter_path: Path) -> List[Path]:
        """Convert all diagrams needed for a chapter"""
        deps = self.manifest.get_chapter_dependencies(chapter_path, self.book_dir)
        diagram_paths = []
        
        for diagram_path_str in deps["diagram_dependencies"]:
            diagram_path = Path(diagram_path_str)
            if not diagram_path.exists():
                continue
            
            # Check if needs conversion
            if self.force or self.cache.is_diagram_changed(diagram_path):
                # Determine output path (preserve directory structure)
                rel_path = diagram_path.relative_to(self.book_dir / "images")
                pdf_path = self.diagrams_dir / rel_path.with_suffix('.pdf')
                pdf_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert
                from convert_svg_to_pdf import convert_svg_to_pdf
                converter = find_converter()
                success, error = convert_svg_to_pdf(diagram_path, pdf_path, converter)
                
                if success:
                    self.cache.update_diagram_cache(diagram_path, pdf_path)
                    self.manifest.update_manifest_from_conversion(diagram_path, pdf_path)
                    diagram_paths.append(pdf_path)
                else:
                    print(f"  Warning: Failed to convert {diagram_path.name}: {error}")
            else:
                # Use cached PDF
                pdf_path = self.cache.get_diagram_pdf_path(diagram_path)
                if pdf_path:
                    diagram_paths.append(pdf_path)
        
        return diagram_paths
    
    def fix_image_references(self, latex_content: str, chapter_path: Path) -> str:
        """Fix image references in LaTeX to point to PDF diagrams with professional scaling"""
        # Pandoc converts markdown images to \includesvg{images/...} or \includegraphics{images/...}
        # We need to replace these with \includegraphics pointing to PDF files in output/diagrams/
        
        def replace_image(match):
            full_match = match.group(0)
            image_path_str = match.group(1)  # The path inside {}
            
            # Skip if already pointing to a PDF or not in images/ directory
            if not image_path_str.startswith('images/') or image_path_str.endswith('.pdf'):
                return full_match
            
            # Convert images/path/to/image.svg to output/diagrams/path/to/image.pdf
            if image_path_str.startswith('images/'):
                rel_path = image_path_str[7:]  # Remove 'images/' prefix
            else:
                return full_match
            
            # Construct PDF path
            pdf_path = self.diagrams_dir / rel_path.replace('.svg', '.pdf')
            
            if not pdf_path.exists():
                # Try to find it - might be in a subdirectory
                return full_match
            
            # Get relative path from main LaTeX directory (not chapter file) to PDF
            # LaTeX resolves paths relative to the main .tex file location, not the chapter file
            rel_pdf_path = self._get_relative_path(self.latex_dir, pdf_path)
            latex_path_str = str(rel_pdf_path).replace('\\', '/')
            
            # Professional figure scaling: use consistent, moderate width
            # 0.65\textwidth provides professional appearance - large enough for detail,
            # but leaves proper margins and ensures consistency across all diagrams
            figure_options = 'width=0.65\\textwidth, keepaspectratio'
            
            if '\\includesvg' in full_match:
                return f'\\includegraphics[{figure_options}]{{{latex_path_str}}}'
            else:
                # Always use professional scaling for figures
                return f'\\includegraphics[{figure_options}]{{{latex_path_str}}}'
        
        # Match \includesvg{images/...} or \includegraphics{images/...}
        image_pattern = r'\\(?:includesvg|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}'
        latex_content = re.sub(image_pattern, replace_image, latex_content)
        
        # Professional table image handling: detect column count and scale appropriately
        def fix_table_images(match):
            table_content = match.group(0)
            
            # Detect number of columns in the table
            # Method 1: Count column specifications: >{...}p{...} or >{...}c patterns
            column_matches = re.findall(r'>\{[^}]*\}[cp]', table_content)
            
            # Method 2: Count standalone column types (c, l, r, p{...}) in table definition
            # Look for patterns like @{}c@{} or p{...} or c|l|r
            if not column_matches:
                # Check for simple column definitions like @{}c@{} or c|l|r
                table_def_match = re.search(r'\\begin\{(?:longtable|table|tabular)\}.*?\{([^}]+)\}', table_content)
                if table_def_match:
                    col_def = table_def_match.group(1)
                    # Count column separators (|) and column types
                    # Each | or column type (c, l, r, p) indicates a column
                    col_count = len(re.findall(r'[clrp]\{', col_def)) + len(re.findall(r'[clr](?![a-z])', col_def))
                    if col_count > 0:
                        num_columns = col_count
                    else:
                        # Fallback: count images in the table
                        image_count = len(re.findall(r'\\includegraphics', table_content))
                        num_columns = image_count if image_count > 0 else 1
                else:
                    # Fallback: count images in the table
                    image_count = len(re.findall(r'\\includegraphics', table_content))
                    num_columns = image_count if image_count > 0 else 1
            else:
                num_columns = len(column_matches)
            
            # Professional scaling based on column count
            # Use consistent, moderate scaling that works well for most diagram types
            # All use keepaspectratio to maintain proper proportions
            if num_columns >= 3:
                # 3+ columns: use 0.30\textwidth per image to ensure all fit on one page
                table_image_options = 'width=0.30\\textwidth, keepaspectratio'
            elif num_columns == 2:
                # 2 columns: use 0.45\textwidth per image
                table_image_options = 'width=0.45\\textwidth, keepaspectratio'
            else:
                # Single column: use 0.65\textwidth (professional, consistent size)
                # Slightly reduced from 0.70 to ensure better consistency across different image types
                table_image_options = 'width=0.65\\textwidth, keepaspectratio'
            
            # Replace any existing image width/height specifications with professional constraints
            # Escape the options string properly for regex replacement
            escaped_options = table_image_options.replace('\\', '\\\\')
            table_content = re.sub(
                r'\\includegraphics\[[^\]]+\]',
                f'\\\\includegraphics[{escaped_options}]',
                table_content
            )
            return table_content
        
        # Match table environments and fix images inside them
        table_pattern = r'(\\begin\{(?:longtable|table|tabular)\}.*?\\end\{(?:longtable|table|tabular)\})'
        latex_content = re.sub(table_pattern, fix_table_images, latex_content, flags=re.DOTALL)
        
        return latex_content
    
    def _get_relative_path(self, from_path: Path, to_path: Path) -> Path:
        """Get relative path from one file/directory to another"""
        # Determine if from_path is a file or directory
        # If it has a file extension or doesn't exist, treat as file (use parent)
        # Otherwise, treat as directory
        if from_path.suffix or not from_path.exists():
            # Treat as file - get path relative to its parent directory
            from_dir = from_path.parent
        else:
            # Treat as directory
            from_dir = from_path
        
        try:
            # Try direct relative path
            return to_path.relative_to(from_dir)
        except ValueError:
            # If not in same tree, find common ancestor and construct path
            from_parts = from_dir.parts
            to_parts = to_path.parts
            
            # Find common prefix
            common_len = 0
            for i, (f, t) in enumerate(zip(from_parts, to_parts)):
                if f == t:
                    common_len = i + 1
                else:
                    break
            
            # Calculate how many levels up we need to go
            up_levels = len(from_parts) - common_len
            
            # Get the path from common ancestor to target
            target_parts = to_parts[common_len:]
            
            # Construct relative path: ../.../target
            if up_levels > 0:
                rel_parts = ['..'] * up_levels + list(target_parts)
            else:
                rel_parts = list(target_parts)
            
            return Path(*rel_parts)
    
    def convert_chapter_to_latex(self, chapter_path: Path) -> Optional[Path]:
        """Convert a single chapter from Markdown to LaTeX"""
        if not chapter_path.exists():
            print(f"  Warning: Chapter not found: {chapter_path}")
            return None
        
        output_tex = self.latex_dir / "chapters" / f"{chapter_path.stem}.tex"
        
        # Check if needs conversion
        deps = self.manifest.get_chapter_dependencies(chapter_path, self.book_dir)
        if not self.force and not self.cache.is_chapter_changed(chapter_path, 
                                                               deps["diagram_dependencies"]):
            return output_tex
        
        # Convert diagrams first
        self.convert_diagrams_for_chapter(chapter_path)
        
        # Convert Markdown to LaTeX using pandoc
        # Don't use --standalone so chapters can be included with \input
        try:
            result = subprocess.run([
                'pandoc',
                str(chapter_path),
                '-f', 'markdown+tex_math_dollars+raw_tex',
                '-t', 'latex',
                '--wrap=none',
                # No --standalone flag - chapters will be included in main document
                '-o', str(output_tex)
            ], capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  Error converting {chapter_path.name}: {e.stderr}")
            return None
        except FileNotFoundError:
            print("  Error: pandoc not found. Install with: sudo apt-get install pandoc")
            return None
        
        # Read generated LaTeX and fix image references
        with open(output_tex, 'r', encoding='utf-8') as f:
            latex_content = f.read()
        
        # Special handling for front matter: remove duplicate headings
        if chapter_path.stem == "00a-preface":
            # Remove first \chapter, \section, or similar heading command
            latex_content = re.sub(r'\\chapter\*?\{[^}]+\}.*?\n', '', latex_content, count=1)
            latex_content = re.sub(r'\\section\*?\{[^}]+\}.*?\n', '', latex_content, count=1)
        elif chapter_path.stem == "00b-toc":
            # For TOC, remove the book title heading but keep the "Table of Contents" section
            # Remove first heading (usually the book title)
            latex_content = re.sub(r'\\chapter\*?\{[^}]+\}.*?\n', '', latex_content, count=1)
            # Keep the "Table of Contents" section heading
        
        # Fix image references
        latex_content = self.fix_image_references(latex_content, chapter_path)
        
        # Write back
        with open(output_tex, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        # Update cache
        self.cache.update_file_cache(output_tex)
        self.manifest.update_manifest_from_chapter(chapter_path, output_tex, self.book_dir)
        
        return output_tex
    
    def build_complete_latex(self) -> Optional[Path]:
        """Build complete LaTeX book from all chapters"""
        # Process front matter
        front_matter_files = []
        for chapter_name in self.get_front_matter_order():
            chapter_path = self.book_dir / chapter_name
            if chapter_path.exists():
                front_matter_files.append(chapter_path)
        
        # Process main matter
        main_matter_files = []
        for chapter_name in self.get_main_matter_order():
            chapter_path = self.book_dir / chapter_name
            if chapter_path.exists():
                main_matter_files.append(chapter_path)
        
        # Process back matter
        back_matter_files = []
        for chapter_name in self.get_back_matter_order():
            chapter_path = self.book_dir / chapter_name
            if chapter_path.exists():
                back_matter_files.append(chapter_path)
        
        # Convert all chapters
        front_matter_latex = []
        for chapter_path in front_matter_files:
            latex_path = self.convert_chapter_to_latex(chapter_path)
            if latex_path:
                front_matter_latex.append(latex_path)
        
        main_matter_latex = []
        for chapter_path in main_matter_files:
            latex_path = self.convert_chapter_to_latex(chapter_path)
            if latex_path:
                main_matter_latex.append(latex_path)
        
        back_matter_latex = []
        for chapter_path in back_matter_files:
            latex_path = self.convert_chapter_to_latex(chapter_path)
            if latex_path:
                back_matter_latex.append(latex_path)
        
        # Build main LaTeX file
        main_tex = self.latex_dir / "book.tex"
        self._create_main_latex(main_tex, front_matter_latex, main_matter_latex, back_matter_latex)
        
        return main_tex
    
    def _create_main_latex(self, main_tex: Path, front_matter_files: List[Path], 
                          main_matter_files: List[Path], back_matter_files: List[Path]):
        """Create main LaTeX file that includes all chapters in proper order"""
        # Select template based on sage_template flag
        if self.sage_template:
            template_name = "book_template_sage.tex"
        else:
            template_name = "book_template.tex"
        template_path = Path(__file__).parent.parent / "templates" / template_name
        
        # Read template
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()
        else:
            # Fallback template
            template = """\\documentclass[11pt,openany]{book}
\\usepackage[letterpaper,margin=1in]{geometry}
\\usepackage{amsmath}
\\usepackage{graphicx}
\\usepackage{hyperref}
\\usepackage{listings}
\\begin{document}
\\frontmatter
\\maketitle
\\tableofcontents
\\mainmatter
{chapters}
\\backmatter
\\end{document}
"""
        
        # Generate front matter includes (preface)
        front_matter_includes = []
        for chapter_file in front_matter_files:
            rel_path = chapter_file.relative_to(main_tex.parent)
            front_matter_includes.append(f"\\input{{{rel_path.as_posix()}}}")
        
        # Generate main matter includes
        main_matter_includes = []
        for chapter_file in main_matter_files:
            rel_path = chapter_file.relative_to(main_tex.parent)
            main_matter_includes.append(f"\\input{{{rel_path.as_posix()}}}")
        
        # Generate back matter includes
        back_matter_includes = []
        for chapter_file in back_matter_files:
            rel_path = chapter_file.relative_to(main_tex.parent)
            back_matter_includes.append(f"\\input{{{rel_path.as_posix()}}}")
        
        # Insert front matter (preface and TOC) after the Preface chapter declaration
        if front_matter_includes:
            # Separate preface and TOC files
            preface_files = [f for f in front_matter_includes if '00a-preface' in f]
            toc_files = [f for f in front_matter_includes if '00b-toc' in f]
            
            # Insert preface after Preface chapter declaration
            if preface_files:
                preface_pattern = r'(% Preface content will be included here by build script\n)'
                preface_content = '\n'.join(preface_files) + '\n'
                def preface_replacer(match):
                    return match.group(1) + preface_content
                template = re.sub(preface_pattern, preface_replacer, template)
            
            # Insert TOC as a separate chapter (optional - since LaTeX auto-generates TOC)
            # We'll include it as "Detailed Table of Contents" for completeness
            if toc_files:
                toc_pattern = r'(% TOC content will be included here by build script if present\n)'
                toc_content = '\\chapter*{Detailed Table of Contents}\n\\addcontentsline{toc}{chapter}{Detailed Table of Contents}\n' + '\n'.join(toc_files) + '\n'
                def toc_replacer(match):
                    return match.group(1) + toc_content
                template = re.sub(toc_pattern, toc_replacer, template)
        
        # Insert main matter chapters after \mainmatter
        if main_matter_includes:
            template = template.replace("\\mainmatter", 
                                      "\\mainmatter\n" + "\n".join(main_matter_includes))
        
        # Insert back matter chapters before bibliography
        if back_matter_includes:
            # Insert before \bibliographystyle
            template = template.replace("\\bibliographystyle{plain}", 
                                      "\n".join(back_matter_includes) + "\n\\bibliographystyle{plain}")
        
        # Write main file
        with open(main_tex, 'w', encoding='utf-8') as f:
            f.write(template)
        
        # Update cache
        all_files = front_matter_files + main_matter_files + back_matter_files
        source_deps = [str(f) for f in all_files]
        self.cache.update_output_cache(main_tex, source_deps)


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build LaTeX book from Markdown chapters')
    parser.add_argument('--book-dir', type=Path, default=Path("book"),
                       help='Directory containing Markdown chapters')
    parser.add_argument('--output-dir', type=Path, default=Path("output"),
                       help='Output directory for LaTeX files')
    parser.add_argument('--force', action='store_true',
                       help='Force rebuild all chapters')
    parser.add_argument('--standard', action='store_true',
                       help='Use standard template instead of SAGE (SAGE is default)')
    # SAGE is default
    parser.set_defaults(sage=True)
    
    args = parser.parse_args()
    
    # Initialize cache and manifest
    cache_dir = Path(".build")
    cache = BuildCache(cache_dir)
    manifest = BuildManifest(cache)
    
    # Determine template: SAGE is default unless --standard is specified
    use_sage = not getattr(args, 'standard', False)
    
    # Build LaTeX book
    builder = LaTeXBookBuilder(args.book_dir, args.output_dir, cache, manifest, args.force, use_sage)
    main_tex = builder.build_complete_latex()
    
    if main_tex:
        print(f"✓ LaTeX book built: {main_tex}")
        cache.save_cache()
    else:
        print("✗ Failed to build LaTeX book")
        sys.exit(1)


if __name__ == '__main__':
    main()

