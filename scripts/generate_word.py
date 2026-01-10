#!/usr/bin/env python3
"""
Word Document Generator

Converts Markdown book to Word (.docx) format with high-resolution
diagram embedding. Converts SVG diagrams to PNG for Word compatibility.
"""

import subprocess
import sys
import re
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
from build_cache import BuildCache
from build_manifest import BuildManifest
from convert_svg_to_pdf import find_converter


def convert_svg_to_png(svg_path: Path, png_path: Path, dpi: int = 600) -> Tuple[bool, str]:
    """Convert SVG to high-resolution PNG"""
    # Try inkscape first
    try:
        result = subprocess.run([
            'inkscape',
            '--export-filename', str(png_path),
            '--export-type', 'png',
            f'--export-dpi={dpi}',
            str(svg_path)
        ], capture_output=True, text=True, check=True)
        return True, ""
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Try rsvg-convert
    try:
        # rsvg-convert uses width/height, calculate from DPI
        # Approximate: 600 DPI for letter size (8.5x11) = 5100x6600
        result = subprocess.run([
            'rsvg-convert',
            '-f', 'png',
            '-o', str(png_path),
            '--width', '5100',
            '--height', '6600',
            str(svg_path)
        ], capture_output=True, text=True, check=True)
        return True, ""
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Try cairosvg
    try:
        import cairosvg
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), dpi=dpi)
        return True, ""
    except ImportError:
        pass
    except Exception as e:
        return False, f"cairosvg error: {str(e)}"
    
    return False, "No SVG to PNG converter found"


class WordGenerator:
    """Generates Word document from Markdown"""
    
    def __init__(self, book_dir: Path, output_dir: Path, cache: BuildCache,
                 manifest: BuildManifest, force: bool = False):
        self.book_dir = book_dir
        self.output_dir = output_dir
        self.word_dir = output_dir / "word"
        self.images_dir = output_dir / "word" / "images"
        self.cache = cache
        self.manifest = manifest
        self.force = force
        
        # Ensure directories exist
        self.word_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
    
    def get_chapter_order(self) -> List[str]:
        """Get ordered list of chapter files"""
        return [
            "00a-preface.md",
            "00b-toc.md",
            "00c-introduction.md",
            "01-neural-networks-perceptron.md",
            "02-probability-statistics.md",
            "03-multilayer-networks-architecture.md",
            "04-learning-algorithms.md",
            "05-training-neural-networks.md",
            "06-embeddings.md",
            "07-attention-intuition.md",
            "08-why-transformers.md",
            "09-example1-forward-pass.md",
            "10-example2-single-step.md",
            "11-example3-full-backprop.md",
            "12-example4-multiple-patterns.md",
            "13-example5-feedforward.md",
            "14-example6-complete.md",
            "15-example7-character-recognition.md",
            "appendix-a-matrix-calculus.md",
            "appendix-b-terminology-reference.md",
            "appendix-c-hand-calculation-tips.md",
            "appendix-d-common-mistakes.md",
            "conclusion.md",
        ]
    
    def convert_diagrams_for_word(self, chapter_path: Path) -> List[Path]:
        """Convert all diagrams needed for a chapter to PNG"""
        deps = self.manifest.get_chapter_dependencies(chapter_path, self.book_dir)
        png_paths = []
        
        for diagram_path_str in deps["diagram_dependencies"]:
            diagram_path = Path(diagram_path_str)
            if not diagram_path.exists():
                continue
            
            # Determine output PNG path
            rel_path = diagram_path.relative_to(self.book_dir / "images")
            png_path = self.images_dir / rel_path.with_suffix('.png')
            png_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if needs conversion
            if self.force or not png_path.exists():
                success, error = convert_svg_to_png(diagram_path, png_path, dpi=600)
                if success:
                    png_paths.append(png_path)
                else:
                    print(f"  Warning: Failed to convert {diagram_path.name}: {error}")
            else:
                png_paths.append(png_path)
        
        return png_paths
    
    def fix_image_references(self, markdown_content: str, chapter_path: Path) -> str:
        """Fix image references in Markdown to point to PNG versions"""
        deps = self.manifest.get_chapter_dependencies(chapter_path, self.book_dir)
        
        for image_ref in deps["image_references"]:
            # Resolve image path
            image_path = self.manifest.resolve_image_path(image_ref, self.book_dir)
            if not image_path or image_path.suffix != '.svg':
                continue
            
            # Get PNG path
            rel_path = image_path.relative_to(self.book_dir / "images")
            png_path = self.images_dir / rel_path.with_suffix('.png')
            
            if png_path.exists():
                # Update markdown reference to use PNG
                # Convert relative path for Word document
                rel_png_ref = f"images/{rel_path.with_suffix('.png').as_posix()}"
                pattern = r'!\[([^\]]*)\]\([^)]*' + re.escape(image_ref) + r'[^)]*\)'
                replacement = f'![\\1]({rel_png_ref})'
                markdown_content = re.sub(pattern, replacement, markdown_content)
        
        return markdown_content
    
    def build_word_document(self) -> Optional[Path]:
        """Build complete Word document from all chapters"""
        chapter_files = []
        for chapter_name in self.get_chapter_order():
            chapter_path = self.book_dir / chapter_name
            if chapter_path.exists():
                chapter_files.append(chapter_path)
        
        # Create temporary combined Markdown file
        combined_md = []
        
        for chapter_path in chapter_files:
            # Read chapter
            with open(chapter_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remove navigation links
            content = re.sub(r'\n---\n\*\*Navigation:\*\*.*?\n---\n', '', content, flags=re.DOTALL)
            
            # Convert diagrams
            self.convert_diagrams_for_word(chapter_path)
            
            # Fix image references
            content = self.fix_image_references(content, chapter_path)
            
            combined_md.append(content)
            combined_md.append("\n\n---\n\n")
        
        # Write combined Markdown to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, 
                                        encoding='utf-8') as tmp_file:
            tmp_file.write('\n'.join(combined_md))
            tmp_md = tmp_file.name
        
        # Convert to Word using pandoc
        output_docx = self.word_dir / "book.docx"
        
        try:
            result = subprocess.run([
                'pandoc',
                tmp_md,
                '-o', str(output_docx),
                '--from', 'markdown+tex_math_dollars',
                '--to', 'docx',
                '--standalone',
                '--toc',
                '--toc-depth', '3',
            ], capture_output=True, text=True, check=True)
            
            # Clean up temp file
            Path(tmp_md).unlink()
            
            # Update cache
            source_deps = [str(f) for f in chapter_files]
            self.cache.update_output_cache(output_docx, source_deps)
            
            return output_docx
            
        except subprocess.CalledProcessError as e:
            print(f"Error converting to Word: {e.stderr}")
            Path(tmp_md).unlink()
            return None
        except FileNotFoundError:
            print("Error: pandoc not found. Install with: sudo apt-get install pandoc")
            Path(tmp_md).unlink()
            return None


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Word document from Markdown')
    parser.add_argument('--book-dir', type=Path, default=Path("book"),
                       help='Directory containing Markdown chapters')
    parser.add_argument('--output-dir', type=Path, default=Path("output"),
                       help='Output directory')
    parser.add_argument('--force', action='store_true',
                       help='Force rebuild all diagrams')
    
    args = parser.parse_args()
    
    # Initialize cache and manifest
    cache_dir = Path(".build")
    cache = BuildCache(cache_dir)
    manifest = BuildManifest(cache)
    
    # Generate Word document
    generator = WordGenerator(args.book_dir, args.output_dir, cache, manifest, args.force)
    output_docx = generator.build_word_document()
    
    if output_docx:
        print(f"✓ Word document generated: {output_docx}")
        cache.save_cache()
    else:
        print("✗ Failed to generate Word document")
        sys.exit(1)


if __name__ == '__main__':
    main()

