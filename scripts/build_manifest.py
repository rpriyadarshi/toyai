#!/usr/bin/env python3
"""
Build Manifest System

Tracks dependencies between source files, diagrams, and outputs
for accurate incremental builds.
"""

from pathlib import Path
from typing import Dict, List, Set, Optional
import re
from build_cache import BuildCache


class BuildManifest:
    """Manages build manifest and dependency graph"""
    
    def __init__(self, cache: BuildCache):
        self.cache = cache
    
    def extract_image_references(self, markdown_path: Path) -> List[str]:
        """Extract all image references from Markdown file"""
        if not markdown_path.exists():
            return []
        
        image_refs = []
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Match markdown image syntax: ![alt](path)
        pattern = r'!\[[^\]]*\]\(([^)]+)\)'
        matches = re.findall(pattern, content)
        
        for match in matches:
            # Handle relative paths
            if match.startswith('images/'):
                image_refs.append(match)
            elif match.startswith('../images/'):
                # Normalize path
                image_refs.append(match.replace('../images/', 'images/'))
        
        return image_refs
    
    def resolve_image_path(self, image_ref: str, book_dir: Path) -> Optional[Path]:
        """Resolve image reference to actual file path"""
        # Remove 'images/' prefix if present
        if image_ref.startswith('images/'):
            image_ref = image_ref[7:]  # Remove 'images/'
        elif image_ref.startswith('../images/'):
            image_ref = image_ref[11:]  # Remove '../images/'
        
        # Try to find the image file
        images_dir = book_dir / "images"
        image_path = images_dir / image_ref
        
        if image_path.exists():
            return image_path
        
        # Try with .svg extension if not present
        if not image_path.suffix:
            image_path = image_path.with_suffix('.svg')
            if image_path.exists():
                return image_path
        
        return None
    
    def get_chapter_dependencies(self, chapter_path: Path, book_dir: Path) -> Dict:
        """Get all dependencies for a chapter"""
        image_refs = self.extract_image_references(chapter_path)
        
        # Resolve image paths
        diagram_deps = []
        for image_ref in image_refs:
            image_path = self.resolve_image_path(image_ref, book_dir)
            if image_path and image_path.suffix == '.svg':
                diagram_deps.append(str(image_path))
        
        return {
            "source_file": str(chapter_path),
            "diagram_dependencies": diagram_deps,
            "image_references": image_refs
        }
    
    def build_dependency_graph(self, book_dir: Path, chapter_files: List[Path]) -> Dict:
        """Build complete dependency graph for all chapters"""
        graph = {
            "chapters": {},
            "diagrams": {},
            "outputs": {}
        }
        
        # Process each chapter
        for chapter_path in chapter_files:
            deps = self.get_chapter_dependencies(chapter_path, book_dir)
            graph["chapters"][str(chapter_path)] = deps
        
        # Collect all unique diagram dependencies
        all_diagrams = set()
        for chapter_deps in graph["chapters"].values():
            all_diagrams.update(chapter_deps["diagram_dependencies"])
        
        # Build diagram dependency info
        for diagram_path_str in all_diagrams:
            diagram_path = Path(diagram_path_str)
            graph["diagrams"][diagram_path_str] = {
                "source_file": diagram_path_str,
                "output_file": None  # Will be set during conversion
            }
        
        return graph
    
    def get_diagrams_needing_conversion(self, book_dir: Path, 
                                       output_diagrams_dir: Path) -> List[Path]:
        """Get list of diagrams that need conversion"""
        images_dir = book_dir / "images"
        diagrams_to_convert = []
        
        # Find all SVG files
        for svg_path in images_dir.rglob("*.svg"):
            if self.cache.is_diagram_changed(svg_path):
                diagrams_to_convert.append(svg_path)
        
        return diagrams_to_convert
    
    def get_chapters_needing_processing(self, book_dir: Path, 
                                       chapter_files: List[Path]) -> List[Path]:
        """Get list of chapters that need processing"""
        chapters_to_process = []
        
        for chapter_path in chapter_files:
            deps = self.get_chapter_dependencies(chapter_path, book_dir)
            diagram_deps = deps["diagram_dependencies"]
            
            if self.cache.is_chapter_changed(chapter_path, diagram_deps):
                chapters_to_process.append(chapter_path)
        
        return chapters_to_process
    
    def update_manifest_from_conversion(self, svg_path: Path, pdf_path: Path):
        """Update manifest after diagram conversion"""
        self.cache.update_diagram_cache(svg_path, pdf_path)
    
    def update_manifest_from_chapter(self, chapter_path: Path, latex_path: Path,
                                    book_dir: Path):
        """Update manifest after chapter processing"""
        deps = self.get_chapter_dependencies(chapter_path, book_dir)
        diagram_deps = deps["diagram_dependencies"]
        self.cache.update_chapter_cache(chapter_path, latex_path, diagram_deps)
    
    def get_output_dependencies(self, output_path: Path, 
                               source_files: List[str]) -> List[str]:
        """Get dependencies for an output file"""
        return source_files
    
    def update_manifest_from_output(self, output_path: Path, source_deps: List[str]):
        """Update manifest after output generation"""
        self.cache.update_output_cache(output_path, source_deps)

