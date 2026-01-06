#!/usr/bin/env python3
"""
Build Cache and Dependency Tracking System

Tracks file hashes, timestamps, and dependencies to enable incremental builds.
Provides compiler-like dependency tracking for the book build system.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime


class BuildCache:
    """Manages build cache and dependency tracking"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_file = cache_dir / "cache.json"
        self.cache: Dict = {}
        self._ensure_cache_dir()
        self._load_cache()
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "diagrams").mkdir(exist_ok=True)
        (self.cache_dir / "chapters").mkdir(exist_ok=True)
    
    def _load_cache(self):
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.cache = self._empty_cache()
        else:
            self.cache = self._empty_cache()
    
    def _empty_cache(self) -> Dict:
        """Return empty cache structure"""
        return {
            "version": "1.0",
            "files": {},  # file path -> {hash, mtime, dependencies}
            "diagrams": {},  # svg path -> {hash, pdf_path, mtime}
            "chapters": {},  # chapter path -> {hash, latex_path, mtime, diagram_deps}
            "outputs": {},  # output path -> {hash, mtime, source_deps}
            "last_build": None
        }
    
    def save_cache(self):
        """Save cache to disk"""
        self.cache["last_build"] = datetime.now().isoformat()
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)
    
    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file"""
        if not file_path.exists():
            return ""
        
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def get_file_mtime(self, file_path: Path) -> float:
        """Get file modification time"""
        if not file_path.exists():
            return 0.0
        return os.path.getmtime(file_path)
    
    def get_file_info(self, file_path: Path) -> Tuple[str, float]:
        """Get file hash and modification time"""
        return (self.compute_file_hash(file_path), self.get_file_mtime(file_path))
    
    def is_file_changed(self, file_path: Path) -> bool:
        """Check if file has changed since last build"""
        file_str = str(file_path)
        current_hash, current_mtime = self.get_file_info(file_path)
        
        if file_str not in self.cache["files"]:
            return True
        
        cached = self.cache["files"][file_str]
        return (cached.get("hash") != current_hash or 
                cached.get("mtime") != current_mtime)
    
    def update_file_cache(self, file_path: Path, dependencies: Optional[List[str]] = None):
        """Update cache entry for a file"""
        file_str = str(file_path)
        current_hash, current_mtime = self.get_file_info(file_path)
        
        self.cache["files"][file_str] = {
            "hash": current_hash,
            "mtime": current_mtime,
            "dependencies": dependencies or []
        }
    
    def update_diagram_cache(self, svg_path: Path, pdf_path: Path):
        """Update cache for diagram conversion"""
        svg_str = str(svg_path)
        svg_hash, svg_mtime = self.get_file_info(svg_path)
        pdf_hash, pdf_mtime = self.get_file_info(pdf_path)
        
        self.cache["diagrams"][svg_str] = {
            "hash": svg_hash,
            "mtime": svg_mtime,
            "pdf_path": str(pdf_path),
            "pdf_hash": pdf_hash,
            "pdf_mtime": pdf_mtime
        }
    
    def is_diagram_changed(self, svg_path: Path) -> bool:
        """Check if diagram needs conversion"""
        svg_str = str(svg_path)
        
        if svg_str not in self.cache["diagrams"]:
            return True
        
        cached = self.cache["diagrams"][svg_str]
        current_hash, current_mtime = self.get_file_info(svg_path)
        
        return (cached.get("hash") != current_hash or 
                cached.get("mtime") != current_mtime)
    
    def update_chapter_cache(self, chapter_path: Path, latex_path: Path, 
                            diagram_deps: Optional[List[str]] = None):
        """Update cache for chapter processing"""
        chapter_str = str(chapter_path)
        chapter_hash, chapter_mtime = self.get_file_info(chapter_path)
        latex_hash, latex_mtime = self.get_file_info(latex_path)
        
        self.cache["chapters"][chapter_str] = {
            "hash": chapter_hash,
            "mtime": chapter_mtime,
            "latex_path": str(latex_path),
            "latex_hash": latex_hash,
            "latex_mtime": latex_mtime,
            "diagram_dependencies": diagram_deps or []
        }
    
    def is_chapter_changed(self, chapter_path: Path, diagram_deps: Optional[List[str]] = None) -> bool:
        """Check if chapter needs reprocessing"""
        chapter_str = str(chapter_path)
        
        if chapter_str not in self.cache["chapters"]:
            return True
        
        if self.is_file_changed(chapter_path):
            return True
        
        # Check if any diagram dependencies changed
        if diagram_deps:
            cached = self.cache["chapters"][chapter_str]
            cached_deps = set(cached.get("diagram_dependencies", []))
            current_deps = set(diagram_deps)
            
            if cached_deps != current_deps:
                return True
            
            # Check if any dependency diagram changed
            for dep in diagram_deps:
                if dep in self.cache["diagrams"]:
                    dep_path = Path(dep)
                    if self.is_diagram_changed(dep_path):
                        return True
        
        return False
    
    def update_output_cache(self, output_path: Path, source_deps: List[str]):
        """Update cache for output file"""
        output_str = str(output_path)
        output_hash, output_mtime = self.get_file_info(output_path)
        
        self.cache["outputs"][output_str] = {
            "hash": output_hash,
            "mtime": output_mtime,
            "source_dependencies": source_deps
        }
    
    def is_output_changed(self, output_path: Path, source_deps: List[str]) -> bool:
        """Check if output needs regeneration"""
        output_str = str(output_path)
        
        if not output_path.exists():
            return True
        
        if output_str not in self.cache["outputs"]:
            return True
        
        # Check if any source dependency changed
        for dep in source_deps:
            dep_path = Path(dep)
            if self.is_file_changed(dep_path):
                return True
        
        # Check if output file itself changed (shouldn't happen, but check anyway)
        cached = self.cache["outputs"][output_str]
        current_hash, current_mtime = self.get_file_info(output_path)
        
        return (cached.get("hash") != current_hash or 
                cached.get("mtime") != current_mtime)
    
    def get_diagram_pdf_path(self, svg_path: Path) -> Optional[Path]:
        """Get cached PDF path for SVG diagram"""
        svg_str = str(svg_path)
        if svg_str in self.cache["diagrams"]:
            pdf_path_str = self.cache["diagrams"][svg_str].get("pdf_path")
            if pdf_path_str:
                pdf_path = Path(pdf_path_str)
                if pdf_path.exists():
                    return pdf_path
        return None
    
    def clear_cache(self):
        """Clear all cache"""
        self.cache = self._empty_cache()
        self.save_cache()
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about cache"""
        return {
            "files_tracked": len(self.cache.get("files", {})),
            "diagrams_tracked": len(self.cache.get("diagrams", {})),
            "chapters_tracked": len(self.cache.get("chapters", {})),
            "outputs_tracked": len(self.cache.get("outputs", {})),
            "last_build": self.cache.get("last_build")
        }

