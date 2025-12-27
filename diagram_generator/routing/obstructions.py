#!/usr/bin/env python3
"""
Obstruction Grid - Grid-based collision detection for label placement.

This module provides a grid-based system for tracking obstructions in SVG
diagrams, enabling efficient collision detection and optimal label placement.
"""

from typing import Tuple, Optional, List, Set
import math

from diagram_generator.core.diagram import BoundingBox


class ObstructionGrid:
    """Grid-based obstruction tracking for label placement."""
    
    def __init__(self, width: float, height: float, cell_size: float = 5.0):
        """
        Initialize obstruction grid.
        
        Args:
            width: Canvas width
            height: Canvas height
            cell_size: Size of each grid cell (default: 5px)
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid: Set[Tuple[int, int]] = set()  # Set of (x, y) cell coordinates
        self.obstructions: List[BoundingBox] = []  # List of bounding boxes
    
    def bbox_to_cells(self, bbox: BoundingBox) -> List[Tuple[int, int]]:
        """
        Convert bounding box to list of grid cells it occupies.
        
        Args:
            bbox: Bounding box to convert
            
        Returns:
            List of (x, y) cell coordinates
        """
        cells = []
        start_x = int(bbox.left / self.cell_size)
        end_x = int(math.ceil(bbox.right / self.cell_size))
        start_y = int(bbox.top / self.cell_size)
        end_y = int(math.ceil(bbox.bottom / self.cell_size))
        
        for x in range(start_x, end_x + 1):
            for y in range(start_y, end_y + 1):
                cells.append((x, y))
        
        return cells
    
    def mark_obstructed(self, bbox: BoundingBox):
        """
        Mark cells as obstructed based on bounding box.
        
        Args:
            bbox: Bounding box to mark as obstructed
        """
        cells = self.bbox_to_cells(bbox)
        for cell in cells:
            self.grid.add(cell)
        self.obstructions.append(bbox)
    
    def is_obstructed(self, bbox: BoundingBox) -> bool:
        """
        Check if bounding box overlaps with obstructions.
        
        Args:
            bbox: Bounding box to check
            
        Returns:
            True if obstructed, False otherwise
        """
        cells = self.bbox_to_cells(bbox)
        return any(cell in self.grid for cell in cells)
    
    def find_nearest_open_space(
        self,
        target_pos: Tuple[float, float],
        label_bbox: BoundingBox,
        max_distance: float = 200.0,
        angle_step: float = 15.0
    ) -> Optional[Tuple[float, float]]:
        """
        Find nearest open space for label placement using spiral search.
        
        Args:
            target_pos: Target position (x, y)
            label_bbox: Label bounding box (width/height used, position will be set)
            max_distance: Maximum distance to search
            angle_step: Angle step for spiral search (degrees)
            
        Returns:
            (x, y) position if found, None otherwise
        """
        # Start close and spiral outward
        distance = 10.0
        angle = 0.0
        
        while distance < max_distance:
            # Calculate position at current angle and distance
            angle_rad = math.radians(angle)
            x = target_pos[0] + distance * math.cos(angle_rad)
            y = target_pos[1] + distance * math.sin(angle_rad)
            
            # Check if position is within canvas bounds
            if x < 0 or y < 0 or x + label_bbox.width > self.width or y + label_bbox.height > self.height:
                angle += angle_step
                if angle >= 360:
                    angle = 0
                    distance += 10
                continue
            
            # Create candidate bounding box
            candidate_bbox = BoundingBox(x, y, label_bbox.width, label_bbox.height)
            
            # Check if candidate position is obstructed
            if not self.is_obstructed(candidate_bbox):
                return (x, y)
            
            # Move to next angle
            angle += angle_step
            if angle >= 360:
                angle = 0
                distance += 10
        
        return None
    
    def clear(self):
        """Clear all obstructions."""
        self.grid.clear()
        self.obstructions.clear()
    
    def rebuild(self, obstructions: List[BoundingBox]):
        """
        Rebuild grid from list of obstructions.
        
        Args:
            obstructions: List of bounding boxes to mark as obstructed
        """
        self.clear()
        for bbox in obstructions:
            self.mark_obstructed(bbox)

