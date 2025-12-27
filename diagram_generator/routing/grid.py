#!/usr/bin/env python3
"""
Routing Grid - Grid matrix system for A* pathfinding.

This module provides a grid-based system for routing connections in SVG diagrams,
enabling A* pathfinding with adaptive cell sizing and comprehensive obstruction tracking.
"""

from typing import Tuple, List, Optional, Dict, Any
import math

from diagram_generator.core.diagram import BoundingBox, SVGDataset, Component


class RoutingGrid:
    """
    Grid matrix system for routing connections.
    
    Provides adaptive cell sizing and obstruction marking for A* pathfinding.
    """
    
    # Cell states
    FREE = 0
    OBSTRUCTED = 1
    PATH = 2
    
    def __init__(self, width: float, height: float, config: Dict[str, Any]):
        """
        Initialize routing grid.
        
        Args:
            width: Canvas width
            height: Canvas height
            config: Configuration dictionary with routing parameters
        """
        self.width = width
        self.height = height
        self.config = config
        
        # Get routing config
        routing_config = config.get("routing", {})
        self.min_cell_size = routing_config.get("min_cell_size", 2.0)
        self.max_cell_size = routing_config.get("max_cell_size", 10.0)
        self.obstruction_padding = routing_config.get("obstruction_padding", 5.0)
        
        # Cell size will be calculated adaptively
        self.cell_size = 5.0  # Default, will be recalculated
        
        # Grid matrix: grid[y][x] where each cell is FREE, OBSTRUCTED, or PATH
        # Initialize as empty (will be built when needed)
        self.grid: List[List[int]] = []
        self.grid_width = 0
        self.grid_height = 0
        
    def _calculate_cell_size(self, dataset: SVGDataset) -> float:
        """
        Calculate adaptive cell size based on component spacing and sizes.
        
        Args:
            dataset: SVG dataset with components
            
        Returns:
            Optimal cell size
        """
        if not dataset.components:
            # Default cell size if no components
            return min(self.max_cell_size, max(self.min_cell_size, 5.0))
        
        # Get all component bounding boxes
        bboxes = []
        for comp in dataset.components:
            bbox = comp.get_absolute_bbox(dataset)
            bboxes.append(bbox)
        
        # Find minimum spacing between components
        min_spacing = float('inf')
        for i, bbox1 in enumerate(bboxes):
            for bbox2 in bboxes[i+1:]:
                # Calculate distance between bounding boxes
                if bbox1.right < bbox2.left:
                    spacing = bbox2.left - bbox1.right
                elif bbox2.right < bbox1.left:
                    spacing = bbox1.left - bbox2.right
                elif bbox1.bottom < bbox2.top:
                    spacing = bbox2.top - bbox1.bottom
                elif bbox2.bottom < bbox1.top:
                    spacing = bbox1.top - bbox2.bottom
                else:
                    spacing = 0  # Overlapping
                
                if spacing > 0:
                    min_spacing = min(min_spacing, spacing)
        
        # Find average component size
        total_size = sum(bbox.width + bbox.height for bbox in bboxes)
        avg_size = total_size / (len(bboxes) * 2) if bboxes else 50.0
        
        # Calculate base cell size
        if min_spacing != float('inf') and min_spacing > 0:
            base_cell_size = min(min_spacing / 4, avg_size / 20)
        else:
            base_cell_size = avg_size / 20
        
        # Clamp between min and max
        cell_size = max(self.min_cell_size, min(self.max_cell_size, base_cell_size))
        
        # Round to reasonable value
        if cell_size <= 2.5:
            cell_size = 2.0
        elif cell_size <= 3.75:
            cell_size = 2.5
        elif cell_size <= 7.5:
            cell_size = 5.0
        else:
            cell_size = 10.0
        
        return cell_size
    
    def build_grid(self, dataset: SVGDataset, existing_paths: Optional[List[List[Tuple[float, float]]]] = None):
        """
        Build grid from all obstructions.
        
        Args:
            dataset: SVG dataset with components and containers
            existing_paths: Optional list of existing connection paths (waypoints)
        """
        # Calculate adaptive cell size
        self.cell_size = self._calculate_cell_size(dataset)
        
        # Calculate grid dimensions
        self.grid_width = int(math.ceil(self.width / self.cell_size)) + 1
        self.grid_height = int(math.ceil(self.height / self.cell_size)) + 1
        
        # Initialize grid as all FREE
        self.grid = [[self.FREE for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        
        # Mark components as obstructions
        for comp in dataset.components:
            bbox = comp.get_absolute_bbox(dataset)
            self.mark_component(bbox, self.obstruction_padding)
        
        # Mark containers as obstructions (but exclude their children)
        for container in dataset.containers:
            bbox = container.get_absolute_bbox(dataset)
            self.mark_component(bbox, self.obstruction_padding)
        
        # Mark existing paths as occupied
        if existing_paths:
            for path_waypoints in existing_paths:
                self.mark_path(path_waypoints, 2.0)  # Default stroke width
        
        # Mark canvas edges as obstructions
        self._mark_edges()
    
    def _mark_edges(self):
        """Mark canvas edges as obstructions."""
        # Mark top and bottom edges
        for x in range(self.grid_width):
            if 0 < self.grid_height:
                self.grid[0][x] = self.OBSTRUCTED
            if self.grid_height > 1:
                self.grid[self.grid_height - 1][x] = self.OBSTRUCTED
        
        # Mark left and right edges
        for y in range(self.grid_height):
            if 0 < self.grid_width:
                self.grid[y][0] = self.OBSTRUCTED
            if self.grid_width > 1:
                self.grid[y][self.grid_width - 1] = self.OBSTRUCTED
    
    def mark_component(self, bbox: BoundingBox, padding: float):
        """
        Mark component area as obstructed.
        
        Args:
            bbox: Component bounding box
            padding: Padding around component
        """
        # Expand bounding box by padding
        expanded_bbox = BoundingBox(
            bbox.left - padding,
            bbox.top - padding,
            bbox.width + 2 * padding,
            bbox.height + 2 * padding
        )
        
        # Convert to grid cells
        start_x = max(0, int(expanded_bbox.left / self.cell_size))
        end_x = min(self.grid_width - 1, int(math.ceil(expanded_bbox.right / self.cell_size)))
        start_y = max(0, int(expanded_bbox.top / self.cell_size))
        end_y = min(self.grid_height - 1, int(math.ceil(expanded_bbox.bottom / self.cell_size)))
        
        # Mark cells as obstructed
        for y in range(start_y, end_y + 1):
            for x in range(start_x, end_x + 1):
                if 0 <= y < self.grid_height and 0 <= x < self.grid_width:
                    self.grid[y][x] = self.OBSTRUCTED
    
    def mark_path(self, waypoints: List[Tuple[float, float]], stroke_width: float):
        """
        Mark connection path as occupied.
        
        Args:
            waypoints: List of (x, y) waypoints
            stroke_width: Stroke width of the path
        """
        if len(waypoints) < 2:
            return
        
        padding = stroke_width / 2.0 + 2.0  # Half stroke width + extra spacing
        
        # Mark each segment
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            
            # Calculate bounding box for this segment
            min_x = min(start[0], end[0]) - padding
            max_x = max(start[0], end[0]) + padding
            min_y = min(start[1], end[1]) - padding
            max_y = max(start[1], end[1]) + padding
            
            # Convert to grid cells
            start_x = max(0, int(min_x / self.cell_size))
            end_x = min(self.grid_width - 1, int(math.ceil(max_x / self.cell_size)))
            start_y = max(0, int(min_y / self.cell_size))
            end_y = min(self.grid_height - 1, int(math.ceil(max_y / self.cell_size)))
            
            # Mark cells as PATH (can be crossed but with penalty)
            for y in range(start_y, end_y + 1):
                for x in range(start_x, end_x + 1):
                    if 0 <= y < self.grid_height and 0 <= x < self.grid_width:
                        if self.grid[y][x] == self.FREE:
                            self.grid[y][x] = self.PATH
    
    def get_cell_size(self) -> float:
        """Return current adaptive cell size."""
        return self.cell_size
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to grid coordinates.
        
        Args:
            x: World x coordinate
            y: World y coordinate
            
        Returns:
            (grid_x, grid_y) tuple
        """
        gx = int(x / self.cell_size)
        gy = int(y / self.cell_size)
        return (gx, gy)
    
    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """
        Convert grid coordinates to world coordinates.
        
        Args:
            gx: Grid x coordinate
            gy: Grid y coordinate
            
        Returns:
            (world_x, world_y) tuple (center of cell)
        """
        x = (gx + 0.5) * self.cell_size
        y = (gy + 0.5) * self.cell_size
        return (x, y)
    
    def is_valid(self, gx: int, gy: int) -> bool:
        """
        Check if grid cell is within bounds and free.
        
        Args:
            gx: Grid x coordinate
            gy: Grid y coordinate
            
        Returns:
            True if cell is valid for routing
        """
        if gx < 0 or gx >= self.grid_width or gy < 0 or gy >= self.grid_height:
            return False
        return self.grid[gy][gx] != self.OBSTRUCTED
    
    def get_neighbors(self, gx: int, gy: int) -> List[Tuple[int, int]]:
        """
        Get valid neighboring cells (4-directional for Manhattan routing).
        
        Args:
            gx: Grid x coordinate
            gy: Grid y coordinate
            
        Returns:
            List of (neighbor_gx, neighbor_gy) tuples
        """
        neighbors = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Up, Down, Right, Left
        
        for dx, dy in directions:
            ngx = gx + dx
            ngy = gy + dy
            if self.is_valid(ngx, ngy):
                neighbors.append((ngx, ngy))
        
        return neighbors
    
    def get_cell_cost(self, gx: int, gy: int) -> float:
        """
        Get cost for traversing a cell (for A* pathfinding).
        
        Args:
            gx: Grid x coordinate
            gy: Grid y coordinate
            
        Returns:
            Cost value (higher = more expensive)
        """
        if not self.is_valid(gx, gy):
            return float('inf')
        
        cell_state = self.grid[gy][gx]
        if cell_state == self.OBSTRUCTED:
            return float('inf')
        elif cell_state == self.PATH:
            # Path crossing penalty
            routing_config = self.config.get("routing", {})
            return 1.0 + routing_config.get("path_crossing_penalty", 5.0)
        else:
            # Base cost
            base_cost = 1.0
            
            # Add penalty for cells near obstructions (encourages routing further away)
            routing_config = self.config.get("routing", {})
            nearby_penalty = routing_config.get("obstruction_nearby_penalty", 10.0)
            
            # Check nearby cells for obstructions (within 2 cell radius)
            nearby_obstructions = 0
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    ngx = gx + dx
                    ngy = gy + dy
                    if 0 <= ngx < self.grid_width and 0 <= ngy < self.grid_height:
                        if self.grid[ngy][ngx] == self.OBSTRUCTED:
                            # Penalty decreases with distance
                            distance = abs(dx) + abs(dy)  # Manhattan distance
                            if distance == 1:
                                nearby_obstructions += 1.0
                            elif distance == 2:
                                nearby_obstructions += 0.5
            
            if nearby_obstructions > 0:
                # Apply penalty based on number of nearby obstructions
                cost_penalty = nearby_obstructions * nearby_penalty / 10.0
                return base_cost + cost_penalty
            
            return base_cost

