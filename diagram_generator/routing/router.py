#!/usr/bin/env python3
"""
Steiner Router - A* pathfinding for optimal connection routing.

This module implements A* pathfinding algorithm for finding optimal paths
between connection points while avoiding obstructions.
"""

from typing import Tuple, List, Optional, Dict, Any
import heapq
import math

from diagram_generator.routing.grid import RoutingGrid
from diagram_generator.core.diagram import SVGDataset


class AStarNode:
    """Node in A* pathfinding."""
    
    def __init__(self, gx: int, gy: int, g_score: float = float('inf'), h_score: float = 0.0, parent: Optional['AStarNode'] = None):
        self.gx = gx
        self.gy = gy
        self.g_score = g_score  # Cost from start
        self.h_score = h_score  # Heuristic to goal
        self.f_score = g_score + h_score  # Total estimated cost
        self.parent = parent
    
    def __lt__(self, other):
        """Comparison for priority queue."""
        if self.f_score != other.f_score:
            return self.f_score < other.f_score
        return self.h_score < other.h_score
    
    def __eq__(self, other):
        """Equality check."""
        return self.gx == other.gx and self.gy == other.gy
    
    def __hash__(self):
        """Hash for set operations."""
        return hash((self.gx, self.gy))


class SteinerRouter:
    """
    A* pathfinding router for optimal connection paths.
    
    Uses A* algorithm with Manhattan distance heuristic to find optimal
    paths between connection points while avoiding obstructions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Steiner router.
        
        Args:
            config: Configuration dictionary with routing parameters
        """
        self.config = config
        routing_config = config.get("routing", {})
        self.path_crossing_penalty = routing_config.get("path_crossing_penalty", 5.0)
        self.obstruction_nearby_penalty = routing_config.get("obstruction_nearby_penalty", 10.0)
        self.curved_corners = routing_config.get("curved_corners", True)
        self.corner_radius = routing_config.get("corner_radius", 5)
        self.steiner_point_offset = routing_config.get("steiner_point_offset", 18.0)
    
    def find_path(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        grid: RoutingGrid,
        dataset: SVGDataset
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Find optimal path using A* pathfinding.
        
        Args:
            start: Start point (world coordinates)
            end: End point (world coordinates)
            grid: RoutingGrid instance
            dataset: SVG dataset
            
        Returns:
            List of waypoints (world coordinates) or None if no path found
        """
        # Convert start/end to grid coordinates
        start_gx, start_gy = grid.world_to_grid(start[0], start[1])
        end_gx, end_gy = grid.world_to_grid(end[0], end[1])
        
        # Validate start and end points
        if not grid.is_valid(start_gx, start_gy):
            # Try to find nearest valid start point
            nearest = self._find_nearest_valid(grid, start_gx, start_gy)
            if nearest is None:
                return None
            start_gx, start_gy = nearest
        
        if not grid.is_valid(end_gx, end_gy):
            # Try to find nearest valid end point
            nearest = self._find_nearest_valid(grid, end_gx, end_gy)
            if nearest is None:
                return None
            end_gx, end_gy = nearest
        
        # Initialize open set (priority queue) with start node
        start_node = AStarNode(start_gx, start_gy, g_score=0.0, h_score=self._heuristic(start_gx, start_gy, end_gx, end_gy))
        open_set = [start_node]
        heapq.heapify(open_set)
        
        # Initialize closed set (visited nodes)
        closed_set = set()
        
        # Track best g-scores for each node
        g_scores = {(start_gx, start_gy): 0.0}
        
        # A* main loop
        while open_set:
            # Pop node with lowest f-score
            current = heapq.heappop(open_set)
            
            # Skip if already processed
            if (current.gx, current.gy) in closed_set:
                continue
            
            # Add to closed set
            closed_set.add((current.gx, current.gy))
            
            # Check if we reached the goal
            if current.gx == end_gx and current.gy == end_gy:
                # Reconstruct path
                path = self._reconstruct_path(current, grid)
                # Post-process path
                path = self._post_process_path(path, start, end, grid)
                return path
            
            # Explore neighbors
            neighbors = grid.get_neighbors(current.gx, current.gy)
            for ngx, ngy in neighbors:
                if (ngx, ngy) in closed_set:
                    continue
                
                # Calculate cost to reach neighbor
                move_cost = grid.get_cell_cost(ngx, ngy)
                tentative_g = current.g_score + move_cost
                
                # Check if this is a better path
                neighbor_key = (ngx, ngy)
                if neighbor_key not in g_scores or tentative_g < g_scores[neighbor_key]:
                    g_scores[neighbor_key] = tentative_g
                    h_score = self._heuristic(ngx, ngy, end_gx, end_gy)
                    neighbor_node = AStarNode(ngx, ngy, g_score=tentative_g, h_score=h_score, parent=current)
                    heapq.heappush(open_set, neighbor_node)
        
        # No path found
        return None
    
    def _find_nearest_valid(self, grid: RoutingGrid, gx: int, gy: int, max_radius: int = 10) -> Optional[Tuple[int, int]]:
        """
        Find nearest valid cell to given coordinates.
        
        Args:
            grid: RoutingGrid instance
            gx: Grid x coordinate
            gy: Grid y coordinate
            max_radius: Maximum search radius
            
        Returns:
            (valid_gx, valid_gy) or None if not found
        """
        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) == radius:  # Manhattan distance
                        ngx = gx + dx
                        ngy = gy + dy
                        if grid.is_valid(ngx, ngy):
                            return (ngx, ngy)
        return None
    
    def _heuristic(self, gx1: int, gy1: int, gx2: int, gy2: int) -> float:
        """
        Manhattan distance heuristic.
        
        Args:
            gx1, gy1: Start grid coordinates
            gx2, gy2: Goal grid coordinates
            
        Returns:
            Estimated cost to goal
        """
        return abs(gx1 - gx2) + abs(gy1 - gy2)
    
    def _reconstruct_path(self, goal_node: AStarNode, grid: RoutingGrid) -> List[Tuple[float, float]]:
        """
        Reconstruct path from goal to start.
        
        Args:
            goal_node: Goal node with parent chain
            grid: RoutingGrid instance
            
        Returns:
            List of waypoints (world coordinates)
        """
        path = []
        current = goal_node
        
        while current is not None:
            # Convert grid coordinates to world coordinates
            world_pos = grid.grid_to_world(current.gx, current.gy)
            path.append(world_pos)
            current = current.parent
        
        # Reverse to get path from start to end
        path.reverse()
        return path
    
    def _post_process_path(
        self,
        path: List[Tuple[float, float]],
        start: Tuple[float, float],
        end: Tuple[float, float],
        grid: RoutingGrid
    ) -> List[Tuple[float, float]]:
        """
        Post-process path: remove redundant waypoints, add arrowhead offset.
        
        Args:
            path: Raw path from A*
            start: Original start point (world coordinates)
            end: Original end point (world coordinates)
            grid: RoutingGrid instance
            
        Returns:
            Processed path with redundant points removed and arrowhead offset
        """
        if len(path) < 2:
            return path
        
        # Ensure Manhattan routing: add intermediate waypoints where needed
        # A* path through grid is Manhattan, but world coordinates may not be aligned
        processed = [path[0]]
        for i in range(1, len(path)):
            prev = processed[-1]
            curr = path[i]
            
            # Check if segment is Manhattan (horizontal or vertical)
            dx = abs(curr[0] - prev[0])
            dy = abs(curr[1] - prev[1])
            
            if dx < 0.1:
                # Vertical segment - already Manhattan
                processed.append(curr)
            elif dy < 0.1:
                # Horizontal segment - already Manhattan
                processed.append(curr)
            else:
                # Diagonal segment - add intermediate waypoint to make it Manhattan
                # Choose the direction that maintains the primary movement
                if dx > dy:
                    # Primarily horizontal - add vertical then horizontal
                    intermediate = (prev[0], curr[1])
                else:
                    # Primarily vertical - add horizontal then vertical
                    intermediate = (curr[0], prev[1])
                
                processed.append(intermediate)
                processed.append(curr)
        
        # Remove redundant waypoints (collinear points) after Manhattan alignment
        final_processed = [processed[0]]
        for i in range(1, len(processed) - 1):
            prev = final_processed[-1]
            curr = processed[i]
            next_pt = processed[i + 1]
            
            # Check if three points are collinear
            if not self._is_collinear(prev, curr, next_pt):
                final_processed.append(curr)
        
        final_processed.append(processed[-1])
        processed = final_processed
        
        # Replace first and last points with actual start/end BEFORE adding arrowhead offset
        # This ensures Manhattan alignment is maintained
        processed[0] = start
        processed[-1] = end
        
        # After replacing start/end, verify Manhattan alignment is maintained
        # If replacing start/end created diagonal segments, fix them
        final_processed = [processed[0]]
        for i in range(1, len(processed)):
            prev = final_processed[-1]
            curr = processed[i]
            
            # Check if segment is Manhattan (horizontal or vertical)
            dx = abs(curr[0] - prev[0])
            dy = abs(curr[1] - prev[1])
            
            if dx < 0.1:
                # Vertical segment - already Manhattan
                final_processed.append(curr)
            elif dy < 0.1:
                # Horizontal segment - already Manhattan
                final_processed.append(curr)
            else:
                # Diagonal segment - add intermediate waypoint to make it Manhattan
                if dx > dy:
                    # Primarily horizontal - add vertical then horizontal
                    intermediate = (prev[0], curr[1])
                else:
                    # Primarily vertical - add horizontal then vertical
                    intermediate = (curr[0], prev[1])
                
                final_processed.append(intermediate)
                final_processed.append(curr)
        
        processed = final_processed
        
        # Add arrowhead offset before entry point (after Manhattan alignment is verified)
        if len(processed) >= 2:
            # Calculate direction of last segment
            last_seg_start = processed[-2]
            last_seg_end = processed[-1]
            
            dx = last_seg_end[0] - last_seg_start[0]
            dy = last_seg_end[1] - last_seg_start[1]
            
            # Normalize direction
            length = math.sqrt(dx * dx + dy * dy)
            if length > 0:
                dx /= length
                dy /= length
                
                # Calculate Steiner point offset from entry
                offset_x = last_seg_end[0] - dx * self.steiner_point_offset
                offset_y = last_seg_end[1] - dy * self.steiner_point_offset
                
                # Insert Steiner point before entry
                processed.insert(-1, (offset_x, offset_y))
        
        return processed
    
    def _is_collinear(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float], tolerance: float = 0.1) -> bool:
        """
        Check if three points are collinear.
        
        Args:
            p1, p2, p3: Three points
            tolerance: Tolerance for collinearity check
            
        Returns:
            True if points are collinear
        """
        # Calculate cross product
        cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        return abs(cross) < tolerance

