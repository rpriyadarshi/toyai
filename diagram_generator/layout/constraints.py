#!/usr/bin/env python3
"""
Constraint Solver - Spacing constraint satisfaction for component placement.

This module provides constraint satisfaction algorithms to ensure proper
spacing between components while minimizing connection lengths.
"""

import math
from typing import Dict, Any, List, Tuple, Optional

from diagram_generator.core.diagram import Component, Connection, BoundingBox


class ConstraintSolver:
    """Solve spacing constraints for component placement."""
    
    def __init__(self, min_spacing: float = 50.0, border_padding: float = 20.0):
        """
        Initialize constraint solver.
        
        Args:
            min_spacing: Minimum spacing between components
            border_padding: Border padding from canvas edges
        """
        self.min_spacing = min_spacing
        self.border_padding = border_padding
    
    def solve_spacing_constraints(
        self,
        components: List[Component],
        connections: List[Connection],
        container_bounds: Optional[BoundingBox] = None
    ) -> Dict[str, Tuple[float, float]]:
        """
        Solve spacing constraints to find valid positions.
        
        Uses iterative constraint satisfaction:
        1. Start with current positions
        2. Detect violations
        3. Adjust positions to resolve violations
        4. Minimize connection lengths
        
        Args:
            components: List of components to place
            connections: List of connections between components
            container_bounds: Optional container bounding box to constrain positions
            
        Returns:
            Dictionary mapping component ID to (x, y) position
        """
        positions = {comp.id: comp.position for comp in components}
        
        # #region agent log
        try:
            with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                import json
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H7", "location": "constraint_solver.py:53", "message": "solve_spacing_constraints entry", "data": {"num_components": len(components), "initial_positions": {k: list(v) for k, v in positions.items()}, "min_spacing": self.min_spacing}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
        except: pass
        # #endregion
        
        # Detect if this is a vertical layout (all components at same X or mostly vertical connections)
        is_vertical_layout = self._is_vertical_layout(components, positions, connections)
        
        # If vertical layout, determine component order from connections
        vertical_order = None
        if is_vertical_layout:
            vertical_order = self._determine_vertical_order(components, connections, positions)
            # Only reorder if there are actual overlaps that need fixing
            # Check for overlaps first
            has_overlaps = False
            for i, comp1 in enumerate(components):
                pos1 = positions[comp1.id]
                bbox1 = BoundingBox(pos1[0], pos1[1], comp1.width, comp1.height)
                for comp2 in components[i+1:]:
                    pos2 = positions[comp2.id]
                    bbox2 = BoundingBox(pos2[0], pos2[1], comp2.width, comp2.height)
                    if bbox1.intersects(bbox2):
                        has_overlaps = True
                        break
                if has_overlaps:
                    break
            
            # Only reorder if there are overlaps - otherwise preserve exact positions
            if has_overlaps:
                positions = self._reorder_vertical_layout(components, connections, positions, vertical_order)
        
        # Iterative improvement with more aggressive resolution
        max_iterations = 200
        for iteration in range(max_iterations):
            violations = self._detect_violations(components, positions, container_bounds)
            
            # #region agent log
            if iteration == 0 or iteration == max_iterations - 1 or len(violations) == 0:
                try:
                    with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                        import json
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H7", "location": "constraint_solver.py:58", "message": "Constraint solver iteration", "data": {"iteration": iteration, "violations_count": len(violations), "positions": {k: list(v) for k, v in positions.items()}}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                except: pass
            # #endregion
            
            if not violations:
                break
            
            # Resolve violations more aggressively
            self._resolve_violations(components, positions, violations, container_bounds, is_vertical_layout, vertical_order)
            
            # If we're not making progress, try more aggressive moves
            if iteration > 50 and len(violations) > 0:
                # Apply larger adjustments
                for violation in violations:
                    comp_id = violation["comp1"]
                    if comp_id in positions:
                        comp = next((c for c in components if c.id == comp_id), None)
                        if comp:
                            needed = violation["needed"]
                            if violation["type"] == "horizontal":
                                # Move further apart
                                other_comp = next((c for c in components if c.id == violation["comp2"]), None)
                                if other_comp and other_comp.id in positions:
                                    other_pos = positions[other_comp.id]
                                    my_pos = positions[comp_id]
                                    if my_pos[0] < other_pos[0]:
                                        positions[comp_id] = (my_pos[0] - needed, my_pos[1])
                                    else:
                                        positions[comp_id] = (my_pos[0] + needed, my_pos[1])
                            elif violation["type"] == "vertical":
                                # Move further apart
                                other_comp = next((c for c in components if c.id == violation["comp2"]), None)
                                if other_comp and other_comp.id in positions:
                                    other_pos = positions[other_comp.id]
                                    my_pos = positions[comp_id]
                                    if my_pos[1] < other_pos[1]:
                                        # comp1 is above comp2 - move comp1 DOWN (increase y) to increase spacing
                                        positions[comp_id] = (my_pos[0], my_pos[1] + needed)
                                    else:
                                        # comp1 is below comp2 - move comp1 UP (decrease y) to increase spacing
                                        positions[comp_id] = (my_pos[0], my_pos[1] - needed)
                            
                            # Constrain to bounds
                            if container_bounds:
                                new_pos = positions[comp_id]
                                new_x = max(container_bounds.left + comp.width/2,
                                          min(container_bounds.right - comp.width/2, new_pos[0]))
                                new_y = max(container_bounds.top + comp.height/2,
                                          min(container_bounds.bottom - comp.height/2, new_pos[1]))
                                positions[comp_id] = (new_x, new_y)
        
        return positions
    
    def _detect_violations(
        self,
        components: List[Component],
        positions: Dict[str, Tuple[float, float]],
        container_bounds: Optional[BoundingBox]
    ) -> List[Dict[str, Any]]:
        # #region agent log
        try:
            with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                import json
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H7", "location": "constraint_solver.py:109", "message": "_detect_violations entry", "data": {"num_components": len(components), "min_spacing": self.min_spacing, "component_ids": [c.id for c in components], "positions": {k: list(v) for k, v in positions.items()}}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
        except: pass
        # #endregion
        # #region agent log
        import json
        log_path = "/home/rohit/src/toyai-1/.cursor/debug.log"
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H8", "location": "constraint_solver.py:107", "message": "_detect_violations entry", "data": {"num_components": len(components), "min_spacing": self.min_spacing, "component_ids": [c.id for c in components]}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
        except: pass
        # #endregion
        """Detect spacing violations."""
        violations = []
        
        # #region agent log
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H8", "location": "constraint_solver.py:122", "message": "checking violations", "data": {"num_components": len(components)}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
        except: pass
        # #endregion
        
        for i, comp1 in enumerate(components):
            pos1 = positions[comp1.id]
            bbox1 = BoundingBox(pos1[0], pos1[1], comp1.width, comp1.height)
            
            for comp2 in components[i+1:]:
                pos2 = positions[comp2.id]
                bbox2 = BoundingBox(pos2[0], pos2[1], comp2.width, comp2.height)
                
                # Skip overlap check if comp1 is a container and comp2 is its child (or vice versa)
                # Containers have type="container", children have parent_id set
                if (comp1.type == "container" and comp2.parent_id == comp1.id) or \
                   (comp2.type == "container" and comp1.parent_id == comp2.id):
                    continue
                
                # Check if boxes overlap
                if bbox1.intersects(bbox2):
                    # Overlapping - need to separate
                    # Calculate overlap amounts
                    overlap_x = min(bbox1.right, bbox2.right) - max(bbox1.left, bbox2.left)
                    overlap_y = min(bbox1.bottom, bbox2.bottom) - max(bbox1.top, bbox2.top)
                    
                    # Determine separation direction (prefer vertical for stacked, horizontal for side-by-side)
                    if overlap_x > overlap_y:
                        # More horizontal overlap - separate vertically
                        center1_y = bbox1.center_y
                        center2_y = bbox2.center_y
                        if center1_y < center2_y:
                            # comp1 above comp2
                            needed = overlap_y + self.min_spacing
                            violations.append({
                                "type": "vertical",
                                "comp1": comp1.id,
                                "comp2": comp2.id,
                                "current_spacing": -overlap_y,
                                "needed": needed
                            })
                        else:
                            # comp2 above comp1
                            needed = overlap_y + self.min_spacing
                            violations.append({
                                "type": "vertical",
                                "comp1": comp2.id,
                                "comp2": comp1.id,
                                "current_spacing": -overlap_y,
                                "needed": needed
                            })
                    else:
                        # More vertical overlap - separate horizontally
                        center1_x = bbox1.center_x
                        center2_x = bbox2.center_x
                        if center1_x < center2_x:
                            # comp1 left of comp2
                            needed = overlap_x + self.min_spacing
                            violations.append({
                                "type": "horizontal",
                                "comp1": comp1.id,
                                "comp2": comp2.id,
                                "current_spacing": -overlap_x,
                                "needed": needed
                            })
                        else:
                            # comp2 left of comp1
                            needed = overlap_x + self.min_spacing
                            violations.append({
                                "type": "horizontal",
                                "comp1": comp2.id,
                                "comp2": comp1.id,
                                "current_spacing": -overlap_x,
                                "needed": needed
                            })
                    continue
                
                # Check horizontal spacing (non-overlapping)
                if bbox1.right <= bbox2.left:
                    spacing = bbox2.left - bbox1.right
                    if spacing < self.min_spacing:
                        violations.append({
                            "type": "horizontal",
                            "comp1": comp1.id,
                            "comp2": comp2.id,
                            "current_spacing": spacing,
                            "needed": self.min_spacing - spacing
                        })
                elif bbox2.right <= bbox1.left:
                    spacing = bbox1.left - bbox2.right
                    if spacing < self.min_spacing:
                        violations.append({
                            "type": "horizontal",
                            "comp1": comp2.id,
                            "comp2": comp1.id,
                            "current_spacing": spacing,
                            "needed": self.min_spacing - spacing
                        })
                
                # Check vertical spacing (non-overlapping)
                if bbox1.bottom <= bbox2.top:
                    spacing = bbox2.top - bbox1.bottom
                    if spacing < self.min_spacing:
                        violation = {
                            "type": "vertical",
                            "comp1": comp1.id,
                            "comp2": comp2.id,
                            "current_spacing": spacing,
                            "needed": self.min_spacing - spacing
                        }
                        violations.append(violation)
                        # #region agent log
                        try:
                            with open(log_path, "a") as f:
                                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H8", "location": "constraint_solver.py:220", "message": "vertical violation detected", "data": violation, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                        except: pass
                        # #endregion
                elif bbox2.bottom <= bbox1.top:
                    spacing = bbox1.top - bbox2.bottom
                    if spacing < self.min_spacing:
                        violations.append({
                            "type": "vertical",
                            "comp1": comp2.id,
                            "comp2": comp1.id,
                            "current_spacing": spacing,
                            "needed": self.min_spacing - spacing
                        })
        
        return violations
    
    def _is_vertical_layout(
        self,
        components: List[Component],
        positions: Dict[str, Tuple[float, float]],
        connections: List[Connection]
    ) -> bool:
        """Check if components are arranged vertically (same X or mostly vertical connections)."""
        if not components:
            return False
        
        # Check if all components are at same X position (within tolerance)
        x_positions = [pos[0] for pos in positions.values()]
        if len(set(round(x, 1) for x in x_positions)) == 1:
            return True
        
        # Check if connections are mostly vertical
        if connections:
            vertical_connections = 0
            for conn in connections:
                from_pos = positions.get(conn.from_component_id)
                to_pos = positions.get(conn.to_component_id)
                if from_pos and to_pos:
                    dx = abs(to_pos[0] - from_pos[0])
                    dy = abs(to_pos[1] - from_pos[1])
                    if dy > dx * 2:  # More vertical than horizontal
                        vertical_connections += 1
            if vertical_connections >= len(connections) * 0.7:  # 70% vertical
                return True
        
        return False
    
    def _determine_vertical_order(
        self,
        components: List[Component],
        connections: List[Connection],
        positions: Dict[str, Tuple[float, float]]
    ) -> Dict[str, int]:
        """Determine top-to-bottom order for vertical layout from connections or positions."""
        order = {}
        
        # Try to determine order from connections (follow forward flow, ignore loops)
        if connections:
            # Build forward flow graph (ignore backward/loop connections)
            forward_edges = {}
            for conn in connections:
                from_id = conn.from_component_id
                to_id = conn.to_component_id
                # Only consider forward connections (not training/loop connections)
                if conn.style != "training" and from_id in [c.id for c in components] and to_id in [c.id for c in components]:
                    if from_id not in forward_edges:
                        forward_edges[from_id] = []
                    forward_edges[from_id].append(to_id)
            
            # Find start nodes (no incoming forward edges)
            in_degree = {comp.id: 0 for comp in components}
            for from_id, to_ids in forward_edges.items():
                for to_id in to_ids:
                    if to_id in in_degree:
                        in_degree[to_id] += 1
            
            start_ids = [comp.id for comp in components if in_degree[comp.id] == 0]
            if not start_ids:
                # No clear start, use initial Y positions
                sorted_comps = sorted(components, key=lambda c: positions.get(c.id, (0, 0))[1])
            else:
                # Traverse forward flow
                sorted_comps = []
                visited = set()
                queue = start_ids[:]
                
                while queue:
                    comp_id = queue.pop(0)
                    if comp_id in visited:
                        continue
                    visited.add(comp_id)
                    comp = next((c for c in components if c.id == comp_id), None)
                    if comp:
                        sorted_comps.append(comp)
                    # Add forward neighbors
                    for neighbor_id in forward_edges.get(comp_id, []):
                        if neighbor_id not in visited:
                            queue.append(neighbor_id)
                
                # Add any unvisited components (disconnected or loop-only)
                for comp in components:
                    if comp.id not in visited:
                        sorted_comps.append(comp)
            
            for idx, comp in enumerate(sorted_comps):
                order[comp.id] = idx
        else:
            # No connections: use Y position
            sorted_comps = sorted(components, key=lambda c: positions.get(c.id, (0, 0))[1])
            for idx, comp in enumerate(sorted_comps):
                order[comp.id] = idx
        
        return order
    
    def _reorder_vertical_layout(
        self,
        components: List[Component],
        connections: List[Connection],
        positions: Dict[str, Tuple[float, float]],
        vertical_order: Dict[str, int]
    ) -> Dict[str, Tuple[float, float]]:
        """Completely reorder components in vertical layout to fix overlaps."""
        # Sort components by order
        sorted_comps = sorted(components, key=lambda c: vertical_order.get(c.id, 999))
        
        # Find average X position (preserve horizontal alignment)
        avg_x = sum(pos[0] for pos in positions.values()) / len(positions) if positions else 0
        
        # Place components sequentially from top to bottom, starting at border_padding
        new_positions = {}
        current_y = self.border_padding
        
        for comp in sorted_comps:
            new_positions[comp.id] = (avg_x, current_y)
            current_y += comp.height + self.min_spacing
        
        return new_positions
    
    def _resolve_violations(
        self,
        components: List[Component],
        positions: Dict[str, Tuple[float, float]],
        violations: List[Dict[str, Any]],
        container_bounds: Optional[BoundingBox],
        is_vertical_layout: bool = False,
        vertical_order: Optional[Dict[str, int]] = None
    ):
        """Resolve spacing violations by adjusting positions."""
        # Group violations by component
        comp_violations: Dict[str, List[Dict[str, Any]]] = {}
        for violation in violations:
            comp_id = violation["comp1"]
            if comp_id not in comp_violations:
                comp_violations[comp_id] = []
            comp_violations[comp_id].append(violation)
        
        # Adjust positions
        for comp_id, comp_viols in comp_violations.items():
            comp = next((c for c in components if c.id == comp_id), None)
            if not comp:
                continue
            
            dx = 0.0
            dy = 0.0
            
            for violation in comp_viols:
                if violation["type"] == "horizontal":
                    # Move away horizontally
                    other_comp = next((c for c in components if c.id == violation["comp2"]), None)
                    if other_comp:
                        other_pos = positions[other_comp.id]
                        my_pos = positions[comp_id]
                        
                        if my_pos[0] < other_pos[0]:
                            # Move left
                            dx -= violation["needed"] / 2
                        else:
                            # Move right
                            dx += violation["needed"] / 2
                
                elif violation["type"] == "vertical":
                    other_comp = next((c for c in components if c.id == violation["comp2"]), None)
                    if other_comp:
                        other_pos = positions[other_comp.id]
                        my_pos = positions[comp_id]
                        
                        if is_vertical_layout and vertical_order:
                            # For vertical layouts, enforce order: move components to maintain top-to-bottom order
                            my_order = vertical_order.get(comp_id, 999)
                            other_order = vertical_order.get(other_comp.id, 999)
                            
                            if my_order < other_order:
                                # I should be above other - move other DOWN
                                # But we're adjusting comp_id, so move myself UP if I'm too low
                                if my_pos[1] >= other_pos[1]:
                                    # I'm below where I should be - move UP
                                    dy -= violation["needed"]
                                else:
                                    # I'm above but too close - move other DOWN (adjust myself slightly)
                                    dy -= violation["needed"] / 4
                            else:
                                # I should be below other - move myself DOWN
                                if my_pos[1] <= other_pos[1]:
                                    # I'm above where I should be - move DOWN
                                    dy += violation["needed"]
                                else:
                                    # I'm below but too close - move myself DOWN more
                                    dy += violation["needed"] / 4
                        else:
                            # Non-vertical layout: move away from each other
                            if my_pos[1] < other_pos[1]:
                                # comp1 is above comp2 - move comp1 DOWN (increase y) to increase spacing
                                dy += violation["needed"] / 2
                            else:
                                # comp1 is below comp2 - move comp1 UP (decrease y) to increase spacing
                                dy -= violation["needed"] / 2
            
            # Apply adjustment
            new_x = positions[comp_id][0] + dx
            new_y = positions[comp_id][1] + dy
            
            # Constrain to container bounds if provided
            if container_bounds:
                new_x = max(container_bounds.left + comp.width/2, 
                           min(container_bounds.right - comp.width/2, new_x))
                new_y = max(container_bounds.top + comp.height/2,
                           min(container_bounds.bottom - comp.height/2, new_y))
            
            positions[comp_id] = (new_x, new_y)

