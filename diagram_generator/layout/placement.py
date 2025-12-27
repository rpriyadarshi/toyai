#!/usr/bin/env python3
"""
Placement Engine - High-quality placement algorithms for diagram layout.

This module provides multiple placement algorithms:
- Force-directed: Spring forces and repulsion
- Hierarchical: Graphviz-style level-based layout
- Grid-based: Constraint satisfaction on grid
"""

import math
import random
from typing import Dict, Any, List, Tuple, Optional

from diagram_generator.core.diagram import SVGDataset, Component, Connection, BoundingBox
from diagram_generator.layout.hierarchy import HierarchyAnalyzer
from diagram_generator.layout.constraints import ConstraintSolver


class PlacementEngine:
    """Placement algorithms for diagram components."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize placement engine.
        
        Args:
            config: Diagram configuration
        """
        self.config = config
        spacing = config.get("spacing", {})
        self.min_spacing = spacing.get("min_box_spacing", 50)
        self.border_padding = spacing.get("border_padding", 20)
    
    def place_force_directed(
        self, 
        dataset: SVGDataset, 
        diagram_json: Dict[str, Any],
        iterations: int = 100,
        temperature: float = 1.0
    ) -> SVGDataset:
        """
        Force-directed placement algorithm.
        
        Uses spring forces for connections and repulsion for all components.
        Gradually cools down to find stable positions.
        
        Args:
            dataset: SVG dataset with components
            diagram_json: Diagram JSON definition
            iterations: Number of iterations
            temperature: Initial temperature (controls movement)
            
        Returns:
            Updated dataset with new positions
        """
        # Initialize positions if not set
        for comp in dataset.components:
            if comp.position == (0, 0) and comp.id not in [c["id"] for c in diagram_json.get("components", []) if "position" in c]:
                # Random initial position
                comp.position = (
                    random.uniform(100, dataset.width - 100),
                    random.uniform(100, dataset.height - 100)
                )
        
        # Build connection graph
        connections_by_comp = {}
        for conn in dataset.connections:
            if conn.from_component_id not in connections_by_comp:
                connections_by_comp[conn.from_component_id] = []
            connections_by_comp[conn.from_component_id].append(conn.to_component_id)
            
            if conn.to_component_id not in connections_by_comp:
                connections_by_comp[conn.to_component_id] = []
            connections_by_comp[conn.to_component_id].append(conn.from_component_id)
        
        # Force-directed simulation
        for iteration in range(iterations):
            forces = {}
            current_temp = temperature * (1 - iteration / iterations)
            
            # Initialize forces
            for comp in dataset.components:
                forces[comp.id] = [0.0, 0.0]  # [fx, fy]
            
            # Spring forces (attraction for connected components)
            spring_constant = 0.1
            ideal_length = self.min_spacing * 3
            
            for conn in dataset.connections:
                from_comp = dataset.get_component(conn.from_component_id)
                to_comp = dataset.get_component(conn.to_component_id)
                
                if not from_comp or not to_comp:
                    continue
                
                dx = to_comp.bbox.center_x - from_comp.bbox.center_x
                dy = to_comp.bbox.center_y - from_comp.bbox.center_y
                distance = math.sqrt(dx*dx + dy*dy) or 1.0
                
                # Spring force
                force_magnitude = spring_constant * (distance - ideal_length)
                fx = force_magnitude * dx / distance
                fy = force_magnitude * dy / distance
                
                forces[from_comp.id][0] += fx
                forces[from_comp.id][1] += fy
                forces[to_comp.id][0] -= fx
                forces[to_comp.id][1] -= fy
            
            # Repulsion forces (between all components)
            repulsion_constant = 1000.0
            
            for i, comp1 in enumerate(dataset.components):
                for comp2 in dataset.components[i+1:]:
                    dx = comp2.bbox.center_x - comp1.bbox.center_x
                    dy = comp2.bbox.center_y - comp1.bbox.center_y
                    distance_sq = dx*dx + dy*dy
                    
                    if distance_sq < 1.0:
                        distance_sq = 1.0
                    
                    # Repulsion force (inverse square)
                    force_magnitude = repulsion_constant / distance_sq
                    fx = force_magnitude * dx / math.sqrt(distance_sq)
                    fy = force_magnitude * dy / math.sqrt(distance_sq)
                    
                    forces[comp1.id][0] -= fx
                    forces[comp1.id][1] -= fy
                    forces[comp2.id][0] += fx
                    forces[comp2.id][1] += fy
            
            # Apply forces with temperature
            for comp in dataset.components:
                fx, fy = forces[comp.id]
                
                # Limit force magnitude
                force_mag = math.sqrt(fx*fx + fy*fy)
                if force_mag > 10.0:
                    fx = fx * 10.0 / force_mag
                    fy = fy * 10.0 / force_mag
                
                # Update position
                new_x = comp.position[0] + fx * current_temp
                new_y = comp.position[1] + fy * current_temp
                
                # Keep within bounds
                new_x = max(comp.width/2, min(dataset.width - comp.width/2, new_x))
                new_y = max(comp.height/2, min(dataset.height - comp.height/2, new_y))
                
                comp.position = (new_x, new_y)
        
        return dataset
    
    def place_hierarchical(
        self,
        dataset: SVGDataset,
        diagram_json: Dict[str, Any],
        direction: str = "top-down"
    ) -> SVGDataset:
        """
        Hierarchical (Graphviz-style) placement algorithm.
        
        Assigns components to levels based on connection graph,
        then positions them within each level to minimize crossings.
        
        Args:
            dataset: SVG dataset with components
            diagram_json: Diagram JSON definition
            direction: Layout direction ("top-down", "left-right")
            
        Returns:
            Updated dataset with new positions
        """
        # Build dependency graph
        in_degree = {comp.id: 0 for comp in dataset.components}
        out_edges = {comp.id: [] for comp in dataset.components}
        
        for conn in dataset.connections:
            if conn.from_component_id in in_degree and conn.to_component_id in in_degree:
                out_edges[conn.from_component_id].append(conn.to_component_id)
                in_degree[conn.to_component_id] += 1
        
        # Topological sort to assign levels
        levels = []
        remaining = set(comp.id for comp in dataset.components)
        current_level = [comp_id for comp_id, degree in in_degree.items() if degree == 0]
        
        while current_level:
            levels.append(current_level)
            remaining -= set(current_level)
            next_level = []
            
            for comp_id in current_level:
                for target_id in out_edges[comp_id]:
                    in_degree[target_id] -= 1
                    if in_degree[target_id] == 0 and target_id in remaining:
                        next_level.append(target_id)
            
            current_level = next_level
        
        # Add remaining components (cycles)
        if remaining:
            levels.append(list(remaining))
        
        # Position components within levels
        if direction == "top-down":
            level_height = (dataset.height - 2 * self.border_padding) / max(len(levels), 1)
            y_start = self.border_padding
            
            for level_idx, level_comps in enumerate(levels):
                y = y_start + level_idx * level_height
                
                # Calculate total width needed
                total_width = sum(dataset.get_component(cid).width for cid in level_comps)
                total_width += self.min_spacing * (len(level_comps) - 1)
                
                # Center level
                x_start = (dataset.width - total_width) / 2
                x = x_start
                
                for comp_id in level_comps:
                    comp = dataset.get_component(comp_id)
                    if comp:
                        comp.position = (x, y)
                        x += comp.width + self.min_spacing
        
        elif direction == "left-right":
            level_width = (dataset.width - 2 * self.border_padding) / max(len(levels), 1)
            x_start = self.border_padding
            
            for level_idx, level_comps in enumerate(levels):
                x = x_start + level_idx * level_width
                
                # Calculate total height needed
                total_height = sum(dataset.get_component(cid).height for cid in level_comps)
                total_height += self.min_spacing * (len(level_comps) - 1)
                
                # Center level
                y_start = (dataset.height - total_height) / 2
                y = y_start
                
                for comp_id in level_comps:
                    comp = dataset.get_component(comp_id)
                    if comp:
                        comp.position = (x, y)
                        y += comp.height + self.min_spacing
        
        return dataset
    
    def place_grid_based(
        self,
        dataset: SVGDataset,
        diagram_json: Dict[str, Any],
        grid_size: int = 20
    ) -> SVGDataset:
        """
        Grid-based constraint satisfaction placement.
        
        Divides space into grid cells and places components on grid points,
        satisfying spacing constraints and minimizing connection lengths.
        
        Args:
            dataset: SVG dataset with components
            diagram_json: Diagram JSON definition
            grid_size: Size of grid cells in pixels
            
        Returns:
            Updated dataset with new positions
        """
        # Create grid
        grid_cols = int(dataset.width / grid_size)
        grid_rows = int(dataset.height / grid_size)
        
        # Initialize grid (True = occupied)
        grid = [[False for _ in range(grid_cols)] for _ in range(grid_rows)]
        
        # Place components on grid
        placed = []
        
        for comp in dataset.components:
            # Find best grid position
            best_pos = None
            best_score = float('inf')
            
            # Try each grid cell
            for row in range(grid_rows):
                for col in range(grid_cols):
                    x = col * grid_size
                    y = row * grid_size
                    
                    # Check if this position is valid
                    if self._is_valid_grid_position(comp, x, y, grid, grid_size, placed, dataset):
                        # Score: minimize connection length
                        score = self._calculate_position_score(comp, x, y, dataset)
                        
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y)
            
            if best_pos:
                comp.position = best_pos
                placed.append(comp)
                
                # Mark grid cells as occupied
                self._mark_grid_occupied(comp, best_pos[0], best_pos[1], grid, grid_size)
        
        return dataset
    
    def _is_valid_grid_position(
        self,
        comp: Component,
        x: float,
        y: float,
        grid: List[List[bool]],
        grid_size: float,
        placed: List[Component],
        dataset: SVGDataset
    ) -> bool:
        """Check if position is valid (no overlaps, respects spacing)."""
        # Check bounds
        if x + comp.width > dataset.width or y + comp.height > dataset.height:
            return False
        
        # Check spacing with placed components
        for placed_comp in placed:
            dx = abs(placed_comp.position[0] - x)
            dy = abs(placed_comp.position[1] - y)
            
            if dx < comp.width/2 + placed_comp.width/2 + self.min_spacing:
                if dy < comp.height/2 + placed_comp.height/2 + self.min_spacing:
                    return False
        
        return True
    
    def _calculate_position_score(self, comp: Component, x: float, y: float, dataset: SVGDataset) -> float:
        """Calculate score for a position (lower is better)."""
        score = 0.0
        
        # Find connections involving this component
        for conn in dataset.connections:
            if conn.from_component_id == comp.id:
                target = dataset.get_component(conn.to_component_id)
                if target:
                    dx = target.bbox.center_x - (x + comp.width/2)
                    dy = target.bbox.center_y - (y + comp.height/2)
                    score += math.sqrt(dx*dx + dy*dy)
            
            elif conn.to_component_id == comp.id:
                source = dataset.get_component(conn.from_component_id)
                if source:
                    dx = (x + comp.width/2) - source.bbox.center_x
                    dy = (y + comp.height/2) - source.bbox.center_y
                    score += math.sqrt(dx*dx + dy*dy)
        
        return score
    
    def _mark_grid_occupied(self, comp: Component, x: float, y: float, grid: List[List[bool]], grid_size: float):
        """Mark grid cells as occupied by component."""
        start_col = int(x / grid_size)
        start_row = int(y / grid_size)
        end_col = int((x + comp.width) / grid_size) + 1
        end_row = int((y + comp.height) / grid_size) + 1
        
        for row in range(start_row, min(end_row, len(grid))):
            for col in range(start_col, min(end_col, len(grid[0]))):
                if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
                    grid[row][col] = True
    
    def place_hierarchical_bottom_up(
        self,
        dataset: SVGDataset,
        diagram_json: Dict[str, Any]
    ) -> SVGDataset:
        """
        Bottom-up hierarchical placement algorithm.
        
        Algorithm:
        1. Global placement of all top-level elements
        2. For each container (bottom-up):
           a. Group connected components
           b. Detail placement with spacing rules
           c. Place labels
           d. Compute bounding box
           e. Move up hierarchy
        
        Args:
            dataset: SVG dataset with components
            diagram_json: Diagram JSON definition
            
        Returns:
            Updated dataset with new positions
        """
        # Build hierarchy
        analyzer = HierarchyAnalyzer(diagram_json)
        constraint_solver = ConstraintSolver(self.min_spacing, self.border_padding)
        
        # Store original container positions from JSON BEFORE any placement
        original_container_positions = {}
        for container_def in diagram_json.get("containers", []):
            if "position" in container_def:
                original_container_positions[container_def["id"]] = (
                    container_def["position"]["x"],
                    container_def["position"]["y"]
                )
        
        # Step 1: Global placement of top-level elements
        # Separate containers from components - containers keep JSON positions
        top_level_ids = analyzer.get_top_level_elements()
        top_level_containers = []
        top_level_components = []
        
        for elem_id in top_level_ids:
            # Check if it's a container
            container = next((c for c in dataset.containers if c.id == elem_id), None)
            if container:
                top_level_containers.append(container)
            else:
                # Try component
                elem = dataset.get_component(elem_id)
                if elem:
                    top_level_components.append(elem)
        
        # Containers keep their original positions from JSON
        for container in top_level_containers:
            if container.id in original_container_positions:
                container.position = original_container_positions[container.id]
        
        # Only apply placement to components (not containers)
        if top_level_components:
            # Use manual positions as seed
            manual_positions = {}
            for elem in top_level_components:
                manual_positions[elem.id] = elem.position
            
            # Get connections between top-level components
            top_level_connections = [
                conn for conn in dataset.connections
                if (conn.from_component_id in [e.id for e in top_level_components] and
                    conn.to_component_id in [e.id for e in top_level_components])
            ]
            
            # Check if ALL components have explicit seed positions
            # If so, preserve exact positions regardless of topology
            all_have_seeds = all(comp.id in manual_positions for comp in top_level_components)
            
            # Detect topology and choose appropriate algorithm
            topology = self._detect_topology(top_level_components, top_level_connections)
            
            # #region agent log
            try:
                with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                    import json
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1", "location": "placement_engine.py:447", "message": "Topology detected", "data": {"topology": topology, "all_have_seeds": all_have_seeds, "component_count": len(top_level_components), "connection_count": len(top_level_connections), "component_ids": [c.id for c in top_level_components], "manual_positions": {k: list(v) for k, v in manual_positions.items()}}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
            except: pass
            # #endregion
            
            # If all components have seed positions, preserve them exactly
            if all_have_seeds:
                # Use exact seed positions - no placement algorithm needed
                for comp in top_level_components:
                    if comp.id in manual_positions:
                        comp.position = manual_positions[comp.id]
            elif topology == "linear":
                # Linear sequence: preserve seed order and relative spacing
                dataset = self._place_linear_with_seeds(
                    dataset,
                    top_level_components,
                    manual_positions,
                    top_level_connections
                )
            elif topology in ["branching", "converging"]:
                # Branching or converging: use hierarchical with seed anchors
                # #region agent log
                try:
                    with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                        import json
                        positions_before = {c.id: list(c.position) for c in top_level_components}
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H2", "location": "placement_engine.py:459", "message": "Before _place_hierarchical_with_anchors", "data": {"topology": topology, "positions_before": positions_before, "manual_positions": {k: list(v) for k, v in manual_positions.items()}}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                except: pass
                # #endregion
                dataset = self._place_hierarchical_with_anchors(
                    dataset,
                    top_level_components,
                    manual_positions,
                    diagram_json
                )
                # #region agent log
                try:
                    with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                        import json
                        positions_after = {c.id: list(c.position) for c in top_level_components}
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H2", "location": "placement_engine.py:464", "message": "After _place_hierarchical_with_anchors", "data": {"positions_after": positions_after}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                except: pass
                # #endregion
            else:
                # Complex: use force-directed with strong seed constraints
                # #region agent log
                try:
                    with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                        import json
                        positions_before = {c.id: list(c.position) for c in top_level_components}
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H3", "location": "placement_engine.py:467", "message": "Before _place_force_directed_with_seed", "data": {"topology": topology, "positions_before": positions_before, "manual_positions": {k: list(v) for k, v in manual_positions.items()}}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                except: pass
                # #endregion
                dataset = self._place_force_directed_with_seed(
                    dataset, 
                    top_level_components,
                    manual_positions,
                    diagram_json
                )
                # #region agent log
                try:
                    with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                        import json
                        positions_after = {c.id: list(c.position) for c in top_level_components}
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H3", "location": "placement_engine.py:472", "message": "After _place_force_directed_with_seed", "data": {"positions_after": positions_after}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                except: pass
                # #endregion
                
                # Check for collapse
                if self._detect_collapse(top_level_components, dataset):
                    # Fallback to hierarchical
                    dataset = self._place_hierarchical_with_anchors(
                        dataset,
                        top_level_components,
                        manual_positions,
                        diagram_json
                    )
            
            # Apply constraint solver to ensure spacing is honored at top level
            # BUT skip for linear sequences with all seed positions (preserve exact positions)
            all_top_level = top_level_containers + top_level_components
            top_level_connections = [
                conn for conn in dataset.connections
                if (conn.from_component_id in [e.id for e in all_top_level] and
                    conn.to_component_id in [e.id for e in all_top_level])
            ]
            
            # Check if this is a linear sequence with all seed positions
            # OR if all components have seeds (already preserved above, regardless of topology)
            is_linear_with_seeds = False
            if all_have_seeds:
                # All positions already preserved - skip constraint solver
                is_linear_with_seeds = True
            elif topology == "linear" and top_level_components:
                all_have_seeds_check = all(comp.id in manual_positions for comp in top_level_components)
                is_linear_with_seeds = all_have_seeds_check
            
            # Only run constraint solver if not a linear sequence with all seeds
            # #region agent log
            try:
                with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                    import json
                    positions_before_solver = {c.id: list(c.position) for c in top_level_components}
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H4", "location": "placement_engine.py:500", "message": "Before constraint solver", "data": {"is_linear_with_seeds": is_linear_with_seeds, "topology": topology, "will_run_solver": not is_linear_with_seeds, "positions_before": positions_before_solver, "manual_positions": {k: list(v) for k, v in manual_positions.items()}}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
            except: pass
            # #endregion
            if not is_linear_with_seeds:
                # Create bounding box for top-level constraint (entire diagram)
                top_level_bbox = BoundingBox(0, 0, dataset.width, dataset.height)
                
                # Solve spacing constraints for all top-level elements (containers + components)
                new_positions = constraint_solver.solve_spacing_constraints(
                    all_top_level,
                    top_level_connections,
                    top_level_bbox
                )
                
                # #region agent log
                try:
                    with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                        import json
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H4", "location": "placement_engine.py:509", "message": "Constraint solver returned new positions", "data": {"new_positions": {k: list(v) for k, v in new_positions.items()}, "component_ids": [c.id for c in top_level_components]}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                except: pass
                # #endregion
                
                # Apply new positions (but restore container positions from JSON)
                for elem in all_top_level:
                    if elem.id in new_positions:
                        # Containers keep JSON positions, components get new positions
                        if elem.id not in original_container_positions:
                            # #region agent log
                            try:
                                with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                                    import json
                                    old_pos = list(elem.position)
                                    new_pos = list(new_positions[elem.id])
                                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H4", "location": "placement_engine.py:516", "message": "Applying constraint solver position", "data": {"component_id": elem.id, "old_position": old_pos, "new_position": new_pos, "is_container": elem.id in original_container_positions}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                            except: pass
                            # #endregion
                            elem.position = new_positions[elem.id]
                        else:
                            # Restore original container position
                            elem.position = original_container_positions[elem.id]
        
        # Step 2: Bottom-up container processing
        container_levels = analyzer.get_container_levels()
        
        # After all containers are processed, re-apply spacing constraints at top level
        # with updated container sizes
        processed_containers = []
        
        for level_containers in container_levels:
            for container_id in level_containers:
                container = next((c for c in dataset.containers if c.id == container_id), None)
                if not container:
                    continue
                
                # Ensure original position is preserved (should already be set in Step 1)
                if container_id in original_container_positions:
                    container.position = original_container_positions[container_id]
                
                # Get child components
                child_ids = analyzer.get_container_children(container_id)
                child_components = [dataset.get_component(cid) for cid in child_ids]
                child_components = [c for c in child_components if c]
                
                if not child_components:
                    continue
                
                # Get connections within this container
                container_connections = [
                    conn for conn in dataset.connections
                    if (conn.container_id == container_id or
                        (conn.from_component_id in child_ids and 
                         conn.to_component_id in child_ids))
                ]
                
                # Group connected components
                connected_groups = analyzer.find_connected_groups(child_ids, container_connections)
                
                # Detail placement within container
                # For now, use constraint solver on all components
                # TODO: Could optimize by placing groups separately
                container_def = analyzer.get_container_definition(container_id)
                if container_def:
                    # Use container's initial bounds as constraint
                    # Container position is absolute (top-level), children are relative
                    # For constraint solving, we need absolute positions
                    # So we'll work with relative positions within container bounds
                    container_bbox = BoundingBox(
                        0, 0,  # Container bounds start at (0,0) for relative coordinates
                        container.width,
                        container.height
                    )
                else:
                    container_bbox = None
                
                # Solve spacing constraints
                new_positions = constraint_solver.solve_spacing_constraints(
                    child_components,
                    container_connections,
                    container_bbox
                )
                
                # Apply new positions
                for comp in child_components:
                    if comp.id in new_positions:
                        comp.position = new_positions[comp.id]
                
                # Place container-scoped labels
                # Import here to avoid circular dependency
                from diagram_generator.labeling.placer import LabelPlacer
                from diagram_generator.layout.engine import LayoutEngine
                
                # Get container-scoped labels that haven't been placed yet
                container_labels = [
                    label for label in dataset.labels 
                    if label.container_id == container_id and label.position is None
                ]
                
                if container_labels:
                    placer = LabelPlacer(dataset, self.config)
                    placer.place_labels(container_labels)
                
                # Compute container bounding box
                layout_engine = LayoutEngine(self.config)
                container_labels_list = [
                    label for label in dataset.labels 
                    if label.container_id == container_id
                ]
                
                container_bbox = layout_engine.calculate_container_bbox(
                    container,
                    child_components,
                    container_labels=container_labels_list,
                    dataset=dataset,
                    preserve_position=True  # Keep original position from JSON
                )
                
                # Update container dimensions only (keep original position like reference)
                # The reference SVG keeps containers at their original positions from JSON
                # Only update width and height to encompass children
                container.width = container_bbox.width
                container.height = container_bbox.height
                
                # Keep original position from JSON (already restored above)
                # Don't update container.position - it should stay as in JSON
                
                # Adjust child component positions to be relative to new container position
                # Children are in absolute coordinates, so we need to keep them absolute
                # The container transform will handle the positioning
                # Actually, we keep children in absolute coordinates - the container transform
                # is applied separately in SVG generation
                
                # Update structure
                if "rect" in container.structure:
                    container.structure["rect"]["width"] = container.width
                    container.structure["rect"]["height"] = container.height
                
                if "text" in container.structure and len(container.structure["text"]) > 0:
                    container.structure["text"][0]["x"] = container.width / 2
                
                processed_containers.append(container)
        
        # Step 3: Simple spacing enforcement for top-level containers
        # Preserve JSON positions as much as possible, only adjust to enforce 20px spacing
        # Sort containers by y-position and enforce spacing top-to-bottom
        # Only enforce vertical spacing for containers that are actually stacked (similar y-coordinates)
        top_level_containers_sorted = sorted(
            [c for c in top_level_containers if c.id in original_container_positions],
            key=lambda c: original_container_positions[c.id][1]  # Sort by original y
        )
        
        # #region agent log
        import json
        log_path = "/home/rohit/src/toyai-1/.cursor/debug.log"
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H9", "location": "placement_engine.py:604", "message": "simple spacing enforcement", "data": {"num_containers": len(top_level_containers_sorted), "container_ids": [c.id for c in top_level_containers_sorted]}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
        except: pass
        # #endregion
        
        # Restore all original positions first
        for container in top_level_containers_sorted:
            if container.id in original_container_positions:
                container.position = original_container_positions[container.id]
        
        # Enforce 20px spacing between containers (top-to-bottom)
        # Check spacing against ALL previous containers that overlap horizontally
        # This ensures containers like training (which overlaps with transformer-block-2) get proper spacing
        required_spacing = self.border_padding  # 20px
        for i in range(len(top_level_containers_sorted)):
            container = top_level_containers_sorted[i]
            
            # Check spacing with ALL previous containers that overlap horizontally
            max_required_y = container.position[1]  # Don't move up, only down
            for j in range(i):
                prev_container = top_level_containers_sorted[j]
                # Use UPDATED positions and heights (prev_container may have been moved and resized)
                prev_bottom = prev_container.position[1] + prev_container.height
                current_top = container.position[1]
                current_spacing = current_top - prev_bottom
                
                # Only enforce spacing if containers are actually stacked (not side-by-side)
                # Check if they overlap horizontally (actual overlap, not just close)
                prev_right = prev_container.position[0] + prev_container.width
                current_left = container.position[0]
                prev_left = prev_container.position[0]
                current_right = container.position[0] + container.width
                # Actual horizontal overlap: containers' x ranges intersect
                horizontal_overlap = (prev_left < current_right and current_left < prev_right)
                
                # #region agent log
                try:
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H9", "location": "placement_engine.py:640", "message": "checking spacing", "data": {"prev_id": prev_container.id, "current_id": container.id, "prev_bottom": prev_bottom, "current_top": current_top, "current_spacing": current_spacing, "required": required_spacing, "horizontal_overlap": horizontal_overlap}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                except: pass
                # #endregion
                
                # If spacing is less than required AND containers are stacked vertically, calculate required y
                if horizontal_overlap and current_spacing < required_spacing:
                    required_y = prev_bottom + required_spacing
                    max_required_y = max(max_required_y, required_y)
            
            # Apply adjustment if needed
            if max_required_y > container.position[1]:
                adjustment = max_required_y - container.position[1]
                container.position = (container.position[0], max_required_y)
                
                # #region agent log
                try:
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H9", "location": "placement_engine.py:665", "message": "spacing adjusted", "data": {"container_id": container.id, "old_y": container.position[1] - adjustment, "new_y": max_required_y, "adjustment": adjustment}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                except: pass
                # #endregion
        
        return dataset
    
    def _place_force_directed_with_seed(
        self,
        dataset: SVGDataset,
        elements: List[Component],
        seed_positions: Dict[str, Tuple[float, float]],
        diagram_json: Dict[str, Any]
    ) -> SVGDataset:
        """Place elements using force-directed with seed positions."""
        # Initialize positions from seed
        for elem in elements:
            if elem.id in seed_positions:
                elem.position = seed_positions[elem.id]
        
        # Run force-directed (simplified version for top-level)
        # Get connections between top-level elements
        top_level_ids = {elem.id for elem in elements}
        relevant_connections = [
            conn for conn in dataset.connections
            if conn.from_component_id in top_level_ids and 
               conn.to_component_id in top_level_ids
        ]
        
        # Simple force-directed iteration
        iterations = 50
        temperature = 1.0
        
        for iteration in range(iterations):
            forces = {elem.id: [0.0, 0.0] for elem in elements}
            current_temp = temperature * (1 - iteration / iterations)
            
            # Spring forces
            spring_constant = 0.1
            ideal_length = self.min_spacing * 3
            
            for conn in relevant_connections:
                from_elem = next((e for e in elements if e.id == conn.from_component_id), None)
                to_elem = next((e for e in elements if e.id == conn.to_component_id), None)
                
                if not from_elem or not to_elem:
                    continue
                
                dx = to_elem.bbox.center_x - from_elem.bbox.center_x
                dy = to_elem.bbox.center_y - from_elem.bbox.center_y
                distance = math.sqrt(dx*dx + dy*dy) or 1.0
                
                force_magnitude = spring_constant * (distance - ideal_length)
                fx = force_magnitude * dx / distance
                fy = force_magnitude * dy / distance
                
                forces[from_elem.id][0] += fx
                forces[from_elem.id][1] += fy
                forces[to_elem.id][0] -= fx
                forces[to_elem.id][1] -= fy
            
            # Seed position constraints (penalize moving too far from seed)
            seed_constraint_constant = 0.5  # Stronger constraint to keep near seeds
            for elem in elements:
                if elem.id in seed_positions:
                    seed_x, seed_y = seed_positions[elem.id]
                    dx = elem.position[0] - seed_x
                    dy = elem.position[1] - seed_y
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    # Only apply constraint if moved more than 10% of average component size
                    avg_size = (elem.width + elem.height) / 2
                    threshold = avg_size * 0.1
                    
                    if distance > threshold:
                        # Pull back toward seed position
                        force_magnitude = seed_constraint_constant * (distance - threshold)
                        fx = -force_magnitude * dx / distance if distance > 0 else 0
                        fy = -force_magnitude * dy / distance if distance > 0 else 0
                        
                        forces[elem.id][0] += fx
                        forces[elem.id][1] += fy
            
            # Repulsion
            repulsion_constant = 1000.0
            for i, elem1 in enumerate(elements):
                for elem2 in elements[i+1:]:
                    dx = elem2.bbox.center_x - elem1.bbox.center_x
                    dy = elem2.bbox.center_y - elem1.bbox.center_y
                    distance_sq = dx*dx + dy*dy
                    
                    if distance_sq < 1.0:
                        distance_sq = 1.0
                    
                    force_magnitude = repulsion_constant / distance_sq
                    fx = force_magnitude * dx / math.sqrt(distance_sq)
                    fy = force_magnitude * dy / math.sqrt(distance_sq)
                    
                    forces[elem1.id][0] -= fx
                    forces[elem1.id][1] -= fy
                    forces[elem2.id][0] += fx
                    forces[elem2.id][1] += fy
            
            # Apply forces
            for elem in elements:
                fx, fy = forces[elem.id]
                force_mag = math.sqrt(fx*fx + fy*fy)
                if force_mag > 10.0:
                    fx = fx * 10.0 / force_mag
                    fy = fy * 10.0 / force_mag
                
                new_x = elem.position[0] + fx * current_temp
                new_y = elem.position[1] + fy * current_temp
                
                # Keep within bounds
                new_x = max(elem.width/2, min(dataset.width - elem.width/2, new_x))
                new_y = max(elem.height/2, min(dataset.height - elem.height/2, new_y))
                
                elem.position = (new_x, new_y)
        
        return dataset
    
    def _detect_topology(
        self,
        components: List[Component],
        connections: List[Connection]
    ) -> str:
        """
        Detect connection topology to choose optimal placement algorithm.
        
        Returns:
            "linear": All components have max in-degree <= 1 and max out-degree <= 1 (chain)
            "branching": Max out-degree > 1 (one-to-many)
            "converging": Max in-degree > 1 (many-to-one)
            "complex": Mixed or high-degree nodes
        """
        if not connections:
            return "linear"  # No connections = linear by default
        
        # Build degree counts
        in_degree = {comp.id: 0 for comp in components}
        out_degree = {comp.id: 0 for comp in components}
        
        for conn in connections:
            from_id = conn.from_component_id
            to_id = conn.to_component_id
            if from_id in out_degree:
                out_degree[from_id] += 1
            if to_id in in_degree:
                in_degree[to_id] += 1
        
        max_in = max(in_degree.values()) if in_degree else 0
        max_out = max(out_degree.values()) if out_degree else 0
        
        # Classify topology
        if max_in <= 1 and max_out <= 1:
            return "linear"
        elif max_out > 1 and max_in <= 1:
            return "branching"
        elif max_in > 1 and max_out <= 1:
            return "converging"
        else:
            return "complex"
    
    def _place_linear_with_seeds(
        self,
        dataset: SVGDataset,
        components: List[Component],
        seed_positions: Dict[str, Tuple[float, float]],
        connections: List[Connection]
    ) -> SVGDataset:
        """
        Place components in linear sequence preserving seed order and relative spacing.
        
        Algorithm:
        1. Determine component order from connections (or seed x-coordinates)
        2. Preserve relative spacing between components
        3. Only adjust positions to satisfy spacing constraints if needed
        """
        # #region agent log
        try:
            with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                import json
                positions_before = {c.id: list(c.position) for c in components}
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H8", "location": "placement_engine.py:937", "message": "_place_linear_with_seeds entry", "data": {"component_ids": [c.id for c in components], "seed_positions": {k: list(v) for k, v in seed_positions.items()}, "positions_before": positions_before}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
        except: pass
        # #endregion
        if not components:
            return dataset
        
        # If no connections, use x-coordinate order from seeds
        if not connections:
            sorted_comps = sorted(components, key=lambda c: seed_positions.get(c.id, (0, 0))[0])
        else:
            # Build order from connections
            # Find start node (in-degree = 0)
            in_degree = {comp.id: 0 for comp in components}
            for conn in connections:
                to_id = conn.to_component_id
                if to_id in in_degree:
                    in_degree[to_id] += 1
            
            # Find starting component(s)
            start_ids = [comp.id for comp in components if in_degree[comp.id] == 0]
            if not start_ids:
                # No clear start, use x-coordinate order
                sorted_comps = sorted(components, key=lambda c: seed_positions.get(c.id, (0, 0))[0])
            else:
                # Traverse from start
                sorted_comps = []
                visited = set()
                queue = start_ids[:]
                
                # Build adjacency list
                adj = {comp.id: [] for comp in components}
                for conn in connections:
                    from_id = conn.from_component_id
                    to_id = conn.to_component_id
                    if from_id in adj:
                        adj[from_id].append(to_id)
                
                # BFS traversal
                while queue:
                    comp_id = queue.pop(0)
                    if comp_id in visited:
                        continue
                    visited.add(comp_id)
                    comp = next((c for c in components if c.id == comp_id), None)
                    if comp:
                        sorted_comps.append(comp)
                    # Add neighbors
                    for neighbor_id in adj.get(comp_id, []):
                        if neighbor_id not in visited:
                            queue.append(neighbor_id)
                
                # Add any unvisited components (disconnected)
                for comp in components:
                    if comp.id not in visited:
                        sorted_comps.append(comp)
        
        # Calculate average spacing from seeds
        if len(sorted_comps) > 1:
            spacings = []
            for i in range(len(sorted_comps) - 1):
                comp1 = sorted_comps[i]
                comp2 = sorted_comps[i + 1]
                if comp1.id in seed_positions and comp2.id in seed_positions:
                    pos1 = seed_positions[comp1.id]
                    pos2 = seed_positions[comp2.id]
                    # Calculate spacing (center-to-center distance minus half widths)
                    spacing = abs(pos2[0] - pos1[0]) - (comp1.width / 2 + comp2.width / 2)
                    if spacing > 0:
                        spacings.append(spacing)
            
            avg_spacing = sum(spacings) / len(spacings) if spacings else self.min_spacing
            # Ensure minimum spacing
            target_spacing = max(avg_spacing, self.min_spacing)
        else:
            target_spacing = self.min_spacing
        
        # For linear sequences, preserve exact seed positions if all components have seeds
        # This ensures manual positioning is respected
        all_have_seeds = all(comp.id in seed_positions for comp in sorted_comps)
        
        if all_have_seeds:
            # Use exact seed positions
            for comp in sorted_comps:
                comp.position = seed_positions[comp.id]
        else:
            # Some components missing seeds - use calculated placement
            # Use first component's seed position as anchor
            if sorted_comps[0].id in seed_positions:
                current_x, current_y = seed_positions[sorted_comps[0].id]
            else:
                current_x = self.border_padding + sorted_comps[0].width / 2
                current_y = dataset.height / 2
            
            sorted_comps[0].position = (current_x, current_y)
            
            # Place subsequent components
            for i in range(1, len(sorted_comps)):
                prev_comp = sorted_comps[i - 1]
                curr_comp = sorted_comps[i]
                
                # Calculate next position
                next_x = prev_comp.position[0] + prev_comp.width / 2 + target_spacing + curr_comp.width / 2
                
                # Try to preserve seed y-coordinate if available
                if curr_comp.id in seed_positions:
                    seed_y = seed_positions[curr_comp.id][1]
                    # Use seed y if it's reasonable, otherwise use previous y
                    if abs(seed_y - prev_comp.position[1]) < dataset.height / 2:
                        next_y = seed_y
                    else:
                        next_y = prev_comp.position[1]
                else:
                    next_y = prev_comp.position[1]
                
                curr_comp.position = (next_x, next_y)
        
        return dataset
    
    def _detect_collapse(self, elements: List[Component], dataset: SVGDataset) -> bool:
        """Detect if force-directed collapsed (overlapping elements)."""
        if len(elements) < 2:
            return False
        
        # Check for significant overlaps
        overlap_count = 0
        for i, elem1 in enumerate(elements):
            for elem2 in elements[i+1:]:
                if elem1.bbox.intersects(elem2.bbox):
                    overlap_count += 1
        
        # If more than 10% of pairs overlap, consider it collapsed
        total_pairs = len(elements) * (len(elements) - 1) / 2
        return overlap_count > total_pairs * 0.1
    
    def _place_hierarchical_with_anchors(
        self,
        dataset: SVGDataset,
        elements: List[Component],
        anchor_positions: Dict[str, Tuple[float, float]],
        diagram_json: Dict[str, Any]
    ) -> SVGDataset:
        """Place elements using hierarchical algorithm with anchor positions."""
        # Use hierarchical placement but respect anchor positions as constraints
        # For now, just use the anchor positions
        # #region agent log
        try:
            with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                import json
                positions_before = {e.id: list(e.position) for e in elements}
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H5", "location": "placement_engine.py:1028", "message": "_place_hierarchical_with_anchors entry", "data": {"element_ids": [e.id for e in elements], "anchor_positions": {k: list(v) for k, v in anchor_positions.items()}, "positions_before": positions_before}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
        except: pass
        # #endregion
        for elem in elements:
            if elem.id in anchor_positions:
                old_pos = elem.position
                elem.position = anchor_positions[elem.id]
                # #region agent log
                try:
                    with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                        import json
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H5", "location": "placement_engine.py:1030", "message": "Setting anchor position", "data": {"element_id": elem.id, "old_position": list(old_pos), "new_position": list(elem.position), "anchor_position": list(anchor_positions[elem.id])}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                except: pass
                # #endregion
        
        # Could enhance this to use hierarchical layout but keep anchors fixed
        return dataset

