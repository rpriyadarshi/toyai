#!/usr/bin/env python3
"""
Label Placer - Priority-based label placement system.

This module provides automatic label placement with priorities and constraints,
using the obstruction grid to find optimal positions.
"""

from typing import Dict, Any, List, Optional, Tuple

from diagram_generator.core.diagram import SVGDataset, Label, BoundingBox
from diagram_generator.routing.obstructions import ObstructionGrid
from diagram_generator.labeling.bbox import calculate_label_bbox, calculate_text_bbox


class LabelPlacer:
    """Priority-based label placement system."""
    
    def __init__(self, diagram: SVGDataset, config: Dict[str, Any]):
        """
        Initialize label placer.
        
        Args:
            diagram: SVG diagram dataset
            config: Diagram configuration
        """
        self.diagram = diagram
        self.config = config
        self.priorities = config.get("label_priorities", {})
        self.constraints = config.get("label_constraints", {})
        self.grid = ObstructionGrid(diagram.width, diagram.height)
        
        # Mark component bounding boxes as obstructions (use absolute bboxes)
        # This enables the obstruction grid to find open space for label placement
        for component in diagram.components:
            abs_bbox = component.get_absolute_bbox(diagram)
            self.grid.mark_obstructed(abs_bbox)
        
        # Mark container bounding boxes as obstructions (use absolute bboxes)
        # CRITICAL: Containers must be marked to prevent title overlap with containers
        for container in diagram.containers:
            abs_bbox = container.get_absolute_bbox(diagram)
            self.grid.mark_obstructed(abs_bbox)
        
        # #region agent log
        try:
            with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                f.write(__import__("json").dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "label_placer.py:43", "message": "Marked components and containers as obstructions", "data": {"component_count": len(diagram.components), "container_count": len(diagram.containers), "containers_marked": True}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
        except: pass
        # #endregion
    
    def mark_connection_paths_as_obstructions(self, connection_waypoints: Dict):
        """
        Mark connection paths as obstructions in the grid.
        
        Args:
            connection_waypoints: Dictionary mapping connection IDs to (connection, waypoints) tuples
        """
        from diagram_generator.core.diagram import BoundingBox
        
        # #region agent log
        try:
            with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                f.write(__import__("json").dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "C", "location": "label_placer.py:mark_connection_paths", "message": "Marking connection paths as obstructions", "data": {"connection_count": len(connection_waypoints)}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
        except: pass
        # #endregion
        
        # Default stroke width for connections (from config or default)
        spacing_config = self.config.get("spacing", {})
        stroke_widths = spacing_config.get("stroke_width", {})
        standard_stroke = stroke_widths.get("standard", 2.0)
        
        for conn_id, value in connection_waypoints.items():
            # Handle both formats: (connection, waypoints) tuple or just waypoints
            if value is None:
                continue
            if isinstance(value, tuple) and len(value) == 2:
                connection, waypoints = value
            else:
                # Fallback: assume value is waypoints directly
                waypoints = value
                connection = None
            
            if not waypoints or len(waypoints) < 2:
                continue
            
            # Get stroke width for this connection
            if connection:
                stroke_width = getattr(connection, 'stroke_width', standard_stroke)
            else:
                stroke_width = standard_stroke
            # Add padding around stroke (half stroke width on each side)
            padding = stroke_width / 2.0 + 2.0  # 2px extra for visual spacing
            
            # Mark each segment as obstruction
            for i in range(len(waypoints) - 1):
                start = waypoints[i]
                end = waypoints[i + 1]
                
                # Calculate bounding box for this segment
                min_x = min(start[0], end[0]) - padding
                max_x = max(start[0], end[0]) + padding
                min_y = min(start[1], end[1]) - padding
                max_y = max(start[1], end[1]) + padding
                
                # Create bounding box for this segment
                segment_bbox = BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y)
                self.grid.mark_obstructed(segment_bbox)
        
        # #region agent log
        try:
            with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                f.write(__import__("json").dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "C", "location": "label_placer.py:mark_connection_paths", "message": "Finished marking connection paths", "data": {"connections_marked": len(connection_waypoints)}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
        except: pass
        # #endregion
    
    def place_labels(self, labels: List[Label]):
        """
        Place labels in priority order.
        Container-scoped labels are placed first, then global labels.
        
        Args:
            labels: List of labels to place
        """
        # Separate container-scoped and global labels
        container_labels = [l for l in labels if l.container_id]
        global_labels = [l for l in labels if not l.container_id]
        
        # Sort by priority (lower = higher priority)
        sorted_container_labels = sorted(container_labels, key=lambda l: l.priority)
        sorted_global_labels = sorted(global_labels, key=lambda l: l.priority)
        
        # Place container labels first (they affect container bboxes)
        for label in sorted_container_labels:
            position = self.find_position(label, label.container_id)
            if position:
                label.position = position
                # Calculate and set bounding box
                label.bbox = self._calculate_label_bbox(label)
                # Mark label as obstruction (within container scope if applicable)
                if label.bbox:
                    self.grid.mark_obstructed(label.bbox)
        
        # Then place global labels
        for label in sorted_global_labels:
            position = self.find_position(label)
            if position:
                label.position = position
                # Calculate and set bounding box
                label.bbox = self._calculate_label_bbox(label)
                # Mark label as obstruction
                if label.bbox:
                    self.grid.mark_obstructed(label.bbox)
    
    def find_position(self, label: Label, container_id: Optional[str] = None) -> Optional[Tuple[float, float]]:
        """
        Find position for label based on type and constraints.
        
        Args:
            label: Label to place
            container_id: Optional container ID for container-scoped labels
            
        Returns:
            (x, y) position if found, None otherwise
        """
        if label.type == "axis":
            return self.find_axis_position(label, container_id)
        elif label.type == "title":
            return self.find_title_position(label, container_id)
        elif label.type == "equation":
            return self.find_equation_position(label, container_id)
        elif label.type == "element":
            return self.find_element_position(label, container_id)
        elif label.type == "note":
            return self.find_note_position(label, container_id)
        else:
            return self.find_element_position(label, container_id)  # Default
    
    def find_axis_position(self, label: Label, container_id: Optional[str] = None) -> Optional[Tuple[float, float]]:
        """
        Find position for axis label (must stay near axis).
        
        Args:
            label: Axis label
            container_id: Optional container ID (axis labels are typically global)
            
        Returns:
            (x, y) position
        """
        if not label.target_position:
            return None
        
        constraint = self.constraints.get("axis", {})
        max_distance = constraint.get("max_distance", 30)
        
        # Calculate label bbox at target position
        bbox = self._calculate_label_bbox_at(label, label.target_position[0], label.target_position[1])
        if not bbox:
            return None
        
        # Try to find position near target
        position = self.grid.find_nearest_open_space(
            label.target_position,
            bbox,
            max_distance=max_distance
        )
        
        return position or label.target_position
    
    def find_title_position(self, label: Label, container_id: Optional[str] = None) -> Optional[Tuple[float, float]]:
        """
        Find position for title using obstruction grid-based algorithm.
        
        Algorithm for global titles:
        1. Mark all components and containers as obstructions in grid
        2. Calculate initial position based on border_padding and font metrics
        3. Check if initial position is obstructed using grid
        4. If obstructed:
           a. Find topmost container
           b. Calculate required position (title bottom above container with spacing)
           c. Verify required position respects border_padding
           d. If still obstructed, use find_nearest_open_space to find alternative
           e. Prioritize avoiding overlap over strict border_padding when needed
        5. Return position that avoids obstructions and respects spacing
        
        If container_id is provided, title is centered within that container (simpler algorithm).
        
        Args:
            label: Title label
            container_id: Optional container ID for container-scoped titles
            
        Returns:
            (x, y) position
        """
        constraint = self.constraints.get("title", {})
        
        if container_id:
            # Container-scoped title: center within container
            container = None
            for c in self.diagram.containers:
                if c.id == container_id:
                    container = c
                    break
            if container:
                x = container.bbox.center_x
                y = container.position[1] + constraint.get("y_offset", {}).get("small", 20)
                return (x, y)
        
        # Global title: top center of diagram - USE OBSTRUCTION GRID
        diagram_size = self._detect_diagram_size()
        y_offset = constraint.get("y_offset", {}).get(diagram_size, 30)
        
        # Get border padding and font size
        spacing = self.config.get("spacing", {})
        border_padding = spacing.get("border_padding", 20)
        typography = self.config.get("typography", {})
        font_size = typography.get("sizes", {}).get("title", {}).get(diagram_size, 18)
        
        # Text metrics
        text_ascent = font_size * 0.8
        text_descent = font_size * 0.2
        
        # Calculate initial position based on border_padding
        min_y = border_padding + text_ascent
        desired_y = y_offset + text_ascent
        initial_y = max(min_y, desired_y)
        
        # Title is always centered horizontally
        x = self.diagram.width / 2
        
        # Calculate title bounding box at initial position
        title_bbox = self._calculate_label_bbox_at(label, x, initial_y)
        if not title_bbox:
            return (x, initial_y)
        
        # #region agent log
        try:
            with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                f.write(__import__("json").dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "label_placer.py:172", "message": "find_title_position using obstruction grid", "data": {"diagram_size": diagram_size, "y_offset": y_offset, "using_obstruction_grid": True, "initial_y": initial_y, "title_bbox": {"left": title_bbox.left, "top": title_bbox.top, "width": title_bbox.width, "height": title_bbox.height}}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
        except: pass
        # #endregion
        
        # Check if initial position is obstructed
        is_obstructed = self.grid.is_obstructed(title_bbox)
        
        if is_obstructed:
            # #region agent log
            try:
                with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                    f.write(__import__("json").dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "C", "location": "label_placer.py:188", "message": "Title position obstructed, finding alternative", "data": {"initial_y": initial_y, "title_bbox": {"left": title_bbox.left, "top": title_bbox.top, "width": title_bbox.width, "height": title_bbox.height}}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
            except: pass
            # #endregion
            
            # Find the topmost container to calculate required spacing
            if self.diagram.containers:
                topmost_container = min(self.diagram.containers, key=lambda c: c.position[1])
                container_bbox = topmost_container.get_absolute_bbox(self.diagram)
                container_top = container_bbox.top
                min_spacing = 10  # Minimum spacing between title bottom and container top
                
                # Calculate required y position: title bottom should be above container top with spacing
                required_title_bottom = container_top - min_spacing
                required_y = required_title_bottom - text_descent
                required_title_top = required_y - text_ascent
                
                # #region agent log
                try:
                    with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                        f.write(__import__("json").dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "C", "location": "label_placer.py:217", "message": "Calculating required position", "data": {"container_top": container_top, "required_y": required_y, "required_title_top": required_title_top, "border_padding": border_padding}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                except: pass
                # #endregion
                
                # Check if required position respects border_padding
                if required_title_top >= border_padding:
                    # Good position - check if it's obstructed
                    candidate_bbox = self._calculate_label_bbox_at(label, x, required_y)
                    if candidate_bbox and not self.grid.is_obstructed(candidate_bbox):
                        y = required_y
                    else:
                        # Still obstructed, try finding open space
                        target_pos = (x, required_y - 10)
                        max_distance = 50
                        alternative_pos = self.grid.find_nearest_open_space(
                            target_pos,
                            title_bbox,
                            max_distance=max_distance,
                            angle_step=15.0
                        )
                        if alternative_pos:
                            alt_y = alternative_pos[1]
                            alt_title_top = alt_y - text_ascent
                            if alt_title_top >= border_padding:
                                y = alt_y
                            else:
                                y = required_y  # Use required position even if slightly violates border_padding
                        else:
                            y = required_y  # Use required position to avoid overlap
                else:
                    # Required position would violate border_padding, but we must avoid overlap
                    # Use required position (slightly violates border_padding but creates spacing)
                    # This is acceptable when container is at border_padding
                    y = required_y
                    # #region agent log
                    try:
                        with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                            f.write(__import__("json").dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "C", "location": "label_placer.py:250", "message": "Using required position despite border_padding violation", "data": {"required_y": required_y, "required_title_top": required_title_top, "border_padding": border_padding}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                    except: pass
                    # #endregion
            else:
                # No containers, just use minimum
                y = min_y
        else:
            y = initial_y
        
        # #region agent log
        try:
            final_bbox = self._calculate_label_bbox_at(label, x, y)
            final_is_obstructed = final_bbox and self.grid.is_obstructed(final_bbox) if final_bbox else False
            with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                f.write(__import__("json").dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "C", "location": "label_placer.py:230", "message": "Final title position", "data": {"x": x, "y": y, "title_bbox": {"left": final_bbox.left, "top": final_bbox.top, "width": final_bbox.width, "height": final_bbox.height} if final_bbox else None, "is_obstructed": final_is_obstructed, "checked_obstructions": True}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
        except: pass
        # #endregion
        
        return (x, y)
    
    def find_equation_position(self, label: Label, container_id: Optional[str] = None) -> Optional[Tuple[float, float]]:
        """
        Find position for equation (below title).
        If container_id is provided, equation is centered within that container.
        
        Args:
            label: Equation label
            container_id: Optional container ID for container-scoped equations
            
        Returns:
            (x, y) position
        """
        constraint = self.constraints.get("equation", {})
        
        if container_id:
            # Container-scoped equation: center within container, below container title
            container = None
            for c in self.diagram.containers:
                if c.id == container_id:
                    container = c
                    break
            if container:
                x = container.bbox.center_x
                # Find container title to place below it
                container_title_y = container.position[1] + constraint.get("y_offset", {}).get("small", 20)
                y = container_title_y + 20  # Below title
                return (x, y)
        
        # Global equation: below global title
        diagram_size = self._detect_diagram_size()
        y_offset = constraint.get("y_offset", {}).get(diagram_size, 50)
        
        # Equation is always centered
        x = self.diagram.width / 2
        y = y_offset
        
        return (x, y)
    
    def find_element_position(self, label: Label, container_id: Optional[str] = None) -> Optional[Tuple[float, float]]:
        """
        Find position for element label (near component).
        If container_id is provided, search is constrained within container bounds.
        
        Args:
            label: Element label
            container_id: Optional container ID for container-scoped labels
            
        Returns:
            (x, y) position
        """
        # Get target position
        if label.target_component_id:
            component = self.diagram.get_component(label.target_component_id)
            if component:
                # Use absolute bbox for target position
                abs_bbox = component.get_absolute_bbox(self.diagram)
                target_pos = (abs_bbox.center_x, abs_bbox.center_y)
            else:
                # Component not found - can't place label
                print(f"Warning: Component '{label.target_component_id}' not found for label '{label.text}'")
                return None
        else:
            if not label.target_position:
                print(f"Warning: Element label '{label.text}' has no target")
                return None
            target_pos = label.target_position
        
        constraint = self.constraints.get("element", {})
        max_distance = constraint.get("max_distance", 200)
        
        # Calculate label bbox
        bbox = self._calculate_label_bbox_at(label, target_pos[0], target_pos[1])
        if not bbox:
            return None
        
        # If container-scoped, limit search to container bounds
        if container_id:
            container = None
            for c in self.diagram.containers:
                if c.id == container_id:
                    container = c
                    break
            if container:
                # Constrain max_distance to container size
                container_width = container.bbox.width
                container_height = container.bbox.height
                max_distance = min(max_distance, max(container_width, container_height) / 2)
        
        # Find nearest open space
        position = self.grid.find_nearest_open_space(
            target_pos,
            bbox,
            max_distance=max_distance
        )
        
        return position
    
    def find_note_position(self, label: Label, container_id: Optional[str] = None) -> Optional[Tuple[float, float]]:
        """
        Find position for note (bottom).
        If container_id is provided, note is at bottom of container.
        
        Args:
            label: Note label
            container_id: Optional container ID for container-scoped notes
            
        Returns:
            (x, y) position
        """
        constraint = self.constraints.get("note", {})
        y_offset = constraint.get("y_offset", 30)
        
        if container_id:
            # Container-scoped note: bottom of container
            container = None
            for c in self.diagram.containers:
                if c.id == container_id:
                    container = c
                    break
            if container:
                x = container.bbox.center_x
                y = container.bbox.bottom - y_offset
                return (x, y)
        
        # Global note: bottom of diagram
        x = self.diagram.width / 2
        y = self.diagram.height - y_offset
        
        return (x, y)
    
    def _calculate_label_bbox(self, label: Label) -> Optional[BoundingBox]:
        """Calculate bounding box for label at its current position."""
        if label.position:
            return self._calculate_label_bbox_at(label, label.position[0], label.position[1])
        return None
    
    def _calculate_label_bbox_at(self, label: Label, x: float, y: float) -> Optional[BoundingBox]:
        """Calculate bounding box for label at specific position."""
        typography = self.config.get("typography", {})
        sizes = typography.get("sizes", {})
        font_family = typography.get("font_family", "Arial, sans-serif")
        weights = typography.get("weights", {})
        
        if label.type == "title":
            diagram_size = self._detect_diagram_size()
            font_size = sizes.get("title", {}).get(diagram_size, 16)
        elif label.type == "equation":
            font_size = sizes.get("equation", 12)
        elif label.type == "axis":
            font_size = sizes.get("axis", 12)
        elif label.type == "note":
            font_size = sizes.get("note", 11)
        else:
            font_size = sizes.get("label", 11)
        
        font_weight = weights.get(label.type, "normal")
        line_count = label.text.count("\n") + 1
        
        return calculate_text_bbox(
            label.text, x, y, font_size, font_family, font_weight,
            text_anchor="middle", line_count=line_count
        )
    
    def _detect_diagram_size(self) -> str:
        """Detect diagram size category."""
        if self.diagram.width < 400:
            return "small"
        elif self.diagram.width < 700:
            return "medium"
        else:
            return "large"

