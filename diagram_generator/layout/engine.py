#!/usr/bin/env python3
"""
Layout Engine - Algorithmic layout calculation for SVG diagrams.

This module provides automatic layout calculation for components and containers,
ensuring consistent spacing and proper bounding box calculations.
"""

from typing import Dict, Any, List, Tuple, Optional

from diagram_generator.core.diagram import SVGDataset, Component, BoundingBox, Label
from diagram_generator.labeling.bbox import estimate_text_width, estimate_text_height


class LayoutEngine:
    """Calculate positions and sizes for diagram elements algorithmically."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize layout engine.
        
        Args:
            config: Diagram configuration with spacing rules
        """
        self.config = config
        spacing = config.get("spacing", {})
        self.horizontal_spacing = spacing.get("min_box_spacing", 50)
        self.vertical_spacing = spacing.get("min_box_spacing", 50)
        self.border_padding = spacing.get("border_padding", 20)
        self.box_padding = spacing.get("box_padding", {}).get("medium", 15)
    
    def calculate_component_bbox_with_text(self, component: Component, dataset: SVGDataset) -> BoundingBox:
        """
        Calculate complete bounding box for a component including all text.
        
        Text that overflows the component rect is included.
        Text that is within the component rect does not expand the bbox.
        
        Args:
            component: Component to calculate bbox for
            dataset: SVGDataset for resolving absolute positions
            
        Returns:
            Bounding box encompassing component and all its text (including overflow)
        """
        # Start with component's base bounding box (absolute coordinates)
        abs_bbox = component.get_absolute_bbox(dataset)
        min_x = abs_bbox.left
        min_y = abs_bbox.top
        max_x = abs_bbox.right
        max_y = abs_bbox.bottom
        
        if "text" not in component.structure:
            return abs_bbox
        
        # Include text elements - only expand bbox if text overflows component bounds
        for text_elem in component.structure["text"]:
            text_content = text_elem.get("content", "")
            if not text_content:
                continue
                
            text_x = text_elem.get("x", component.width / 2)
            text_y = text_elem.get("y", component.height / 2)
            text_class = text_elem.get("class", "box-label")
            text_anchor = text_elem.get("text-anchor", "middle")
            
            # Determine font size based on class
            if text_class == "box-label":
                font_size = 14
            elif text_class == "box-text":
                font_size = 11
            else:
                font_size = 12
            
            text_width = estimate_text_width(text_content, font_size)
            text_height = estimate_text_height(font_size, 1)
            
            # Text position is relative to component, convert to absolute
            abs_pos = component.get_absolute_position(dataset)
            abs_text_x = abs_pos[0] + text_x
            abs_text_y = abs_pos[1] + text_y
            
            # Calculate text bounding box
            if text_anchor == "middle":
                text_left = abs_text_x - text_width / 2
                text_right = abs_text_x + text_width / 2
            elif text_anchor == "start":
                text_left = abs_text_x
                text_right = abs_text_x + text_width
            else:  # "end"
                text_left = abs_text_x - text_width
                text_right = abs_text_x
            
            # Text vertical bounds (y is baseline)
            text_top = abs_text_y - font_size * 0.8  # Baseline to top
            text_bottom = abs_text_y + font_size * 0.2  # Baseline to bottom
            
            # Expand bbox to include text (even if it overflows component)
            min_x = min(min_x, text_left)
            min_y = min(min_y, text_top)
            max_x = max(max_x, text_right)
            max_y = max(max_y, text_bottom)
        
        return BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y)
    
    def calculate_container_bbox(
        self, 
        container: Component, 
        child_components: List[Component],
        container_labels: Optional[List[Label]] = None,
        dataset: Optional[SVGDataset] = None,
        preserve_position: bool = True
    ) -> BoundingBox:
        """
        Calculate container bounding box to encompass all children with padding.
        Includes child components, their text, labels associated with container,
        and internal connections.
        
        Args:
            container: Container component
            child_components: List of child components
            container_labels: Optional list of labels belonging to this container
            dataset: Optional dataset for finding internal connections
            preserve_position: If True, keep container's original position and only update size
            
        Returns:
            Bounding box for container
        """
        if not child_components and not container_labels:
            return container.bbox
        
        # Calculate bounding box of all children
        # Use component rect bounds (not text-expanded bounds) for child components
        # Text inside component rects is part of the component and doesn't expand the bbox
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        # #region agent log
        import json
        log_path = "/home/rohit/src/toyai-1/.cursor/debug.log"
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1", "location": "layout_engine.py:133", "message": "calculate_container_bbox entry", "data": {"container_id": container.id, "num_children": len(child_components), "border_padding": self.border_padding}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
        except: pass
        # #endregion
        
        for child in child_components:
            # Get box bounds from the actual rect structure, not just child.width/height
            # This ensures we use the actual rendered dimensions
            # #region agent log
            try:
                with open(log_path, "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H6", "location": "layout_engine.py:150", "message": "child component check", "data": {"child_id": child.id, "child_width": child.width, "child_height": child.height, "has_rect": "rect" in child.structure, "rect_width": child.structure.get("rect", {}).get("width") if "rect" in child.structure else None}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
            except: pass
            # #endregion
            
            # Use component.width/height as the authoritative source - this matches what gets rendered
            # The structure rect width might be stale or incorrect, so we always use component dimensions
            box_width = child.width
            box_height = child.height
            
            # Get box bounds in absolute coordinates (x1, y1, x2, y2)
            # Children have relative positions, so we need absolute position
            abs_pos = child.get_absolute_position(dataset)
            box_x1 = abs_pos[0]
            box_y1 = abs_pos[1]
            box_x2 = abs_pos[0] + box_width
            box_y2 = abs_pos[1] + box_height
            
            # #region agent log
            try:
                with open(log_path, "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1", "location": "layout_engine.py:152", "message": "child box bounds", "data": {"child_id": child.id, "position": child.position, "child_width": child.width, "child_height": child.height, "box_width": box_width, "box_height": box_height, "box_x1": box_x1, "box_y1": box_y1, "box_x2": box_x2, "box_y2": box_y2, "max_y_before": max_y}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
            except: pass
            # #endregion
            
            # Update bounds: check ALL four coordinates to handle edge cases
            min_x = min(min_x, box_x1, box_x2)
            max_x = max(max_x, box_x1, box_x2)
            min_y = min(min_y, box_y1, box_y2)
            max_y = max(max_y, box_y1, box_y2)
            
            # #region agent log
            try:
                with open(log_path, "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H5", "location": "layout_engine.py:178", "message": "after bounds update", "data": {"child_id": child.id, "box_x2": box_x2, "max_x_after": max_x, "container_id": container.id}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
            except: pass
            # #endregion
            
            # #region agent log
            try:
                with open(log_path, "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1", "location": "layout_engine.py:162", "message": "after box bounds update", "data": {"child_id": child.id, "max_y_after": max_y}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
            except: pass
            # #endregion
            
            # Include ALL text in the component (user requirement: "all text")
            if "text" in child.structure:
                for text_elem in child.structure["text"]:
                    text_content = text_elem.get("content", "")
                    if not text_content:
                        continue
                    
                    text_x = text_elem.get("x", child.width / 2)
                    text_y = text_elem.get("y", child.height / 2)
                    text_class = text_elem.get("class", "box-label")
                    text_anchor = text_elem.get("text-anchor", "middle")
                    
                    # Determine font size
                    if text_class == "box-label":
                        font_size = 14
                    elif text_class == "box-text":
                        font_size = 11
                    else:
                        font_size = 12
                    
                    text_width = estimate_text_width(text_content, font_size)
                    
                    # Text position (absolute) - use absolute position API
                    child_abs_pos = child.get_absolute_position(dataset)
                    abs_text_x = child_abs_pos[0] + text_x
                    abs_text_y = child_abs_pos[1] + text_y
                    
                    # Calculate text bounds (x1, y1, x2, y2)
                    if text_anchor == "middle":
                        text_x1 = abs_text_x - text_width / 2
                        text_x2 = abs_text_x + text_width / 2
                    elif text_anchor == "start":
                        text_x1 = abs_text_x
                        text_x2 = abs_text_x + text_width
                    else:  # "end"
                        text_x1 = abs_text_x - text_width
                        text_x2 = abs_text_x
                    
                    text_y1 = abs_text_y - font_size * 0.8
                    text_y2 = abs_text_y + font_size * 0.2
                    
                    # #region agent log
                    try:
                        with open(log_path, "a") as f:
                            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H2", "location": "layout_engine.py:201", "message": "text bounds", "data": {"child_id": child.id, "text_content": text_content[:20], "abs_text_y": abs_text_y, "font_size": font_size, "text_y1": text_y1, "text_y2": text_y2, "box_y2": box_y2, "max_y_before": max_y}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                    except: pass
                    # #endregion
                    
                    # Update bounds: check ALL four coordinates to handle edge cases
                    min_x = min(min_x, text_x1, text_x2)
                    max_x = max(max_x, text_x1, text_x2)
                    min_y = min(min_y, text_y1, text_y2)
                    max_y = max(max_y, text_y1, text_y2)
                    
                    # #region agent log
                    try:
                        with open(log_path, "a") as f:
                            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H2", "location": "layout_engine.py:208", "message": "after text bounds update", "data": {"child_id": child.id, "max_y_after": max_y}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                    except: pass
                    # #endregion
        
        # Include labels associated with this container
        if container_labels:
            for label in container_labels:
                if label.bbox:
                    # Get label bounds (x1, y1, x2, y2)
                    label_x1 = label.bbox.left
                    label_y1 = label.bbox.top
                    label_x2 = label.bbox.right
                    label_y2 = label.bbox.bottom
                    
                    # Update bounds: check ALL four coordinates to handle edge cases
                    min_x = min(min_x, label_x1, label_x2)
                    max_x = max(max_x, label_x1, label_x2)
                    min_y = min(min_y, label_y1, label_y2)
                    max_y = max(max_y, label_y1, label_y2)
        
        # Container title is rendered at a fixed position within the container
        # and doesn't affect the container size calculation
        # The container size should be based on children + padding only
        
        # Account for internal connections (add extra padding for arrowheads)
        if dataset:
            connection_padding = 10  # Extra space for arrowheads
            for conn in dataset.connections:
                # Check if connection is within this container
                from_comp = dataset.get_component(conn.from_component_id)
                to_comp = dataset.get_component(conn.to_component_id)
                if from_comp and to_comp:
                    # Both endpoints must be in child_components
                    if (from_comp in child_components and to_comp in child_components):
                        # Connection is internal - ensure we have space for it
                        # This is handled by the padding, but we could add more if needed
                        pass
        
        # #region agent log
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H3", "location": "layout_engine.py:244", "message": "before padding", "data": {"container_id": container.id, "min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y, "border_padding": self.border_padding}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
        except: pass
        # #endregion
        
        # Add padding
        min_x -= self.border_padding
        min_y -= self.border_padding
        max_x += self.border_padding
        max_y += self.border_padding
        
        # #region agent log
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H3", "location": "layout_engine.py:248", "message": "after padding", "data": {"container_id": container.id, "min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
        except: pass
        # #endregion
        
        # Calculate dimensions needed to encompass all children
        # If preserving position, calculate dimensions relative to container's original position
        # Otherwise, use the padded min/max bounds
        if preserve_position:
            # Container position is preserved, so calculate dimensions from container position
            orig_x = container.position[0]
            orig_y = container.position[1]
            width = max_x - orig_x  # Width from container x to max_x
            height = max_y - orig_y  # Height from container y to max_y
        else:
            # Use padded bounds
            width = max_x - min_x
            height = max_y - min_y
        
        # #region agent log
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H4", "location": "layout_engine.py:252", "message": "calculated dimensions", "data": {"container_id": container.id, "width": width, "height": height, "preserve_position": preserve_position, "container_pos": container.position, "max_x": max_x, "max_y": max_y, "min_x": min_x, "min_y": min_y}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
        except: pass
        # #endregion
        
        # Ensure minimum size (at least accommodate title)
        min_width = 200
        min_height = 40
        
        width = max(width, min_width)
        height = max(height, min_height)
        
        # If preserving position, use container's original position
        # Otherwise, use calculated min position
        if preserve_position:
            # Keep original position, use calculated dimensions
            orig_x = container.position[0]
            orig_y = container.position[1]
            result_bbox = BoundingBox(orig_x, orig_y, width, height)
        else:
            # Return bounding box with calculated position
            result_bbox = BoundingBox(min_x, min_y, width, height)
        
        # #region agent log
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H4", "location": "layout_engine.py:265", "message": "final bbox result", "data": {"container_id": container.id, "bbox_x": result_bbox.x, "bbox_y": result_bbox.y, "bbox_width": result_bbox.width, "bbox_height": result_bbox.height, "bbox_bottom": result_bbox.bottom}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
        except: pass
        # #endregion
        
        return result_bbox
    
    def validate_spacing(self, components: List[Component], container_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate spacing between sibling components and return violations.
        
        Spacing is measured between component RECT boundaries, not including text
        that extends beyond the box (text is inside the box).
        
        Args:
            components: List of components to check (should be siblings)
            container_id: Optional container ID for context in error messages
            
        Returns:
            Dictionary with spacing violations and recommendations
        """
        violations = []
        
        for i, comp1 in enumerate(components):
            # Use actual component bbox (rect), not text-expanded bbox for spacing
            bbox1 = comp1.bbox
            
            for j, comp2 in enumerate(components[i+1:], start=i+1):
                # Use actual component bbox (rect), not text-expanded bbox for spacing
                bbox2 = comp2.bbox
                
                # Check horizontal spacing (left-right or right-left)
                if bbox1.right <= bbox2.left:
                    # comp1 is to the left of comp2
                    spacing = bbox2.left - bbox1.right
                    if spacing < self.horizontal_spacing:
                        violations.append({
                            "type": "horizontal",
                            "components": [comp1.id, comp2.id],
                            "current": spacing,
                            "required": self.horizontal_spacing,
                            "difference": self.horizontal_spacing - spacing,
                            "container": container_id
                        })
                elif bbox2.right <= bbox1.left:
                    # comp2 is to the left of comp1
                    spacing = bbox1.left - bbox2.right
                    if spacing < self.horizontal_spacing:
                        violations.append({
                            "type": "horizontal",
                            "components": [comp2.id, comp1.id],
                            "current": spacing,
                            "required": self.horizontal_spacing,
                            "difference": self.horizontal_spacing - spacing,
                            "container": container_id
                        })
                
                # Check vertical spacing (top-bottom or bottom-top)
                if bbox1.bottom <= bbox2.top:
                    # comp1 is above comp2
                    spacing = bbox2.top - bbox1.bottom
                    if spacing < self.vertical_spacing:
                        violations.append({
                            "type": "vertical",
                            "components": [comp1.id, comp2.id],
                            "current": spacing,
                            "required": self.vertical_spacing,
                            "difference": self.vertical_spacing - spacing,
                            "container": container_id
                        })
                elif bbox2.bottom <= bbox1.top:
                    # comp2 is above comp1
                    spacing = bbox1.top - bbox2.bottom
                    if spacing < self.vertical_spacing:
                        violations.append({
                            "type": "vertical",
                            "components": [comp2.id, comp1.id],
                            "current": spacing,
                            "required": self.vertical_spacing,
                            "difference": self.vertical_spacing - spacing,
                            "container": container_id
                        })
        
        return {
            "violations": violations,
            "total": len(violations),
            "spacing_standard": {
                "horizontal": self.horizontal_spacing,
                "vertical": self.vertical_spacing
            }
        }
    
    def validate_hierarchical_spacing(self, dataset: SVGDataset, diagram_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate spacing at each hierarchy level (siblings within containers).
        
        Args:
            dataset: SVG dataset
            diagram_json: Diagram JSON definition
            
        Returns:
            Dictionary with all spacing violations organized by container
        """
        all_violations = []
        container_defs = {c["id"]: c for c in diagram_json.get("containers", [])}
        
        # Validate spacing within each container
        for container in dataset.containers:
            container_def = container_defs.get(container.id)
            if not container_def:
                continue
            
            # Get child components (siblings)
            child_components = []
            if "contains" in container_def:
                for child_id in container_def["contains"]:
                    child = dataset.get_component(child_id)
                    if child:
                        child_components.append(child)
            
            if len(child_components) > 1:
                # Validate spacing between siblings
                result = self.validate_spacing(child_components, container.id)
                all_violations.extend(result["violations"])
        
        # Also validate spacing between top-level components (not in containers)
        top_level_components = []
        all_contained_ids = set()
        for container_def in diagram_json.get("containers", []):
            all_contained_ids.update(container_def.get("contains", []))
        
        for comp in dataset.components:
            if comp.id not in all_contained_ids:
                top_level_components.append(comp)
        
        if len(top_level_components) > 1:
            result = self.validate_spacing(top_level_components, "root")
            all_violations.extend(result["violations"])
        
        return {
            "violations": all_violations,
            "total": len(all_violations),
            "spacing_standard": {
                "horizontal": self.horizontal_spacing,
                "vertical": self.vertical_spacing
            }
        }

