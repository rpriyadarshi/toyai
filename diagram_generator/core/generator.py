#!/usr/bin/env python3
"""
SVG Generator - Generate Inkscape-compliant SVG from JSON diagram definitions.

This module generates SVG files from JSON diagram definitions, ensuring
full Inkscape compatibility with connectors, namespaces, and proper structure.
"""

import json
import math
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

from diagram_generator.core.diagram import SVGDataset, Component, Connection, Label, BoundingBox
from diagram_generator.core.component_loader import ComponentLoader
from diagram_generator.labeling.placer import LabelPlacer
from diagram_generator.layout.engine import LayoutEngine
from diagram_generator.layout.placement import PlacementEngine
from diagram_generator.routing.grid import RoutingGrid
from diagram_generator.routing.router import SteinerRouter


class SVGGenerator:
    """Generate Inkscape-compliant SVG from JSON diagram definitions."""
    
    def __init__(self, config_path: Optional[Path] = None, components_dir: Optional[Path] = None):
        """
        Initialize SVG generator.
        
        Args:
            config_path: Path to diagram config JSON
            components_dir: Path to components directory
        """
        # Load config
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "book" / "diagrams" / "diagram_config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")
        
        # Initialize component loader
        self.component_loader = ComponentLoader(components_dir)
        
        # Initialize layout engine
        self.layout_engine = LayoutEngine(self.config)
        
        # Initialize placement engine
        self.placement_engine = PlacementEngine(self.config)
        
        # Initialize Steiner router
        self.steiner_router = SteinerRouter(self.config)
    
    def generate(self, diagram_json: Dict[str, Any], diagram_path: Optional[Path] = None) -> str:
        """
        Generate SVG XML from JSON diagram definition.
        
        Args:
            diagram_json: Diagram definition as dictionary
            diagram_path: Optional path to diagram JSON file (used to find local components)
            
        Returns:
            SVG XML string
        """
        # Check for local components directory if diagram_path is provided
        local_components_dir = None
        if diagram_path:
            diagram_path = Path(diagram_path)
            local_components_dir = diagram_path.parent / "components"
            if not local_components_dir.exists():
                local_components_dir = None
        
        # Update component loader with local components directory if found
        if local_components_dir:
            self.component_loader.local_components_dir = local_components_dir
        # Create dataset
        metadata = diagram_json.get("metadata", {})
        dataset = SVGDataset(
            width=metadata.get("width", 800),
            height=metadata.get("height", 600),
            metadata=metadata
        )
        
        # Instantiate containers from templates
        for container_def in diagram_json.get("containers", []):
            container = self.component_loader.instantiate_component(
                component_id=container_def["id"],
                component_type=container_def.get("type", "container"),
                template_name=container_def["template"],
                position=(container_def["position"]["x"], container_def["position"]["y"]),
                config={"label": container_def.get("title", "")}
            )
            # Override dimensions if specified
            if "width" in container_def:
                container.width = container_def["width"]
                # Update structure rect width
                if "rect" in container.structure:
                    container.structure["rect"]["width"] = container.width
                # Update text x position for title centering
                if "text" in container.structure and len(container.structure["text"]) > 0:
                    container.structure["text"][0]["x"] = container.width / 2
            if "height" in container_def:
                container.height = container_def["height"]
                # Update structure rect height
                if "rect" in container.structure:
                    container.structure["rect"]["height"] = container.height
            dataset.add_container(container)
        
        # Instantiate components from templates
        for comp_def in diagram_json.get("components", []):
            # #region agent log
            try:
                with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                    import json
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H6", "location": "svg_generator.py:114", "message": "Creating component from JSON", "data": {"component_id": comp_def["id"], "json_position": comp_def["position"], "has_class": "class" in comp_def, "class_value": comp_def.get("class")}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
            except: pass
            # #endregion
            component = self.component_loader.instantiate_component(
                component_id=comp_def["id"],
                component_type=comp_def["type"],
                template_name=comp_def["template"],
                position=(comp_def["position"]["x"], comp_def["position"]["y"]),
                config=comp_def.get("custom_text")
            )
            # #region agent log
            try:
                with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                    import json
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H6", "location": "svg_generator.py:121", "message": "Component created", "data": {"component_id": component.id, "component_position": list(component.position)}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
            except: pass
            # #endregion
            # Override class if specified in JSON (either top-level or in visual)
            if "class" in comp_def:
                component.visual["class"] = comp_def["class"]
            elif "visual" in comp_def and "class" in comp_def["visual"]:
                component.visual["class"] = comp_def["visual"]["class"]
            # Override dimensions if specified in JSON (for spacing control)
            if "width" in comp_def:
                component.width = comp_def["width"]
                if "rect" in component.structure:
                    component.structure["rect"]["width"] = component.width
                # Update text x positions to center in new width
                if "text" in component.structure:
                    for text_elem in component.structure["text"]:
                        # Re-center text if it was centered in original template
                        if text_elem.get("x") == component.visual.get("width", component.width) / 2:
                            text_elem["x"] = component.width / 2
            if "height" in comp_def:
                component.height = comp_def["height"]
                if "rect" in component.structure:
                    component.structure["rect"]["height"] = component.height
            dataset.add_component(component)
        
        # Convert child positions from absolute to relative
        # This establishes hierarchical grouping in the data model
        dataset.convert_absolute_to_relative(diagram_json)
        
        # Load connections (needed for placement algorithms)
        for conn_data in diagram_json.get("connections", []):
            from_parts = conn_data["from"].split(".")
            to_parts = conn_data["to"].split(".")
            from_comp_id = from_parts[0]
            from_pin_id = from_parts[1] if len(from_parts) > 1 else "output"
            to_comp_id = to_parts[0]
            to_pin_id = to_parts[1] if len(to_parts) > 1 else "input"
            
            connection = Connection(
                from_component_id=from_comp_id,
                from_pin_id=from_pin_id,
                to_component_id=to_comp_id,
                to_pin_id=to_pin_id,
                style=conn_data.get("style", "forward"),
                routing=conn_data.get("routing", "polyline"),
                color=conn_data.get("color"),
                stroke_width=conn_data.get("stroke_width", 2.0),
                container_id=conn_data.get("container_id"),
                label=conn_data.get("label")  # Load label from JSON
            )
            # Store dasharray in connection metadata if provided
            if "dasharray" in conn_data:
                connection.metadata = {"dasharray": conn_data["dasharray"]}
            
            # Auto-detect container_id if not specified
            if not connection.container_id:
                connection.container_id = self._detect_connection_container(
                    connection, dataset, diagram_json
                )
            
            dataset.add_connection(connection)
        
        # Initialize labels (needed for placement - container-scoped labels will be placed during bottom-up)
        labels = [Label(
            type=label_def["type"],
            text=label_def["text"],
            priority=label_def.get("priority", 4),
            target_component_id=label_def.get("target"),
            target_position=(
                (label_def["target_position"]["x"], label_def["target_position"]["y"])
                if label_def.get("target_position") else None
            ),
            container_id=label_def.get("container_id")
        ) for label_def in diagram_json.get("labels", [])]
        
        # Add all labels to dataset (they'll be placed during bottom-up or after)
        for label in labels:
            dataset.add_label(label)
        
        # Run placement algorithm
        placement_algorithm = metadata.get("placement_algorithm", "hierarchical_bottom_up")
        
        if placement_algorithm == "hierarchical_bottom_up":
            # Use new bottom-up hierarchical placement
            # This will place container-scoped labels and calculate bboxes
            dataset = self.placement_engine.place_hierarchical_bottom_up(dataset, diagram_json)
        elif placement_algorithm == "force_directed":
            dataset = self.placement_engine.place_force_directed(dataset, diagram_json)
        elif placement_algorithm == "hierarchical":
            dataset = self.placement_engine.place_hierarchical(dataset, diagram_json)
        elif placement_algorithm == "grid_based":
            dataset = self.placement_engine.place_grid_based(dataset, diagram_json)
        elif placement_algorithm == "manual":
            # Keep manual positions, but still calculate bboxes
            pass
        else:
            print(f"Warning: Unknown placement algorithm '{placement_algorithm}', using hierarchical_bottom_up")
            dataset = self.placement_engine.place_hierarchical_bottom_up(dataset, diagram_json)
        
        # CRITICAL: Generate all geometries (connection paths) BEFORE placing labels
        # This ensures labels honor obstructions from connections
        # Build routing grid once for all connections
        routing_grid = self._build_routing_grid(dataset, [])
        
        connection_waypoints = {}
        for connection in dataset.connections:
            waypoints = self._calculate_connection_waypoints(connection, dataset, routing_grid)
            if waypoints:
                # Use connection ID as key since Connection objects aren't hashable
                connection_waypoints[id(connection)] = (connection, waypoints)
                # Mark this path in the grid for subsequent connections
                routing_grid.mark_path(waypoints, connection.stroke_width)
        
        # Place global labels (not container-scoped)
        global_labels = [l for l in labels if not l.container_id]
        if global_labels:
            placer = LabelPlacer(dataset, self.config)
            # Mark connection paths as obstructions BEFORE placing labels
            # Convert connection_waypoints dict to format expected by label_placer
            # Format: {connection_id: (connection, waypoints)}
            placer.mark_connection_paths_as_obstructions(connection_waypoints)
            placer.place_labels(global_labels)
            
            # Canvas Height Auto-Adjustment Algorithm
            # After placing labels, check if any extend above border_padding and adjust canvas height
            # Algorithm:
            # 1. Find minimum y-coordinate of all global label bounding boxes
            # 2. If min_y < border_padding, calculate extra space needed
            # 3. Increase canvas height by extra_top_space
            # 4. Shift all elements (components, containers, labels) down by extra_top_space
            # 5. This ensures title top is exactly at border_padding
            spacing = self.config.get("spacing", {})
            border_padding = spacing.get("border_padding", 20)
            min_y = 0
            for label in global_labels:
                if label.position and label.bbox:
                    label_top = label.bbox.top
                    min_y = min(min_y, label_top)
            
            # #region agent log
            try:
                with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                    f.write(__import__("json").dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "E", "location": "svg_generator.py:214", "message": "Checking canvas height adjustment", "data": {"min_y": min_y, "border_padding": border_padding, "current_height": dataset.height, "needs_adjustment": min_y < border_padding}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
            except: pass
            # #endregion
            
            # If labels extend above border_padding, adjust canvas height
            if min_y < border_padding:
                # Calculate how much extra space is needed at the top
                extra_top_space = border_padding - min_y
                # Increase canvas height
                dataset.height += extra_top_space
                # Shift all elements down by the extra space
                for component in dataset.components:
                    if component.parent_id is None:  # Only shift top-level components
                        component.position = (component.position[0], component.position[1] + extra_top_space)
                for container in dataset.containers:
                    container.position = (container.position[0], container.position[1] + extra_top_space)
                # Shift global labels
                for label in global_labels:
                    if label.position:
                        label.position = (label.position[0], label.position[1] + extra_top_space)
                        # Recalculate bbox
                        if label.bbox:
                            label.bbox = BoundingBox(
                                label.bbox.left,
                                label.bbox.top + extra_top_space,
                                label.bbox.width,
                                label.bbox.height
                            )
                
                # #region agent log
                try:
                    with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                        f.write(__import__("json").dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "E", "location": "svg_generator.py:240", "message": "Canvas height adjusted", "data": {"extra_top_space": extra_top_space, "new_height": dataset.height, "min_y_after": min_y + extra_top_space}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                except: pass
                # #endregion
        
        # Container bounding boxes are already calculated by placement engine
        # Don't recalculate here as it would overwrite the correct heights calculated during placement
        
        # Validate spacing at each hierarchy level
        spacing_report = self.layout_engine.validate_hierarchical_spacing(dataset, diagram_json)
        if spacing_report["total"] > 0:
            print(f"Warning: {spacing_report['total']} spacing violations detected:")
            # Group by container
            by_container = {}
            for violation in spacing_report["violations"]:
                container = violation.get("container", "root")
                if container not in by_container:
                    by_container[container] = []
                by_container[container].append(violation)
            
            for container_id, violations in by_container.items():
                print(f"  Container '{container_id}': {len(violations)} violations")
                for violation in violations[:3]:  # Show first 3 per container
                    print(f"    - {violation['type']} spacing between {violation['components'][0]} and {violation['components'][1]}: "
                          f"{violation['current']:.1f}px (required: {violation['required']}px, need {violation['difference']:.1f}px more)")
        
        # Generate SVG XML
        return self._to_svg_xml(dataset, diagram_json)
    
    def _detect_connection_container(
        self, 
        connection: Connection, 
        dataset: SVGDataset,
        diagram_json: Dict[str, Any]
    ) -> Optional[str]:
        """
        Auto-detect which container a connection belongs to.
        A connection belongs to a container if both endpoints are in that container.
        
        Args:
            connection: Connection to check
            dataset: SVG dataset
            diagram_json: Diagram JSON definition
            
        Returns:
            Container ID if both endpoints are in same container, None otherwise
        """
        from_comp = dataset.get_component(connection.from_component_id)
        to_comp = dataset.get_component(connection.to_component_id)
        
        if not from_comp or not to_comp:
            return None
        
        # Check each container
        for container_def in diagram_json.get("containers", []):
            if "contains" not in container_def:
                continue
            
            contained_ids = set(container_def["contains"])
            if (connection.from_component_id in contained_ids and 
                connection.to_component_id in contained_ids):
                return container_def["id"]
        
        return None
    
    def _to_svg_xml(self, dataset: SVGDataset, diagram_json: Dict[str, Any] = None) -> str:
        """
        Convert dataset to SVG XML.
        
        Args:
            dataset: SVG dataset
            
        Returns:
            SVG XML string
        """
        lines = []
        
        # XML header
        lines.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
        
        # SVG root with namespaces
        lines.append(f'<svg')
        lines.append(f'   width="{dataset.width}"')
        lines.append(f'   height="{dataset.height}"')
        lines.append(f'   xmlns="http://www.w3.org/2000/svg"')
        lines.append(f'   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"')
        lines.append(f'   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd">')
        
        # Inkscape namedview
        lines.append('  <sodipodi:namedview')
        lines.append('     id="namedview"')
        lines.append('     pagecolor="#ffffff"')
        lines.append('     bordercolor="#000000"')
        lines.append('     borderopacity="0.25"')
        lines.append('     inkscape:showpageshadow="2"')
        lines.append('     inkscape:pageopacity="0.0"')
        lines.append('     inkscape:pagecheckerboard="0"')
        lines.append('     inkscape:deskcolor="#d1d1d1"')
        lines.append('     showgrid="false" />')
        
        # Defs section (styles and markers)
        lines.append('  <defs>')
        lines.extend(self._generate_styles())
        lines.extend(self._generate_markers())
        lines.append('  </defs>')
        
        # Containers (as groups, rendered first so they appear behind components)
        # We'll render containers with their nested connections
        container_connections = {}
        global_connections = []
        
        for connection in dataset.connections:
            if connection.container_id:
                if connection.container_id not in container_connections:
                    container_connections[connection.container_id] = []
                container_connections[connection.container_id].append(connection)
            else:
                global_connections.append(connection)
        
        # Render containers (using absolute positioning like reference, not nested transforms)
        for container in dataset.containers:
            # Container group with NO transform (absolute positioning like reference)
            lines.append(f'  <g id="{container.id}">')
            
            # Container background rect (absolute position)
            if "rect" in container.structure:
                rect = container.structure["rect"]
                lines.append(f'    <rect x="{container.position[0]}" y="{container.position[1]}" width="{container.width}" height="{container.height}" rx="10" class="subgraph-bg" />')
            
            # Container title (absolute position)
            if "text" in container.structure and len(container.structure["text"]) > 0:
                text = container.structure["text"][0]
                text_content = text.get("content", text.get("text", ""))
                # Handle custom text override for containers
                if container.visual.get("custom_text"):
                    text_content = container.visual.get("custom_text", {}).get("label", text_content)
                title_x = container.position[0] + (text.get("x", container.width/2))
                lines.append(f'    <text x="{title_x}" y="{container.position[1] + 20}" class="subgraph-title" text-anchor="middle">{text_content}</text>')
            
            # Render child components within this container (absolute positioning)
            container_def = None
            if diagram_json:
                for cdef in diagram_json.get("containers", []):
                    if cdef["id"] == container.id:
                        container_def = cdef
                        break
            
            if container_def:
                child_ids = container_def.get("contains", [])
                for child_id in child_ids:
                    child = dataset.get_component(child_id)
                    if child:
                        # Children use absolute positioning (like reference)
                        child_lines = self._generate_component(child, dataset)
                        lines.extend([f'    {line}' for line in child_lines])
            
            # Render connections within this container (absolute positioning)
            if container.id in container_connections:
                lines.append(f'    <!-- Connections within container: {container.id} -->')
                for connection in container_connections[container.id]:
                    conn_xml = self._generate_connection(connection, dataset)
                    if conn_xml:
                        lines.append(f'    {conn_xml}')
            
            # Close container group
            lines.append('  </g>')
        
        # Components not in containers (top-level)
        container_child_ids = set()
        if diagram_json:
            for container_def in diagram_json.get("containers", []):
                container_child_ids.update(container_def.get("contains", []))
        
        for component in dataset.components:
            if component.id not in container_child_ids:
                lines.extend(self._generate_component(component, dataset))
        
        # Render global connections (between containers or top-level)
        if global_connections:
            lines.append('  <!-- Global connections -->')
            for connection in global_connections:
                conn_xml = self._generate_connection(connection, dataset)
                if conn_xml:
                    lines.append(conn_xml)
        
        # Labels (text elements)
        for label in dataset.labels:
            if label.position:
                lines.extend(self._generate_label(label))
        
        # Close SVG
        lines.append('</svg>')
        
        return '\n'.join(lines)
    
    def _generate_styles(self) -> list:
        """Generate CSS styles."""
        styles = ['    <style>']
        
        # Box styles
        colors = self.config.get("colors", {})
        backgrounds = colors.get("background", {})
        borders = colors.get("border", {})
        
        # Mapping from component class names to config keys
        class_to_config = {
            "ln-box": "layer_norm",
            "qkv-box": "qkv",
            "attention-box": "attention",
            "softmax-box": "softmax",
            "context-box": "context",
            "residual-box": "residual",
            "ffn-box": "ffn",
            "input-box": "input",
            "output-box": "output",
            "embed-box": "embed",
            "loss-box": "loss",
            "update-box": "update",
            "start-box": "input",
            "forward-box": "embed",
            "backward-box": "qkv",
            "stop-box": "output",
            "state-strong": "state_strong",
            "state-medium": "state_medium",
            "state-weak": "state_weak",
            "check-box": "check_box",
            "output-yes": "output_yes",
            "output-no": "output_no",
            "range-box": "range_box",
            "bias-box": "bias_box",
            "sum-box": "sum_box",
            "activation-box": "activation_box",
            "layer-box": "layer_box"
        }
        
        # Generate box classes based on actual component class names
        # First, generate from config keys (for backward compatibility)
        for box_type, bg_color in backgrounds.items():
            border_color = borders.get(box_type, borders.get("default", "#cbd5e1"))
            class_name = f"{box_type}-box"
            styles.append(f'      .{class_name} {{ fill: {bg_color}; stroke: {border_color}; stroke-width: 2; rx: 6; }}')
        
        # Then, generate styles for component class names that don't match config keys
        for class_name, config_key in class_to_config.items():
            if config_key in backgrounds:
                bg_color = backgrounds[config_key]
                border_color = borders.get(config_key, borders.get("default", "#cbd5e1"))
                styles.append(f'      .{class_name} {{ fill: {bg_color}; stroke: {border_color}; stroke-width: 2; rx: 6; }}')
        
        # Add subgraph styles
        styles.append('      .subgraph-bg { fill: #f8fafc; stroke: #cbd5e1; stroke-width: 2; rx: 10; }')
        styles.append('      .subgraph-title { font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; fill: #1e293b; }')
        
        # Text styles
        typography = self.config.get("typography", {})
        font_family = typography.get("font_family", "Arial, sans-serif")
        sizes = typography.get("sizes", {})
        weights = typography.get("weights", {})
        
        text_colors = colors.get("text", {})
        primary_text = text_colors.get("primary", "#1e293b")
        secondary_text = text_colors.get("secondary", "#64748b")
        
        styles.append(f'      .box-label {{ font-family: {font_family}; font-size: 14px; font-weight: bold; fill: {primary_text}; }}')
        styles.append(f'      .box-text {{ font-family: {font_family}; font-size: 11px; fill: {secondary_text}; }}')
        styles.append(f'      .title {{ font-family: {font_family}; font-size: 18px; font-weight: bold; fill: {primary_text}; }}')
        styles.append(f'      .equation {{ font-family: {font_family}; font-size: 12px; fill: {text_colors.get("muted", "#666")}; }}')
        styles.append(f'      .label {{ font-family: {font_family}; font-size: 11px; fill: {primary_text}; }}')
        styles.append(f'      .note {{ font-family: {font_family}; font-size: 11px; fill: {text_colors.get("muted", "#666")}; }}')
        
        styles.append('    </style>')
        return styles
    
    def _generate_markers(self) -> list:
        """Generate arrow markers."""
        markers = []
        colors = self.config.get("colors", {})
        
        # Blue arrow (forward)
        primary = colors.get("primary", "#2563eb")
        markers.append(f'    <marker id="arrow-blue" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">')
        markers.append(f'      <polygon points="0 0, 8 3, 0 6" fill="{primary}" />')
        markers.append('    </marker>')
        
        # Purple arrow (backward/training)
        secondary = colors.get("secondary", "#9333ea")
        markers.append(f'    <marker id="arrow-purple" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">')
        markers.append(f'      <polygon points="0 0, 8 3, 0 6" fill="{secondary}" />')
        markers.append('    </marker>')
        
        return markers
    
    def _generate_component(self, component: Component, dataset: SVGDataset) -> list:
        """Generate SVG for a component."""
        # Get absolute position for rendering
        abs_pos = component.get_absolute_position(dataset)
        lines = []
        
        # Component group
        lines.append(f'  <g id="{component.id}" transform="translate({abs_pos[0]}, {abs_pos[1]})">')
        
        # Generate structure from template
        structure = component.structure
        
        # Rectangle
        if "rect" in structure:
            rect = structure["rect"]
            # Use component dimensions (which may have been overridden)
            rect_width = rect.get("width", component.width)
            rect_height = rect.get("height", component.height)
            # Ensure structure rect matches component dimensions
            if component.width != rect_width or component.height != rect_height:
                rect_width = component.width
                rect_height = component.height
            lines.append(f'    <rect x="{rect.get("x", 0)}" y="{rect.get("y", 0)}" '
                        f'width="{rect_width}" '
                        f'height="{rect_height}" '
                        f'rx="{rect.get("rx", 6)}" '
                        f'class="{component.visual.get("class", "input-box")}" />')
        
        # Text elements
        if "text" in structure:
            for text_elem in structure["text"]:
                text_content = text_elem.get("content", "")
                # Skip empty text elements
                if not text_content or not str(text_content).strip():
                    continue
                # Handle custom text override for containers
                if component.type == "container" and component.visual.get("custom_text"):
                    if text_elem.get("class") == "subgraph-title":
                        text_content = component.visual.get("custom_text", {}).get("label", text_content)
                lines.append(f'    <text x="{text_elem.get("x", component.width/2)}" '
                           f'y="{text_elem.get("y", component.height/2)}" '
                           f'class="{text_elem.get("class", "box-label")}" '
                           f'text-anchor="middle">{text_content}</text>')
        
        lines.append('  </g>')
        return lines
    
    def _build_routing_grid(
        self,
        dataset: SVGDataset,
        existing_paths: Optional[List[List[Tuple[float, float]]]] = None
    ) -> RoutingGrid:
        """
        Build routing grid from all obstructions.
        
        Args:
            dataset: SVG dataset with components and containers
            existing_paths: Optional list of existing connection paths (waypoints)
            
        Returns:
            RoutingGrid instance ready for pathfinding
        """
        grid = RoutingGrid(dataset.width, dataset.height, self.config)
        grid.build_grid(dataset, existing_paths)
        return grid
    
    def _calculate_connection_waypoints(
        self,
        connection: Connection,
        dataset: SVGDataset,
        routing_grid: Optional[RoutingGrid] = None
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Calculate waypoints for a connection path using A* pathfinding.
        
        This is used to mark connection paths as obstructions for label placement.
        
        Args:
            connection: Connection to calculate waypoints for
            dataset: SVG dataset
            routing_grid: Optional pre-built routing grid (if None, will build one)
            
        Returns:
            List of (x, y) waypoints, or None if connection is invalid
        """
        from_comp = dataset.get_component(connection.from_component_id)
        to_comp = dataset.get_component(connection.to_component_id)
        
        if not from_comp or not to_comp:
            return None
        
        from_abs_bbox = from_comp.get_absolute_bbox(dataset)
        to_abs_bbox = to_comp.get_absolute_bbox(dataset)
        
        from_center = (from_abs_bbox.center_x, from_abs_bbox.center_y)
        to_center = (to_abs_bbox.center_x, to_abs_bbox.center_y)
        
        # ALWAYS calculate exit/entry points using center-to-center algorithm
        # This ensures all connections use center-to-center routing for calculation
        exit_point = self._find_box_exit_point(from_abs_bbox, from_center, to_center)
        entry_point = self._find_box_entry_point(to_abs_bbox, to_center, from_center)
        
        routing_type = connection.routing or "polyline"
        
        if routing_type == "direct":
            # Direct routing: always straight line (boundary to boundary)
            return [exit_point, entry_point]
        else:
            # Check for obstructions along direct path (center-to-center line)
            obstructions = self._detect_path_obstructions(
                from_center, to_center, from_comp, to_comp, dataset
            )
            
            if not obstructions:
                # No obstructions: use simple straight line (boundary to boundary)
                # Exit/entry points already calculated from center-to-center
                return [exit_point, entry_point]
            else:
                # Obstructions detected: choose routing direction first, then calculate exit/entry points
                # This ensures exit/entry points are on the correct edges (e.g., right edge for loop)
                direction = self._choose_routing_direction(
                    from_center, to_center, obstructions, dataset, connection
                )
                
                # Calculate exit/entry points based on routing direction (not center-to-center line)
                exit_point = self._calculate_steiner_exit_point(
                    from_abs_bbox, direction, from_center
                )
                entry_point = self._calculate_steiner_entry_point(
                    to_abs_bbox, direction, to_center
                )
                
                # Now use A* routing from these direction-based exit/entry points
                if routing_grid is None:
                    routing_grid = self._build_routing_grid(dataset, [])
                
                # Use A* pathfinding to find optimal path around obstructions
                path = self.steiner_router.find_path(exit_point, entry_point, routing_grid, dataset)
                
                if path:
                    return path
                else:
                    # Fallback to direct if A* fails
                    return [exit_point, entry_point]
    
    def _generate_connection(self, connection: Connection, dataset: SVGDataset) -> Optional[str]:
        """
        Generate Inkscape connector for a connection using center-to-center algorithm.
        
        Algorithm:
        1. Calculate line from source box center to destination box center
        2. Find where this line exits the source box boundary (arrow start point)
        3. Find where this line enters the destination box boundary (arrow end point)
        4. Use these intersection points as the path start and end
        
        This ensures arrows start and terminate at object boundaries, matching Inkscape's
        connector behavior and ensuring SVG viewers render identically.
        """
        from_comp = dataset.get_component(connection.from_component_id)
        to_comp = dataset.get_component(connection.to_component_id)
        
        if not from_comp or not to_comp:
            return None
        
        # Use center-to-center algorithm: calculate line from box center to box center
        # Then find where this line exits source box boundary (arrow start)
        # and enters destination box boundary (arrow end)
        from_abs_bbox = from_comp.get_absolute_bbox(dataset)
        to_abs_bbox = to_comp.get_absolute_bbox(dataset)
        
        # Get box centers
        from_center = (from_abs_bbox.center_x, from_abs_bbox.center_y)
        to_center = (to_abs_bbox.center_x, to_abs_bbox.center_y)
        
        # Determine style
        if connection.style == "backward" or connection.style == "dashed" or connection.style == "training":
            color = connection.color or self.config.get("colors", {}).get("secondary", "#9333ea")
            marker = "arrow-purple"
            dasharray = connection.metadata.get("dasharray", "4,4")
        else:
            color = connection.color or self.config.get("colors", {}).get("primary", "#2563eb")
            marker = "arrow-blue"
            dasharray = None
        
        # ALWAYS calculate exit/entry points using center-to-center algorithm
        # This ensures all connections use center-to-center routing for calculation
        exit_point = self._find_box_exit_point(from_abs_bbox, from_center, to_center)
        entry_point = self._find_box_entry_point(to_abs_bbox, to_center, from_center)
        
        # Generate path based on routing type
        routing_type = connection.routing or "polyline"
        
        if routing_type == "direct":
            # Direct routing: always straight line (boundary to boundary)
            d = f"M {exit_point[0]},{exit_point[1]} L {entry_point[0]},{entry_point[1]}"
        else:
            # Check for obstructions along direct path (center-to-center line)
            obstructions = self._detect_path_obstructions(
                from_center, to_center, from_comp, to_comp, dataset
            )
            
            if not obstructions:
                # No obstructions: use simple straight line (boundary to boundary)
                # Exit/entry points already calculated from center-to-center
                d = f"M {exit_point[0]},{exit_point[1]} L {entry_point[0]},{entry_point[1]}"
            else:
                # Obstructions detected: choose routing direction first, then calculate exit/entry points
                # This ensures exit/entry points are on the correct edges (e.g., right edge for loop)
                direction = self._choose_routing_direction(
                    from_center, to_center, obstructions, dataset, connection
                )
                
                # Calculate exit/entry points based on routing direction (not center-to-center line)
                exit_point = self._calculate_steiner_exit_point(
                    from_abs_bbox, direction, from_center
                )
                entry_point = self._calculate_steiner_entry_point(
                    to_abs_bbox, direction, to_center
                )
                
                # Now use A* routing from these direction-based exit/entry points
                routing_grid = self._build_routing_grid(dataset, [])
                
                # Use A* pathfinding to find optimal path around obstructions
                path_waypoints = self.steiner_router.find_path(exit_point, entry_point, routing_grid, dataset)
                
                if path_waypoints:
                    # Convert waypoints to SVG path
                    curved = (routing_type == "polyline")
                    d = self._path_waypoints_to_svg(path_waypoints, curved)
                else:
                    # Fallback to direct if A* fails
                    d = f"M {exit_point[0]},{exit_point[1]} L {entry_point[0]},{entry_point[1]}"
        
        # Build connector element
        style_parts = [f"fill:none", f"stroke:{color}", f"stroke-width:{connection.stroke_width}"]
        if dasharray:
            style_parts.append(f"stroke-dasharray:{dasharray}")
        style = ";".join(style_parts)
        
        connector = f'  <path\n'
        connector += f'     d="{d}"\n'
        connector += f'     style="{style}"\n'
        connector += f'     inkscape:connector-type="polyline"\n'
        connector += f'     inkscape:connector-curvature="0"\n'
        connector += f'     inkscape:connection-start="#{from_comp.id}"\n'
        connector += f'     inkscape:connection-end="#{to_comp.id}"\n'
        connector += f'     marker-end="url(#{marker})" />'
        
        # Add label if present
        if connection.label:
            # Calculate label position (midpoint of connection)
            label_x = (exit_point[0] + entry_point[0]) / 2
            label_y = (exit_point[1] + entry_point[1]) / 2
            
            # Offset label slightly perpendicular to the line to avoid overlap
            dx = entry_point[0] - exit_point[0]
            dy = entry_point[1] - exit_point[1]
            length = math.sqrt(dx*dx + dy*dy) if (dx != 0 or dy != 0) else 1.0
            
            # Perpendicular offset (10px above the line)
            offset_x = -dy / length * 10 if length > 0 else 0
            offset_y = dx / length * 10 if length > 0 else 0
            
            label_x += offset_x
            label_y += offset_y
            
            # Determine label color based on connection color
            label_color = color
            if connection.style == "backward" or connection.style == "training":
                label_color = self.config.get("colors", {}).get("secondary", "#9333ea")
            
            connector += f'\n  <text x="{label_x}" y="{label_y}" '
            connector += f'font-family="Arial, sans-serif" font-size="11" '
            connector += f'fill="{label_color}" text-anchor="middle" '
            connector += f'class="arrow-label">{connection.label}</text>'
        
        return connector
    
    def _find_box_exit_point(self, bbox, from_pos, to_pos):
        """Find where line exits bounding box."""
        return self._line_box_intersection(from_pos, to_pos, bbox, exit=True)
    
    def _find_box_entry_point(self, bbox, to_pos, from_pos):
        """Find where line enters bounding box."""
        # Line should go from source to destination, not backwards
        return self._line_box_intersection(from_pos, to_pos, bbox, exit=False)
    
    def _line_box_intersection(self, start, end, bbox, exit=True):
        """
        Find intersection of line segment with bounding box.
        
        Algorithm:
        - For exit points: allows t >= 0 (line starts at center, which may be inside box)
        - For entry points: allows t <= 1 (line ends at center, which may be inside box)
        - Checks all four edges (left, right, top, bottom) for intersections
        - Returns the intersection point closest to start (for exit) or end (for entry)
        
        Args:
            start: Start point of line segment (x, y)
            end: End point of line segment (x, y)
            bbox: BoundingBox to intersect with
            exit: If True, find exit point (closest to start). If False, find entry point (closest to end)
            
        Returns:
            (x, y) intersection point
        """
        # Improved implementation: find intersection with box edges
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        # Handle edge cases: horizontal and vertical lines
        if abs(dx) < 1e-6:  # Vertical line
            if exit:
                if dy > 0:
                    return (start[0], bbox.bottom)
                else:
                    return (start[0], bbox.top)
            else:
                if dy > 0:
                    return (start[0], bbox.top)
                else:
                    return (start[0], bbox.bottom)
        
        if abs(dy) < 1e-6:  # Horizontal line
            if exit:
                if dx > 0:
                    return (bbox.right, start[1])
                else:
                    return (bbox.left, start[1])
            else:
                if dx > 0:
                    return (bbox.left, start[1])
                else:
                    return (bbox.right, start[1])
        
        # Check each edge
        intersections = []
        
        # For exit: allow t >= 0 (line starts at center, which may be inside box)
        # For entry: allow t <= 1 (line ends at center, which may be inside box)
        t_min = 0.0 if exit else -1e10  # Allow t >= 0 for exit
        t_max = 1.0 if not exit else 1e10  # Allow t <= 1 for entry
        
        # Left edge
        if dx != 0:
            t = (bbox.left - start[0]) / dx
            if t_min <= t <= t_max:
                y = start[1] + t * dy
                if bbox.top <= y <= bbox.bottom:
                    intersections.append((bbox.left, y, t))
        
        # Right edge
        if dx != 0:
            t = (bbox.right - start[0]) / dx
            if t_min <= t <= t_max:
                y = start[1] + t * dy
                if bbox.top <= y <= bbox.bottom:
                    intersections.append((bbox.right, y, t))
        
        # Top edge
        if dy != 0:
            t = (bbox.top - start[1]) / dy
            if t_min <= t <= t_max:
                x = start[0] + t * dx
                if bbox.left <= x <= bbox.right:
                    intersections.append((x, bbox.top, t))
        
        # Bottom edge
        if dy != 0:
            t = (bbox.bottom - start[1]) / dy
            if t_min <= t <= t_max:
                x = start[0] + t * dx
                if bbox.left <= x <= bbox.right:
                    intersections.append((x, bbox.bottom, t))
        
        if intersections:
            # Return the intersection point with the smallest t (closest to start) for exit,
            # or largest t (closest to end) for entry
            if exit:
                # For exit, find intersection with smallest t (closest to start)
                closest = min(intersections, key=lambda p: p[2])  # p[2] is the t parameter
                return (closest[0], closest[1])
            else:
                # For entry, find intersection with largest t (closest to end)
                closest = max(intersections, key=lambda p: p[2])  # p[2] is the t parameter
                return (closest[0], closest[1])
        
        # Fallback: if no intersection found, use the edge closest to the line
        # This handles cases where the line starts/ends inside the box
        if exit:
            # Find which edge is closest to start point
            dist_left = abs(start[0] - bbox.left)
            dist_right = abs(start[0] - bbox.right)
            dist_top = abs(start[1] - bbox.top)
            dist_bottom = abs(start[1] - bbox.bottom)
            
            min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
            if min_dist == dist_left:
                return (bbox.left, start[1])
            elif min_dist == dist_right:
                return (bbox.right, start[1])
            elif min_dist == dist_top:
                return (start[0], bbox.top)
            else:
                return (start[0], bbox.bottom)
        else:
            # Find which edge is closest to end point
            dist_left = abs(end[0] - bbox.left)
            dist_right = abs(end[0] - bbox.right)
            dist_top = abs(end[1] - bbox.top)
            dist_bottom = abs(end[1] - bbox.bottom)
            
            min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
            if min_dist == dist_left:
                return (bbox.left, end[1])
            elif min_dist == dist_right:
                return (bbox.right, end[1])
            elif min_dist == dist_top:
                return (end[0], bbox.top)
            else:
                return (end[0], bbox.bottom)
    
    def _get_arrowhead_offset(self) -> float:
        """
        Get the arrowhead offset distance for Steiner points.
        
        The offset accounts for the arrowhead width plus additional spacing
        to ensure the Steiner point is positioned away from the arrowhead.
        
        This value is configurable via routing.steiner_point_offset in diagram_config.json.
        Default is 18px for good visual spacing.
        
        Returns:
            Offset distance in pixels (read from config, default 18px)
        """
        # Get from config, with sensible default
        routing_config = self.config.get("routing", {})
        offset = routing_config.get("steiner_point_offset", 18.0)
        return float(offset)
    
    def _calculate_steiner_exit_point(
        self,
        bbox: BoundingBox,
        direction: str,
        center: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Calculate Steiner point for exit based on routing direction.
        
        For Manhattan routing, the exit point should be on the edge that the
        Manhattan path will use (e.g., right edge when routing right), not on
        the edge determined by the center-to-center line.
        
        The exit point should be AT the box edge (no offset), since there's
        no arrowhead at the start of the path.
        
        Args:
            bbox: Source component bounding box
            direction: Routing direction ("left", "right", "top", "bottom")
            center: Component center point (for vertical/horizontal position)
            
        Returns:
            Exit point (x, y) at the appropriate edge
        """
        if direction == "right":
            # Exit on right edge, at vertical center
            return (bbox.right, center[1])
        elif direction == "left":
            # Exit on left edge, at vertical center
            return (bbox.left, center[1])
        elif direction == "top":
            # Exit on top edge, at horizontal center
            return (center[0], bbox.top)
        else:  # bottom
            # Exit on bottom edge, at horizontal center
            return (center[0], bbox.bottom)
    
    def _calculate_steiner_entry_point(
        self,
        bbox: BoundingBox,
        direction: str,
        center: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Calculate Steiner point for entry based on routing direction.
        
        For Manhattan routing, the entry point should be on the edge that the
        Manhattan path will approach from (e.g., right edge when routing right),
        not on the edge determined by the center-to-center line.
        
        The entry point should be positioned so the arrowhead base is at the
        box edge with proper spacing. For a LEFT-pointing arrow:
        - Arrowhead base vertices are 7 units LEFT of reference point
        - To position base at box edge with spacing, entry = box_edge - spacing - 7
        
        Args:
            bbox: Destination component bounding box
            direction: Routing direction ("left", "right", "top", "bottom")
            center: Component center point (for vertical/horizontal position)
            
        Returns:
            Entry point (x, y) positioned so arrowhead base is at box edge with spacing
        """
        arrowhead_offset = self._get_arrowhead_offset()
        refX = 7.0  # Arrowhead width (refX from marker definition)
        spacing = arrowhead_offset - refX  # Additional spacing beyond arrowhead width
        
        # #region agent log
        try:
            with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                import json
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1", "location": "svg_generator.py:_calculate_steiner_entry_point", "message": "Calculating entry point", "data": {"direction": direction, "bbox_right": bbox.right, "bbox_left": bbox.left, "bbox_top": bbox.top, "bbox_bottom": bbox.bottom, "center": list(center), "arrowhead_offset": arrowhead_offset, "refX": refX, "spacing": spacing}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
        except: pass
        # #endregion
        
        if direction == "right":
            # Routing on RIGHT side, entering RIGHT edge
            # Contact point should be AT the box right edge (no offset)
            entry_x = bbox.right
            # #region agent log
            try:
                with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                    import json
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1", "location": "svg_generator.py:_calculate_steiner_entry_point", "message": "RIGHT routing contact point", "data": {"entry_x": entry_x, "box_edge": bbox.right}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
            except: pass
            # #endregion
            return (entry_x, center[1])
        elif direction == "left":
            # For LEFT-pointing arrow entering RIGHT edge:
            # Arrowhead points LEFT, base extends LEFT from reference (refX units LEFT)
            # User wants arrowhead base positioned with spacing from edge
            # Base = entry - refX
            # To position base with spacing: base = box_edge - spacing
            # So: entry - refX = box_edge - spacing
            # Therefore: entry = box_edge - spacing + refX
            # This positions base at (box_edge - spacing) with proper spacing
            entry_x = bbox.right - spacing + refX
            # #region agent log
            try:
                with open("/home/rohit/src/toyai-1/.cursor/debug.log", "a") as f:
                    import json
                    calculated_base_x = entry_x - refX  # Base is LEFT of reference for LEFT-pointing arrow
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1", "location": "svg_generator.py:_calculate_steiner_entry_point", "message": "LEFT direction entry point (right edge)", "data": {"entry_x": entry_x, "box_edge": bbox.right, "calculated_base_x": calculated_base_x, "base_distance_from_edge": bbox.right - calculated_base_x, "spacing": spacing, "refX": refX, "entry_distance_from_edge": bbox.right - entry_x}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
            except: pass
            # #endregion
            return (entry_x, center[1])
        elif direction == "top":
            # For TOP-pointing arrow: base extends UP from reference
            entry_y = bbox.top + spacing + refX
            return (center[0], entry_y)
        else:  # bottom
            # For BOTTOM-pointing arrow: base extends DOWN from reference
            entry_y = bbox.bottom - spacing - refX
            return (center[0], entry_y)
    
    def _line_segment_intersects_bbox(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        bbox: BoundingBox
    ) -> bool:
        """
        Check if line segment intersects bounding box.
        
        Uses parametric line equation: P(t) = start + t * (end - start), where 0 <= t <= 1
        Checks intersection with each edge of bounding box.
        
        Args:
            start: Start point of line segment (x, y)
            end: End point of line segment (x, y)
            bbox: BoundingBox to check intersection with
            
        Returns:
            True if line segment intersects bounding box, False otherwise
        """
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        # Check each edge for intersection
        # Left edge
        if dx != 0:
            t = (bbox.left - start[0]) / dx
            if 0 <= t <= 1:
                y = start[1] + t * dy
                if bbox.top <= y <= bbox.bottom:
                    return True
        
        # Right edge
        if dx != 0:
            t = (bbox.right - start[0]) / dx
            if 0 <= t <= 1:
                y = start[1] + t * dy
                if bbox.top <= y <= bbox.bottom:
                    return True
        
        # Top edge
        if dy != 0:
            t = (bbox.top - start[1]) / dy
            if 0 <= t <= 1:
                x = start[0] + t * dx
                if bbox.left <= x <= bbox.right:
                    return True
        
        # Bottom edge
        if dy != 0:
            t = (bbox.bottom - start[1]) / dy
            if 0 <= t <= 1:
                x = start[0] + t * dx
                if bbox.left <= x <= bbox.right:
                    return True
        
        return False
    
    def _detect_path_obstructions(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        from_comp: Component,
        to_comp: Component,
        dataset: SVGDataset
    ) -> list:
        """
        Detect components that obstruct the direct path.
        
        Args:
            start: Start point of path (exit point)
            end: End point of path (entry point)
            from_comp: Source component
            to_comp: Destination component
            dataset: SVG dataset containing all components
            
        Returns:
            List of components that obstruct the path
        """
        obstructions = []
        
        for comp in dataset.components:
            # Skip source and destination
            if comp.id == from_comp.id or comp.id == to_comp.id:
                continue
            
            # Get component bounding box
            bbox = comp.get_absolute_bbox(dataset)
            
            # Check if line segment intersects bounding box
            if self._line_segment_intersects_bbox(start, end, bbox):
                obstructions.append(comp)
        
        return obstructions
    
    def _choose_routing_direction(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        obstructions: list,
        dataset: SVGDataset,
        connection: Connection
    ) -> str:
        """
        Choose routing direction (left/right/top/bottom) to avoid obstructions.
        
        Args:
            start: Start point
            end: End point
            obstructions: List of obstructing components
            dataset: SVG dataset
            connection: Connection object (for style-based preferences)
            
        Returns:
            "left", "right", "top", or "bottom"
        """
        if not obstructions:
            # No obstructions, default to right for vertical, bottom for horizontal
            dx = abs(end[0] - start[0])
            dy = abs(end[1] - start[1])
            return "right" if dy > dx else "bottom"
        
        # Get routing configuration
        routing_config = self.config.get("routing", {})
        padding = routing_config.get("routing_padding", 10)
        
        # Calculate available space on each side
        min_x = min(obst.bbox.left for obst in obstructions)
        max_x = max(obst.bbox.right for obst in obstructions)
        min_y = min(obst.bbox.top for obst in obstructions)
        max_y = max(obst.bbox.bottom for obst in obstructions)
        
        # Calculate connection direction
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        is_vertical = abs(dy) > abs(dx)
        
        # For vertical connections, prefer left/right routing
        if is_vertical:
            left_space = min_x - padding
            right_space = dataset.width - max_x - padding
            
            # For training/loop connections, prefer outer edges (right)
            if connection.style == "training":
                return "right" if right_space >= padding else "left"
            
            # Choose side with more space
            return "right" if right_space >= left_space else "left"
        else:
            # For horizontal connections, prefer top/bottom routing
            top_space = min_y - padding
            bottom_space = dataset.height - max_y - padding
            
            # For training/loop connections, prefer outer edges (bottom)
            if connection.style == "training":
                return "bottom" if bottom_space >= padding else "top"
            
            # Choose side with more space
            return "bottom" if bottom_space >= top_space else "top"
    
    def _generate_manhattan_waypoints(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        obstructions: list,
        dataset: SVGDataset,
        curved: bool,
        connection: Connection
    ) -> List[Tuple[float, float]]:
        """
        Generate Manhattan path waypoints (without converting to SVG string).
        
        This is used to get waypoints for obstruction marking.
        
        Args:
            start: Start point (exit point)
            end: End point (entry point)
            obstructions: List of obstructing components
            dataset: SVG dataset
            curved: If True, use curved corners
            connection: Connection object
            
        Returns:
            List of (x, y) waypoints
        """
        if not obstructions:
            return [start, end]
        
        # Get routing configuration
        routing_config = self.config.get("routing", {})
        padding = routing_config.get("routing_padding", 10)
        corner_radius = routing_config.get("corner_radius", 5)
        
        # Choose routing direction
        direction = self._choose_routing_direction(start, end, obstructions, dataset, connection)
        
        # Generate waypoints based on direction
        waypoints = []
        waypoints.append(start)  # Start at exit point
        
        if direction == "right":
            # Route on right side - PROPER MANHATTAN ROUTING
            # Waypoints must be: horizontal → vertical → horizontal (no diagonals!)
            obstruction_max_x = max(obst.get_absolute_bbox(dataset).right for obst in obstructions) if obstructions else 0
            routing_x = max(obstruction_max_x + padding, start[0] + padding)
            
            # 1. Go right (horizontal segment, y stays constant)
            waypoints.append((routing_x, start[1]))
            
            # 2. Go down/up (vertical segment, x stays constant at routing_x)
            # Calculate intermediate y for horizontal alignment step (before final vertical to entry)
            # For proper Manhattan routing, we need an intermediate y that allows horizontal alignment
            if curved:
                corner_radius = routing_config.get("corner_radius", 5)
                # If going down (start[1] > end[1]), intermediate_y should be end[1] + corner_radius
                # If going up (start[1] < end[1]), intermediate_y should be end[1] - corner_radius
                if start[1] > end[1]:  # Going down
                    intermediate_y = end[1] + corner_radius
                else:  # Going up
                    intermediate_y = end[1] - corner_radius
            else:
                # For sharp corners, use entry y directly (no intermediate step needed)
                intermediate_y = end[1]
            waypoints.append((routing_x, intermediate_y))
            
            # 3. Steiner point: offset by arrowhead width from box edge (route OUT first)
            # Entry point is at box right edge, so Steiner point should be offset to the right
            # This routes the path OUT to avoid arrowhead overlap, then back to entry
            arrowhead_offset = self._get_arrowhead_offset()
            if curved:
                corner_radius = routing_config.get("corner_radius", 5)
                steiner_x = end[0] + arrowhead_offset + corner_radius
            else:
                steiner_x = end[0] + arrowhead_offset
            waypoints.append((steiner_x, intermediate_y))   # Horizontal segment OUT (y stays constant)
            
            # 4. Go left to align with entry side (horizontal segment, y stays constant at intermediate_y)
            # This ensures arrow start and end are at same x-coordinate
            waypoints.append((end[0], intermediate_y))
            
            # 5. Entry point (contact point at box edge)
            # Final vertical segment down/up to entry (x stays constant at entry x)
            waypoints.append(end)
        elif direction == "left":
            # Route on left side - PROPER MANHATTAN ROUTING
            obstruction_min_x = min(obst.get_absolute_bbox(dataset).left for obst in obstructions) if obstructions else dataset.width
            routing_x = min(obstruction_min_x - padding, start[0] - padding)
            
            # 1. Go left (horizontal segment, y stays constant)
            waypoints.append((routing_x, start[1]))
            
            # 2. Go down/up (vertical segment, x stays constant at routing_x)
            waypoints.append((routing_x, end[1]))
            
            # 3. Go right to align with entry side (horizontal segment, y stays constant)
            waypoints.append((end[0], end[1]))
            
            # 4. Steiner point before entry
            arrowhead_offset = self._get_arrowhead_offset()
            if curved:
                corner_radius = routing_config.get("corner_radius", 5)
                steiner_x = end[0] - arrowhead_offset - corner_radius
            else:
                steiner_x = end[0] - arrowhead_offset
            waypoints.append((steiner_x, end[1]))
            
            # 5. Entry point
            waypoints.append(end)
        elif direction == "top":
            # Route on top - PROPER MANHATTAN ROUTING
            obstruction_min_y = min(obst.get_absolute_bbox(dataset).top for obst in obstructions) if obstructions else 0
            routing_y = min(obstruction_min_y - padding, start[1] - padding)
            
            # 1. Go up (vertical segment, x stays constant)
            waypoints.append((start[0], routing_y))
            
            # 2. Go left/right (horizontal segment, y stays constant at routing_y)
            waypoints.append((end[0], routing_y))
            
            # 3. Go down to align with entry side (vertical segment, x stays constant)
            waypoints.append((end[0], end[1]))
            
            # 4. Steiner point before entry
            arrowhead_offset = self._get_arrowhead_offset()
            if curved:
                corner_radius = routing_config.get("corner_radius", 5)
                steiner_y = end[1] - arrowhead_offset - corner_radius
            else:
                steiner_y = end[1] - arrowhead_offset
            waypoints.append((end[0], steiner_y))
            
            # 5. Entry point
            waypoints.append(end)
        else:  # bottom
            # Route on bottom - PROPER MANHATTAN ROUTING
            obstruction_max_y = max(obst.get_absolute_bbox(dataset).bottom for obst in obstructions) if obstructions else dataset.height
            routing_y = max(obstruction_max_y + padding, start[1] + padding)
            
            # 1. Go down (vertical segment, x stays constant)
            waypoints.append((start[0], routing_y))
            
            # 2. Go left/right (horizontal segment, y stays constant at routing_y)
            waypoints.append((end[0], routing_y))
            
            # 3. Go up to align with entry side (vertical segment, x stays constant)
            waypoints.append((end[0], end[1]))
            
            # 4. Steiner point before entry
            arrowhead_offset = self._get_arrowhead_offset()
            if curved:
                corner_radius = routing_config.get("corner_radius", 5)
                steiner_y = end[1] + arrowhead_offset + corner_radius
            else:
                steiner_y = end[1] + arrowhead_offset
            waypoints.append((end[0], steiner_y))
            
            # 5. Entry point
            waypoints.append(end)
        
        return waypoints
    
    def _generate_manhattan_path(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        obstructions: list,
        dataset: SVGDataset,
        curved: bool,
        connection: Connection
    ) -> str:
        """
        Generate Manhattan path avoiding obstructions.
        
        Args:
            start: Start point (exit point)
            end: End point (entry point)
            obstructions: List of obstructing components
            dataset: SVG dataset
            curved: If True, use curved corners (Q commands), else sharp corners (L commands)
            connection: Connection object
            
        Returns:
            SVG path string (d attribute)
        """
        waypoints = self._generate_manhattan_waypoints(start, end, obstructions, dataset, curved, connection)
        return self._path_segments_to_svg(waypoints, curved)
    
    def _path_waypoints_to_svg(self, waypoints: List[Tuple[float, float]], curved: bool) -> str:
        """
        Convert waypoints to SVG path string.
        
        Args:
            waypoints: List of (x, y) waypoints from A* pathfinding
            curved: If True, use curved corners
            
        Returns:
            SVG path string (d attribute)
        """
        return self._path_segments_to_svg(waypoints, curved)
    
    def _path_segments_to_svg(self, waypoints: List[Tuple[float, float]], curved: bool) -> str:
        """
        Convert waypoints to SVG path string.
        
        Args:
            waypoints: List of (x, y) waypoints
            curved: If True, use curved corners
            
        Returns:
            SVG path string (d attribute)
        """
        if not waypoints:
            return ""
        
        routing_config = self.config.get("routing", {})
        corner_radius = routing_config.get("corner_radius", 5)
        
        if curved:
            # Use curves at corners (like original: M 170 445 L 180 445 Q 195 445 195 440 L 195 145 Q 195 140 180 140 L 170 140)
            # Q command format: Q control_x,control_y end_x,end_y
            path_parts = [f"M {waypoints[0][0]},{waypoints[0][1]}"]
            
            for i in range(1, len(waypoints)):
                prev = waypoints[i-1]
                curr = waypoints[i]
                
                if i < len(waypoints) - 1:
                    # Not the last point - check if we need a curve
                    next_pt = waypoints[i+1]
                    
                    # Determine segment directions
                    prev_dx = curr[0] - prev[0]
                    prev_dy = curr[1] - prev[1]
                    next_dx = next_pt[0] - curr[0]
                    next_dy = next_pt[1] - curr[1]
                    
                    # Check if we're turning (horizontal to vertical or vice versa)
                    is_horizontal = abs(prev_dx) > abs(prev_dy)
                    is_vertical_next = abs(next_dy) > abs(next_dx)
                    
                    if is_horizontal and is_vertical_next:
                        # Horizontal then vertical - curve at corner
                        # Line to near corner (before curve)
                        if prev_dx > 0:  # Going right
                            line_end_x = curr[0] - corner_radius
                        else:  # Going left
                            line_end_x = curr[0] + corner_radius
                        path_parts.append(f"L {line_end_x},{curr[1]}")
                        
                        # Curve: Q control_x,control_y end_x,end_y
                        # Control point is at the corner (curr)
                        # End point is on the vertical segment, corner_radius into it (toward next point)
                        if next_dy > 0:  # Next segment going down
                            curve_end_y = curr[1] + corner_radius
                        else:  # Next segment going up (next_dy < 0)
                            curve_end_y = curr[1] - corner_radius
                        # CRITICAL: For Manhattan routing, curve must end on vertical line (x = curr[0])
                        # The curve end should be corner_radius distance along the next segment
                        path_parts.append(f"Q {curr[0]},{curr[1]} {curr[0]},{curve_end_y}")
                    elif not is_horizontal and not is_vertical_next:
                        # Vertical then horizontal - curve at corner
                        # Line to near corner (before curve)
                        if prev_dy > 0:  # Going down
                            line_end_y = curr[1] - corner_radius
                        else:  # Going up
                            line_end_y = curr[1] + corner_radius
                        path_parts.append(f"L {curr[0]},{line_end_y}")
                        
                        # Curve
                        if next_dx > 0:  # Next segment going right
                            curve_end_x = curr[0] + corner_radius
                        else:  # Next segment going left
                            curve_end_x = curr[0] - corner_radius
                        path_parts.append(f"Q {curr[0]},{curr[1]} {curve_end_x},{curr[1]}")
                    else:
                        # No turn, just line
                        path_parts.append(f"L {curr[0]},{curr[1]}")
                else:
                    # Last point, just line to it
                    path_parts.append(f"L {curr[0]},{curr[1]}")
            
            return " ".join(path_parts)
        else:
            # Use sharp corners
            path_parts = [f"M {waypoints[0][0]},{waypoints[0][1]}"]
            for i in range(1, len(waypoints)):
                path_parts.append(f"L {waypoints[i][0]},{waypoints[i][1]}")
            return " ".join(path_parts)
    
    def _generate_label(self, label: Label) -> list:
        """Generate SVG for a label."""
        lines = []
        
        if not label.position:
            return lines
        
        typography = self.config.get("typography", {})
        sizes = typography.get("sizes", {})
        font_family = typography.get("font_family", "Arial, sans-serif")
        weights = typography.get("weights", {})
        colors = self.config.get("colors", {}).get("text", {})
        
        # Determine font size and class
        if label.type == "title":
            font_size = sizes.get("title", {}).get("large", 18)
            class_name = "title"
            fill = colors.get("primary", "#1e293b")
        elif label.type == "equation":
            font_size = sizes.get("equation", 12)
            class_name = "equation"
            fill = colors.get("muted", "#666")
        elif label.type == "note":
            font_size = sizes.get("note", 11)
            class_name = "note"
            fill = colors.get("muted", "#666")
        else:
            font_size = sizes.get("label", 11)
            class_name = "label"
            fill = colors.get("primary", "#1e293b")
        
        font_weight = weights.get(label.type, "normal")
        
        lines.append(f'  <text x="{label.position[0]}" y="{label.position[1]}" '
                    f'font-family="{font_family}" '
                    f'font-size="{font_size}" '
                    f'font-weight="{font_weight}" '
                    f'fill="{fill}" '
                    f'text-anchor="middle">{label.text}</text>')
        
        return lines

