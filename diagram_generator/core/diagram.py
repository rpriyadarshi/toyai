#!/usr/bin/env python3
"""
SVG Diagram Data Structure - Manipulable representation of SVG diagrams.

This module provides data structures for representing SVG diagrams in a
manipulable format, enabling programmatic generation, modification, and
regeneration of diagrams.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import json


@dataclass
class BoundingBox:
    """Represents a bounding box."""
    x: float
    y: float
    width: float
    height: float
    
    @property
    def left(self) -> float:
        return self.x
    
    @property
    def right(self) -> float:
        return self.x + self.width
    
    @property
    def top(self) -> float:
        return self.y
    
    @property
    def bottom(self) -> float:
        return self.y + self.height
    
    @property
    def center_x(self) -> float:
        return self.x + self.width / 2
    
    @property
    def center_y(self) -> float:
        return self.y + self.height / 2
    
    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if this bounding box intersects with another."""
        return not (self.right < other.left or
                   self.left > other.right or
                   self.bottom < other.top or
                   self.top > other.bottom)


@dataclass
class Pin:
    """Represents a connection point on a component."""
    id: str
    side: str  # "left", "right", "top", "bottom"
    position: str  # "top", "middle", "bottom", "left", "right", "center"
    offset_x: float = 0
    offset_y: float = 0
    
    def calculate_position(self, component_bbox: BoundingBox) -> Tuple[float, float]:
        """Calculate absolute position of pin based on component bounding box."""
        if self.side == "left":
            x = component_bbox.left + self.offset_x
            if self.position == "top":
                y = component_bbox.top + self.offset_y
            elif self.position == "middle" or self.position == "center":
                y = component_bbox.center_y + self.offset_y
            elif self.position == "bottom":
                y = component_bbox.bottom + self.offset_y
            else:
                y = component_bbox.center_y + self.offset_y
        elif self.side == "right":
            x = component_bbox.right + self.offset_x
            if self.position == "top":
                y = component_bbox.top + self.offset_y
            elif self.position == "middle" or self.position == "center":
                y = component_bbox.center_y + self.offset_y
            elif self.position == "bottom":
                y = component_bbox.bottom + self.offset_y
            else:
                y = component_bbox.center_y + self.offset_y
        elif self.side == "top":
            y = component_bbox.top + self.offset_y
            if self.position == "left":
                x = component_bbox.left + self.offset_x
            elif self.position == "middle" or self.position == "center":
                x = component_bbox.center_x + self.offset_x
            elif self.position == "right":
                x = component_bbox.right + self.offset_x
            else:
                x = component_bbox.center_x + self.offset_x
        elif self.side == "bottom":
            y = component_bbox.bottom + self.offset_y
            if self.position == "left":
                x = component_bbox.left + self.offset_x
            elif self.position == "middle" or self.position == "center":
                x = component_bbox.center_x + self.offset_x
            elif self.position == "right":
                x = component_bbox.right + self.offset_x
            else:
                x = component_bbox.center_x + self.offset_x
        else:
            # Default to center
            x = component_bbox.center_x + self.offset_x
            y = component_bbox.center_y + self.offset_y
        
        return (x, y)


@dataclass
class Component:
    """Represents a component in the diagram."""
    id: str
    type: str
    template: str
    position: Tuple[float, float]  # (x, y) - RELATIVE if parent_id is set, ABSOLUTE if None
    width: float
    height: float
    parent_id: Optional[str] = None  # ID of parent container, None if top-level
    pins: Dict[str, Pin] = field(default_factory=dict)
    visual: Dict[str, Any] = field(default_factory=dict)
    structure: Dict[str, Any] = field(default_factory=dict)
    
    def get_absolute_position(self, dataset: 'SVGDataset') -> Tuple[float, float]:
        """
        Get absolute position, accounting for parent transforms.
        
        Args:
            dataset: SVGDataset containing this component and its parents
            
        Returns:
            Absolute (x, y) position
        """
        if self.parent_id is None:
            return self.position
        
        parent = dataset.get_container(self.parent_id)
        if parent is None:
            # Parent not found, return position as-is (assume absolute)
            return self.position
        
        # Recursively get parent's absolute position
        parent_abs = parent.get_absolute_position(dataset)
        return (parent_abs[0] + self.position[0], parent_abs[1] + self.position[1])
    
    def get_absolute_bbox(self, dataset: 'SVGDataset') -> BoundingBox:
        """
        Get bounding box in absolute coordinates.
        
        Args:
            dataset: SVGDataset containing this component and its parents
            
        Returns:
            BoundingBox in absolute coordinates
        """
        abs_pos = self.get_absolute_position(dataset)
        return BoundingBox(abs_pos[0], abs_pos[1], self.width, self.height)
    
    @property
    def bbox(self) -> BoundingBox:
        """
        Get bounding box of component (relative coordinates).
        
        Note: For absolute bbox, use get_absolute_bbox(dataset) instead.
        """
        return BoundingBox(self.position[0], self.position[1], self.width, self.height)
    
    def get_pin(self, pin_id: str) -> Optional[Pin]:
        """Get pin by ID."""
        return self.pins.get(pin_id)
    
    def get_pin_position(self, pin_id: str, dataset: 'SVGDataset') -> Optional[Tuple[float, float]]:
        """
        Get absolute position of a pin.
        
        Args:
            pin_id: ID of the pin
            dataset: SVGDataset for resolving absolute position
            
        Returns:
            Absolute (x, y) position of pin, or None if pin not found
        """
        pin = self.get_pin(pin_id)
        if pin:
            abs_bbox = self.get_absolute_bbox(dataset)
            return pin.calculate_position(abs_bbox)
        return None


@dataclass
class Connection:
    """Represents a connection between components."""
    from_component_id: str
    from_pin_id: str
    to_component_id: str
    to_pin_id: str
    style: str = "forward"  # "forward", "backward", "dashed", "training"
    routing: str = "polyline"  # "direct", "orthogonal", "polyline" (matches Inkscape connector types)
    color: Optional[str] = None
    stroke_width: float = 2.0
    container_id: Optional[str] = None  # If set, connection belongs to this container
    label: Optional[str] = None  # Optional label text to display along the connection
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Label:
    """Represents a label in the diagram."""
    type: str  # "title", "equation", "axis", "element", "note"
    text: str
    priority: int
    position: Optional[Tuple[float, float]] = None
    target_component_id: Optional[str] = None
    target_position: Optional[Tuple[float, float]] = None
    container_id: Optional[str] = None  # If set, label belongs to this container
    style: Dict[str, Any] = field(default_factory=dict)
    bbox: Optional[BoundingBox] = None


@dataclass
class SVGDataset:
    """Main data structure representing an SVG diagram."""
    width: float
    height: float
    components: List[Component] = field(default_factory=list)
    containers: List[Component] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)
    labels: List[Label] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_component(self, component: Component):
        """Add a component to the dataset."""
        self.components.append(component)
    
    def get_component(self, component_id: str) -> Optional[Component]:
        """Get component by ID."""
        for comp in self.components:
            if comp.id == component_id:
                return comp
        return None
    
    def add_connection(self, connection: Connection):
        """Add a connection to the dataset."""
        self.connections.append(connection)
    
    def add_label(self, label: Label):
        """Add a label to the dataset."""
        self.labels.append(label)
    
    def add_container(self, container: Component):
        """Add a container to the dataset."""
        self.containers.append(container)
    
    def get_container(self, container_id: str) -> Optional[Component]:
        """Get container by ID."""
        for container in self.containers:
            if container.id == container_id:
                return container
        return None
    
    def get_children(self, container_id: str) -> List[Component]:
        """Get all child components of a container."""
        return [comp for comp in self.components if comp.parent_id == container_id]
    
    def convert_absolute_to_relative(self, diagram_json: Dict[str, Any]):
        """
        Convert all child component positions from absolute to relative.
        
        Uses the "contains" field from diagram JSON to identify parent-child relationships.
        After conversion, child components will have:
        - parent_id set to their container's ID
        - position converted from absolute to relative (relative to container)
        
        Args:
            diagram_json: Diagram JSON definition with "containers" field
        """
        for container_def in diagram_json.get("containers", []):
            container = self.get_container(container_def["id"])
            if not container:
                continue
            
            # Container position remains absolute (top-level)
            # Convert children to relative positions
            for child_id in container_def.get("contains", []):
                child = self.get_component(child_id)
                if child:
                    # Convert absolute position to relative
                    child.position = (
                        child.position[0] - container.position[0],
                        child.position[1] - container.position[1]
                    )
                    child.parent_id = container.id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": {
                "width": self.width,
                "height": self.height,
                **self.metadata
            },
            "components": [
                {
                    "id": comp.id,
                    "type": comp.type,
                    "template": comp.template,
                    "position": {"x": comp.position[0], "y": comp.position[1]},
                    "width": comp.width,
                    "height": comp.height,
                    "pins": {
                        pin_id: {
                            "id": pin.id,
                            "side": pin.side,
                            "position": pin.position,
                            "offset_x": pin.offset_x,
                            "offset_y": pin.offset_y
                        }
                        for pin_id, pin in comp.pins.items()
                    }
                }
                for comp in self.components
            ],
            "connections": [
                {
                    "from": f"{conn.from_component_id}.{conn.from_pin_id}",
                    "to": f"{conn.to_component_id}.{conn.to_pin_id}",
                    "style": conn.style,
                    "color": conn.color,
                    "stroke_width": conn.stroke_width
                }
                for conn in self.connections
            ],
            "labels": [
                {
                    "type": label.type,
                    "text": label.text,
                    "priority": label.priority,
                    "position": {"x": label.position[0], "y": label.position[1]} if label.position else None,
                    "target": label.target_component_id,
                    "target_position": {"x": label.target_position[0], "y": label.target_position[1]} if label.target_position else None,
                    "container_id": label.container_id
                }
                for label in self.labels
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SVGDataset':
        """Create SVGDataset from dictionary."""
        metadata = data.get("metadata", {})
        dataset = cls(
            width=metadata.get("width", 800),
            height=metadata.get("height", 600),
            metadata=metadata
        )
        
        # Load components
        for comp_data in data.get("components", []):
            comp = Component(
                id=comp_data["id"],
                type=comp_data["type"],
                template=comp_data["template"],
                position=(comp_data["position"]["x"], comp_data["position"]["y"]),
                width=comp_data["width"],
                height=comp_data["height"]
            )
            
            # Load pins
            for pin_id, pin_data in comp_data.get("pins", {}).items():
                comp.pins[pin_id] = Pin(
                    id=pin_id,
                    side=pin_data["side"],
                    position=pin_data["position"],
                    offset_x=pin_data.get("offset_x", 0),
                    offset_y=pin_data.get("offset_y", 0)
                )
            
            dataset.add_component(comp)
        
        # Load connections
        for conn_data in data.get("connections", []):
            from_parts = conn_data["from"].split(".")
            to_parts = conn_data["to"].split(".")
            conn = Connection(
                from_component_id=from_parts[0],
                from_pin_id=from_parts[1] if len(from_parts) > 1 else "output",
                to_component_id=to_parts[0],
                to_pin_id=to_parts[1] if len(to_parts) > 1 else "input",
                style=conn_data.get("style", "forward"),
                color=conn_data.get("color"),
                stroke_width=conn_data.get("stroke_width", 2.0)
            )
            dataset.add_connection(conn)
        
        # Load labels
        for label_data in data.get("labels", []):
            label = Label(
                type=label_data["type"],
                text=label_data["text"],
                priority=label_data.get("priority", 4),
                position=(
                    (label_data["position"]["x"], label_data["position"]["y"])
                    if label_data.get("position") else None
                ),
                target_component_id=label_data.get("target"),
                target_position=(
                    (label_data["target_position"]["x"], label_data["target_position"]["y"])
                    if label_data.get("target_position") else None
                )
            )
            dataset.add_label(label)
        
        return dataset

