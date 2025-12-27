#!/usr/bin/env python3
"""
Component Loader - Load and manage component templates.

This module provides functionality to load component templates from the
component library and instantiate them with specific positions and configurations.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from diagram_generator.core.diagram import Component, Pin, BoundingBox


class ComponentLoader:
    """Load and manage component templates."""
    
    def __init__(self, components_dir: Optional[Path] = None, local_components_dir: Optional[Path] = None):
        """
        Initialize component loader.
        
        Args:
            components_dir: Directory containing shared component JSON files (default: book/diagrams/components)
            local_components_dir: Optional directory for diagram-specific local components
        """
        if components_dir is None:
            # Default to book/diagrams/components
            self.components_dir = Path(__file__).parent.parent.parent / "book" / "diagrams" / "components"
        else:
            self.components_dir = Path(components_dir)
        
        self.local_components_dir = Path(local_components_dir) if local_components_dir else None
        
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._load_all_templates()
    
    def _load_all_templates(self):
        """Load all component templates from components directory."""
        if not self.components_dir.exists():
            self.components_dir.mkdir(parents=True, exist_ok=True)
            return
        
        for json_file in self.components_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    template_data = json.load(f)
                    # Template file contains single template with key = filename
                    template_name = json_file.stem
                    self._templates[template_name] = template_data
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in template {json_file}: {e}")
            except FileNotFoundError as e:
                print(f"Warning: Template file not found {json_file}: {e}")
            except Exception as e:
                print(f"Warning: Failed to load template {json_file}: {e}")
    
    def get_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Get component template by name.
        
        Resolution strategy:
        1. Check shared library (book/config/components/)
        2. Check local components directory (if available)
        3. Return None if not found
        
        Args:
            template_name: Name of template
            
        Returns:
            Template dictionary or None if not found
        """
        # First check shared library (already loaded)
        if template_name in self._templates:
            return self._templates[template_name]
        
        # Then check local components directory (load on demand)
        if self.local_components_dir and self.local_components_dir.exists():
            local_path = self.local_components_dir / f"{template_name}.json"
            if local_path.exists():
                try:
                    with open(local_path, 'r') as f:
                        template_data = json.load(f)
                        # Cache it for future use
                        self._templates[template_name] = template_data
                        return template_data
                except Exception as e:
                    print(f"Warning: Failed to load local template {local_path}: {e}")
        
        return None
    
    def instantiate_component(
        self,
        component_id: str,
        component_type: str,
        template_name: str,
        position: tuple,
        config: Optional[Dict[str, Any]] = None
    ) -> Component:
        """
        Instantiate a component from a template.
        
        Args:
            component_id: Unique ID for component instance
            component_type: Type of component
            template_name: Name of template to use
            position: (x, y) position
            config: Optional configuration overrides
            
        Returns:
            Component instance
        """
        template = self.get_template(template_name)
        if not template:
            # Provide helpful error message
            shared_path = self.components_dir / f"{template_name}.json"
            local_path = self.local_components_dir / f"{template_name}.json" if self.local_components_dir else None
            error_msg = f"Template '{template_name}' not found"
            if shared_path.exists():
                error_msg += f" (file exists at {shared_path} but failed to load)"
            elif local_path and local_path.exists():
                error_msg += f" (file exists at {local_path} but failed to load)"
            else:
                error_msg += f" in shared library ({self.components_dir})"
                if self.local_components_dir:
                    error_msg += f" or local components ({self.local_components_dir})"
            raise ValueError(error_msg)
        
        visual = template.get("visual", {})
        structure = template.get("structure", {})
        pins_def = template.get("pins", {})
        
        # Get dimensions from visual or structure
        width = visual.get("width", structure.get("rect", {}).get("width", 100))
        height = visual.get("height", structure.get("rect", {}).get("height", 80))
        
        # Note: Auto-sizing is disabled to preserve spacing. Box dimensions from templates
        # should be designed to accommodate their text content. If text overflows, it's
        # a template design issue, not a runtime sizing issue.
        # 
        # Spacing between components is critical and must be maintained at 50px.
        # Auto-sizing would break spacing calculations and require position adjustments,
        # which would make the layout non-deterministic.
        
        # Apply custom text overrides if provided
        if config:
            # Override text content in structure
            structure = structure.copy()
            if "text" in structure:
                text_list = structure["text"].copy()
                for i, text_elem in enumerate(text_list):
                    if i == 0 and "label" in config:
                        text_elem = text_elem.copy()
                        text_elem["content"] = config["label"]
                        text_list[i] = text_elem
                    elif i == 1 and "subtext" in config:
                        text_elem = text_elem.copy()
                        # Handle multiline subtext (split by \n)
                        subtext = config["subtext"]
                        if subtext and ("\n" in subtext or "\\n" in subtext):
                            # For multiline, we'll use the first line here
                            # Additional lines will be added as separate text elements
                            # Handle both actual newlines and escaped \n
                            subtext_lines = subtext.replace("\\n", "\n").split("\n")
                            text_elem["content"] = subtext_lines[0]
                            text_list[i] = text_elem
                            # Add remaining lines as additional text elements
                            for line_idx, line in enumerate(subtext_lines[1:], start=2):
                                if line.strip():  # Only add non-empty lines
                                    line_elem = {
                                        "content": line,
                                        "x": text_elem.get("x", width / 2),
                                        "y": text_elem.get("y", height / 2) + 15 * line_idx,
                                        "class": "box-text"
                                    }
                                    text_list.append(line_elem)
                        else:
                            text_elem["content"] = subtext
                            text_list[i] = text_elem
                structure["text"] = text_list
            
            # Add subtext as new text element if not already present
            if "subtext" in config and config["subtext"]:
                if "text" not in structure:
                    structure["text"] = []
                text_list = structure["text"]
                # Check if subtext already exists (index 1)
                if len(text_list) < 2:
                    # Add subtext element
                    # Get label element for reference
                    label_elem = text_list[0] if text_list else {"x": width / 2, "y": height / 2, "class": "box-label"}
                    subtext_elem = {
                        "content": config["subtext"],
                        "x": label_elem.get("x", width / 2),
                        "y": label_elem.get("y", height / 2) + 15,  # 15px below label
                        "class": "box-text"
                    }
                    text_list.append(subtext_elem)
                    structure["text"] = text_list
            
            # Store custom_text in visual for later use (e.g., containers)
            visual = visual.copy()
            visual["custom_text"] = config
        
        # Recalculate text positions to center them vertically within the box
        # This ensures proper alignment regardless of component height
        # Uses bounding box calculations to ensure all text fits within box boundaries
        if "text" in structure and structure["text"]:
            text_list = structure["text"]
            
            # Font metrics
            label_font_size = 14  # box-label
            subtext_font_size = 11  # box-text
            label_ascent = label_font_size * 0.8  # Baseline to top
            subtext_ascent = subtext_font_size * 0.8
            label_descent = label_font_size * 0.2  # Baseline to bottom
            subtext_descent = subtext_font_size * 0.2
            
            # Line spacing: 1 line height of caption (subtext font size)
            line_spacing = subtext_font_size
            
            # Count non-empty text elements
            text_elements = []
            for t in text_list:
                content = t.get("content", "")
                if content and str(content).strip():
                    text_elements.append(t)
            num_text_lines = len(text_elements)
            
            if num_text_lines == 0:
                # No text, nothing to do
                pass
            elif num_text_lines == 1:
                # Single text element (label only) - center it vertically
                text_elem = text_elements[0]
                text_elem["x"] = width / 2  # Center horizontally
                
                # Calculate text block height (just the label)
                total_text_height = label_font_size
                
                # Center the text block vertically
                box_center_y = height / 2
                text_block_top = box_center_y - (total_text_height / 2)
                
                # Position label baseline: text_block_top + label_ascent
                text_elem["y"] = text_block_top + label_ascent
            else:
                # Multiple text elements (label + subtext, possibly more)
                # Calculate total height of text block
                total_text_height = label_font_size  # Label height
                
                # Add spacing and height for each additional line
                for i in range(1, num_text_lines):
                    total_text_height += line_spacing  # Spacing before this line
                    # Determine font size for this line
                    if text_elements[i].get("class") == "box-label":
                        total_text_height += label_font_size
                    else:
                        total_text_height += subtext_font_size
                
                # Center the text block vertically within the box
                box_center_y = height / 2
                text_block_top = box_center_y - (total_text_height / 2)
                
                # Position each text element relative to the centered block
                # Build a mapping from text_elements to their indices in text_list
                text_elem_to_index = {}
                for idx, t in enumerate(text_list):
                    content = t.get("content", "")
                    if content and str(content).strip():
                        # Find which text_element this corresponds to
                        for j, te in enumerate(text_elements):
                            if te is t:  # Same object reference
                                text_elem_to_index[j] = idx
                                break
                
                current_y = text_block_top
                for i, text_elem in enumerate(text_elements):
                    # Determine font size and ascent for this element
                    if text_elem.get("class") == "box-label":
                        font_size = label_font_size
                        ascent = label_ascent
                    else:
                        font_size = subtext_font_size
                        ascent = subtext_ascent
                    
                    # Position baseline: current_y + ascent
                    text_elem["x"] = width / 2  # Center horizontally
                    text_elem["y"] = current_y + ascent
                    
                    # Also update in text_list if we have the mapping
                    if i in text_elem_to_index:
                        list_idx = text_elem_to_index[i]
                        text_list[list_idx]["x"] = width / 2
                        text_list[list_idx]["y"] = current_y + ascent
                    
                    # Move to next line: current position + font_size + spacing
                    current_y += font_size
                    if i < num_text_lines - 1:
                        current_y += line_spacing
            
            # Update all text elements in text_list (including empty ones) with x positions
            # Empty text elements keep their original y positions but get centered x
            for t in text_list:
                t["x"] = width / 2  # Center horizontally
            
            # Update structure with recalculated positions
            structure["text"] = text_list
        
        # Create component
        component = Component(
            id=component_id,
            type=component_type,
            template=template_name,
            position=position,
            width=width,
            height=height,
            visual=visual,
            structure=structure
        )
        
        # Create pins
        for pin_id, pin_data in pins_def.items():
            component.pins[pin_id] = Pin(
                id=pin_id,
                side=pin_data.get("side", "left"),
                position=pin_data.get("position", "middle"),
                offset_x=pin_data.get("offset", {}).get("x", 0),
                offset_y=pin_data.get("offset", {}).get("y", 0)
            )
        
        return component
    
    def list_templates(self) -> list:
        """List all available template names."""
        return list(self._templates.keys())

