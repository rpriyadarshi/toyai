#!/usr/bin/env python3
"""
Bounding Box Calculator - Calculate text bounding boxes with font metrics.

This module provides functions to calculate bounding boxes for text elements,
accounting for font family, size, weight, and multi-line text.
"""

from typing import Dict, Any, Tuple

from diagram_generator.core.diagram import BoundingBox


def estimate_text_width(text: str, font_size: float, font_family: str = "Arial") -> float:
    """
    Estimate text width based on font size and character count.
    
    This is an approximation. For accurate measurements, would need
    actual font metrics, but this works reasonably well for most cases.
    
    Args:
        text: Text string
        font_size: Font size in pixels
        font_family: Font family (for future font metric lookup)
        
    Returns:
        Estimated width in pixels
    """
    # Approximate character width: font_size * 0.6 for most fonts
    # This is a rough estimate; actual width depends on font metrics
    char_width = font_size * 0.6
    return len(text) * char_width


def estimate_text_height(font_size: float, line_count: int = 1, line_spacing: float = 1.2) -> float:
    """
    Estimate text height based on font size and line count.
    
    Args:
        font_size: Font size in pixels
        line_count: Number of lines
        line_spacing: Line spacing multiplier
        
    Returns:
        Estimated height in pixels
    """
    if line_count == 1:
        return font_size * 1.2  # Add some padding
    else:
        return font_size * line_count * line_spacing


def calculate_text_bbox(
    text: str,
    x: float,
    y: float,
    font_size: float,
    font_family: str = "Arial",
    font_weight: str = "normal",
    text_anchor: str = "middle",
    line_count: int = 1
) -> BoundingBox:
    """
    Calculate bounding box for text element.
    
    Args:
        text: Text string
        x: X position (anchor point)
        y: Y position (baseline)
        font_size: Font size in pixels
        font_family: Font family
        font_weight: Font weight
        text_anchor: Text anchor ("start", "middle", "end")
        line_count: Number of lines
        
    Returns:
        Bounding box for text
    """
    width = estimate_text_width(text, font_size, font_family)
    height = estimate_text_height(font_size, line_count)
    
    # Adjust x based on text anchor
    if text_anchor == "middle":
        bbox_x = x - width / 2
    elif text_anchor == "end":
        bbox_x = x - width
    else:  # "start"
        bbox_x = x
    
    # Adjust y to top of text (y is baseline, need to account for ascent)
    # Most fonts have baseline at ~80% from top
    bbox_y = y - font_size * 0.8
    
    return BoundingBox(bbox_x, bbox_y, width, height)


def calculate_label_bbox(label: Dict[str, Any], config: Dict[str, Any]) -> BoundingBox:
    """
    Calculate bounding box for a label from its definition.
    
    Args:
        label: Label definition dictionary
        config: Diagram configuration
        
    Returns:
        Bounding box for label
    """
    label_type = label.get("type", "element")
    text = label.get("text", "")
    
    # Get font size from config
    typography = config.get("typography", {})
    sizes = typography.get("sizes", {})
    
    if label_type == "title":
        # Determine size based on diagram size
        diagram_size = label.get("diagram_size", "medium")
        font_size = sizes.get("title", {}).get(diagram_size, 16)
    elif label_type == "equation":
        font_size = sizes.get("equation", 12)
    elif label_type == "axis":
        font_size = sizes.get("axis", 12)
    elif label_type == "note":
        font_size = sizes.get("note", 11)
    else:  # element
        font_size = sizes.get("label", 11)
    
    # Get font family
    font_family = typography.get("font_family", "Arial, sans-serif")
    
    # Get font weight
    weights = typography.get("weights", {})
    font_weight = weights.get(label_type, "normal")
    
    # Estimate line count (simple: count newlines + 1)
    line_count = text.count("\n") + 1
    
    # For now, use default position (will be set by label placer)
    x = label.get("position", {}).get("x", 0) if isinstance(label.get("position"), dict) else 0
    y = label.get("position", {}).get("y", 0) if isinstance(label.get("position"), dict) else 0
    
    return calculate_text_bbox(
        text, x, y, font_size, font_family, font_weight,
        text_anchor="middle", line_count=line_count
    )

