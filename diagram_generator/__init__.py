"""
Diagram Generator - SVG diagram generation from JSON definitions.

A professional Python package for generating Inkscape-compliant SVG diagrams
from structured JSON definitions with automatic layout, routing, and labeling.
"""

from diagram_generator.core.generator import SVGGenerator
from diagram_generator.core.diagram import (
    SVGDataset,
    Component,
    Connection,
    Label,
    BoundingBox,
    Pin
)

__all__ = [
    'SVGGenerator',
    'SVGDataset',
    'Component',
    'Connection',
    'Label',
    'BoundingBox',
    'Pin',
]

__version__ = '1.0.0'

