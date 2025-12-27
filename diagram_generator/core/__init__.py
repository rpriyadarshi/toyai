"""
Core modules for diagram generation.

Contains data structures, generator, and component loading.
"""

from diagram_generator.core.diagram import (
    SVGDataset,
    Component,
    Connection,
    Label,
    BoundingBox,
    Pin
)
from diagram_generator.core.generator import SVGGenerator
from diagram_generator.core.component_loader import ComponentLoader

__all__ = [
    'SVGDataset',
    'Component',
    'Connection',
    'Label',
    'BoundingBox',
    'Pin',
    'SVGGenerator',
    'ComponentLoader',
]

