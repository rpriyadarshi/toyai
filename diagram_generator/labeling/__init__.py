"""
Labeling modules for text and label placement.

Contains label placement algorithms and bounding box calculations.
"""

from diagram_generator.labeling.placer import LabelPlacer
from diagram_generator.labeling.bbox import (
    calculate_label_bbox,
    calculate_text_bbox,
    estimate_text_width,
    estimate_text_height,
)

__all__ = [
    'LabelPlacer',
    'calculate_label_bbox',
    'calculate_text_bbox',
    'estimate_text_width',
    'estimate_text_height',
]

