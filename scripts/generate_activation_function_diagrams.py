#!/usr/bin/env python3
"""
Generate Activation Function Diagrams

Creates SVG diagrams for the "How Activation Functions Enable Complex Pattern Learning" section.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
from matplotlib.patches import ConnectionPatch
from pathlib import Path
import matplotlib.patches as mpatches

def generate_linear_layers_collapse(output_path):
    """Generate diagram showing how two linear layers collapse into one."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Layer 1
    layer1 = FancyBboxPatch((0.5, 2), 2, 2, boxstyle="round,pad=0.1", 
                           facecolor='#e1f5ff', edgecolor='#2563eb', linewidth=2)
    ax.add_patch(layer1)
    ax.text(1.5, 3, 'Layer 1\n(Linear)', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(1.5, 1.5, r'$\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$', ha='center', fontsize=10)
    
    # Arrow 1
    arrow1 = FancyArrowPatch((2.5, 3), (3.5, 3), arrowstyle='->', 
                             mutation_scale=20, color='#2563eb', linewidth=2)
    ax.add_patch(arrow1)
    
    # Layer 2
    layer2 = FancyBboxPatch((3.5, 2), 2, 2, boxstyle="round,pad=0.1", 
                           facecolor='#e1f5ff', edgecolor='#2563eb', linewidth=2)
    ax.add_patch(layer2)
    ax.text(4.5, 3, 'Layer 2\n(Linear)', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(4.5, 1.5, r'$\mathbf{W}_2 \mathbf{y}_1 + \mathbf{b}_2$', ha='center', fontsize=10)
    
    # Equals sign
    ax.text(6, 3, '=', ha='center', va='center', fontsize=24, fontweight='bold')
    
    # Combined layer
    combined = FancyBboxPatch((6.5, 2), 2, 2, boxstyle="round,pad=0.1", 
                             facecolor='#fff4e1', edgecolor='#f59e0b', linewidth=2)
    ax.add_patch(combined)
    ax.text(7.5, 3, 'Single Layer\n(Equivalent)', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='#92400e')
    ax.text(7.5, 1.5, r'$\mathbf{W}_{c} \mathbf{x} + \mathbf{b}_{c}$', 
           ha='center', fontsize=10, color='#92400e')
    ax.text(7.5, 1.2, '(combined)', ha='center', fontsize=8, style='italic', color='#92400e')
    
    # Collapse arrow
    collapse_arrow = FancyArrowPatch((5.5, 4.5), (7.5, 4.5), 
                                     arrowstyle='->', mutation_scale=20, 
                                     color='#dc2626', linewidth=2, linestyle='--')
    ax.add_patch(collapse_arrow)
    ax.text(6.5, 4.8, 'Collapse!', ha='center', fontsize=11, fontweight='bold', color='#dc2626')
    
    # Input
    ax.text(0.5, 4.5, 'Input $\\mathbf{x}$', ha='left', fontsize=11, fontweight='bold')
    
    # Output
    ax.text(8.5, 4.5, 'Output $\\mathbf{y}_2$', ha='left', fontsize=11, fontweight='bold')
    
    ax.set_title('Linear Layers Collapse: Two layers = One layer', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=100)
    plt.close()
    print(f"Generated: {output_path}")

def generate_nonlinear_prevents_collapse(output_path):
    """Generate diagram showing how non-linearity prevents collapse."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Layer 1 with activation
    layer1 = FancyBboxPatch((0.5, 2), 2, 2, boxstyle="round,pad=0.1", 
                           facecolor='#e1f5ff', edgecolor='#2563eb', linewidth=2)
    ax.add_patch(layer1)
    ax.text(1.5, 3.3, 'Layer 1', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(1.5, 2.7, r'$f(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$', ha='center', fontsize=10)
    ax.text(1.5, 2.3, '(Non-linear)', ha='center', fontsize=9, style='italic', color='#16a34a')
    
    # Activation symbol
    activation1 = Circle((1.5, 1.5), 0.3, facecolor='#16a34a', edgecolor='#15803d', linewidth=2)
    ax.add_patch(activation1)
    ax.text(1.5, 1.5, 'f', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Arrow 1
    arrow1 = FancyArrowPatch((2.5, 3), (3.5, 3), arrowstyle='->', 
                         mutation_scale=20, color='#2563eb', linewidth=2)
    ax.add_patch(arrow1)
    
    # Layer 2 with activation
    layer2 = FancyBboxPatch((3.5, 2), 2, 2, boxstyle="round,pad=0.1", 
                           facecolor='#e1f5ff', edgecolor='#2563eb', linewidth=2)
    ax.add_patch(layer2)
    ax.text(4.5, 3.3, 'Layer 2', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(4.5, 2.7, r'$f(\mathbf{W}_2 \mathbf{y}_1 + \mathbf{b}_2)$', ha='center', fontsize=10)
    ax.text(4.5, 2.3, '(Non-linear)', ha='center', fontsize=9, style='italic', color='#16a34a')
    
    # Activation symbol
    activation2 = Circle((4.5, 1.5), 0.3, facecolor='#16a34a', edgecolor='#15803d', linewidth=2)
    ax.add_patch(activation2)
    ax.text(4.5, 1.5, 'f', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Cannot simplify
    ax.text(6, 3, '≠', ha='center', va='center', fontsize=24, fontweight='bold', color='#dc2626')
    
    # Single layer (crossed out)
    single = FancyBboxPatch((6.5, 2), 2, 2, boxstyle="round,pad=0.1", 
                           facecolor='#fee2e2', edgecolor='#dc2626', linewidth=2, linestyle='--')
    ax.add_patch(single)
    ax.plot([6.5, 8.5], [4, 2], 'r-', linewidth=3)
    ax.text(7.5, 3, 'Cannot\nSimplify', ha='center', va='center', 
           fontsize=11, fontweight='bold', color='#dc2626')
    
    # Non-linearity breaks composition
    ax.text(6, 0.8, 'Non-linearity breaks composition!', ha='center', 
           fontsize=12, fontweight='bold', color='#16a34a')
    
    # Input
    ax.text(0.5, 4.5, 'Input $\\mathbf{x}$', ha='left', fontsize=11, fontweight='bold')
    
    # Output
    ax.text(8.5, 4.5, 'Output $\\mathbf{y}_2$', ha='left', fontsize=11, fontweight='bold')
    
    ax.set_title('Non-Linear Activation Prevents Collapse', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=100)
    plt.close()
    print(f"Generated: {output_path}")

def generate_hierarchical_pattern_learning(output_path):
    """Generate diagram showing hierarchical pattern learning."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Layer 1 - Simple patterns
    layer1_box = FancyBboxPatch((1, 5.5), 3, 1.5, boxstyle="round,pad=0.1", 
                               facecolor='#e1f5ff', edgecolor='#2563eb', linewidth=2)
    ax.add_patch(layer1_box)
    ax.text(2.5, 6.5, 'Layer 1: Simple Patterns', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    
    # Simple pattern examples
    patterns1 = ['Edges', 'Lines', 'Bright pixels']
    for i, p in enumerate(patterns1):
        ax.text(2.5, 6.0 - i*0.25, f'• {p}', ha='center', fontsize=10)
    
    # Arrow down
    arrow1 = FancyArrowPatch((2.5, 5.5), (2.5, 4.5), arrowstyle='->', 
                             mutation_scale=20, color='#2563eb', linewidth=2)
    ax.add_patch(arrow1)
    
    # Layer 2 - Combined patterns
    layer2_box = FancyBboxPatch((5.5, 3), 3, 1.5, boxstyle="round,pad=0.1", 
                               facecolor='#fff4e1', edgecolor='#f59e0b', linewidth=2)
    ax.add_patch(layer2_box)
    ax.text(7, 4, 'Layer 2: Combined Patterns', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    
    # Combined pattern examples
    patterns2 = ['Corners', 'Shapes', 'Face parts']
    for i, p in enumerate(patterns2):
        ax.text(7, 3.5 - i*0.25, f'• {p}', ha='center', fontsize=10)
    
    # Arrow down
    arrow2 = FancyArrowPatch((7, 3), (7, 2), arrowstyle='->', 
                             mutation_scale=20, color='#f59e0b', linewidth=2)
    ax.add_patch(arrow2)
    
    # Layer 3 - Complex patterns
    layer3_box = FancyBboxPatch((10, 0.5), 3, 1.5, boxstyle="round,pad=0.1", 
                               facecolor='#c8e6c9', edgecolor='#16a34a', linewidth=2)
    ax.add_patch(layer3_box)
    ax.text(11.5, 1.5, 'Layer 3: Complex Patterns', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    
    # Complex pattern examples
    patterns3 = ['Complete faces', 'Sentences', 'Objects']
    for i, p in enumerate(patterns3):
        ax.text(11.5, 1.0 - i*0.25, f'• {p}', ha='center', fontsize=10)
    
    # Progression arrow
    progression = FancyArrowPatch((4, 6.25), (9.5, 1.75), arrowstyle='->', 
                                 mutation_scale=20, color='#7c3aed', linewidth=3, 
                                 linestyle='--', alpha=0.6)
    ax.add_patch(progression)
    ax.text(6.75, 4.5, 'Increasing Complexity', ha='center', fontsize=11, 
           fontweight='bold', color='#7c3aed', rotation=-35)
    
    ax.set_title('Hierarchical Pattern Learning: Simple → Complex', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=100)
    plt.close()
    print(f"Generated: {output_path}")

def generate_digit_recognition_layers(output_path):
    """Generate diagram showing digit recognition through layers."""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Input - digit 6
    input_box = FancyBboxPatch((0.5, 2), 2, 2, boxstyle="round,pad=0.1", 
                              facecolor='#f3f4f6', edgecolor='#6b7280', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 3.5, 'Input', ha='center', va='center', fontsize=12, fontweight='bold')
    # Simple representation of digit 6
    ax.text(1.5, 2.8, '6', ha='center', va='center', fontsize=48, fontweight='bold')
    ax.text(1.5, 1.5, 'Pixel values', ha='center', fontsize=9, style='italic')
    
    # Arrow
    arrow1 = FancyArrowPatch((2.5, 3), (3.5, 3), arrowstyle='->', 
                             mutation_scale=20, color='#2563eb', linewidth=2)
    ax.add_patch(arrow1)
    
    # Layer 1
    layer1 = FancyBboxPatch((3.5, 1.5), 2.5, 3, boxstyle="round,pad=0.1", 
                           facecolor='#e1f5ff', edgecolor='#2563eb', linewidth=2)
    ax.add_patch(layer1)
    ax.text(4.75, 4, 'Layer 1', ha='center', fontsize=11, fontweight='bold')
    ax.text(4.75, 3.5, 'Detects:', ha='center', fontsize=10)
    ax.text(4.75, 3.1, '• Vertical line', ha='center', fontsize=9)
    ax.text(4.75, 2.8, '• Curve', ha='center', fontsize=9)
    ax.text(4.75, 2.5, '• Horizontal line', ha='center', fontsize=9)
    ax.text(4.75, 2, 'Output: [0.8, 0.7, 0.1]', ha='center', fontsize=9, 
           style='italic', color='#1e40af')
    
    # Arrow
    arrow2 = FancyArrowPatch((6, 3), (7, 3), arrowstyle='->', 
                             mutation_scale=20, color='#f59e0b', linewidth=2)
    ax.add_patch(arrow2)
    
    # Layer 2
    layer2 = FancyBboxPatch((7, 1.5), 2.5, 3, boxstyle="round,pad=0.1", 
                           facecolor='#fff4e1', edgecolor='#f59e0b', linewidth=2)
    ax.add_patch(layer2)
    ax.text(8.25, 4, 'Layer 2', ha='center', fontsize=11, fontweight='bold')
    ax.text(8.25, 3.5, 'Combines:', ha='center', fontsize=10)
    ax.text(8.25, 3.1, '• Vertical + Curve', ha='center', fontsize=9)
    ax.text(8.25, 2.8, '= Top of "6"', ha='center', fontsize=9, style='italic')
    ax.text(8.25, 2, 'Output: [0.9, ...]', ha='center', fontsize=9, 
           style='italic', color='#92400e')
    
    # Arrow
    arrow3 = FancyArrowPatch((9.5, 3), (10.5, 3), arrowstyle='->', 
                             mutation_scale=20, color='#16a34a', linewidth=2)
    ax.add_patch(arrow3)
    
    # Layer 3
    layer3 = FancyBboxPatch((10.5, 1.5), 2.5, 3, boxstyle="round,pad=0.1", 
                           facecolor='#c8e6c9', edgecolor='#16a34a', linewidth=2)
    ax.add_patch(layer3)
    ax.text(11.75, 4, 'Layer 3', ha='center', fontsize=11, fontweight='bold')
    ax.text(11.75, 3.5, 'Recognizes:', ha='center', fontsize=10)
    ax.text(11.75, 3.1, '• Complete "6"', ha='center', fontsize=9)
    ax.text(11.75, 2.8, 'pattern', ha='center', fontsize=9, style='italic')
    ax.text(11.75, 2, 'Output: [0.95, ...]', ha='center', fontsize=9, 
           style='italic', color='#166534')
    
    # Arrow
    arrow4 = FancyArrowPatch((13, 3), (14, 3), arrowstyle='->', 
                             mutation_scale=20, color='#7c3aed', linewidth=2)
    ax.add_patch(arrow4)
    
    # Output
    output_box = FancyBboxPatch((14, 2), 1.5, 2, boxstyle="round,pad=0.1", 
                               facecolor='#f3e8ff', edgecolor='#7c3aed', linewidth=2)
    ax.add_patch(output_box)
    ax.text(14.75, 3.5, 'Output', ha='center', fontsize=11, fontweight='bold')
    ax.text(14.75, 3, 'Digit 6', ha='center', fontsize=10, color='#7c3aed')
    ax.text(14.75, 2.5, '90%', ha='center', fontsize=10, fontweight='bold', color='#7c3aed')
    
    ax.set_title('Digit Recognition Through Layers', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=100)
    plt.close()
    print(f"Generated: {output_path}")

def generate_feature_vectors_classification(output_path):
    """Generate diagram showing feature vectors to classification."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Feature vectors through layers
    y_pos = 6
    
    # Layer 1 output
    ax.text(1, y_pos, 'Layer 1:', ha='left', fontsize=11, fontweight='bold')
    vector1 = FancyBboxPatch((1, y_pos-0.8), 3, 0.6, boxstyle="round,pad=0.05", 
                            facecolor='#e1f5ff', edgecolor='#2563eb', linewidth=1.5)
    ax.add_patch(vector1)
    ax.text(2.5, y_pos-0.5, r'$\mathbf{h}_1 = [0.8, 0.2, 0.0, 0.5]$', 
           ha='center', fontsize=10)
    ax.text(4.5, y_pos-0.5, 'Edges, lines', ha='left', fontsize=9, style='italic', color='#6b7280')
    
    # Arrow
    arrow1 = FancyArrowPatch((2.5, y_pos-1.2), (2.5, y_pos-2), arrowstyle='->', 
                             mutation_scale=15, color='#2563eb', linewidth=1.5)
    ax.add_patch(arrow1)
    
    y_pos -= 2
    
    # Layer 2 output
    ax.text(1, y_pos, 'Layer 2:', ha='left', fontsize=11, fontweight='bold')
    vector2 = FancyBboxPatch((1, y_pos-0.8), 3, 0.6, boxstyle="round,pad=0.05", 
                            facecolor='#fff4e1', edgecolor='#f59e0b', linewidth=1.5)
    ax.add_patch(vector2)
    ax.text(2.5, y_pos-0.5, r'$\mathbf{h}_2 = [0.2, 0.9, 0.1, 0.3]$', 
           ha='center', fontsize=10)
    ax.text(4.5, y_pos-0.5, 'Combined features', ha='left', fontsize=9, style='italic', color='#6b7280')
    
    # Arrow
    arrow2 = FancyArrowPatch((2.5, y_pos-1.2), (2.5, y_pos-2), arrowstyle='->', 
                             mutation_scale=15, color='#f59e0b', linewidth=1.5)
    ax.add_patch(arrow2)
    
    y_pos -= 2
    
    # Layer 3 output
    ax.text(1, y_pos, 'Layer 3:', ha='left', fontsize=11, fontweight='bold')
    vector3 = FancyBboxPatch((1, y_pos-0.8), 3, 0.6, boxstyle="round,pad=0.05", 
                            facecolor='#c8e6c9', edgecolor='#16a34a', linewidth=1.5)
    ax.add_patch(vector3)
    ax.text(2.5, y_pos-0.5, r'$\mathbf{h}_3 = [0.1, 0.05, 0.95, 0.2]$', 
           ha='center', fontsize=10)
    ax.text(4.5, y_pos-0.5, 'Complex patterns', ha='left', fontsize=9, style='italic', color='#6b7280')
    
    # Arrow to output layer
    arrow3 = FancyArrowPatch((4, y_pos-0.5), (6, 3.5), arrowstyle='->', 
                             mutation_scale=20, color='#7c3aed', linewidth=2)
    ax.add_patch(arrow3)
    
    # Output layer - logits
    output_box = FancyBboxPatch((6, 2), 3, 3, boxstyle="round,pad=0.1", 
                               facecolor='#f3e8ff', edgecolor='#7c3aed', linewidth=2)
    ax.add_patch(output_box)
    ax.text(7.5, 4.5, 'Output Layer', ha='center', fontsize=12, fontweight='bold')
    ax.text(7.5, 4, 'Logits:', ha='center', fontsize=10)
    ax.text(7.5, 3.6, 'Digit 0: 0.3', ha='center', fontsize=9)
    ax.text(7.5, 3.3, 'Digit 1: 0.1', ha='center', fontsize=9)
    ax.text(7.5, 3.0, 'Digit 2: 0.2', ha='center', fontsize=9)
    ax.text(7.5, 2.6, 'Digit 6: 2.5', ha='center', fontsize=9, fontweight='bold', color='#7c3aed')
    ax.text(7.5, 2.3, '(highest!)', ha='center', fontsize=8, style='italic', color='#7c3aed')
    
    # Arrow to prediction
    arrow4 = FancyArrowPatch((9, 3.5), (10.5, 3.5), arrowstyle='->', 
                             mutation_scale=20, color='#16a34a', linewidth=2)
    ax.add_patch(arrow4)
    
    # Final prediction
    pred_box = FancyBboxPatch((10.5, 2.5), 2, 2, boxstyle="round,pad=0.1", 
                             facecolor='#dcfce7', edgecolor='#16a34a', linewidth=2)
    ax.add_patch(pred_box)
    ax.text(11.5, 4, 'Prediction', ha='center', fontsize=11, fontweight='bold')
    ax.text(11.5, 3.5, 'Digit 6', ha='center', fontsize=14, fontweight='bold', color='#16a34a')
    ax.text(11.5, 3, '90% confidence', ha='center', fontsize=9, style='italic')
    
    ax.set_title('Feature Vectors to Classification Decision', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=100)
    plt.close()
    print(f"Generated: {output_path}")

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / "book" / "images" / "activation-functions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_linear_layers_collapse(output_dir / "linear-layers-collapse.svg")
    generate_nonlinear_prevents_collapse(output_dir / "nonlinear-prevents-collapse.svg")
    generate_hierarchical_pattern_learning(output_dir / "hierarchical-pattern-learning.svg")
    generate_digit_recognition_layers(output_dir / "digit-recognition-layers.svg")
    generate_feature_vectors_classification(output_dir / "feature-vectors-classification.svg")
    
    print("\nAll activation function diagrams generated!")

