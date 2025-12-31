#!/usr/bin/env python3
"""
Interactive 3D Hyperplane Visualization

Launches an interactive 3D plot that you can rotate, zoom, and inspect in real-time.
Use mouse to rotate: click and drag
Use scroll wheel to zoom
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def create_interactive_3d():
    """
    Create an interactive 3D hyperplane visualization.
    The window can be rotated, zoomed, and panned with mouse controls.
    """
    # Create figure with 3D axes
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis limits
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    
    # Define plane: w₁x₁ + w₂x₂ + w₃x₃ + b = 0
    # For visualization, use a plane through origin: x₃ = 0
    x1_range = np.linspace(-1.2, 1.2, 40)
    x2_range = np.linspace(-1.2, 1.2, 40)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    X3 = np.zeros_like(X1)  # Plane at z=0
    
    # Plot decision plane (semi-transparent blue with grid)
    # Use higher zorder so it appears above gradient layers but below labels
    ax.plot_surface(X1, X2, X3, alpha=0.3, color='#2563eb', 
                    edgecolor='#3b82f6', linewidth=0.5, shade=True,
                    rstride=2, cstride=2, zorder=50)
    
    # Draw axes with proper styling
    # x₁-axis - position label to avoid overlap
    ax.plot([-1.2, 1.2], [0, 0], [0, 0], 'k-', linewidth=3, label='x₁', zorder=100)
    ax.text(1.45, -0.15, 0, 'x₁', fontsize=15, fontweight='bold', color='#1f2937', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95), zorder=200)
    
    # x₂-axis - position label to avoid overlap
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], 'k-', linewidth=3, label='x₂', zorder=100)
    ax.text(-0.15, 1.45, 0, 'x₂', fontsize=15, fontweight='bold', color='#1f2937',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95), zorder=200)
    
    # x₃-axis - position label to avoid overlap
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], 'k-', linewidth=3, label='x₃', zorder=100)
    ax.text(0.15, 0, 1.45, 'x₃', fontsize=15, fontweight='bold', color='#1f2937',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95), zorder=200)
    
    # Origin - position to avoid overlap with plane label
    ax.scatter([0], [0], [0], color='black', s=100, zorder=150)
    ax.text(0.25, -0.25, -0.25, '(0,0,0)', fontsize=10, color='#6b7280', fontweight='500',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9), zorder=200)
    
    # Positive region: entire half-space above plane (x₃ > 0)
    # Create gradient cubes that fade with distance from plane
    x1_range_full = np.linspace(-1.2, 1.2, 30)
    x2_range_full = np.linspace(-1.2, 1.2, 30)
    X1_full, X2_full = np.meshgrid(x1_range_full, x2_range_full)
    
    # Create multiple layers at different z-levels with decreasing opacity
    # This creates a gradient effect that fades as we move away from the plane
    # Use lower zorder so labels appear on top
    # Very transparent to avoid obstructing the decision plane
    z_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    for i, z_level in enumerate(z_levels):
        # Opacity decreases with distance from plane - very transparent
        alpha = max(0.01, 0.06 * (1.0 - i / len(z_levels)))
        X3_pos = np.ones_like(X1_full) * z_level
        ax.plot_surface(X1_full, X2_full, X3_pos, alpha=alpha, 
                        color='#16a34a', edgecolor='none', shade=True, zorder=10)
    
    # Negative region: entire half-space below plane (x₃ < 0)
    # Create gradient cubes that fade with distance from plane
    # Very transparent to avoid obstructing the decision plane
    z_levels_neg = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2]
    for i, z_level in enumerate(z_levels_neg):
        # Opacity decreases with distance from plane - very transparent
        alpha = max(0.01, 0.06 * (1.0 - i / len(z_levels_neg)))
        X3_neg = np.ones_like(X1_full) * z_level
        ax.plot_surface(X1_full, X2_full, X3_neg, alpha=alpha, 
                        color='#dc2626', edgecolor='none', shade=True, zorder=10)
    
    # Add arrows pointing to regions
    # Arrow to positive region (pointing up in z direction)
    # Position arrow and label at different corners to avoid overlap
    ax.quiver(1.0, -1.0, 0.5, 0, 0, 0.4, color='#16a34a', 
              arrow_length_ratio=0.3, linewidth=3, alpha=0.9, zorder=150)
    ax.text(1.2, -1.2, 1.2, 'w·x + b > 0\n(all x₃ > 0)', fontsize=11, fontweight='600', 
            color='#16a34a', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='#16a34a', linewidth=2), 
            zorder=200, ha='left', va='bottom')
    
    # Arrow to negative region (pointing down in z direction)
    # Position arrow and label at different corners to avoid overlap
    ax.quiver(-1.0, 1.0, -0.5, 0, 0, -0.4, color='#dc2626', 
              arrow_length_ratio=0.3, linewidth=3, alpha=0.9, zorder=150)
    ax.text(-1.2, 1.2, -1.2, 'w·x + b < 0\n(all x₃ < 0)', fontsize=11, fontweight='600', 
            color='#dc2626', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='#dc2626', linewidth=2),
            zorder=200, ha='right', va='top')
    
    # Set view angle (initial view) - isometric view that shows plane clearly
    ax.view_init(elev=20, azim=30)
    
    # Remove default axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    
    # Set background to white, remove grid planes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.grid(False)
    ax.set_facecolor('white')
    
    # Title and equation
    fig.suptitle('3D Input (d=3) - Interactive View', fontsize=18, fontweight='bold', y=0.98)
    fig.text(0.5, 0.95, 'w₁x₁ + w₂x₂ + w₃x₃ + b = 0', 
             ha='center', fontsize=14, color='#6b7280')
    
    # Add plane label - position it clearly on the plane but not obstructed
    # Offset slightly to avoid overlap with origin label
    ax.text(0.3, -0.3, 0.15, 'Decision Plane', fontsize=13, fontweight='600', 
            color='#2563eb', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.95, edgecolor='#2563eb', linewidth=2),
            zorder=200)
    
    # Instructions
    fig.text(0.5, 0.02, 'Mouse: Click and drag to rotate | Scroll wheel: Zoom | Close window when done', 
             ha='center', fontsize=10, color='#9ca3af', style='italic')
    
    # Make layout tight
    plt.tight_layout()
    
    # Show interactive window
    print("\n" + "="*60)
    print("Interactive 3D Visualization Launched!")
    print("="*60)
    print("Controls:")
    print("  - Click and drag: Rotate the 3D view")
    print("  - Scroll wheel: Zoom in/out")
    print("  - Right-click and drag: Pan the view")
    print("  - Close the window when you're done")
    print("="*60 + "\n")
    
    plt.show()

if __name__ == "__main__":
    create_interactive_3d()

