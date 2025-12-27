# SVG to JSON Conversion Guide

This guide provides analysis of all SVG files in `book/images/` and requirements for converting them to JSON format.

## Overview

**39 SVG files** analyzed (excluding `complete-transformer-architecture-new.svg`) across 5 categories:

1. **Flow/Algorithm Diagrams** (8 files) - ✅ Mostly supported
2. **Graph/Plot Diagrams** (8 files) - ❌ Need new components
3. **Table/Grid Layouts** (5 files) - ⚠️ Need grid system
4. **Network Structure** (5 files) - ✅ Fully supported
5. **Mathematical/Geometric** (13 files) - ⚠️ May need custom components

## Category 1: Flow/Algorithm Diagrams (8 files)

### Files
- `training/gradient-descent-algorithm.svg` - Vertical flowchart with decision boxes
- `training/training-loop.svg` - Loop structure
- `activation-functions/activation-relu.svg` - Decision flowchart with conditional branches
- `activation-functions/activation-sigmoid.svg` - Similar decision flow
- `activation-functions/activation-tanh.svg` - Similar decision flow
- `flow-diagrams/forward-pass-flow.svg` - Horizontal sequence
- `flow-diagrams/backward-pass-flow.svg` - Horizontal sequence
- `other/rnn-limitation.svg` - Flow diagram

### Current Support
- ✅ Boxes, arrows, text labels
- ✅ Conditional branches (via different arrow styles/colors)

### Needed
- ⚠️ **Decision diamond shapes** (currently only rectangles)
- ⚠️ **Loop indicators** (curved arrows back to earlier boxes)

### Conversion Strategy
1. Use existing box components for process steps
2. Create `decision_diamond.json` component for decision points
3. Use connection styles for conditional branches (yes/no, true/false)
4. Add loop connection type with curved paths

## Category 2: Graph/Plot Diagrams (8 files)

### Files
- `training/training-loss-epochs.svg` - Line graph with axes, grid, curve
- `training/training-loss-example4.svg` - Similar line graph
- `training/gradient-descent-path.svg` - 2D plot with contour lines
- `training/gradient-visualization.svg` - Vector field visualization
- `other/hyperplane-1d.svg` - 1D plot with decision boundary
- `other/hyperplane-2d.svg` - 2D plot with axes, decision line, regions
- `other/hyperplane-3d.svg` - 3D visualization
- `other/embedding-space-2d.svg` - 2D scatter plot
- `activation-functions/activation-graph-relu.svg` - Function plot with axes
- `activation-functions/activation-graph-sigmoid.svg` - Function plot
- `activation-functions/activation-graph-tanh.svg` - Function plot
- `other/attention-weights-distribution.svg` - Bar chart with axes

### Current Support
- ❌ None - requires new component types

### Needed Components (Local)
- **`axis_system.json`**: X-axis and Y-axis with configurable range, tick marks, labels, grid lines
- **`plot_line.json`**: Path data (from function or points), stroke color/width, optional markers
- **`plot_region.json`**: Polygon points, fill color, opacity, border
- **`bar_chart.json`**: Data array (values, labels), bar width, spacing, colors
- **`scatter_plot.json`**: Data points (x, y coordinates), point size, color, optional labels

### Conversion Strategy
1. Create local components directory for each graph diagram
2. Implement `axis_system` component with configurable ranges
3. Implement plot components (line, region, bar, scatter)
4. Add mathematical function evaluator for curve generation
5. Store plot data in component configuration

## Category 3: Table/Grid Layouts (5 files)

### Files
- `training/batch-training.svg` - 4-column grid (Batch → 4 Process boxes → 4 Gradient boxes → Average)
- `network-structure/ffn-structure.svg` - Horizontal sequence (7 boxes in a row)
- `flow-diagrams/forward-pass-flow.svg` - Horizontal sequence (5 boxes)
- `flow-diagrams/backward-pass-flow.svg` - Horizontal sequence
- `network-structure/multi-layer-network.svg` - Horizontal sequence with training box above

### Current Support
- ✅ Horizontal sequences via connections

### Needed
- ⚠️ **Grid layout system** (explicit row/column positioning)
- ⚠️ **Size constraints** (equal widths/heights for alignment)
- ⚠️ **Alignment rules** (top-aligned, center-aligned, bottom-aligned)
- ⚠️ **Spacing constraints** (equal spacing between grid items)

### Conversion Strategy
1. Create `grid_container.json` component
2. Add grid placement algorithm
3. Add size constraint solver
4. Use grid coordinates for component positioning

## Category 4: Network Structure Diagrams (5 files)

### Files
- `network-structure/complete-transformer-architecture.svg` - ✅ Already converted
- `network-structure/neural-network-structure.json` - Multi-layer with training feedback
- `network-structure/multi-layer-network.json` - Horizontal layers with training box
- `network-structure/perceptron-diagram.json` - Simple network structure
- `other/attention-example.json` - Attention mechanism visualization

### Current Support
- ✅ Containers, hierarchical grouping
- ✅ Connections between components
- ✅ Training feedback arrows (dashed, different color)
- ✅ Custom text overrides

### Conversion Strategy
1. Use existing container system
2. Use existing connection styles for training feedback
3. Reference shared component library

## Category 5: Mathematical/Geometric Diagrams (13 files)

### Files
- `other/matrix-multiplication-identity.svg` - Matrix visualization
- `other/vector-rotation-90deg.svg` - Vector transformation
- `other/cat-journey.svg` - Conceptual diagram
- `training/cross-entropy-loss.svg` - Mathematical visualization
- `training/loss-correct.svg` - Comparison diagram
- `training/loss-wrong.svg` - Comparison diagram
- `training/gradient-decay-chain.svg` - Chain visualization
- `training/gradient-decay-vanishing.svg` - Gradient flow
- `activation-functions/examples/linear-function.svg` - Function example
- `activation-functions/examples/relu-applied.svg` - Applied function
- `activation-functions/examples/sigmoid-applied.svg` - Applied function
- `activation-functions/examples/tanh-applied.svg` - Applied function

### Current Support
- ⚠️ Limited - may require custom components

### Needed
- ⚠️ **Mathematical shapes** (matrices, vectors, equations)
- ⚠️ **Geometric primitives** (circles, polygons, paths)
- ⚠️ **Comparison layouts** (side-by-side diagrams)

### Conversion Strategy
1. Analyze each diagram individually
2. Create custom components as needed
3. Use local components directory for unique cases

## Conversion Phases

### Phase 1: Current System Compatible (13 files)
- Network structure diagrams (5 files) - ✅ Fully supported
- Simple flow diagrams (8 files) - ✅ Mostly supported (need decision diamonds)

### Phase 2: Extend System for Graphs (8 files)
- Add axis system components
- Add plot components (line, region, bar, scatter)
- Add grid system component
- Mathematical function evaluator for curve generation

### Phase 3: Extend System for Tables/Grids (5 files)
- Add grid container component
- Add grid placement algorithm
- Add size constraint solver

### Phase 4: Special Cases (13 files)
- Mathematical/geometric diagrams - May require custom components
- Comparison diagrams - May require side-by-side layout system

## Required New Component Types

### For Shared Library
- `decision_diamond.json` - For flowcharts (used by activation functions, algorithms)
- `grid_container.json` - For table layouts (used by batch-training, etc.)

### For Local Components
- `axis_system.json` - For graph diagrams (each may have different axis config)
- `plot_line.json` - For line graphs
- `plot_region.json` - For filled regions
- `bar_chart.json` - For bar charts
- `scatter_plot.json` - For scatter plots

## Conversion Workflow

1. **Analyze SVG structure**
   - Identify components (boxes, shapes, text)
   - Identify connections (arrows, lines)
   - Identify containers (groups, subgraphs)

2. **Map to component templates**
   - Use shared library when possible
   - Create local components for unique cases
   - Document component requirements

3. **Create JSON definition**
   - Define metadata (id, title, dimensions)
   - Define components (positions, templates, custom text)
   - Define connections (from/to pins, styles)
   - Define containers (hierarchy, titles)
   - Define labels (titles, equations, notes)

4. **Validate and test**
   - Validate against schema
   - Generate SVG and compare with original
   - Adjust positions and styling as needed

## Tips

- **Start with simple diagrams** - Flow diagrams and network structures are easiest
- **Use existing templates** - Check shared library first before creating new components
- **Preserve spacing** - Maintain original spacing relationships
- **Test incrementally** - Convert one diagram at a time and verify
- **Document custom components** - If creating local components, document their purpose

## See Also

- `README.md` - General documentation
- `schema.json` - JSON schema definition
- `book/diagrams/components/` - Shared component library
- `book/diagrams/diagram_config.json` - Global configuration

