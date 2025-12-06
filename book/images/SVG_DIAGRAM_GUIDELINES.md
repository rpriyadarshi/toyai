# SVG Diagram Guidelines and Best Practices

This document captures all the lessons learned and best practices for creating and fixing SVG diagrams, particularly for complex architecture diagrams with connectors.

## Core Requirements

### 1. All Arrows Must Be Connectors

**Requirement**: Every arrow/path connecting boxes MUST be an Inkscape connector, not a static path.

**Why**: Connectors maintain their connections when boxes are moved, which is essential for maintaining diagram integrity during layout changes.

**Implementation**:
- Use `inkscape:connector-type="polyline"` attribute
- Set `inkscape:connector-curvature="0"` for straight lines (or appropriate value for curves)
- Use `inkscape:connection-start="#group-id"` and `inkscape:connection-end="#group-id"` to reference the parent `<g>` elements (not the `<rect>` elements directly)
- Always reference the group ID with `#` prefix (e.g., `#g22`, not `g22`)

**Example**:
```xml
<path
   style="fill:none;stroke:#2563eb;stroke-width:2px"
   d="M 515,80 L 565,80"
   inkscape:connector-type="polyline"
   inkscape:connector-curvature="0"
   inkscape:connection-start="#g22"
   inkscape:connection-end="#g30"
   marker-end="url(#arrow-blue)" />
```

**Key Points**:
- Connectors reference the parent `<g>` element that contains the box, not the `<rect>` itself
- The `d` attribute should use absolute coordinates (see Coordinate System section)
- Preserve styling (color, stroke-width, marker-end) when converting to connectors

### 2. Box Sizing - Fully Encapsulate Text

**Requirement**: All boxes must be sized such that all text inside is fully visible and properly contained.

**Implementation**:
- Calculate text width and height before setting box dimensions
- Add padding around text (typically 10-20px on each side)
- For multi-line text, account for line height and spacing
- Use `text-anchor="middle"` for centered text
- Ensure text doesn't overflow box boundaries

**Best Practices**:
- Minimum box height: text height + 20px (10px top + 10px bottom padding)
- Minimum box width: text width + 20px (10px left + 10px right padding)
- For boxes with labels and subtext, account for both lines
- Test with longest expected text content

### 3. Spacing Between Boxes

**Requirement**: Maintain reasonable spacing between boxes to allow for full arrow drawings without overlap.

**Implementation**:
- Minimum spacing: 50px between boxes (allows for arrow path and arrowhead)
- For horizontal arrows: ensure space for arrow path + arrowhead (typically 8-10px for arrowhead)
- For vertical arrows: same spacing considerations
- Consider arrow routing when boxes are not directly adjacent

**Guidelines**:
- Standard spacing: 50px between box edges
- For diagonal arrows: increase spacing to 60-70px
- When arrows need to curve or route around obstacles, increase spacing accordingly
- Test arrow visibility after layout changes

### 4. Block Structure - Grouped Children

**Requirement**: Blocks should be built with all children properly grouped.

**Implementation**:
- Each major section (e.g., "Transformer Block 1", "Input Processing") should be a parent `<g>` element
- All boxes within a block should be child `<g>` elements with unique IDs
- Each box group should contain: `<rect>`, `<text>` elements, and any other box content
- Use `transform="translate(x, y)"` on child groups for positioning

**Structure Example**:
```xml
<g id="transformer-block-1">
  <!-- Background rectangle for the block -->
  <rect x="320" y="200" width="560" height="310" class="subgraph-bg" />
  <text x="600" y="220" class="subgraph-title">Transformer Block 1</text>
  
  <!-- Individual box groups -->
  <g transform="translate(340, 240)" id="g64">
    <rect x="0" y="0" width="140" height="50" class="qkv-box" />
    <text x="70" y="20" class="box-label">Q/K/V</text>
    <text x="70" y="35" class="box-text">WQ, WK, WV</text>
  </g>
  
  <!-- More boxes... -->
  
  <!-- Arrows within the block -->
  <path ... inkscape:connection-start="#g64" inkscape:connection-end="#g72" ... />
</g>
```

**Key Points**:
- Each box gets its own `<g>` with unique ID (e.g., `id="g64"`)
- Use `transform="translate(x, y)"` for positioning within the parent block
- Box coordinates are relative to the translate offset
- Connectors reference these group IDs

### 5. Border Padding - Add After Bounding Box Calculation

**Requirement**: Always add a 20px border AFTER computing the bounding box for all drawings.

**Implementation Steps**:
1. Calculate the bounding box of all elements (min x, min y, max x, max y)
2. Calculate content width: `max_x - min_x`
3. Calculate content height: `max_y - min_y`
4. Add 20px padding on all sides:
   - SVG width: `content_width + 40` (20px left + 20px right)
   - SVG height: `content_height + 40` (20px top + 20px bottom)
5. Adjust element positions to center content within the padded area if needed

**Example Calculation**:
```python
# Find bounding box
min_x = min(all_element_x_positions)
min_y = min(all_element_y_positions)
max_x = max(all_element_x_positions + element_widths)
max_y = max(all_element_y_positions + element_heights)

content_width = max_x - min_x
content_height = max_y - min_y

# Add 20px padding
svg_width = content_width + 40
svg_height = content_height + 40

# Center content if needed
offset_x = 20 - min_x
offset_y = 20 - min_y
```

## Coordinate System and Path Format

### Use Absolute Coordinates

**Critical**: All path `d` attributes MUST use absolute coordinates for SVG viewer compatibility.

**Problem**: Relative coordinates (e.g., `m 515,80 h 50`) work in Inkscape but may not render in standard SVG viewers.

**Solution**: Convert all relative paths to absolute:
- `m 515,80 h 50` → `M 515,80 L 565,80`
- `m 720,105 -120,50` → `M 720,105 L 600,155`
- `m 110,390 v 50` → `M 110,390 L 110,440`

**Conversion Rules**:
- `m x,y` (move relative) → `M x,y` (move absolute)
- `h dx` (horizontal relative) → `L x+dx,y` (line absolute)
- `v dy` (vertical relative) → `L x,y+dy` (line absolute)
- `l dx,dy` (line relative) → `L x+dx,y+dy` (line absolute)
- `q dx1,dy1 dx2,dy2` (quadratic relative) → `Q x1+dx1,y1+dy1 x2+dx2,y2+dy2` (quadratic absolute)

### Arrow Connection Points - Center-to-Center Algorithm

**Critical Rule**: Use the **center-to-center algorithm** for all connector paths to ensure SVG viewers render the same as Inkscape.

**The Problem**: Inkscape connectors use `inkscape:connection-start` and `inkscape:connection-end` to dynamically calculate paths based on group bounding boxes. However, the `d` attribute may contain stale coordinates that don't match what Inkscape actually renders. Standard SVG viewers ignore Inkscape attributes and only render the `d` attribute, causing visual mismatches.

**The Solution**: Use the center-to-center algorithm:
1. Calculate the center of the source group's bounding box
2. Calculate the center of the destination group's bounding box
3. Draw a line from center to center
4. Find where this line **exits** the source box boundary
5. Find where this line **enters** the destination box boundary
6. Use these intersection points as the start and end of the `d` attribute

**Algorithm Steps**:
```python
# For a connector from group A to group B:
# 1. Get bounding boxes
bbox_a = get_bbox(group_a)  # {left, right, top, bottom, center_x, center_y}
bbox_b = get_bbox(group_b)

# 2. Calculate center-to-center line
x1, y1 = bbox_a['center_x'], bbox_a['center_y']
x2, y2 = bbox_b['center_x'], bbox_b['center_y']

# 3. Find exit point (where line leaves box A)
exit_point = line_box_intersection(x1, y1, x2, y2, bbox_a)

# 4. Find entry point (where line enters box B)
entry_point = line_box_intersection(x2, y2, x1, y1, bbox_b)

# 5. Update d attribute
d = f"M {exit_point[0]},{exit_point[1]} {entry_point[0]},{entry_point[1]}"
```

**Why This Works**: This matches exactly what Inkscape does internally when rendering connectors. By updating the `d` attribute to match Inkscape's calculation, both Inkscape and standard SVG viewers render identically.

**Simple Cases** (for reference, but use algorithm for accuracy):
- **Horizontal Arrows**: Right edge center → Left edge center
- **Vertical Arrows**: Bottom edge center → Top edge center

**Calculation**:
- Box at `translate(x, y)` with width `w` and height `h`:
  - Center: `(x + w/2, y + h/2)`
  - Right edge center: `(x + w, y + h/2)`
  - Left edge center: `(x, y + h/2)`
  - Bottom edge center: `(x + w/2, y + h)`
  - Top edge center: `(x + w/2, y)`

## Inter-Block Connections

### Placement and Z-Order

**Requirement**: Inter-block connection arrows must be placed in a separate group at the END of the SVG to render in front of all blocks.

**Implementation**:
```xml
<!-- All block groups first -->
<g id="input-processing">...</g>
<g id="transformer-block-1">...</g>
<g id="transformer-block-2">...</g>
<g id="output">...</g>
<g id="training">...</g>

<!-- Inter-block connections LAST (renders on top) -->
<g id="inter-block-connections">
  <path ... inkscape:connection-start="#g44" inkscape:connection-end="#g64" ... />
</g>
```

**Why**: SVG renders elements in document order. Elements defined later appear on top.

### Straight Lines for Inter-Block Arrows

**Requirement**: Inter-block arrows should be straight lines connecting the correct box positions.

**Implementation**:
- Use `inkscape:connector-curvature="0"` for straight lines
- Calculate exact connection points (box edges)
- Avoid unnecessary curves unless routing around obstacles

**Example**:
```xml
<!-- Input Processing → Transformer Block 1 -->
<path
   d="M 600,175 L 410,240"
   inkscape:connector-type="polyline"
   inkscape:connector-curvature="0"
   inkscape:connection-start="#g44"
   inkscape:connection-end="#g64"
   marker-end="url(#arrow-blue)" />
```

## Styling and Markers

### Arrow Markers

**Requirement**: Define arrow markers in `<defs>` section and reference them consistently.

**Implementation**:
```xml
<defs>
  <marker
     id="arrow-blue"
     markerWidth="8"
     markerHeight="8"
     refX="7"
     refY="3"
     orient="auto">
    <polygon points="0 0, 8 3, 0 6" fill="#2563eb" />
  </marker>
</defs>
```

**Usage**:
- Add `marker-end="url(#arrow-blue)"` to connector paths
- Use different markers for different arrow types (forward pass, backward pass, etc.)

### Color Coding

**Common Patterns**:
- Blue (`#2563eb`): Forward pass arrows
- Purple (`#9333ea`): Backward pass / training arrows (often dashed)
- Dashed style: `stroke-dasharray="4,4"` for training/feedback arrows

## Testing and Validation

### Checklist Before Finalizing

1. ✅ All arrows are connectors (check for `inkscape:connector-type`)
2. ✅ All paths use absolute coordinates (no `m`, `h`, `v` commands)
3. ✅ Connector path coordinates match Inkscape rendering (run `fix-svg-arrows.py --fix-paths`)
4. ✅ All boxes fully contain their text
5. ✅ Spacing between boxes is adequate (50px minimum)
6. ✅ All blocks are properly grouped
7. ✅ 20px border added after bounding box calculation
8. ✅ Inter-block arrows are in separate group at end
9. ✅ Arrow connection points use center-to-center algorithm
10. ✅ SVG renders correctly in both Inkscape and standard SVG viewers (verify visually)
11. ✅ All marker references have corresponding definitions

### Common Issues and Fixes

**Issue**: Arrows visible in Inkscape but not in SVG viewers
- **Fix**: Convert relative coordinates to absolute

**Issue**: Arrows render differently in Inkscape vs SVG viewers (coordinates mismatch)
- **Root Cause**: Inkscape dynamically calculates connector paths from `connection-start`/`connection-end`, but the `d` attribute contains stale coordinates
- **Fix Options**:
  1. **Automated (Recommended)**: Use `scripts/fix-svg-arrows.py --fix-paths` to algorithmically calculate correct coordinates
  2. **Manual**: Export from Inkscape as "Plain SVG" to get correct `d` attributes, then copy coordinates back
  3. **Manual Calculation**: Use center-to-center algorithm to calculate boundary intersections

**Issue**: Arrows don't connect properly when boxes are moved
- **Fix**: Ensure arrows are connectors with proper `inkscape:connection-start` and `inkscape:connection-end` attributes

**Issue**: Text overflows box boundaries
- **Fix**: Increase box dimensions and recalculate spacing

**Issue**: Arrows render behind boxes
- **Fix**: Move inter-block connections group to end of SVG

**Issue**: Arrow coordinates are incorrect
- **Fix**: Use the center-to-center algorithm to calculate boundary intersection points (see Arrow Connection Points section)

## Workflow Summary

1. **Design Phase**:
   - Plan block structure and grouping
   - Calculate box sizes based on text content
   - Plan spacing (50px minimum between boxes)

2. **Implementation Phase**:
   - Create block groups with proper IDs
   - Create box groups with `transform="translate(x, y)"`
   - Add boxes with proper sizing
   - Create connector arrows (not static paths)
   - Use absolute coordinates for all paths

3. **Layout Phase**:
   - Position blocks with adequate spacing
   - Calculate bounding box of all elements
   - Add 20px border padding
   - Center content if needed

4. **Inter-Block Connections**:
   - Create separate group for inter-block arrows
   - Place at end of SVG (for z-order)
   - Use straight lines with proper connection points

5. **Fix Connector Paths** (Critical for SVG viewer compatibility):
   - Run `scripts/fix-svg-arrows.py --fix-paths file.svg` to update `d` attributes
   - This ensures SVG viewers render the same as Inkscape
   - Or export as "Plain SVG" from Inkscape and copy coordinates back

6. **Validation**:
   - Test in Inkscape (connector functionality)
   - Test in standard SVG viewers (rendering)
   - Verify all arrows connect correctly
   - Check text visibility and box sizing
   - Verify arrows render identically in both Inkscape and SVG viewers

## Example: Complete Arrow Pattern

```xml
<!-- Within a block -->
<g id="transformer-block-1">
  <!-- Boxes -->
  <g transform="translate(340, 240)" id="g64">
    <rect x="0" y="0" width="140" height="50" class="qkv-box" />
    <text x="70" y="20" class="box-label">Q/K/V</text>
  </g>
  
  <g transform="translate(530, 240)" id="g72">
    <rect x="0" y="0" width="140" height="50" class="attention-box" />
    <text x="70" y="20" class="box-label">Attention</text>
  </g>
  
  <!-- Connector arrow -->
  <path
     style="fill:none;stroke:#2563eb;stroke-width:2px"
     d="M 480,265 L 530,265"
     inkscape:connector-type="polyline"
     inkscape:connector-curvature="0"
     inkscape:connection-start="#g64"
     inkscape:connection-end="#g72"
     marker-end="url(#arrow-blue)" />
</g>
```

## Automated Tools

### fix-svg-arrows.py

Consolidated script for fixing SVG arrows and connectors:

```bash
# Convert static arrows to connectors
python3 scripts/fix-svg-arrows.py --to-connectors file.svg

# Fix connector path coordinates (matches Inkscape rendering)
python3 scripts/fix-svg-arrows.py --fix-paths file.svg

# Do both operations (recommended)
python3 scripts/fix-svg-arrows.py --all file.svg

# Fix all files in a directory
python3 scripts/fix-svg-arrows.py --all book/images/*.svg
```

**What it does**:
- `--to-connectors`: Converts static arrow paths to Inkscape connectors
- `--fix-paths`: Algorithmically calculates correct connector path coordinates using center-to-center algorithm
- `--all`: Runs both operations in sequence

**When to use**:
- After creating new SVG diagrams with arrows
- After moving boxes in Inkscape (connector paths may need recalculation)
- Before committing SVG files to ensure SVG viewer compatibility

### Alternative: Export Plain SVG from Inkscape

If automated scripts don't work, you can export from Inkscape:

1. Open file in Inkscape
2. File → Save As → Choose "Plain SVG (*.svg)"
3. This converts connectors to static paths with correct `d` attributes
4. Copy the `d` attributes back to your original file (preserving connector attributes)

See `scripts/export-plain-svg.md` for detailed instructions.

## Notes

- Always preserve Inkscape-specific attributes for connector functionality
- Test changes in both Inkscape and standard SVG viewers
- Keep connector attributes even when using absolute coordinates
- The `d` attribute must match what Inkscape renders for SVG viewer compatibility
- Use automated tools to ensure consistency
- Document any deviations from these guidelines with justification

---

**Last Updated**: Based on fixes to `complete-transformer-architecture.svg` and connector path coordinate mismatch resolution
**Key Learnings**: 
- Connector format, absolute coordinates, proper grouping
- Center-to-center algorithm for connection points
- Connector path coordinate mismatch between Inkscape and SVG viewers
- Automated tools for fixing connector paths

