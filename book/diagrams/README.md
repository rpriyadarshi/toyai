# Diagram JSON Database

This directory contains JSON definitions for all diagrams in the book. Diagrams are organized by category, mirroring the structure of `book/images/`.

## Directory Structure

```
book/diagrams/
  components/              # Shared component templates
  diagram_config.json      # Global configuration
  schema.json              # JSON schema definition
  README.md                # This file
  CONVERSION_GUIDE.md      # SVG to JSON conversion guide
  example-diagram.json     # Example diagram for reference
  generated/               # Generated SVG files
  
  network-structure/       # Network architecture diagrams
  training/                 # Training and optimization diagrams
  flow-diagrams/           # Flowchart and algorithm diagrams
  activation-functions/    # Activation function visualizations
  other/                   # Miscellaneous diagrams
```

## Component System

### Hybrid Component Strategy

The system uses a **hybrid approach** for component templates:

1. **Shared Component Library** (`book/diagrams/components/`)
   - Common, reusable components used by multiple diagrams
   - Examples: `input_box.json`, `qkv_box.json`, `attention_box.json`
   - Referenced via `"template": "input_box"` in diagram JSON

2. **Local Components** (`{diagram-dir}/components/`)
   - Diagram-specific components used by only one diagram
   - Examples: `axis_system.json`, `plot_line.json`, `bar_chart.json`
   - Automatically resolved if template not found in shared library

### Component Resolution

When a diagram references a template:

1. **Primary**: Check shared library (`book/diagrams/components/`)
2. **Fallback**: Check local components directory (`{diagram-dir}/components/`)
3. **Error**: Raise helpful error if not found in either location

### When to Use Shared vs Local Components

**Add to Shared Library** when:
- Component is used by 2+ diagrams
- Component represents a common pattern
- Component is part of standard vocabulary (e.g., `input_box`, `output_box`)

**Use Local Components** when:
- Component is truly unique to one diagram
- Component has diagram-specific configuration
- Component is experimental or under development

## JSON Format

### Basic Structure

```json
{
  "metadata": {
    "id": "diagram-id",
    "title": "Diagram Title",
    "width": 1200,
    "height": 800
  },
  "components": [...],
  "connections": [...],
  "containers": [...],
  "labels": [...]
}
```

### Components

Each component references a template from the component library:

```json
{
  "id": "component-id",
  "type": "input",
  "template": "input_box",
  "position": {"x": 100, "y": 200},
  "custom_text": {
    "label": "Custom Label",
    "subtext": "Custom subtext"
  }
}
```

### Connections

Connections link components via pins:

```json
{
  "from": "comp1.output",
  "to": "comp2.input",
  "style": "forward",
  "color": "#2563eb",
  "stroke_width": 2.0,
  "dasharray": "4,4"
}
```

### Containers

Containers group related components:

```json
{
  "id": "container-id",
  "type": "container",
  "template": "subgraph_container",
  "position": {"x": 50, "y": 50},
  "width": 500,
  "height": 300,
  "title": "Container Title",
  "contains": ["comp1", "comp2", "comp3"]
}
```

## Usage

### Generate SVG from JSON

```bash
python3 -m diagram_generator generate book/diagrams/network-structure/complete-transformer-architecture.json -o output.svg
```

### Validate JSON

```bash
python3 -m diagram_generator validate book/diagrams/network-structure/complete-transformer-architecture.json
```

## Schema Validation

The `schema.json` file defines the complete JSON schema. Use it to validate diagram definitions:

- Required fields: `metadata`, `components`, `connections`, `labels`
- Optional fields: `containers`, `metadata.diagram_type`, `metadata.placement_algorithm`
- Component properties: `width`, `height`, `custom_text`, `pins` (all optional)

## See Also

- `CONVERSION_GUIDE.md` - Guide for converting SVG files to JSON
- `components/` - Shared component library
- `diagram_config.json` - Global configuration (colors, fonts, spacing)
- `generated/` - Generated SVG files

