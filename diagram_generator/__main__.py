"""
Entry point for running diagram_generator as a module.

Usage:
    python -m diagram_generator generate <diagram.json> [-o output.svg]
    python -m diagram_generator validate <diagram.json>
"""

from diagram_generator.cli import main

if __name__ == "__main__":
    main()

