#!/bin/bash
# Helper script to find Cursor workspace storage for this project

PROJECT_PATH="/home/rohit/src/toyai-1"
STORAGE_BASE="$HOME/.config/Cursor/User/workspaceStorage"

echo "Looking for workspace storage for: $PROJECT_PATH"
echo ""

# Calculate what the workspace hash might be (Cursor uses a hash of the path)
# Note: This is approximate - Cursor's actual hashing may differ
PROJECT_HASH=$(echo -n "$PROJECT_PATH" | sha256sum | cut -d' ' -f1 | cut -c1-32)
echo "Expected hash pattern (first 32 chars): $PROJECT_HASH"
echo ""

# Check each workspace directory for clues
echo "Checking workspace directories for this project..."
echo ""

for dir in "$STORAGE_BASE"/*; do
    if [ -d "$dir" ]; then
        # Check if there's a workspace.json or state.vscdb that might reference our path
        if [ -f "$dir/workspace.json" ]; then
            if grep -q "toyai-1" "$dir/workspace.json" 2>/dev/null; then
                echo "✓ Found matching workspace: $(basename "$dir")"
                echo "  Path: $dir"
                ls -lh "$dir" 2>/dev/null | head -5
                echo ""
            fi
        fi
    fi
done

echo ""
echo "To view all workspace directories:"
echo "  ls -la $STORAGE_BASE"
echo ""
echo "Chat history is typically in:"
echo "  <workspace-hash>/state.vscdb"
echo "  or"
echo "  <workspace-hash>/cursor/chat/"

