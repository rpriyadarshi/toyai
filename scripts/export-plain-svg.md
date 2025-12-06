# How to Export Plain SVG from Inkscape

## Problem
Inkscape connectors use `inkscape:connection-start` and `inkscape:connection-end` attributes that Inkscape dynamically interprets. Standard SVG viewers ignore these and only render the static `d` attribute, causing rendering differences.

## Solution: Export as Plain SVG

1. **Open the file in Inkscape**
   ```bash
   inkscape book/images/complete-transformer-architecture.svg
   ```

2. **Export as Plain SVG** (this converts connectors to static paths):
   - File → Save As
   - Choose "Plain SVG (*.svg)" as the format
   - Save to a new file (e.g., `complete-transformer-architecture-plain.svg`)
   - OR use command line:
   ```bash
   inkscape --export-type=svg \
     --export-plain-svg \
     book/images/complete-transformer-architecture.svg \
     -o book/images/complete-transformer-architecture-plain.svg
   ```

3. **Verify the export**:
   - The exported file should have the same visual appearance
   - All `d` attributes should contain the actual rendered path coordinates
   - The `inkscape:connection-*` attributes may still be present but the `d` will match what Inkscape renders

4. **Replace the original**:
   ```bash
   mv book/images/complete-transformer-architecture-plain.svg \
      book/images/complete-transformer-architecture.svg
   ```

## Alternative: Use Inkscape's "Object to Path" for Connectors

If exporting as Plain SVG doesn't work, you can convert connectors to paths:

1. Select all connector paths
2. Path → Object to Path (or Shift+Ctrl+C)
3. This converts connectors to regular paths with static `d` attributes
4. Save the file

## Verification

After export, check that:
- The file renders identically in both Inkscape and standard SVG viewers
- All arrow paths have correct `d` attributes matching the visual appearance
- No arrows are missing or misaligned

