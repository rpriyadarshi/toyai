# Professional Book Publishing Build System

Compiler-like build system for generating professional publishing formats from Markdown source.

## Features

- **Incremental Builds**: Only rebuilds changed files
- **Dependency Tracking**: Tracks file hashes and dependencies
- **Parallel Processing**: Converts diagrams in parallel
- **Multiple Formats**: LaTeX, PDF, PostScript, Word
- **Quality Validation**: Validates diagram conversion quality
- **Compiler-Like Interface**: Clear error messages, exit codes, progress reporting

## Quick Start

```bash
# Build all formats
python3 scripts/build_book.py --all

# Build specific formats
python3 scripts/build_book.py --latex --pdf
python3 scripts/build_book.py --word

# Force full rebuild
python3 scripts/build_book.py --all --force

# Clean build artifacts
python3 scripts/build_book.py --clean
```

## Build Targets

- `--latex`: Generate LaTeX source
- `--pdf`: Generate PDF (requires LaTeX)
- `--ps`: Generate PostScript (requires PDF)
- `--word`: Generate Word document
- `--all`: Generate all formats

## Build Options

- `--force`: Force rebuild everything (ignore cache)
- `--clean`: Remove all build artifacts
- `--verbose`: Verbose output
- `--quiet`: Quiet mode (errors only)

## Output Structure

```
output/
  latex/
    book.tex              # Main LaTeX file
    chapters/             # Individual chapter files
  diagrams/               # Converted PDF diagrams
  pdf/
    book.pdf              # Final PDF
  postscript/
    book.ps                # PostScript version
  word/
    book.docx              # Word document
    images/                # PNG diagrams for Word
```

## Build Cache

The build system maintains a cache in `.build/` directory:

```
.build/
  cache.json              # Build manifest with hashes
  diagrams/               # Diagram metadata
  chapters/               # Chapter metadata
  errors.log              # Error log
```

## Dependencies

### Required

- **pandoc**: Markdown to LaTeX/Word conversion
  ```bash
  sudo apt-get install pandoc
  ```

- **LaTeX**: For PDF generation
  ```bash
  sudo apt-get install texlive-xetex texlive-latex-base texlive-latex-extra
  ```

### Optional (for diagram conversion)

- **inkscape**: Best quality SVG conversion (recommended)
  ```bash
  sudo apt-get install inkscape
  ```

- **rsvg-convert**: Alternative SVG converter
  ```bash
  sudo apt-get install librsvg2-bin
  ```

- **cairosvg**: Python fallback (if others not available)
  ```bash
  pip install cairosvg
  ```

### Optional (for PostScript)

- **poppler-utils**: PDF to PostScript conversion
  ```bash
  sudo apt-get install poppler-utils
  ```

## Usage Examples

### First Build

```bash
$ python3 scripts/build_book.py --all
======================================================================
Building Complete Book
======================================================================

Converting 44 diagram(s)...
[1/44] diagram1.svg... ✓
[2/44] diagram2.svg... ✓
...
Building LaTeX source...
Generating PDF...
✓ Build complete in 45.2s
```

### Incremental Build

```bash
$ python3 scripts/build_book.py --pdf
✓ All diagrams up-to-date
~ Chapter 02 changed, rebuilding...
~ LaTeX source changed, recompiling...
✓ Build complete in 3.1s (42.1s saved)
```

### Quality Validation

```bash
$ python3 scripts/validate_diagram_quality.py
======================================================================
Diagram Quality Validation Report
======================================================================

Total SVG diagrams: 44

PDF Validation:
  ✓ Valid: 44
  ✗ Missing: 0
  ✗ Errors: 0

PNG Validation (for Word):
  ✓ Valid: 44
  ✗ Missing: 0
  ✗ Errors: 0
```

## Architecture

### Core Components

1. **build_cache.py**: Dependency tracking and file hash management
2. **build_manifest.py**: Build manifest and dependency graph
3. **convert_svg_to_pdf.py**: SVG to PDF conversion with multiple backends
4. **build_latex_book.py**: Markdown to LaTeX conversion
5. **generate_pdf.py**: LaTeX to PDF compilation
6. **generate_postscript.py**: PDF to PostScript conversion
7. **generate_word.py**: Markdown to Word with diagram embedding
8. **validate_diagram_quality.py**: Quality validation
9. **build_book.py**: Main build script (unified entry point)

### Build Flow

```
Markdown Chapters
    ↓
SVG Diagrams → PDF (for LaTeX) / PNG (for Word)
    ↓
LaTeX Source (with diagram references)
    ↓
PDF (via XeLaTeX/pdflatex)
    ↓
PostScript (optional)
```

## Troubleshooting

### "pandoc not found"
```bash
sudo apt-get install pandoc
```

### "LaTeX engine not found"
```bash
sudo apt-get install texlive-xetex texlive-latex-base
```

### "No SVG converter found"
Install one of:
- `sudo apt-get install inkscape` (recommended)
- `sudo apt-get install librsvg2-bin`
- `pip install cairosvg`

### Diagrams not appearing in PDF
- Check that diagrams were converted: `ls output/diagrams/`
- Verify image references in LaTeX source
- Check LaTeX compilation logs for missing file errors

### Build cache issues
```bash
# Clean and rebuild
python3 scripts/build_book.py --clean
python3 scripts/build_book.py --all --force
```

## Advanced Usage

### Custom Paths

```bash
python3 scripts/build_book.py --all \
  --book-dir custom/book \
  --output-dir custom/output \
  --cache-dir custom/.build
```

### Parallel Diagram Conversion

The build system automatically uses parallel processing for diagram conversion. The number of workers defaults to CPU count, but can be controlled via the `convert_svg_to_pdf` module.

### Incremental Build Details

The build system tracks:
- File hashes (SHA256)
- Modification times
- Dependency relationships
- Output file states

Only changed files and their dependents are rebuilt.

## Integration

The build system can be integrated into CI/CD pipelines:

```bash
# In CI script
python3 scripts/build_book.py --all --quiet
if [ $? -eq 0 ]; then
    echo "Build successful"
    # Deploy or publish outputs
else
    echo "Build failed"
    exit 1
fi
```

## See Also

- `templates/book_template.tex`: LaTeX book template
- Individual script files for detailed documentation
- Plan file: `.cursor/plans/professional_book_publishing_pipeline_*.plan.md`

