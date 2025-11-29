# Quick Start: Building the Complete Book

## One Command to Build Everything

```bash
python3 scripts/build_book.py
```

This generates:
- `output/COMPLETE_BOOK.md` - Combined Markdown source
- `output/Understanding_Transformers_Complete.pdf` - Professional PDF book (ready to download/print)

## What's Included

The complete book contains:

1. **Title Page** - Book metadata
2. **Table of Contents** - Auto-generated with page numbers
3. **Part I: Foundations** (Chapters 1-4)
4. **Part II: Progressive Examples** (Examples 1-6)
5. **Appendices** (A, B, C)
6. **Conclusion**
7. **Part: Worksheets** - All 6 hand-calculation guides
8. **Part: Code Examples** - Complete C++ code for all examples

## Requirements

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install pandoc texlive-xetex texlive-latex-base

# Verify installation
pandoc --version
xelatex --version
```

## Output Formats

```bash
# PDF (default)
python3 scripts/build_book.py

# HTML
python3 scripts/build_book.py --format html

# Both
python3 scripts/build_book.py --format both

# Just Markdown (skip PDF)
python3 scripts/build_book.py --no-pdf
```

## After Making Changes

1. Edit chapters in `book/` directory
2. Edit worksheets in `worksheets/` directory
3. Edit code in `examples/` directory
4. Run: `python3 scripts/build_book.py`
5. Check: `output/Understanding_Transformers_Complete.pdf`

## File Locations

- **Source chapters**: `book/*.md`
- **Worksheets**: `worksheets/example*_worksheet.md`
- **Code**: `examples/example*/main.cpp`
- **Output**: `output/` (excluded from git)

## Troubleshooting

**"pandoc not found"**
```bash
sudo apt-get install pandoc
```

**"xelatex not found"**
```bash
sudo apt-get install texlive-xetex
```

**PDF has formatting issues**
- Check Markdown syntax in source files
- Ensure proper code block formatting
- Verify LaTeX math syntax (`$...$` for inline, `$$...$$` for display)

## Customization

Edit `scripts/build_book.py`:
- Change chapter order: modify `get_chapter_order()`
- Adjust PDF formatting: modify pandoc command options
- Change fonts: modify font settings

## Full Documentation

See `scripts/README.md` for detailed documentation.
See `scripts/UPDATE_BOOK.md` for update workflow.

