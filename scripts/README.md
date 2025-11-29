# Book Building Scripts

## build_book.py

Compiles all chapter files into a complete, professionally formatted book.

### Features

- **Combines all chapters** into a single document
- **Organizes content**: Foundations → Examples → Appendices → Worksheets → Code
- **Professional PDF** with proper LaTeX formatting
- **Working links** (converted for PDF format)
- **Table of contents** with proper numbering
- **Code syntax highlighting**
- **Maintainable**: Run script to regenerate book as you update chapters

### Usage

```bash
# Generate PDF (default)
python3 scripts/build_book.py

# Generate HTML
python3 scripts/build_book.py --format html

# Generate both PDF and HTML
python3 scripts/build_book.py --format both

# Just create Markdown (skip PDF generation)
python3 scripts/build_book.py --no-pdf
```

### Output

Files are generated in `output/` directory:
- `COMPLETE_BOOK.md` - Combined Markdown source
- `Understanding_Transformers_Complete.pdf` - Professional PDF book
- `Understanding_Transformers_Complete.html` - HTML version (if requested)

### Requirements

For PDF generation:
```bash
sudo apt-get install pandoc texlive-latex-base texlive-latex-extra
```

For HTML generation:
```bash
sudo apt-get install pandoc
```

### Book Structure

The compiled book includes:

1. **Title Page** - Book title and metadata
2. **Table of Contents** - Auto-generated
3. **Part I: Foundations** - Chapters 1-4
4. **Part II: Progressive Examples** - Examples 1-6
5. **Appendices** - Reference materials
6. **Conclusion**
7. **Part: Worksheets** - All hand-calculation guides
8. **Part: Code Examples** - Complete C++ implementations

### Updating the Book

Simply run the script after making changes to any chapter:

```bash
python3 scripts/build_book.py
```

The script automatically:
- Reads all current chapter files
- Combines them in the correct order
- Fixes links for PDF format
- Generates fresh output

### Customization

Edit `build_book.py` to:
- Change chapter order
- Modify PDF formatting
- Add custom sections
- Adjust LaTeX settings

