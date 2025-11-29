# How to Update the Book

This guide explains how to keep the complete PDF book up-to-date as you work through the project.

## Quick Update

After making changes to any chapter, worksheet, or code example:

```bash
python3 scripts/build_book.py
```

This will:
- Combine all chapters into a single document
- Include all worksheets and code examples
- Generate a fresh PDF in `output/Understanding_Transformers_Complete.pdf`

## What Gets Included

The book automatically includes:

1. **All chapters** from `book/` directory (in order)
2. **All worksheets** from `worksheets/` directory
3. **All code examples** from `examples/` directory

## File Organization

- **Chapters**: Edit files in `book/` directory
- **Worksheets**: Edit files in `worksheets/` directory  
- **Code**: Edit files in `examples/exampleX_*/main.cpp`
- **Output**: Generated files go to `output/` directory

## Output Formats

Generate different formats:

```bash
# PDF only (default)
python3 scripts/build_book.py

# HTML only
python3 scripts/build_book.py --format html

# Both PDF and HTML
python3 scripts/build_book.py --format both

# Just Markdown (skip PDF generation)
python3 scripts/build_book.py --no-pdf
```

## Adding New Chapters

1. Create new file in `book/` directory (e.g., `11-new-chapter.md`)
2. Add filename to `get_chapter_order()` in `scripts/build_book.py`
3. Run build script

## Troubleshooting

### PDF Generation Fails

**Error**: "pandoc not found"
```bash
sudo apt-get install pandoc texlive-latex-base texlive-xetex
```

**Error**: "xelatex not found"
```bash
sudo apt-get install texlive-xetex
```

**Error**: Unicode characters not rendering
- The script uses XeLaTeX for Unicode support
- If issues persist, check font availability

### Links Not Working in PDF

- Internal links (to sections) work automatically
- Links to worksheets/code are converted to anchors
- External links work as-is

### Formatting Issues

- Check Markdown syntax in source files
- LaTeX math should use `$...$` for inline, `$$...$$` for display
- Code blocks should use triple backticks with language tags

## Best Practices

1. **Update regularly**: Run build script after significant changes
2. **Check output**: Review generated PDF to catch formatting issues
3. **Version control**: Commit source files, not generated PDFs (add `output/` to `.gitignore`)
4. **Test links**: Verify internal links work in PDF
5. **Consistent style**: Follow existing chapter format

## Customization

Edit `scripts/build_book.py` to:
- Change chapter order
- Modify PDF formatting (margins, fonts, etc.)
- Add custom sections
- Adjust LaTeX settings

## Example Workflow

```bash
# 1. Edit a chapter
vim book/05-example1-forward-pass.md

# 2. Update worksheet
vim worksheets/example1_worksheet.md

# 3. Rebuild book
python3 scripts/build_book.py

# 4. Check output
evince output/Understanding_Transformers_Complete.pdf

# 5. Commit changes
git add book/ worksheets/
git commit -m "Update example 1 chapter and worksheet"
```

