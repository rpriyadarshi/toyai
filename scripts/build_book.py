#!/usr/bin/env python3
"""
Build Complete PDF Book from Chapter Files

This script:
1. Combines all chapter files into a single document
2. Organizes content (foundations, examples, appendices, worksheets, code)
3. Generates professional PDF with proper formatting
4. Ensures all links work in PDF format
5. Creates a maintainable, updatable book

Usage:
    python3 scripts/build_book.py [--format pdf|html|docx]
"""

import os
import sys
import subprocess
import re
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime

# Configuration
BOOK_DIR = Path(__file__).parent.parent / "book"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
WORKSHEETS_DIR = Path(__file__).parent.parent / "worksheets"
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

def get_chapter_order():
    """Define the order of chapters in the final book"""
    return [
        # Part I: Foundations
        "00-index.md",
        "01-terminology-foundations.md",  # Terminology FIRST - explains all concepts
        "02-matrix-core.md",              # Matrix operations - foundation for everything
        "03-embeddings.md",               # Tokens to vectors
        "04-attention-intuition.md",      # Attention mechanism
        "05-why-transformers.md",         # Why transformers (moved from 01)
        
        # Part II: Examples
        "06-example1-forward-pass.md",    # Renumbered from 05
        "07-example2-single-step.md",     # Renumbered from 06
        "08-example3-full-backprop.md",   # Renumbered from 07
        "09-example4-multiple-patterns.md", # Renumbered from 08
        "10-example5-feedforward.md",     # Renumbered from 09
        "11-example6-complete.md",        # Renumbered from 10
        
        # Appendices
        "appendix-a-matrix-calculus.md",
        "appendix-b-hand-calculation-tips.md",
        "appendix-c-common-mistakes.md",
        
        # Conclusion
        "conclusion.md",
    ]

def read_chapter(filename):
    """Read a chapter file and return its content"""
    filepath = BOOK_DIR / filename
    if not filepath.exists():
        print(f"Warning: {filename} not found")
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove navigation links (not needed in compiled book)
    content = re.sub(r'\n---\n\*\*Navigation:\*\*.*?\n---\n', '', content, flags=re.DOTALL)
    
    return content

def read_worksheet(example_num):
    """Read a worksheet file and fix heading levels"""
    filepath = WORKSHEETS_DIR / f"example{example_num}_worksheet.md"
    if not filepath.exists():
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix heading levels: convert # Hand Calculation Worksheet to ## (section level)
    # This prevents creating separate chapters
    content = re.sub(r'^# Hand Calculation Worksheet: Example \d+', 
                     r'## Hand Calculation Worksheet', content, flags=re.MULTILINE)
    
    # Convert all other headings down one level to maintain hierarchy
    # ## becomes ###, ### becomes ####, etc.
    lines = content.split('\n')
    fixed_lines = []
    for line in lines:
        if line.startswith('## '):
            # Skip the main heading we already fixed, convert others
            if not line.startswith('## Hand Calculation Worksheet'):
                fixed_lines.append('###' + line[2:])  # ## -> ###
            else:
                fixed_lines.append(line)
        elif line.startswith('### '):
            fixed_lines.append('####' + line[3:])  # ### -> ####
        elif line.startswith('#### '):
            fixed_lines.append('#####' + line[4:])  # #### -> #####
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def read_code_example(example_num):
    """Read a code example file"""
    example_dirs = {
        1: "example1_forward_only",
        2: "example2_single_step",
        3: "example3_full_backprop",
        4: "example4_multiple_patterns",
        5: "example5_feedforward",
        6: "example6_complete",
    }
    
    if example_num not in example_dirs:
        return None
    
    filepath = EXAMPLES_DIR / example_dirs[example_num] / "main.cpp"
    if not filepath.exists():
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return content

def convert_mermaid_to_images(content, output_dir):
    """Convert Mermaid diagrams to images and replace in content"""
    # DISABLED: Skip Mermaid conversion for now (only needed for PDF)
    # Mermaid diagrams will remain as code blocks in Markdown
    # This can be re-enabled when PDF generation is needed
    return content
    
    # BELOW CODE IS DISABLED - uncomment when PDF generation is needed
    # The code below converts Mermaid diagrams to PNG images for PDF rendering
    # For now, we keep Mermaid as code blocks which work fine in Markdown viewers
    """
    print("  Converting Mermaid diagrams to images...")
    
    mermaid_dir = output_dir / "mermaid_images"
    mermaid_dir.mkdir(exist_ok=True)
    
    mermaid_pattern = r'```mermaid\n(.*?)```'
    
    def replace_mermaid(match):
        mermaid_code = match.group(1)
        mermaid_code = re.sub(r",?\s*['\"]?fontSize['\"]?:\s*['\"]?\d+px", "", mermaid_code)
        mermaid_code = re.sub(r"'themeVariables':\s*\{\s*\}", "", mermaid_code)
        mermaid_code = re.sub(r",\s*'themeVariables':\s*\{\s*\}", "", mermaid_code)
        
        mermaid_hash = hashlib.md5(mermaid_code.encode()).hexdigest()[:8]
        image_filename = f"mermaid_{mermaid_hash}.png"
        image_path = mermaid_dir / image_filename
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as tmp_file:
            tmp_file.write(mermaid_code)
            tmp_mmd = tmp_file.name
        
        try:
            puppeteer_config = OUTPUT_DIR / "puppeteer-config.json"
            with open(puppeteer_config, 'w') as f:
                f.write('{"args": ["--no-sandbox", "--disable-setuid-sandbox"]}\n')
            
            result = subprocess.run([
                'mmdc', '-i', tmp_mmd, '-o', str(image_path),
                '-w', '2400', '-H', '1800', '-b', 'white', '-s', '4',
                '--puppeteerConfigFile', str(puppeteer_config)
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and image_path.exists():
                rel_path = f"mermaid_images/{image_filename}"
                return f'![Mermaid Diagram]({rel_path})'
            else:
                return match.group(0)
        except Exception as e:
            return match.group(0)
        finally:
            try:
                os.unlink(tmp_mmd)
            except:
                pass
    
    content = re.sub(mermaid_pattern, replace_mermaid, content, flags=re.DOTALL)
    return content
    """

def fix_links_for_pdf(content):
    """Fix links to work in PDF format"""
    # Convert relative links to work in compiled document
    # For PDF, pandoc will handle internal links automatically
    
    # Fix links to other chapters (remove .md extension, pandoc handles section links)
    # Keep chapter links as-is - pandoc will convert them to section references
    content = re.sub(r'\[([^\]]+)\]\(([^)]+)\.md\)', r'[\1](\2)', content)
    
    # Fix links to worksheets (will be in same document with anchors)
    content = re.sub(r'\[worksheet\]\(\.\./worksheets/example(\d+)_worksheet\.md\)', 
                     r'[worksheet](#worksheet-example\1_worksheet)', content)
    
    # Fix links to code (will be in same document with anchors)
    content = re.sub(r'\[code\]\(\.\./examples/([^)]+)/main\.cpp\)', 
                     r'[code](#code-\1/main.cpp)', content)
    
    # Fix any remaining relative paths
    content = re.sub(r'\(\.\./worksheets/([^)]+)\)', r'(#worksheet-\1)', content)
    content = re.sub(r'\(\.\./examples/([^)]+)\)', r'(#code-\1)', content)
    
    # Escape special LaTeX characters in code blocks (but not in math mode)
    # This is handled by pandoc, but we can ensure proper encoding
    # Replace problematic Unicode arrows with LaTeX equivalents in non-code contexts
    # (pandoc should handle this, but just in case)
    
    return content

def build_complete_book():
    """Build the complete book document"""
    print("Building complete book...")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Start building the document
    book_content = []
    
    # Title page
    book_content.append("""---
title: "Understanding Transformers: From First Principles to Mastery"
subtitle: "A Progressive Learning System with Hand-Calculable Examples"
author: "ToyAI Educational Project"
date: """ + datetime.now().strftime("%Y-%m-%d") + """
documentclass: book
geometry: margin=1in
fontsize: 11pt
linestretch: 1.2
numbersections: true
header-includes:
  - \\usepackage{hyperref}
  - \\usepackage{bookmark}
  - \\usepackage{enumitem}
  - \\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue, bookmarksopen=true, bookmarksopenlevel=2}
  - \\bookmarksetup{startatroot}
  - \\setlist{nosep,leftmargin=*,itemsep=0.5em}
  - \\renewenvironment{itemize}{\\begin{list}{$\\bullet$}{\\setlength{\\leftmargin}{1.5em}\\setlength{\\itemsep}{0.3em}\\setlength{\\parsep}{0pt}\\setlength{\\topsep}{0.5em}}}{\\end{list}}
  - \\AtBeginDocument{\\pdfbookmark[0]{Title Page}{titlepage}}
  - \\let\\oldmaketitle\\maketitle
  - \\renewcommand\\maketitle{\\oldmaketitle}
  - \\let\\oldtoc\\tableofcontents
  - \\renewcommand\\tableofcontents{\\cleardoublepage\\phantomsection\\pdfbookmark[0]{Table of Contents}{toc}\\oldtoc}
---

\\frontmatter

\\begingroup
\\hypersetup{bookmarks=false}
\\addtocontents{toc}{}
# Understanding Transformers: From First Principles to Mastery
\\endgroup

**A Progressive Learning System with Hand-Calculable Examples**

*Learn transformers from first principles using 2x2 matrices you can compute by hand.*

\\mainmatter

""")
    
    # Add all chapters
    chapter_order = get_chapter_order()
    
    for i, chapter_file in enumerate(chapter_order):
        print(f"  Adding {chapter_file}...")
        content = read_chapter(chapter_file)
        
        if content:
            # Skip the index file entirely (we've already added "How to Use" section)
            if chapter_file == "00-index.md":
                continue
            
            # Check if this is an example chapter
            is_example = chapter_file.startswith('0') and 'example' in chapter_file.lower()
            example_num = None
            if is_example:
                # Extract example number from filename (e.g., "06-example1-forward-pass.md" -> 1)
                match = re.search(r'example(\d+)', chapter_file)
                if match:
                    example_num = int(match.group(1))
            
            # Fix links for PDF
            content = fix_links_for_pdf(content)
            
            # Convert chapter headings from ## to # (level 2 to level 1)
            # This ensures chapters are properly recognized in LaTeX book class
            # Pattern: ## Chapter X: or ## Example X: or ## Appendix X:
            content = re.sub(r'^## (Chapter \d+:|Example \d+:|Appendix [A-Z]:)', r'# \1', content, flags=re.MULTILINE)
            # Also handle standalone chapter titles like "## Conclusion"
            if 'conclusion' in chapter_file.lower() or 'appendix' in chapter_file.lower():
                # For conclusion and appendices, convert first ## to #
                lines = content.split('\n')
                if lines and lines[0].startswith('## '):
                    lines[0] = lines[0].replace('## ', '# ', 1)
                    content = '\n'.join(lines)
            
            # For example chapters, wrap existing content in Theory section and add worksheet/code
            if is_example and example_num:
                # Remove navigation links at the end
                content = re.sub(r'\n---\n\*\*Navigation:.*?\n---\n?$', '', content, flags=re.DOTALL)
                
                # Wrap existing content in ## Theory section
                # Find where the content starts (after the chapter heading)
                lines = content.split('\n')
                theory_start = 0
                for j, line in enumerate(lines):
                    if line.startswith('# Example'):
                        theory_start = j + 1
                        break
                
                # Get everything after the chapter heading
                theory_content = '\n'.join(lines[theory_start:])
                theory_content = theory_content.strip()
                
                # Remove any standalone "## Theory" or "### Theory" subsections since we're wrapping everything in Theory
                # These will become redundant
                theory_content = re.sub(r'^## Theory\s*$', '', theory_content, flags=re.MULTILINE)
                theory_content = re.sub(r'^### Theory\s*$', '', theory_content, flags=re.MULTILINE)
                
                # Remove old "## Code Implementation" section with just a link - we'll add the actual code later
                # Match the entire section including the heading, link, and blank lines
                # Handle various whitespace patterns
                theory_content = re.sub(r'\n## Code Implementation\s*\n+See \[code\][^\n]*\n+', '\n\n', theory_content, flags=re.MULTILINE)
                
                # Convert first level of subsections from ### to ## (level 3 to level 2)
                # In LaTeX book class: # = Chapter, ## = Section (1.1), ### = Subsection (1.1.1)
                # Convert the top-level ### in the original content to ##
                theory_lines = theory_content.split('\n')
                fixed_theory = []
                for line in theory_lines:
                    if line.startswith('### ') and not line.startswith('#### '):
                        # This is a top-level subsection in the original, convert to ##
                        fixed_theory.append('##' + line[3:])
                    else:
                        fixed_theory.append(line)
                theory_content = '\n'.join(fixed_theory)
                
                # Now remove the old "## Code Implementation" section that was just a link
                # (it was originally ### Code Implementation, now converted to ##)
                theory_content = re.sub(r'\n## Code Implementation\s*\n+See \[code\][^\n]*\n+', '\n\n', theory_content, flags=re.MULTILINE)
                
                # Rebuild content with Theory section wrapper
                content = '\n'.join(lines[:theory_start]) + '\n\n## Theory\n\n' + theory_content
                
                # Add worksheet section (after Theory, before Code)
                worksheet = read_worksheet(example_num)
                if worksheet:
                    content += "\n\n" + worksheet
                
                # Add code section (after Worksheet)
                code = read_code_example(example_num)
                if code:
                    content += "\n\n## Code Implementation\n\n"
                    content += "```cpp\n"
                    content += code
                    content += "\n```\n"
            else:
                # For non-example chapters, just convert subsection levels
                # Convert first level of subsections from ### to ## (level 3 to level 2)
                content = re.sub(r'^### ', r'## ', content, flags=re.MULTILINE)
            
            # Append content for all chapters (both examples and non-examples)
            book_content.append(content + "\n\n")
    
    # Combine everything
    full_book = ''.join(book_content)
    
    # Convert Mermaid diagrams to images BEFORE writing
    full_book = convert_mermaid_to_images(full_book, OUTPUT_DIR)
    
    # Write to file
    output_md = OUTPUT_DIR / "COMPLETE_BOOK.md"
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write(full_book)
    
    print(f"\n✓ Complete book written to: {output_md}")
    return output_md

def generate_pdf(md_file):
    """Generate PDF from Markdown using pandoc"""
    print("\nGenerating PDF...")
    
    pdf_file = OUTPUT_DIR / "Understanding_Transformers_Complete.pdf"
    
    try:
        # Try XeLaTeX first (better Unicode support), fallback to pdflatex
        # Use XeLaTeX for Unicode support
        pdf_engine = 'xelatex'
        try:
            subprocess.run(['which', 'xelatex'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to pdflatex if xelatex not available
            pdf_engine = 'pdflatex'
            print("  Note: xelatex not found, using pdflatex (Unicode may be limited)")
        
        cmd = [
            'pandoc',
            str(md_file),
            '-o', str(pdf_file),
            f'--pdf-engine={pdf_engine}',
            '--from=markdown+tex_math_dollars+raw_tex+lists_without_preceding_blankline',
            '--to=pdf',
            '-V', 'geometry:margin=1in',
            '-V', 'fontsize=11pt',
            '-V', 'linestretch=1.2',
            '-V', 'documentclass=book',
            '-V', 'classoption=openany',  # Allow chapters to start on any page
            '-V', 'colorlinks=true',
            '-V', 'linkcolor=blue',
            '-V', 'urlcolor=blue',
            '--toc',
            '--toc-depth=3',
            '--number-sections',
            '--highlight-style=tango',
            '--standalone',
        ]
        
        # Add Unicode support only for XeLaTeX
        if pdf_engine == 'xelatex':
            # Use system fonts that are typically available
            # XeLaTeX will use default fonts if these aren't found
            cmd.extend([
            '-V', 'mainfont=Liberation Serif',
            '-V', 'sansfont=Liberation Sans',
            '-V', 'monofont=Liberation Mono',
            ])
        
        # Add LaTeX packages for better code/diagram rendering
        header_includes = r'''
\usepackage{fancyvrb}
\usepackage{adjustbox}
\fvset{fontsize=\normalsize}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{commandchars=\\\{\},formatcom=\color[rgb]{0,0,0}}
'''
        cmd.extend(['-H', '/dev/stdin'])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ PDF generated: {pdf_file}")
            print(f"  File size: {pdf_file.stat().st_size / 1024 / 1024:.2f} MB")
            return True
        else:
            print(f"✗ PDF generation failed:")
            print(result.stderr)
            return False
            
    except FileNotFoundError:
        print("✗ pandoc not found. Install with: sudo apt-get install pandoc texlive-latex-base")
        return False

def generate_html(md_file):
    """Generate HTML version"""
    print("\nGenerating HTML...")
    
    html_file = OUTPUT_DIR / "Understanding_Transformers_Complete.html"
    
    try:
        cmd = [
            'pandoc',
            str(md_file),
            '-o', str(html_file),
            '--standalone',
            '--toc',
            '--toc-depth=3',
            '--mathjax',
            '--css', 'https://cdn.jsdelivr.net/npm/water.css@2/out/water.css',
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ HTML generated: {html_file}")
            return True
        else:
            print(f"✗ HTML generation failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("✗ pandoc not found")
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build complete book from chapters')
    parser.add_argument('--pdf', action='store_true',
                       help='Generate PDF (requires pandoc and LaTeX)')
    parser.add_argument('--html', action='store_true',
                       help='Generate HTML (requires pandoc)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Building Complete Book")
    print("=" * 70)
    print()
    
    # Build the complete book
    md_file = build_complete_book()
    
    if not md_file:
        print("✗ Failed to build book")
        sys.exit(1)
    
    # Generate output formats only if explicitly requested
    if args.pdf:
        generate_pdf(md_file)
    
    if args.html:
        generate_html(md_file)
    
    print("\n" + "=" * 70)
    print("Book build complete!")
    print("=" * 70)
    print(f"\nOutput files in: {OUTPUT_DIR}")
    print(f"  - COMPLETE_BOOK.md (source)")
    if args.pdf:
        print(f"  - Understanding_Transformers_Complete.pdf")
    if args.html:
        print(f"  - Understanding_Transformers_Complete.html")
    if not args.pdf and not args.html:
        print("\nNote: PDF/HTML generation skipped. Use --pdf or --html to generate.")
    print()

if __name__ == '__main__':
    main()

