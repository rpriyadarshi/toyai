#!/usr/bin/env python3
"""
Convert BOOK.md to Word/PDF formats

This script converts the Markdown book to other formats using pandoc.
"""

import subprocess
import sys
import os

def convert_to_word(md_file, output_file):
    """Convert Markdown to Word (.docx)"""
    try:
        subprocess.run([
            'pandoc',
            md_file,
            '-o', output_file,
            '--reference-doc', 'reference.docx' if os.path.exists('reference.docx') else None
        ], check=True, stderr=subprocess.PIPE)
        print(f"✓ Converted to {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error converting to Word: {e}")
        return False
    except FileNotFoundError:
        print("✗ pandoc not found. Install with: sudo apt-get install pandoc")
        return False

def convert_to_pdf(md_file, output_file):
    """Convert Markdown to PDF"""
    try:
        subprocess.run([
            'pandoc',
            md_file,
            '-o', output_file,
            '--pdf-engine=pdflatex',
            '-V', 'geometry:margin=1in'
        ], check=True, stderr=subprocess.PIPE)
        print(f"✓ Converted to {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error converting to PDF: {e}")
        return False
    except FileNotFoundError:
        print("✗ pandoc not found. Install with: sudo apt-get install pandoc")
        return False

def main():
    md_file = 'BOOK.md'
    
    if not os.path.exists(md_file):
        print(f"✗ {md_file} not found")
        sys.exit(1)
    
    print("Converting BOOK.md to other formats...\n")
    
    # Convert to Word
    word_file = 'BOOK.docx'
    convert_to_word(md_file, word_file)
    
    # Convert to PDF
    pdf_file = 'BOOK.pdf'
    convert_to_pdf(md_file, pdf_file)
    
    print("\nConversion complete!")

if __name__ == '__main__':
    main()

