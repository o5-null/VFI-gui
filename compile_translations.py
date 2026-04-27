#!/usr/bin/env python3
"""Compile PO files to MO files for VFI-gui i18n.

Usage:
    python compile_translations.py
    
This script compiles all .po files in the locales directory
to .mo files that can be loaded by gettext at runtime.
"""

import os
import subprocess
import sys
from pathlib import Path


def compile_po_to_mo(po_path: Path, mo_path: Path) -> bool:
    """Compile a PO file to MO file using msgfmt.
    
    Args:
        po_path: Path to the .po file
        mo_path: Path to output .mo file
        
    Returns:
        True if compilation succeeded
    """
    # Try using msgfmt from gettext tools
    try:
        result = subprocess.run(
            ["msgfmt", str(po_path), "-o", str(mo_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True
        print(f"msgfmt failed: {result.stderr}")
    except FileNotFoundError:
        print("msgfmt not found, trying Python-based compilation...")
    
    # Fallback: Use Python's polib if available
    try:
        import polib
        po = polib.pofile(str(po_path))
        po.save_as_mofile(str(mo_path))
        return True
    except ImportError:
        print("polib not installed. Install with: pip install polib")
    
    # Manual compilation using gettext module (limited support)
    # This is a simplified approach that works for basic translations
    print(f"Warning: Could not compile {po_path}")
    print("Please install gettext tools or polib package")
    return False


def compile_all_translations():
    """Compile all translation files in the locales directory."""
    locales_dir = Path(__file__).parent / "locales"
    
    if not locales_dir.exists():
        print(f"Locales directory not found: {locales_dir}")
        return False
    
    compiled_count = 0
    
    # Find all .po files
    for po_file in locales_dir.glob("**/*.po"):
        # Corresponding .mo file path
        mo_file = po_file.with_suffix(".mo")
        
        print(f"Compiling: {po_file.relative_to(locales_dir)}")
        
        if compile_po_to_mo(po_file, mo_file):
            print(f"  -> {mo_file.relative_to(locales_dir)}")
            compiled_count += 1
        else:
            print(f"  Failed to compile {po_file}")
    
    print(f"\nCompiled {compiled_count} translation files")
    return compiled_count > 0


if __name__ == "__main__":
    success = compile_all_translations()
    sys.exit(0 if success else 1)