#!/usr/bin/env python3
"""
Syntax checker to verify Python files have correct syntax.
This doesn't require torch or other dependencies.
"""

import os
import sys
import py_compile
from pathlib import Path

def check_python_file(filepath):
    """Check if a Python file has valid syntax"""
    try:
        py_compile.compile(filepath, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)

def check_directory(directory, extensions=('.py',)):
    """Recursively check all Python files in a directory"""
    errors = []
    checked = 0
    
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ and .git directories
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', '.eggs', 'build', 'dist']]
        
        for file in files:
            if file.endswith(extensions):
                filepath = os.path.join(root, file)
                checked += 1
                success, error = check_python_file(filepath)
                if not success:
                    errors.append((filepath, error))
                    print(f"✗ {filepath}")
                    print(f"  Error: {error}")
                else:
                    print(f"✓ {filepath}")
    
    return checked, errors

def main():
    """Check all Python files in the project"""
    print("="*70)
    print("GOAT Python Syntax Check")
    print("="*70)
    
    project_root = Path(__file__).parent.parent
    
    # Check main directories
    directories_to_check = [
        project_root / 'goat',
        project_root / 'data',
        project_root / 'scripts',
        project_root / 'tests',
    ]
    
    total_checked = 0
    all_errors = []
    
    for directory in directories_to_check:
        if directory.exists():
            print(f"\nChecking {directory}...")
            checked, errors = check_directory(directory)
            total_checked += checked
            all_errors.extend(errors)
    
    print("\n" + "="*70)
    print(f"Checked {total_checked} Python files")
    
    if not all_errors:
        print("✓ ALL FILES HAVE VALID SYNTAX!")
        print("The code is syntactically correct and ready to run.")
        return 0
    else:
        print(f"✗ FOUND {len(all_errors)} FILES WITH SYNTAX ERRORS!")
        print("\nErrors:")
        for filepath, error in all_errors:
            print(f"  - {filepath}")
            print(f"    {error}")
        return 1
    
    print("="*70)

if __name__ == "__main__":
    sys.exit(main())

