#!/usr/bin/env python
"""Lint and type-check runner for VFI-gui.

Usage:
    python scripts/run_lint.py          # Run all checks
    python scripts/run_lint.py --ruff   # Run ruff only
    python scripts/run_lint.py --types  # Run pyright only
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    """Run lint and type checking tools."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    exit_code = 0
    check_ruff = "--types" not in sys.argv
    check_types = "--ruff" not in sys.argv

    # 1. Ruff lint
    if check_ruff:
        print("\n" + "=" * 60, flush=True)
        print(" [VFI-gui] ruff check (lint + format)", flush=True)
        print("=" * 60, flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "core/", "tests/", "scripts/", "--fix"],
            cwd=project_root,
        )
        if result.returncode != 0:
            exit_code = result.returncode
            print("  -> ruff found issues", flush=True)
        else:
            print("  -> clean", flush=True)

        print("\n" + "=" * 60, flush=True)
        print(" [VFI-gui] ruff format check", flush=True)
        print("=" * 60, flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "format", "core/", "tests/", "scripts/", "--check"],
            cwd=project_root,
        )
        if result.returncode != 0:
            exit_code = result.returncode
            print("  -> formatting issues found (run `ruff format core/ tests/`)", flush=True)
        else:
            print("  -> clean", flush=True)

    # 2. PyRight type checking
    if check_types:
        print("\n" + "=" * 60, flush=True)
        print(" [VFI-gui] pyright type check", flush=True)
        print("=" * 60, flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pyright", "core/", "tests/"],
            cwd=project_root,
        )
        if result.returncode != 0:
            exit_code = result.returncode
            print("  -> type errors found", flush=True)
        else:
            print("  -> clean", flush=True)

    print("\n" + "=" * 60, flush=True)
    if exit_code == 0:
        print(" [VFI-gui] All checks passed!", flush=True)
    else:
        print(f" [VFI-gui] Some checks failed (exit code: {exit_code})", flush=True)
    print("=" * 60, flush=True)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
