#!/usr/bin/env python
"""Coverage runner — run tests and generate coverage reports.

Usage:
    python scripts/run_coverage.py              # Full coverage
    python scripts/run_coverage.py --html       # Generate HTML report
    python scripts/run_coverage.py --min=80     # Set threshold (default: 70)
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    """Run tests with coverage and generate reports."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    min_coverage = 70
    for arg in sys.argv:
        if arg.startswith("--min="):
            min_coverage = int(arg.split("=")[1])

    args = [
        sys.executable, "-m", "pytest",
        "tests/",
        "--cov=core",
        "--cov-report=term-missing",
        "--cov-report=xml:coverage.xml",
        f"--cov-fail-under={min_coverage}",
        "-v",
        "--tb=short",
    ]

    if "--html" in sys.argv:
        args.append("--cov-report=html:coverage_html")

    print(f"[VFI-gui] Coverage run (min: {min_coverage}%)", flush=True)
    result = subprocess.run(args, cwd=project_root)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
