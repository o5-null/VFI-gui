#!/usr/bin/env python
"""Test runner with coverage and CI-friendly output.

Usage:
    python scripts/run_tests.py                  # Full suite with coverage
    python scripts/run_tests.py --no-cov          # Without coverage
    python scripts/run_tests.py --verbose         # Verbose output
    python scripts/run_tests.py --fail-fast       # Stop on first failure
    python scripts/run_tests.py -- -k "test_type" # Run matching tests
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    """Run the test suite with coverage and CI-friendly output."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Detect CI environment
    is_ci = os.environ.get("CI", "").lower() in ("true", "1")

    # Build pytest args
    args = [
        sys.executable, "-m", "pytest",
        "tests/",
        "--tb=short",
        "--strict-config",
    ]

    # CI-specific flags
    if is_ci:
        args.extend([
            "-v",                       # verbose in CI
            "--color=yes",
            "-p", "no:cacheprovider",   # disable cache for CI
        ])
    else:
        args.append("--tb=short")

    # Parse custom flags from argv
    if "--no-cov" not in sys.argv:
        args.append("--cov=core")
        args.append("--cov-report=term-missing")
        args.append("--cov-report=xml:coverage.xml")
        args.append("--cov-fail-under=70")

    if "--verbose" in sys.argv:
        args.append("-v")

    if "--fail-fast" in sys.argv:
        args.append("--maxfail=1")
    else:
        args.append("--maxfail=10")

    # Forward remaining args to pytest (after --)
    if "--" in sys.argv:
        dash_idx = sys.argv.index("--")
        args.extend(sys.argv[dash_idx + 1:])

    print(f"[VFI-gui] Running: {' '.join(args)}", flush=True)

    result = subprocess.run(args, cwd=project_root)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
