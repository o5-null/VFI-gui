#!/usr/bin/env python
"""End-to-end smoke test for the ONNX Runtime backend.

Exercises the real backend factory/registration path
(``BackendFactory.create(BackendType.ONNX, config)`` -> ``OnnxBackend``) with a
local RIFE ONNX model and runs a small battery of sanity checks:

    - shape / dtype / finite output
    - identity property (frame0 == frame1 == same frame)
    - range sanity
    - timestep sensitivity (informational)
    - provider report
    - timing over several runs

The model file must already exist locally; this script does NOT download it.

Usage:
    python scripts/test_onnx_backend.py
    python scripts/test_onnx_backend.py --model models/rife_v2/rife_v4.26.onnx
    python scripts/test_onnx_backend.py --device cpu --runs 10
    python scripts/test_onnx_backend.py --height 512 --width 512 --timestep 0.5

Exit codes:
    0  all hard checks passed
    1  one or more hard checks failed
    2  model file missing (nothing to run)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Tunable thresholds (module-level so they are easy to adjust)
# ---------------------------------------------------------------------------

# Loose upper bound on mean-abs error for the identity property (frame0 ==
# frame1): a correct interpolator should return (approximately) the input.
IDENTITY_MAE_THRESHOLD = 0.02

# Rough acceptable output range. Small overshoot beyond [0, 1] is tolerated.
RANGE_MIN = -0.05
RANGE_MAX = 1.05

# Seed for reproducible synthetic frames.
RANDOM_SEED = 1234


def _resolve_runtime_python() -> str:
    """Find the GPU runtime python executable (mirrors run_inference.py)."""
    project_root = Path(__file__).resolve().parent.parent
    for candidate in [
        project_root / "runtime" / "xpu" / "Scripts" / "python.exe",
        project_root / "runtime" / "cuda" / "Scripts" / "python.exe",
    ]:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _report(name: str, passed: bool, detail: str = "") -> bool:
    """Print a PASS/FAIL line and return the boolean result."""
    icon = "PASS" if passed else "FAIL"
    line = f"  [{icon}] {name}"
    if detail:
        line += f" — {detail}"
    print(line, flush=True)
    return passed


def _main_impl(args_list: List[str]) -> int:
    project_root = Path(__file__).resolve().parent.parent
    # Ensure the project root is importable before importing core modules.
    sys.path.insert(0, str(project_root))

    import numpy as np
    import torch

    from core.types import BackendType, BackendConfig, InferenceRequest
    from core.backends import BackendFactory

    # ---- Parse ----
    parser = argparse.ArgumentParser(
        description="ONNX Runtime backend end-to-end smoke test"
    )
    parser.add_argument(
        "--model", default="models/rife_v2/rife_v4.26.onnx",
        help="Path to RIFE ONNX model (relative to project root or absolute)",
    )
    parser.add_argument("--height", type=int, default=256, help="Frame height")
    parser.add_argument("--width", type=int, default=256, help="Frame width")
    parser.add_argument(
        "--timestep", type=float, default=0.5, help="Interpolation timestep (0-1)"
    )
    parser.add_argument("--runs", type=int, default=5, help="Timed runs")
    parser.add_argument(
        "--device", default="auto", help="Force device: auto, cpu, cuda:0, xpu:0"
    )
    parser.add_argument(
        "--provider", default=None,
        help="Comma-separated execution providers to force "
             "(e.g. CUDAExecutionProvider,DmlExecutionProvider)",
    )
    parser.add_argument(
        "--list-providers", action="store_true",
        help="Print available ONNX Runtime providers and exit",
    )
    args = parser.parse_args(args_list)

    # ---- --list-providers: report and exit without loading a model ----
    if args.list_providers:
        import onnxruntime as ort

        print("Available ONNX Runtime providers:")
        for name in ort.get_available_providers():
            print(f"  - {name}")
        return 0

    # Resolve model path (relative paths are relative to the project root).
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = project_root / model_path

    print("=" * 60)
    print("ONNX Runtime backend smoke test")
    print("=" * 60)
    print(f"  model     : {model_path}")
    print(f"  frame     : {args.width}x{args.height}")
    print(f"  timestep  : {args.timestep}")
    print(f"  runs      : {args.runs}")
    print(f"  device    : {args.device}")
    print(f"  provider  : {args.provider or '(auto)'}")
    print()

    # ---- Model presence check (exit 2, no traceback) ----
    if not model_path.exists():
        print(f"ERROR: ONNX model not found at: {model_path}")
        print(
            "Download/export the RIFE ONNX model first, or pass --model <path>. "
            "This script does not download models."
        )
        return 2

    # ---- Build config + create backend through the real factory ----
    extra: dict = {}
    if args.provider:
        extra["execution_providers"] = [
            p.strip() for p in args.provider.split(",") if p.strip()
        ]
    config = BackendConfig(
        backend_type=BackendType.ONNX,
        models_dir=str(project_root / "models"),
        device=args.device,
        extra=extra,
    )

    available = [t.value for t in BackendFactory.get_available_backends()]
    if BackendType.ONNX not in BackendFactory.get_available_backends():
        print(f"ERROR: ONNX backend not registered. Available: {available}")
        return 1
    print(f"  registered backends: {available}")

    backend = BackendFactory.create(BackendType.ONNX, config)

    failures = 0

    # ---- initialize() ----
    if not _report("initialize()", backend.initialize() is True):
        print("ERROR: backend.initialize() failed; aborting.")
        return 1

    # ---- load_model() ----
    model_config = {
        "model_type": "rife",
        "model_version": "4.26",
        "onnx_path": str(model_path),
    }
    if not _report("load_model()", backend.load_model(model_config) is True):
        print("ERROR: backend.load_model() failed; aborting.")
        return 1

    # ---- Provider report (informational; warn if CPU-only) ----
    import onnxruntime as ort

    available_providers = ort.get_available_providers()
    active_providers = backend._session.get_providers()
    print(f"  available providers : {available_providers}")
    print(f"  session providers   : {active_providers}")
    _accel = {"TensorrtExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider"}
    if not (_accel & set(active_providers)):
        print("  WARNING: running on CPU only (no accelerator provider active)")
    print()

    # ---- Synthetic frames (seeded for reproducibility) ----
    torch.manual_seed(RANDOM_SEED)
    h, w = args.height, args.width
    frame_a = torch.rand(3, h, w, dtype=torch.float32)
    frame_b = torch.rand(3, h, w, dtype=torch.float32)

    # ---- Test 1: shape / dtype / finite ----
    req = InferenceRequest(frame0=frame_a, frame1=frame_b, timestep=0.5, model_config={})
    res = backend.infer(req)
    if not res.success:
        failures += 1
        _report("shape/dtype/finite", False, f"infer failed: {res.error}")
    else:
        out = res.output_frame
        ok_shape = tuple(out.shape) == (3, h, w)
        ok_dtype = out.dtype == torch.float32
        ok_finite = bool(torch.isfinite(out).all().item())
        if not _report(
            "shape/dtype/finite",
            ok_shape and ok_dtype and ok_finite,
            f"shape={tuple(out.shape)} dtype={out.dtype} finite={ok_finite}",
        ):
            failures += 1

    # ---- Test 2: identity property ----
    identity_req = InferenceRequest(
        frame0=frame_a, frame1=frame_a, timestep=0.5, model_config={}
    )
    id_res = backend.infer(identity_req)
    if not id_res.success:
        failures += 1
        _report("identity property", False, f"infer failed: {id_res.error}")
    else:
        mae = float(torch.mean(torch.abs(id_res.output_frame - frame_a)).item())
        if not _report(
            "identity property",
            mae < IDENTITY_MAE_THRESHOLD,
            f"mean-abs-error={mae:.5f} (threshold {IDENTITY_MAE_THRESHOLD})",
        ):
            failures += 1

    # ---- Test 3: range sanity ----
    if res.success:
        out = res.output_frame
        lo = float(out.min().item())
        hi = float(out.max().item())
        if not _report(
            "range sanity",
            lo >= RANGE_MIN and hi <= RANGE_MAX,
            f"min={lo:.4f} max={hi:.4f} (expect [{RANGE_MIN}, {RANGE_MAX}])",
        ):
            failures += 1

    # ---- Test 4: timestep sensitivity (informational, non-fatal) ----
    print("  timestep sensitivity (informational):")
    prev = None
    for t in (0.25, 0.5, 0.75):
        t_req = InferenceRequest(
            frame0=frame_a, frame1=frame_b, timestep=t, model_config={}
        )
        t_res = backend.infer(t_req)
        if not t_res.success:
            print(f"    t={t:.2f}: infer failed: {t_res.error}")
            continue
        cur = t_res.output_frame
        if prev is not None:
            diff = float(torch.mean(torch.abs(cur - prev)).item())
            print(f"    t={t:.2f}: mean-abs diff vs previous t = {diff:.5f}")
        else:
            print(f"    t={t:.2f}: baseline")
        prev = cur

    # ---- Test 5: timing over --runs ----
    times_ms: List[float] = []
    timing_ok = True
    for _ in range(max(1, args.runs)):
        t_res = backend.infer(req)
        if not t_res.success:
            timing_ok = False
            break
        times_ms.append(float(t_res.inference_time_ms))
    if timing_ok and times_ms:
        mean_ms = sum(times_ms) / len(times_ms)
        print(
            f"  timing: {mean_ms:.2f} ms mean over {len(times_ms)} runs "
            f"(min {min(times_ms):.2f} / max {max(times_ms):.2f})"
        )
    else:
        failures += 1
        _report("timing", False, "one or more timed runs failed")

    # ---- Cleanup ----
    backend.unload_model()
    backend.cleanup()

    print()
    print("=" * 60)
    if failures == 0:
        print("RESULT: ALL HARD CHECKS PASSED")
        return 0
    print(f"RESULT: {failures} HARD CHECK(S) FAILED")
    return 1


def main() -> int:
    script_args = sys.argv[1:]
    python_exe = _resolve_runtime_python()
    if python_exe != sys.executable:
        import subprocess

        cmd = [python_exe, __file__] + script_args
        print(f"[test_onnx_backend] Re-executing with: {python_exe}", flush=True)
        result = subprocess.run(cmd)
        return result.returncode
    return _main_impl(script_args)


if __name__ == "__main__":
    sys.exit(main())
