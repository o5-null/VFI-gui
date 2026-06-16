#!/usr/bin/env python
"""Model inference smoke test.

Loads example frames from *VFI-gui/example/*, runs each registered VFI model
with available checkpoint weights, and reports timing.

Checkpoint paths are auto-resolved from *VFI-gui/models/*.

Usage:
    python scripts/run_inference.py                     # All models, default
    python scripts/run_inference.py --models rife,amt   # Selected only
    python scripts/run_inference.py --device cpu         # Force CPU
    python scripts/run_inference.py --timestep 0.5      # Interpolation factor
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


# Default checkpoint paths relative to project root.
# Models without weights here will attempt load_model() with their fallback.
DEFAULT_CKPTS: dict[str, str] = {
    "RIFE": "models/rife/rife47.pth",
    "FILM": "models/film/film_net_fp32.pt",
    "IFRNET": "models/ifrnet/IFRNet_L_Vimeo90K.pth",
    "AMT": "models/amt/amt-g.pth",
    "XVFI": "models/xvfi/XVFInet_Vimeo_exp1_latest.pt",
}

# Models that need special config overrides.
MODEL_SPECIAL_CONFIG: dict[str, dict] = {
    "GMFSS": {"model_version": "fortuna", "checkpoint_path": "models/gmfss_fortuna"},
}

# Models listed in ModelType but not yet registered.
MODEL_SKIP: set = {"STMFNET", "FLAVR", "CAIN"}


def _resolve_runtime_python() -> str:
    """Find the GPU runtime python executable."""
    project_root = Path(__file__).resolve().parent.parent
    for candidate in [
        project_root / "runtime" / "xpu" / "Scripts" / "python.exe",
        project_root / "runtime" / "cuda" / "Scripts" / "python.exe",
    ]:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _main_impl(args_list: list[str]) -> int:
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.pytorch_models.vfi_torch import (
        ModelType, VFIConfig, get_model, MODEL_REGISTRY,
    )

    # ---- Parse ----
    parser = argparse.ArgumentParser(description="VFI model inference smoke test")
    parser.add_argument("--models", default="",
                        help="Comma-separated model names (default: all)")
    parser.add_argument("--device", default="auto",
                        help="Force device: cpu, cuda:0, xpu:0")
    parser.add_argument("--timestep", type=float, default=0.5,
                        help="Interpolation timestep (0-1)")
    parser.add_argument("--example-dir", default="",
                        help="Example frames directory")
    args = parser.parse_args(args_list)

    # ---- Device ----
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu:0")
            device_name = f"XPU ({torch.xpu.get_device_name(0)})"
        else:
            device = torch.device("cpu")
            device_name = "CPU"
    else:
        device = torch.device(args.device)
        device_name = str(device)
    print(f"[run_inference] Device: {device_name}")
    print()

    # ---- Example frames ----
    example_dir = Path(args.example_dir) if args.example_dir else (
        Path(__file__).resolve().parent.parent / "example"
    )
    frame_dir = example_dir / "1080p"
    if not frame_dir.is_dir():
        print(f"[run_inference] WARNING: Example directory not found at {frame_dir}")
        print("[run_inference] Using random tensors instead")
        frame0 = torch.rand(3, 1080, 1920)
        frame1 = torch.rand(3, 1080, 1920)
    else:
        from PIL import Image
        import numpy as np
        pngs = sorted(frame_dir.glob("*.png"))
        png0_path, png1_path = pngs[0], pngs[1]
        np0 = np.array(Image.open(str(png0_path)), dtype=np.float32)
        np1 = np.array(Image.open(str(png1_path)), dtype=np.float32)
        frame0 = torch.from_numpy(np0).permute(2, 0, 1) / 255.0 * 2.0 - 1.0
        frame1 = torch.from_numpy(np1).permute(2, 0, 1) / 255.0 * 2.0 - 1.0
        print(f"[run_inference] Loaded: {png0_path.name} + {png1_path.name} "
              f"({frame0.shape[2]}x{frame0.shape[1]})")
    frame0 = frame0.to(device)
    frame1 = frame1.to(device)

    # ---- Model filter ----
    if args.models:
        selected = [m.strip().upper() for m in args.models.split(",")]
    else:
        selected = [m.name for m in ModelType]
    print(f"[run_inference] Models to test: {', '.join(selected)}")
    print()

    # ---- Run ----
    results: list[dict] = []
    for model_type in ModelType:
        if model_type.name not in selected:
            continue
        if model_type.name in MODEL_SKIP:
            print(f"  ⏭ SKIP  {model_type.name:12s} — not yet in MODEL_REGISTRY")
            continue
        if model_type not in MODEL_REGISTRY:
            print(f"  ⏭ SKIP  {model_type.name:12s} — not in MODEL_REGISTRY")
            continue

        try:
            # Build config with checkpoint path
            overrides = MODEL_SPECIAL_CONFIG.get(model_type.name, {})
            ckpt_path = overrides.get(
                "checkpoint_path",
                DEFAULT_CKPTS.get(model_type.name, ""),
            )
            config = VFIConfig(
                model_type=model_type,
                device=str(device),
                model_version=overrides.get("model_version", model_type.value),
                checkpoint_path=ckpt_path,
            )
            model = get_model(model_type, config)
            model.eval()

            # Load checkpoint
            try:
                model.load_model()
            except TypeError:
                # load_model requires positional argument
                model.load_model(str(Path(config.checkpoint_path).resolve()))

            # Warmup
            with torch.no_grad():
                _ = model.interpolate(frame0, frame1, timestep=args.timestep)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elif hasattr(torch, "xpu") and device.type == "xpu":
                torch.xpu.synchronize()

            # Timed run
            num_iters = 3
            t0 = time.perf_counter()
            out = None
            with torch.no_grad():
                for _ in range(num_iters):
                    out = model.interpolate(frame0, frame1, timestep=args.timestep)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif hasattr(torch, "xpu") and device.type == "xpu":
                torch.xpu.synchronize()
            elapsed = time.perf_counter() - t0
            if out is None:
                raise RuntimeError("Model returned None")

            avg_ms = elapsed / num_iters * 1000
            results.append({
                "name": model_type.name,
                "out_shape": list(out.shape),
                "avg_ms": avg_ms,
                "fps": 1000.0 / avg_ms,
                "status": "OK",
            })
            print(f"  ✅ {model_type.name:12s}  {avg_ms:6.0f} ms  "
                  f"{str(list(out.shape)):20s}  ({1000.0/avg_ms:.1f} FPS)")
        except Exception as e:
            msg = str(e).split("\n")[0]
            results.append({"name": model_type.name, "status": "FAIL", "error": msg})
            print(f"  ❌ {model_type.name:12s}  {msg}")

    # ---- Summary ----
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r["status"] == "OK")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    for r in results:
        icon = "✅" if r["status"] == "OK" else "❌"
        if r["status"] == "OK":
            print(f"  {icon} {r['name']:12s}  {r['avg_ms']:6.0f} ms  {r['out_shape']}")
        else:
            print(f"  {icon} {r['name']:12s}  FAILED: {r['error']}")
    print(f"  Passed: {passed}/{len(results)}  Failed: {failed}/{len(results)}")
    return 1 if failed > 0 else 0


def main() -> int:
    script_args = sys.argv[1:]
    python_exe = _resolve_runtime_python()
    if python_exe != sys.executable:
        import subprocess
        cmd = [python_exe, __file__] + script_args
        print(f"[run_inference] Re-executing with: {python_exe}", flush=True)
        result = subprocess.run(cmd)
        return result.returncode
    return _main_impl(script_args)


if __name__ == "__main__":
    sys.exit(main())
