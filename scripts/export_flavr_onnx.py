#!/usr/bin/env python
"""Export FLAVR checkpoints (2x / 4x / 8x) to ONNX.

FLAVR is a 4-frame video frame interpolation network (UNet3D3D). Unlike the
per-pair models, its native interface takes 4 input frames and returns
``n_outputs`` intermediate frames in one shot:

    FLAVR_2x.pth -> n_outputs = 1   (output [1,  3, H, W])
    FLAVR_4x.pth -> n_outputs = 3   (output [1,  9, H, W])
    FLAVR_8x.pth -> n_outputs = 7   (output [1, 21, H, W])

ONNX interface (matches the project's uniform RGB [0,1] / NCHW contract):

    input  ``frames`` [1, 12, H, W] float32   (4 frames x 3ch, channel-concat)
    output ``output`` [1, 3*n_outputs, H, W] float32

Verification for every exported file:
    1. onnxruntime load + random-input run (shape / dtype / finite)
    2. mean-abs-error against the PyTorch reference on the same input
    3. on-disk file size and I/O signature

Usage (must run under the CUDA runtime python):
    D:\\code\\VFI\\runtime\\cuda\\Scripts\\python.exe scripts/export_flavr_onnx.py
    # optional:
    ... scripts/export_flavr_onnx.py --size 256 --out-dir models/flavr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Model / checkpoint map: filename -> (n_outputs, display version)
# Kept in sync with core/pytorch_models/vfi_torch/flavr/__init__.py CKPT_CONFIGS
# ---------------------------------------------------------------------------
CKPT_CONFIGS: Dict[str, Dict[str, int]] = {
    "FLAVR_2x.pth": {"n_outputs": 1, "version": 2},
    "FLAVR_4x.pth": {"n_outputs": 3, "version": 4},
    "FLAVR_8x.pth": {"n_outputs": 7, "version": 8},
}

# Default static export resolution (must be divisible by 16 for the encoder).
DEFAULT_SIZE = 256
# Fixed seed so dummy input and the MAE check are reproducible.
RANDOM_SEED = 1234


def _resolve_runtime_python() -> str:
    """Locate the GPU runtime python (runtime/ is a sibling of the project root)."""
    project_root = Path(__file__).resolve().parent.parent
    workspace_root = project_root.parent
    for candidate in [
        workspace_root / "runtime" / "cuda" / "Scripts" / "python.exe",
        workspace_root / "runtime" / "xpu" / "Scripts" / "python.exe",
        project_root / "runtime" / "cuda" / "Scripts" / "python.exe",
    ]:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _strip_prefix(state_dict: dict) -> dict:
    """Drop the DataParallel ``module.`` prefix from checkpoint keys."""
    out = {}
    for k, v in state_dict.items():
        out[k[len("module."):] if k.startswith("module.") else k] = v
    return out


def _install_bn_free_encoder(resnet_3d) -> None:
    """Force the 3D-resnet encoder to build WITHOUT BatchNorm.

    The shipped FLAVR checkpoints contain no encoder BatchNorm tensors, and the
    upstream reference (ComfyUI-Frame-Interpolation) builds the encoder with
    ``bn=False``. The project's ``resnet_3d.unet_18`` unconditionally resets the
    module-global ``batchnorm`` to ``nn.BatchNorm3d``, so patching the global is
    not enough -- we replace ``unet_18`` with a variant that keeps it ``None``.
    """
    def _unet_18_no_bn(pretrained=False, progress=True, **kwargs):
        resnet_3d.batchnorm = None
        return resnet_3d._video_resnet(
            "r3d_18", pretrained, progress,
            block=resnet_3d.BasicBlock,
            conv_makers=[resnet_3d.Conv3DSimple] * 4,
            layers=[2, 2, 2, 2],
            stem=resnet_3d.BasicStem,
            **kwargs,
        )

    resnet_3d.unet_18 = _unet_18_no_bn


def build_reference_model(n_outputs: int, ckpt_path: Path):
    """Build a UNet3D3D and load the checkpoint weights.

    Two encoder construction details are forced explicitly so a single-process
    export of all three checkpoints stays correct:

    * ``useBias`` -- the 3D-resnet reads this module-level global at conv
      construction time. 2x checkpoints have no conv biases; 4x/8x do. The
      project only ever flips it to ``True`` (it leaks across builds), so we set
      it per model.
    * encoder BatchNorm -- disabled (see ``_install_bn_free_encoder``) so the
      state_dict loads 1:1 with no missing keys.
    """
    import torch

    from core.pytorch_models.vfi_torch.flavr import resnet_3d
    from core.pytorch_models.vfi_torch.flavr.flavr_arch import UNet3D3D

    # 2x checkpoints have no conv biases; 4x/8x do (verified on disk).
    resnet_3d.useBias = n_outputs > 1
    _install_bn_free_encoder(resnet_3d)

    model = UNet3D3D(
        "unet_18",
        n_inputs=4,
        n_outputs=n_outputs,
        join_type="concat",
        upmode="transpose",
    )

    state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = _strip_prefix(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading {ckpt_path.name}: {missing[:5]}")
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys when loading {ckpt_path.name}: {unexpected[:5]}"
        )

    model.eval()
    return model


def make_wrapper(model, n_outputs: int):
    """Create the ONNX-exportable wrapper around a reference model."""
    import torch
    from torch import nn

    class _Wrapper(nn.Module):
        def __init__(self, ref_model, n_out: int):
            super().__init__()
            self.model = ref_model
            self.n_outputs = n_out

        def forward(self, frames: "torch.Tensor") -> "torch.Tensor":
            # Split the 12-channel tensor back into 4 RGB frames, run the
            # native 4-frame FLAVR forward, then channel-concat the outputs.
            images = list(torch.split(frames, 3, dim=1))  # 4 x [B,3,H,W]
            outs = self.model(images)                      # list of n x [B,3,H,W]
            return torch.cat(outs, dim=1)                  # [B, 3*n, H, W]

    return _Wrapper(model, n_outputs)


def export_one(
    ckpt_path: Path,
    onnx_path: Path,
    n_outputs: int,
    size: int,
    opset: int,
) -> Path:
    """Export a single checkpoint to ``onnx_path``."""
    import torch

    model = build_reference_model(n_outputs, ckpt_path)
    wrapper = make_wrapper(model, n_outputs)
    wrapper.eval()

    dummy = torch.rand(1, 12, size, size, dtype=torch.float32)

    export_kwargs = dict(
        input_names=["frames"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
    )

    with torch.no_grad():
        try:
            # Prefer the legacy (TorchScript) exporter: clean static shapes.
            torch.onnx.export(wrapper, (dummy,), str(onnx_path), dynamo=False, **export_kwargs)
        except TypeError:
            # Older/newer torch without the dynamo flag -> default exporter.
            torch.onnx.export(wrapper, (dummy,), str(onnx_path), **export_kwargs)

    return onnx_path


def verify_one(onnx_path: Path, ckpt_path: Path, n_outputs: int, size: int) -> dict:
    """Run onnxruntime + PyTorch comparison and return verification metrics."""
    import numpy as np
    import onnxruntime as ort
    import torch

    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)

    # ---- I/O signature ----
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    signature = {
        "input_name": inp.name,
        "input_shape": inp.shape,
        "input_type": inp.type,
        "output_name": out.name,
        "output_shape": out.shape,
        "output_type": out.type,
        "providers": sess.get_providers(),
    }

    # ---- Shared random input (same tensor for both runtimes) ----
    torch.manual_seed(RANDOM_SEED)
    dummy = torch.rand(1, 12, size, size, dtype=torch.float32)

    ort_out = sess.run(["output"], {"frames": dummy.numpy()})[0]

    model = build_reference_model(n_outputs, ckpt_path)
    wrapper = make_wrapper(model, n_outputs)
    wrapper.eval()
    with torch.no_grad():
        torch_out = wrapper(dummy).numpy()

    mae = float(np.mean(np.abs(ort_out - torch_out)))

    return {
        "signature": signature,
        "onnx_shape": tuple(ort_out.shape),
        "onnx_dtype": str(ort_out.dtype),
        "finite": bool(np.isfinite(ort_out).all()),
        "mae": mae,
        "expected_channels": 3 * n_outputs,
    }


def _main_impl(argv: List[str]) -> int:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    parser = argparse.ArgumentParser(description="Export FLAVR 2x/4x/8x to ONNX")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE,
                        help=f"Static H=W export resolution (default {DEFAULT_SIZE})")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset (default 17)")
    parser.add_argument("--out-dir", default="models/flavr",
                        help="Directory for the .onnx files (default models/flavr)")
    parser.add_argument("--models-dir", default="models/flavr",
                        help="Directory containing the .pth checkpoints")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip export if the .onnx already exists")
    args = parser.parse_args(argv)

    if args.size % 16 != 0:
        print(f"ERROR: --size must be divisible by 16, got {args.size}")
        return 1

    models_dir = Path(args.models_dir)
    if not models_dir.is_absolute():
        models_dir = project_root / models_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("FLAVR -> ONNX export")
    print("=" * 64)
    print(f"  project   : {project_root}")
    print(f"  ckpt dir  : {models_dir}")
    print(f"  out dir   : {out_dir}")
    print(f"  size      : {args.size}x{args.size}")
    print(f"  opset     : {args.opset}")
    print()

    results: List[dict] = []
    failures = 0

    import torch
    print(f"  torch     : {torch.__version__}")

    for ckpt_name, cfg in CKPT_CONFIGS.items():
        n_outputs = int(cfg["n_outputs"])
        version = int(cfg["version"])
        ckpt_path = models_dir / ckpt_name
        onnx_path = out_dir / f"FLAVR_{version}x.onnx"

        print("-" * 64)
        print(f"[{version}x] n_outputs={n_outputs}  {ckpt_path.name}")

        if not ckpt_path.exists():
            print(f"  FAIL: checkpoint not found: {ckpt_path}")
            failures += 1
            continue

        if args.skip_existing and onnx_path.exists():
            print(f"  SKIP: {onnx_path.name} already exists")
        else:
            try:
                export_one(ckpt_path, onnx_path, n_outputs, args.size, args.opset)
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  FAIL: export error: {exc!r}")
                failures += 1
                continue
            print(f"  exported -> {onnx_path}")

        size_bytes = onnx_path.stat().st_size
        print(f"  file size : {size_bytes / 1e6:.1f} MB")

        try:
            info = verify_one(onnx_path, ckpt_path, n_outputs, args.size)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  FAIL: verification error: {exc!r}")
            failures += 1
            continue

        sig = info["signature"]
        print(f"  input     : {sig['input_name']} {sig['input_shape']} {sig['input_type']}")
        print(f"  output    : {sig['output_name']} {sig['output_shape']} {sig['output_type']}")
        print(f"  providers : {sig['providers']}")
        print(f"  ort shape : {info['onnx_shape']} dtype={info['onnx_dtype']} finite={info['finite']}")
        print(f"  MAE vs torch : {info['mae']:.3e}")

        ok = (
            info["finite"]
            and info["onnx_dtype"] == "float32"
            and len(info["onnx_shape"]) == 4
            and info["onnx_shape"][0] == 1
            and info["onnx_shape"][1] == info["expected_channels"]
            and info["onnx_shape"][2] == args.size
            and info["onnx_shape"][3] == args.size
            and info["mae"] < 1e-3
        )
        if not ok:
            print("  FAIL: verification checks did not pass")
            failures += 1
            continue
        print("  PASS")

        results.append({
            "version": version,
            "onnx_path": onnx_path,
            "size_bytes": size_bytes,
            "channels": info["expected_channels"],
            "mae": info["mae"],
        })

    print("=" * 64)
    if failures:
        print(f"RESULT: {failures} failure(s)")
        return 1
    print("RESULT: ALL EXPORTS PASSED")
    for r in results:
        print(f"  {r['onnx_path'].name:16s} out_ch={r['channels']:2d} "
              f"size={r['size_bytes']/1e6:6.1f}MB mae={r['mae']:.2e}")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    python_exe = _resolve_runtime_python()
    if python_exe != sys.executable:
        import subprocess

        print(f"[export_flavr_onnx] Re-executing with: {python_exe}", flush=True)
        return subprocess.run([python_exe, __file__] + argv).returncode
    return _main_impl(argv)


if __name__ == "__main__":
    sys.exit(main())
