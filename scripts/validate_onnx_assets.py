"""批量校验 models/ 下的 ONNX 资产。

用途
----
项目新增了多种 ONNX 模型文件（RIFE 的版本/lite/heavy/ensemble 变体，
以及上游下载的 IFRNet / CAIN）。本项目推理后端只支持 RIFE，故此脚本
在 ONNX Runtime 层做独立校验，不经过 BackendFactory：

- 对 RIFE 系列（文件名含 ``rife``）：做「恒等」检查——令 frame0 == frame1，
  插值结果应约等于原帧，报告 mean-abs-error（越小越好）。
- 对其它模型（IFRNet / CAIN 等）：仅做结构检查——能否加载、输入/输出名与
  形状、dtype，不保证语义。

用法
----
    D:\\code\\VFI\\runtime\\cuda\\Scripts\\python.exe scripts/validate_onnx_assets.py
    ... scripts/validate_onnx_assets.py --root models/rife_v2

退出码：0=全部通过；1=存在失败。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

# 允许以 `scripts/` 为工作目录时仍能 import 项目 `core` 包
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 恒等检查的 MAE 阈值（与 scripts/test_onnx_backend.py 保持一致）
IDENTITY_MAE_THRESHOLD = 0.02
# 默认测试分辨率（RIFE 对分辨率无强约束，64 的倍数较稳妥）
DEFAULT_H = 128
DEFAULT_W = 128


def _build_rife_input(shape: list, frame: np.ndarray, timestep: float) -> np.ndarray:
    """按通道数构造 RIFE 输入张量。

    直接复用项目的 ``core.backends.rife_input.pack_rife_input``，保证
    7ch / 11ch 的通道布局与后端完全一致（frame0 == frame1 -> 期望恒等）。
    """
    from core.backends.rife_input import pack_rife_input

    c = shape[1] if isinstance(shape[1], int) else 7
    f = frame[0]  # [3, H, W]
    return pack_rife_input(f, f, timestep, in_channels=c)


def _validate_rife(path: Path) -> tuple[bool, str]:
    """对单个 RIFE ONNX 做恒等检查，返回 (是否通过, 描述)。"""
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp_meta = sess.get_inputs()[0]
    shape = list(inp_meta.shape)
    # 动态维度用默认分辨率补齐
    h = shape[2] if isinstance(shape[2], int) else DEFAULT_H
    w = shape[3] if isinstance(shape[3], int) else DEFAULT_W
    frame = np.random.rand(1, 3, h, w).astype(np.float32)
    feed = _build_rife_input(shape, frame, timestep=0.5)
    out = sess.run(None, {inp_meta.name: feed})[0]
    ok_shape = tuple(out.shape) == (1, 3, h, w)
    finite = bool(np.isfinite(out).all())
    mae = float(np.mean(np.abs(out - frame)))
    passed = ok_shape and finite and mae < IDENTITY_MAE_THRESHOLD
    desc = (
        f"{path.name}: in={shape} out={tuple(out.shape)} "
        f"identity_MAE={mae:.5f} finite={finite} -> "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return passed, desc


def _validate_structural(path: Path) -> tuple[bool, str]:
    """对非 RIFE ONNX 做结构检查（能否加载 + I/O 签名）。"""
    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001 - 汇总所有加载失败
        return False, f"{path.name}: LOAD-FAIL {exc}"
    ins = ", ".join(f"{i.name}{list(i.shape)}:{i.type}" for i in sess.get_inputs())
    outs = ", ".join(f"{o.name}{list(o.shape)}:{o.type}" for o in sess.get_outputs())
    return True, f"{path.name}: OK\n    in : {ins}\n    out: {outs}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="models",
        help="扫描根目录（默认 models/）",
    )
    parser.add_argument(
        "--pattern",
        default="**/*.onnx",
        help="相对 root 的 glob（默认 **/*.onnx）",
    )
    args = parser.parse_args()

    root = Path(args.root)
    files = sorted(root.glob(args.pattern))
    if not files:
        print(f"未找到 ONNX 文件：{root}/{args.pattern}")
        return 1

    failures = 0
    for f in files:
        name = f.name.lower()
        try:
            if "rife" in name:
                passed, desc = _validate_rife(f)
            else:
                passed, desc = _validate_structural(f)
        except Exception as exc:  # noqa: BLE001 - 单文件失败不影响整体
            passed, desc = False, f"{f.name}: ERROR {exc}"
        print(desc)
        if not passed:
            failures += 1

    print(f"\n共 {len(files)} 个文件，失败 {failures} 个。")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
