"""模型资产解析器 — 版本令牌 / checkpoint / ONNX 路径的单一真相源。

契约见 ``docs/ONNX_TRT_MULTI_MODEL_SPEC.md`` §4。

设计要点
--------
- **版本令牌**（version token）是面向用户/配置的规范标识，例如 RIFE 的 ``"4.22"``、
  FILM 的 ``"fp32"``、AMT 的 ``"s"``。checkpoint 文件名与 ONNX 路径都是**派生物**。
- ``MODEL_ASSET_TABLE`` 只收录**有后端适配实现**的 9 个类型（不含 cain/sepconv）。
- ``checkpoint`` 字段为相对 ``models/<model_type>/`` 的权重文件名；无对应权重时为 ``None``。
- ``onnx`` 字段为相对 ``models_dir`` 的 ONNX 路径；目录中不存在的资产填 ``None``。
  本表内容基于对 ``models/`` 目录的**实际扫描**（2026-10 核对）。

禁止模块级 ``import torch``（``core/backends/AGENTS.md`` 红线），本模块纯 Python。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 版本令牌 -> 资产
# ---------------------------------------------------------------------------

def _rife_assets() -> Dict[str, Dict[str, Optional[str]]]:
    """基于 ``models/rife_v2`` 与 ``models/rife_ensemble`` 的真实文件名构造 RIFE 表。

    基础版本 ONNX：``rife_v2/rife_v<ver>.onnx``；
    变体（lite/heavy/ensemble）以 ``<ver>_lite`` / ``<ver>_heavy`` / ``<ver>_ensemble``
    作为独立版本令牌。
    """
    onnx: Dict[str, str] = {}

    # 基础版本（models/rife_v2 中实际存在的文件，缺 4.1）
    for ver in [
        "4.0", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7", "4.8", "4.9",
        "4.10", "4.11", "4.12", "4.13", "4.14", "4.15", "4.16", "4.17",
        "4.18", "4.19", "4.20", "4.21", "4.22", "4.23", "4.24", "4.25", "4.26",
    ]:
        onnx[ver] = f"rife_v2/rife_v{ver}.onnx"

    # lite 变体（models/rife_v2）
    for ver in ["4.12", "4.13", "4.14", "4.15", "4.16", "4.17", "4.22", "4.25"]:
        onnx[f"{ver}_lite"] = f"rife_v2/rife_v{ver}_lite.onnx"
    # 4.12~4.17 另有 lite_ensemble
    for ver in ["4.12", "4.13", "4.14", "4.15", "4.16", "4.17"]:
        onnx[f"{ver}_lite_ensemble"] = f"rife_v2/rife_v{ver}_lite_ensemble.onnx"
    # heavy 变体
    for ver in ["4.25", "4.26"]:
        onnx[f"{ver}_heavy"] = f"rife_v2/rife_v{ver}_heavy.onnx"
    # ensemble 变体（models/rife_ensemble/rife，4.0~4.6）
    for ver in ["4.0", "4.2", "4.3", "4.4", "4.5", "4.6"]:
        onnx[f"{ver}_ensemble"] = f"rife_ensemble/rife/rife_v{ver}_ensemble.onnx"

    # 已知 checkpoint 映射（取自 model_manager.CHECKPOINT_VERSION_MAP 与
    # rife/__init__.py 的 CKPT_VERSION_MAP）。
    ckpt = {
        "4.0": "sudo_rife4_269.662_testV1_scale1.pth",
        "4.7": "rife47.pth",
        "4.9": "rife49.pth",
        "4.17": "rife417.pth",
        "4.26": "rife426.pth",
    }
    return {v: {"checkpoint": ckpt.get(v), "onnx": p} for v, p in onnx.items()}


MODEL_ASSET_TABLE: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {
    "rife": _rife_assets(),
    "film": {
        # 真实资产：models/film/film_net_fp32.onnx（输入 img0/img1/timestep，输出 output）
        "fp32": {"checkpoint": "film_net_fp32.pt", "onnx": "film/film_net_fp32.onnx"},
    },
    "amt": {
        # 仅权重，目录中无 ONNX 资产
        "s": {"checkpoint": "amt-s.pth", "onnx": None},
        "l": {"checkpoint": "amt-l.pth", "onnx": None},
        "g": {"checkpoint": "amt-g.pth", "onnx": None},
        "gopro": {"checkpoint": "gopro_amt-s.pth", "onnx": None},
    },
    "m2m": {
        "default": {"checkpoint": "M2M.pth", "onnx": None},
    },
    "flavr": {
        # 真实资产：models/flavr/FLAVR_{2x,4x,8x}.onnx（单输入 frames=[1,12,H,W]）
        "2x": {"checkpoint": "FLAVR_2x.pth", "onnx": "flavr/FLAVR_2x.onnx"},
        "4x": {"checkpoint": "FLAVR_4x.pth", "onnx": "flavr/FLAVR_4x.onnx"},
        "8x": {"checkpoint": "FLAVR_8x.pth", "onnx": "flavr/FLAVR_8x.onnx"},
    },
    "stmfnet": {
        "v1": {"checkpoint": "stmfnet.pth", "onnx": None},
    },
    "atm": {
        "base": {"checkpoint": "atm-vfi-base.pt", "onnx": None},
        "lite": {"checkpoint": "atm-vfi-lite.pt", "onnx": None},
        "base_pct": {"checkpoint": "atm-vfi-base-pct.pt", "onnx": None},
    },
    "momo": {
        # MODEL_DEFINITIONS 记为 momo.pth，但实际资产为 momo-base/lite.pth
        "base": {"checkpoint": "momo-base.pth", "onnx": None},
        "lite": {"checkpoint": "momo-lite.pth", "onnx": None},
    },
    "xvfi": {
        # MODEL_DEFINITIONS 记为 xvfi.pth，但实际资产为两个 XVFInet 权重
        "x4k1000fps": {"checkpoint": "XVFInet_X4K1000FPS_exp1_latest.pt", "onnx": None},
        "vimeo": {"checkpoint": "XVFInet_Vimeo_exp1_latest.pt", "onnx": None},
    },
}


# 默认版本令牌：model_manager.DEFAULT_VERSIONS 的超集，补齐其余已适配类型。
DEFAULT_VERSIONS: Dict[str, str] = {
    "rife": "4.22",
    "film": "fp32",
    "amt": "s",
    "m2m": "default",
    "flavr": "2x",
    "stmfnet": "v1",
    "atm": "base",
    "momo": "base",
    "xvfi": "x4k1000fps",
}


# 无 PyTorch 实现、明确拒绝的类型。
UNSUPPORTED_MODEL_TYPES = ("cain", "sepconv")


# ---------------------------------------------------------------------------
# 解析 API
# ---------------------------------------------------------------------------

def _entry(model_type: str, version: str) -> Optional[Dict[str, Optional[str]]]:
    """返回 ``MODEL_ASSET_TABLE`` 条目（大小写不敏感），不存在返回 ``None``。"""
    table = MODEL_ASSET_TABLE.get((model_type or "").lower())
    if not table:
        return None
    return table.get(version)


def checkpoint_to_version(model_type: str, checkpoint: str) -> Optional[str]:
    """checkpoint 文件名 -> 版本令牌。

    Args:
        model_type: 模型类型（如 ``"rife"``）。
        checkpoint: checkpoint 文件名或含目录的路径（只取 basename 匹配）。

    Returns:
        版本令牌；未命中返回 ``None``。
    """
    if not checkpoint:
        return None
    name = Path(checkpoint).name
    table = MODEL_ASSET_TABLE.get((model_type or "").lower(), {})
    for version, asset in table.items():
        if asset.get("checkpoint") == name:
            return version
    return None


def version_to_checkpoint(model_type: str, version: str) -> Optional[str]:
    """版本令牌 -> checkpoint 文件名。未命中返回 ``None``。"""
    entry = _entry(model_type, version)
    if entry is None:
        return None
    return entry.get("checkpoint")


def resolve_checkpoint_path(
    model_type: str, version: str, models_dir: str
) -> Optional[Path]:
    """版本令牌（或 checkpoint 名）-> 绝对 checkpoint 路径候选。

    路径形如 ``<models_dir>/<model_type>/<checkpoint>``。即使文件不存在也返回候选
    （便于日志）；无法解析返回 ``None``。
    """
    ver = checkpoint_to_version(model_type, version) or version
    checkpoint = version_to_checkpoint(model_type, ver)
    if not checkpoint:
        return None
    return Path(models_dir) / (model_type or "").lower() / checkpoint


def resolve_onnx_path(
    model_type: str,
    version: str,
    models_dir: str,
    explicit: Optional[str] = None,
) -> Optional[Path]:
    """解析 ONNX 文件路径候选。

    Args:
        model_type: 模型类型。
        version: 版本令牌或 checkpoint 名（先经 :func:`checkpoint_to_version` 归一化）。
        models_dir: ``models/`` 目录。
        explicit: 显式覆盖路径（``onnx_path`` / ``checkpoint_path``）。若以 ``.onnx``
            结尾则直接作为 ONNX 候选；否则视为 checkpoint 覆盖，先映射到版本再查表。

    Returns:
        候选路径（即使不存在，便于日志）；无法解析返回 ``None``。
    """
    models_dir_path = Path(models_dir)

    if explicit:
        p = Path(explicit)
        if p.suffix.lower() == ".onnx":
            # 显式 ONNX 路径优先。
            return p if p.is_absolute() else models_dir_path / p
        # 显式 checkpoint 覆盖：映射到版本后查 ONNX 表。
        ver = checkpoint_to_version(model_type, p.name) or p.stem
        entry = _entry(model_type, ver)
        if entry and entry.get("onnx"):
            return models_dir_path / entry["onnx"]
        # 无 ONNX 资产：返回显式路径本身作为候选（调用方可据此报错）。
        return p if p.is_absolute() else models_dir_path / p

    # 版本令牌或 checkpoint 名归一化。
    ver = checkpoint_to_version(model_type, version) or version
    entry = _entry(model_type, ver)
    if entry and entry.get("onnx"):
        return models_dir_path / entry["onnx"]
    return None


def supported_model_types() -> List[str]:
    """返回有适配/资产定义的类型（不含 cain/sepconv），已排序。"""
    return sorted(
        mt for mt in MODEL_ASSET_TABLE if mt not in UNSUPPORTED_MODEL_TYPES
    )
