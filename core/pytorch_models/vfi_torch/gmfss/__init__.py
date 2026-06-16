"""
GMFSS Fortuna — Animation-dedicated Video Frame Interpolation.

Port of GMFSS Fortuna with both union (IFNet-guided) and basic variants.
Reference: https://github.com/98mxr/GMFSS_Fortuna

Architecture:
  - GMFlow: Global matching optical flow (transformer-based)
  - FeatureNet: Multi-scale feature extraction backbone
  - MetricNet: Flow confidence estimation
  - FusionNet/GridNet: Frame blending from warped features
  - IFNet (union only): RIFE-based flow initialization

Model files (5-6 .pkl files):
  - flownet.pkl: GMFlow optical flow
  - metricnet.pkl: Metric/confidence estimation
  - feat_ext.pkl: Feature extraction backbone
  - fusionnet.pkl: Frame fusion network
  - ifnet.pkl (union only): RIFE IFNet weights
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from ..base import PyTorchVFIModel, VFIConfig, ModelType
from ..utils import get_device
from .arch import GMFSSUnionModel, GMFSSModel as GMFSSBaseModel


GMFSS_CKPTS = {
    "GMFSS_fortuna_union": {
        "ifnet": "rife/rife46.pth",
        "flownet": "gmfss/GMFSS_fortuna_flownet.pkl",
        "metricnet": "gmfss/GMFSS_fortuna_union_metric.pkl",
        "feat_ext": "gmfss/GMFSS_fortuna_union_feat.pkl",
        "fusionnet": "gmfss/GMFSS_fortuna_union_fusionnet.pkl",
    },
    "GMFSS_fortuna": {
        "flownet": "gmfss/GMFSS_fortuna_flownet.pkl",
        "metricnet": "gmfss/GMFSS_fortuna_metric.pkl",
        "feat_ext": "gmfss/GMFSS_fortuna_feat.pkl",
        "fusionnet": "gmfss/GMFSS_fortuna_fusionnet.pkl",
    },
}

GMFSS_MODEL_URLS = {
    "GMFSS_fortuna_flownet.pkl": "https://github.com/98mxr/GMFSS_Fortuna/releases/download/v1.0/GMFSS_fortuna_flownet.pkl",
    "GMFSS_fortuna_metric.pkl": "https://github.com/98mxr/GMFSS_Fortuna/releases/download/v1.0/GMFSS_fortuna_metric.pkl",
    "GMFSS_fortuna_feat.pkl": "https://github.com/98mxr/GMFSS_Fortuna/releases/download/v1.0/GMFSS_fortuna_feat.pkl",
    "GMFSS_fortuna_fusionnet.pkl": "https://github.com/98mxr/GMFSS_Fortuna/releases/download/v1.0/GMFSS_fortuna_fusionnet.pkl",
    "GMFSS_fortuna_union_metric.pkl": "https://github.com/98mxr/GMFSS_Fortuna/releases/download/v1.0/GMFSS_fortuna_union_metric.pkl",
    "GMFSS_fortuna_union_feat.pkl": "https://github.com/98mxr/GMFSS_Fortuna/releases/download/v1.0/GMFSS_fortuna_union_feat.pkl",
    "GMFSS_fortuna_union_fusionnet.pkl": "https://github.com/98mxr/GMFSS_Fortuna/releases/download/v1.0/GMFSS_fortuna_union_fusionnet.pkl",
}


class GMFSSModel(PyTorchVFIModel):
    """GMFSS Fortuna — Animation-focused VFI model.

    Supports two variants:
    - "GMFSS_fortuna": Base 4-component (flownet + metricnet + feat_ext + fusionnet)
    - "GMFSS_fortuna_union": + IFNet for flow guidance (recommended)

    Requires 5-6 separate checkpoint files loaded per-component.
    """

    MODEL_NAME = "gmfss"
    SUPPORTED_VERSIONS = ["fortuna", "fortuna_union"]
    DEFAULT_VERSION = "fortuna_union"
    MIN_INPUT_FRAMES = 2

    def __init__(self, config: VFIConfig):
        super().__init__(config)
        self._variant = config.model_version if config.model_version in ("fortuna", "fortuna_union") else "fortuna_union"
        self._model_dict: dict = {}  # component_name -> nn.Module

    # ====================
    # Public API
    # ====================

    def load_model(self, checkpoint_path: str = "", **kwargs) -> None:
        """Load all GMFSS component checkpoints.

        Expects a directory containing the .pkl files listed in
        GMFSS_CKPTS[self._variant].

        Args:
            checkpoint_path: Directory containing checkpoint files.
                If empty, uses self._config.checkpoint_path.
        """
        if self._model_dict:
            return

        ckpt_dir = Path(checkpoint_path or (self._config.checkpoint_path or ""))
        self._load_components(ckpt_dir)
        self._is_loaded = True

    def interpolate(
        self,
        frame0: torch.Tensor,
        frame1: torch.Tensor,
        timestep: float = 0.5,
        **kwargs,
    ) -> torch.Tensor:
        """Interpolate a single frame between two inputs.

        Uses the two-step reuse + inference pattern:
        1. reuse(): Compute flows + features from both frames (once per pair)
        2. inference(): Generate interpolation at given timestep

        Args:
            frame0: First input frame [C, H, W] or [B, C, H, W]
            frame1: Second input frame [C, H, W] or [B, C, H, W]
            timestep: Interpolation timestep (0.0 to 1.0)

        Returns:
            Interpolated frame tensor
        """
        if not self._is_loaded:
            self.load_model()

        squeeze_output = False
        if frame0.dim() == 3:
            frame0 = frame0.unsqueeze(0)
            frame1 = frame1.unsqueeze(0)
            squeeze_output = True

        frame0, frame1 = self.prepare_frames(frame0, frame1)
        scale = self._config.scale

        # Validate and pad
        n, c, h, w = frame0.shape
        tmp = max(64, int(64 / scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)
        I0 = F.pad(frame0, padding)
        I1 = F.pad(frame1, padding)

        # Step 1: reuse — compute flows and features
        flow_data = self._reuse(I0, I1, scale)

        # Step 2: inference — generate interpolated frame at given timestep
        with torch.no_grad():
            output = self._inference(I0, I1, timestep, flow_data)

        output = output[:, :, :h, :w]

        if squeeze_output:
            output = output.squeeze(0)

        return output

    def unload(self) -> None:
        """Unload all model components."""
        for key in list(self._model_dict.keys()):
            del self._model_dict[key]
        self._model_dict.clear()
        self._is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        import gc
        gc.collect()

    # ====================
    # Component Loading
    # ====================

    def _load_components(self, ckpt_dir: Path) -> None:
        """Load component checkpoints from directory."""
        if self._variant == "fortuna_union":
            model = GMFSSUnionModel()
        else:
            model = GMFSSBaseModel()
        model.eval()

        # Build path dict: component -> full path
        ckpt_config = GMFSS_CKPTS[self._variant]
        path_dict: dict = {}
        for comp_name, rel_path in ckpt_config.items():
            ckpt_path = ckpt_dir / rel_path
            # Also try just the filename
            if not ckpt_path.exists():
                ckpt_path = ckpt_dir / Path(rel_path).name
            if ckpt_path.exists():
                path_dict[comp_name] = str(ckpt_path)

        if not path_dict:
            raise FileNotFoundError(
                f"No GMFSS checkpoints found in {ckpt_dir}. "
                f"Expected: {list(ckpt_config.values())}"
            )

        model.load_model(path_dict)
        model.to(self.device)
        if self.dtype != torch.float32:
            model = model.to(self.torch_dtype)
        model.eval()
        self._model_dict["main"] = model

    # ====================
    # Inference Pipeline
    # ====================

    def _reuse(self, I0: torch.Tensor, I1: torch.Tensor, scale: float) -> dict:
        """Compute flows and features (pre-computation step)."""
        model = self._get_model()
        with torch.no_grad():
            (flow01, flow10, metric0, metric1,
             feat11, feat12, feat13,
             feat21, feat22, feat23) = model.reuse(I0, I1, scale)
        return {
            "flow01": flow01,
            "flow10": flow10,
            "metric0": metric0,
            "metric1": metric1,
            "feat11": feat11, "feat12": feat12, "feat13": feat13,
            "feat21": feat21, "feat22": feat22, "feat23": feat23,
        }

    def _inference(
        self,
        I0: torch.Tensor,
        I1: torch.Tensor,
        timestep: float,
        flow_data: dict,
    ) -> torch.Tensor:
        """Generate interpolated frame at given timestep."""
        model = self._get_model()
        with torch.no_grad():
            output = model.inference(
                I0, I1,
                flow_data["flow01"], flow_data["flow10"],
                flow_data["metric0"], flow_data["metric1"],
                flow_data["feat11"], flow_data["feat12"], flow_data["feat13"],
                flow_data["feat21"], flow_data["feat22"], flow_data["feat23"],
                timestep,
            )
        return output

    def _get_model(self):
        """Get the main model component."""
        model = self._model_dict.get("main")
        if model is None:
            raise RuntimeError("GMFSS model not loaded. Call load_model() first.")
        return model


# ====================
# Convenience: Model Type Registration
# ====================

def get_gmfss_model_info() -> dict:
    """Get model info for registry."""
    return {
        "name": "gmfss",
        "display_name": "GMFSS Fortuna",
        "variants": ["fortuna", "fortuna_union"],
        "default_variant": "fortuna_union",
        "description": "Animation-dedicated VFI with global matching optical flow",
        "checkpoint_urls": GMFSS_MODEL_URLS,
    }
