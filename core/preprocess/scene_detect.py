"""Scene detection module for VFI-gui.

Provides three detection strategies covering different precision and performance needs:
1. PlaneStatsSceneDetector - Fast, simple pixel difference (misc.SCDetect equivalent)
2. NeuralSceneDetector - ONNX neural network for higher accuracy
3. VapourSynthSceneDetector - Uses VapourSynth's built-in scene change detection

Algorithm equivalence:
    PlaneStatsSceneDetector uses normalized SAD = mean(|frame_curr - frame_prev|) / 255.0,
    which is equivalent to VapourSynth's misc.SCDetect algorithm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from core.types import SceneDetectionMethod, ProcessingConfig


# Project root resolution (VFI-gui/ from core/preprocess/scene_detect.py)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SceneDetectorBase(ABC):
    """Abstract base class for scene detectors.

    All scene detectors must implement is_scene_cut() method.
    The reset() method clears internal state for new tasks.
    """

    def __init__(self):
        """Initialize base detector with no previous frame."""
        self._prev_frame: Optional[np.ndarray] = None

    @abstractmethod
    def is_scene_cut(self, frame: np.ndarray) -> bool:
        """Detect if current frame is a scene change point.

        Args:
            frame: Current frame as numpy array [H, W, C] uint8

        Returns:
            True if scene change detected, False otherwise
        """
        ...

    def reset(self) -> None:
        """Reset internal state for a new task.

        Clears the stored previous frame reference.
        """
        self._prev_frame = None


class PlaneStatsSceneDetector(SceneDetectorBase):
    """Pixel-based scene detector using normalized SAD.

    Equivalent to VapourSynth's misc.SCDetect algorithm.
    Uses normalized SAD = mean(|frame_curr - frame_prev|) / 255.0

    This is a fast, simple approach suitable for most content.
    """

    def __init__(self, threshold: float = 0.1):
        """Initialize PlaneStats detector.

        Args:
            threshold: Scene change threshold (0.0-1.0).
                       Higher values = fewer scene cuts detected.
                       Default 0.1 is equivalent to VapourSynth's default.
        """
        super().__init__()
        self._threshold = threshold

    def is_scene_cut(self, frame: np.ndarray) -> bool:
        """Detect scene change using normalized SAD.

        Args:
            frame: Current frame as numpy array [H, W, C] uint8

        Returns:
            True if normalized SAD > threshold
        """
        if self._prev_frame is None:
            self._prev_frame = frame.copy()
            return False

        # Compute normalized SAD = mean(|curr - prev|) / 255.0
        # This is equivalent to VapourSynth misc.SCDetect
        diff = np.mean(np.abs(
            frame.astype(np.float64) - self._prev_frame.astype(np.float64)
        )) / 255.0

        self._prev_frame = frame.copy()
        return bool(diff > self._threshold)

    def reset(self) -> None:
        """Reset detector state."""
        super().reset()


class NeuralSceneDetector(SceneDetectorBase):
    """ONNX neural network scene detector.

    Port of VSGAN's scene_detect.py with support for 5 ONNX models.
    Uses 6-channel input (concatenated prev + curr frames) in CHW format.

    Available models:
        | Model | Architecture          | Resolution | Recommended threshold |
        | 0     | EfficientFormerV2-S0  | 224x224    | ~0.93                |
        | 6     | EfficientNetV2-B0     | 48x27      | ~0.9                 |
        | 12    | MaxViT+RIFE+Sobel     | 256x256    | ~0.93                |
        | 14    | Shift-LPIPS-Alex      | 256x256    | ~0.45                |
        | 16    | DISTS                 | 256x256    | ~0.25                |
    """

    # Model file names mapping (from VSGAN scene_detect)
    MODEL_FILES: Dict[int, str] = {
        0: "sc_efficientformerv2_s0_12263_224_CHW_6ch_clamp_softmax_op17_fp16_sim.onnx",
        6: "sc_tf_efficientnetv2_b0.in1k_48x27_b100_30k_coloraug0.4_6ch_clamp_softmax_fp16_op17_onnxslim.onnx",
        12: "sc_mobilevitv2_050.cvnets_in1k+efficientvit_b2.r288_in1k_rife422_sobel_256px_5k_CHW_6ch_clamp_softmax_op20_fp16_onnxslim.onnx",
        14: "sc_shift_lpips_alex_256px_CHW_6ch_clamp_op20_fp16_onnxslim.onnx",
        16: "sc_dists_256px_CHW_6ch_clamp_op20_fp16_onnxslim.onnx",
    }

    # Model input resolutions mapping (height, width)
    MODEL_RESOLUTIONS: Dict[int, tuple[int, int]] = {
        0: (224, 224),
        6: (48, 27),
        12: (256, 256),
        14: (256, 256),
        16: (256, 256),
    }

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.93,
        fp16: bool = True,
        device: str = "cpu",
        model_resolution: tuple[int, int] = (256, 256),
    ):
        """Initialize neural scene detector.

        Args:
            model_path: Path to the ONNX model file.
            threshold: Scene change threshold (0.0-1.0). Default 0.93.
            fp16: Use FP16 inference for speed.
            device: Device for inference ("cpu" or "cuda").
            model_resolution: Input resolution (height, width) for the model.
        """
        super().__init__()
        self._threshold = threshold
        self._resolution = model_resolution
        self._dtype = np.float16 if fp16 else np.float32

        # Load ONNX session
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for NeuralSceneDetector. "
                "Install with: pip install onnxruntime or onnxruntime-gpu"
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device != "cpu"
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(model_path, providers=providers)

    def is_scene_cut(self, frame: np.ndarray) -> bool:
        """Detect scene change using neural network.

        Algorithm: frame_prev + frame_curr -> 6-channel CHW concat
                   -> ONNX inference -> softmax probability

        Args:
            frame: Current frame [H, W, 3] RGB float32 [0, 1]

        Returns:
            True if probability > threshold
        """
        if self._prev_frame is None:
            self._prev_frame = frame
            return False

        # Resize and convert to CHW format
        i0 = self._resize(self._prev_frame)   # [3, H, W]
        i1 = self._resize(frame)               # [3, H, W]

        # Concatenate to 6-channel input [1, 6, H, W]
        input_tensor = np.concatenate([i0, i1], axis=0)[np.newaxis]
        input_tensor = input_tensor.astype(self._dtype)

        # Run inference
        result = self._session.run(None, {"input": input_tensor})[0]
        prob = float(result[0][0])

        self._prev_frame = frame
        return bool(prob > self._threshold)

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to model input resolution and convert to CHW.

        Args:
            frame: Input frame [H, W, 3] RGB float32 [0, 1]

        Returns:
            Resized frame [3, H, W] float32 [0, 1]
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "PIL is required for NeuralSceneDetector. "
                "Install with: pip install Pillow"
            )

        # Convert float32 [0, 1] to uint8 for PIL
        img = Image.fromarray(
            (frame * 255).clip(0, 255).astype(np.uint8)
        )
        # Resize: PIL uses (width, height)
        img = img.resize(
            (self._resolution[1], self._resolution[0]),
            Image.Resampling.BILINEAR,
        )
        # Convert back to float32 [0, 1] and transpose to CHW
        arr = np.array(img).astype(np.float32) / 255.0
        return arr.transpose(2, 0, 1)  # [3, H, W]

    def reset(self) -> None:
        """Reset detector state."""
        super().reset()


class VapourSynthSceneDetector(SceneDetectorBase):
    """VapourSynth-based scene detector.

    Uses VapourSynth's built-in scene change detection via frame properties.
    This detector reads the _SceneChangeNext frame property set by VapourSynth.

    Note: is_scene_cut() is not implemented - use is_scene_cut_from_props()
    when working with VapourSynth clips.
    """

    def __init__(self, threshold: float = 0.5):
        """Initialize VapourSynth scene detector.

        Args:
            threshold: Fallback threshold if _SceneChangeNext not available.
        """
        super().__init__()
        self._threshold = threshold

    def is_scene_cut(self, frame: np.ndarray) -> bool:
        """Not implemented for VapourSynth detector.

        VapourSynth scene detection works via frame properties, not raw frames.
        Use is_scene_cut_from_props() when working with VapourSynth clips.

        Raises:
            NotImplementedError: Always, since this detector requires VS props.
        """
        raise NotImplementedError(
            "VapourSynthSceneDetector requires frame properties. "
            "Use is_scene_cut_from_props() with a VapourSynth frame."
        )

    def is_scene_cut_from_props(self, frame_props: Dict[str, Any]) -> bool:
        """Detect scene change from VapourSynth frame properties.

        Args:
            frame_props: VapourSynth frame properties dict

        Returns:
            True if _SceneChangeNext property is set/True
        """
        # Check _SceneChangeNext property (standard VS scene change marker)
        if "_SceneChangeNext" in frame_props:
            return bool(frame_props["_SceneChangeNext"])

        # Fallback to threshold-based detection if property not available
        return False

    def reset(self) -> None:
        """Reset detector state. No state to reset for VS detector."""
        super().reset()


class SceneDetectorFactory:
    """Factory for creating scene detector instances.

    Supports three detection methods:
    - planestats: Fast pixel-based (PlaneStatsSceneDetector)
    - neural: ONNX neural network (NeuralSceneDetector)
    - vapoursynth: VapourSynth built-in (VapourSynthSceneDetector)
    - auto: Automatically select best method based on backend and config
    """

    # Models base directory resolved relative to project root
    MODELS_BASE_DIR = os.path.join(_PROJECT_ROOT, "models", "scene_detect")

    @staticmethod
    def create(
        config: ProcessingConfig,
        backend_type: Any = None,
    ) -> Optional[SceneDetectorBase]:
        """Create a scene detector based on config and backend type.

        Auto-selection logic (method="auto"):
            1. VapourSynth backend -> vapoursynth (zero overhead)
            2. Neural model specified -> neural (best accuracy)
            3. Otherwise -> planestats (always available)

        Args:
            config: ProcessingConfig with scene_detection settings dict.
                Expected keys: enabled, method, threshold, model, fp16,
                device, dup_threshold
            backend_type: BackendType enum value. Used for auto-selection
                when method="auto".

        Returns:
            Scene detector instance, or None if scene detection is disabled
        """
        sc_config = config.scene_detection
        if not sc_config.get("enabled", False):
            return None

        method = sc_config.get("method", "auto")

        # Auto-select method based on backend and available models
        if method == "auto":
            # Import locally to avoid circular dependency
            from core.types import BackendType

            if backend_type == BackendType.VAPOURSYNTH:
                method = "vapoursynth"
            elif sc_config.get("model", None) is not None:
                method = "neural"
            else:
                method = "planestats"

        # Create appropriate detector
        if method == "vapoursynth":
            return VapourSynthSceneDetector(
                threshold=sc_config.get("threshold", 0.1),
            )
        elif method == "neural":
            model_idx = sc_config.get("model", 12)
            model_path = SceneDetectorFactory._resolve_model_path(model_idx)
            resolution = NeuralSceneDetector.MODEL_RESOLUTIONS.get(
                model_idx, (256, 256)
            )
            return NeuralSceneDetector(
                model_path=model_path,
                threshold=sc_config.get("threshold", 0.93),
                fp16=sc_config.get("fp16", True),
                device=sc_config.get("device", "cpu"),
                model_resolution=resolution,
            )
        else:
            return PlaneStatsSceneDetector(
                threshold=sc_config.get("threshold", 0.1),
            )

    @staticmethod
    def _resolve_model_path(model_index: int) -> str:
        """Resolve ONNX model file path from model index.

        Args:
            model_index: Model index (0, 6, 12, 14, 16)

        Returns:
            Absolute path to the ONNX model file

        Raises:
            ValueError: If model_index is not a valid model
        """
        filename = NeuralSceneDetector.MODEL_FILES.get(model_index)
        if filename is None:
            raise ValueError(
                f"Unknown scene detection model: {model_index}. "
                f"Valid options: {list(NeuralSceneDetector.MODEL_FILES.keys())}"
            )
        return os.path.join(SceneDetectorFactory.MODELS_BASE_DIR, filename)

    @staticmethod
    def get_available_methods() -> List[str]:
        """Get list of available detection methods.

        Returns:
            List of method names that are usable in current environment.
        """
        methods = ["planestats"]  # Always available

        # Check neural availability
        try:
            import onnxruntime
            models_dir = SceneDetectorFactory.MODELS_BASE_DIR
            if os.path.isdir(models_dir):
                methods.append("neural")
        except ImportError:
            pass

        # Check VapourSynth availability
        try:
            import vapoursynth
            methods.append("vapoursynth")
        except ImportError:
            pass

        return methods


def main() -> None:
    """CLI entry point for scene detection.

    Usage:
        python -m core.preprocess.scene_detect --input video.mp4 --method neural --json
    """
    parser = argparse.ArgumentParser(
        description="Scene detection CLI for VFI-gui",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m core.preprocess.scene_detect --input video.mp4 --method planestats
  python -m core.preprocess.scene_detect --input video.mp4 --method neural --model 12 --json
  python -m core.preprocess.scene_detect --input video.mp4 --threshold 0.15 --json
        """,
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input video path",
    )
    parser.add_argument(
        "--method", "-m",
        default="planestats",
        choices=["planestats", "neural", "auto"],
        help="Detection method (default: planestats)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.1,
        help="Scene change threshold (default: 0.1)",
    )
    parser.add_argument(
        "--model",
        type=int,
        default=12,
        choices=[0, 6, 12, 14, 16],
        help="Neural model index (default: 12)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 inference for neural method (default: True)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Try to read video with PyAV
    try:
        import av
    except ImportError:
        print("Error: PyAV is required for CLI usage. Install with: pip install av", file=sys.stderr)
        sys.exit(1)

    # Create config for factory
    scene_config = ProcessingConfig(
        scene_detection={
            "enabled": True,
            "method": args.method,
            "threshold": args.threshold,
            "model": args.model,
            "fp16": args.fp16,
        }
    )

    # Create detector
    try:
        detector = SceneDetectorFactory.create(scene_config)
        if detector is None:
            print("Error: Could not create detector", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error creating detector: {e}", file=sys.stderr)
        sys.exit(1)

    # Open video and process frames
    scene_cuts: List[int] = []
    frame_count = 0

    try:
        container = av.open(args.input)
        stream = container.streams.video[0]

        for frame in container.decode(video=0):
            # Convert to numpy array
            img = frame.to_ndarray(format="rgb24")

            # Detect scene change
            if detector.is_scene_cut(img):
                scene_cuts.append(frame_count)
                if args.verbose and not args.json:
                    print(f"Scene cut detected at frame {frame_count}")

            frame_count += 1

        container.close()

    except Exception as e:
        print(f"Error processing video: {e}", file=sys.stderr)
        sys.exit(1)

    # Output results
    if args.json:
        result = {
            "input": args.input,
            "method": args.method,
            "threshold": args.threshold,
            "model": args.model if args.method in ["neural", "auto"] else None,
            "total_frames": frame_count,
            "scene_cuts": scene_cuts,
            "scene_cut_count": len(scene_cuts),
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"Processed {frame_count} frames")
        print(f"Detected {len(scene_cuts)} scene cuts")
        if scene_cuts:
            print(f"Scene cut frames: {scene_cuts}")


if __name__ == "__main__":
    main()