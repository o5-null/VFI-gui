"""PyTorch 视频处理器。

提供基于 PyTorch 的视频处理管道，作为 VapourSynth 的替代方案。
与现有 VideoProcessor 接口兼容。
"""

import gc
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from PyQt6.QtCore import QThread, QObject, pyqtSignal

from .torch_backend import (
    # 配置和结果
    VFIConfig,
    VFIResult,
    ModelType,
    # 模型
    get_model,
    MODEL_REGISTRY,
    # 工具
    get_device,
    clear_cache,
    preprocess_frames,
    postprocess_frames,
    InputPadder,
)


# 支持的插值模型
SUPPORTED_TORCH_MODELS = {
    "rife": {
        "name": "RIFE",
        "description": "Real-Time Intermediate Flow Estimation",
        "versions": ["4.0", "4.6", "4.7", "4.17", "4.22", "4.26"],
        "default_version": "4.22",
    },
    "film": {
        "name": "FILM",
        "description": "Frame Interpolation for Large Motion",
        "versions": ["fp32"],
        "default_version": "fp32",
    },
    "ifrnet": {
        "name": "IFRNet",
        "description": "Intermediate Feature Refine Network",
        "versions": ["S_Vimeo90K", "L_Vimeo90K"],
        "default_version": "L_Vimeo90K",
    },
    "amt": {
        "name": "AMT",
        "description": "All-Pairs Multi-Field Transforms",
        "versions": ["s", "l", "g"],
        "default_version": "s",
    },
}


class TorchProcessor(QThread):
    """PyTorch 视频处理线程。

    使用 PyTorch 模型进行视频帧插值，
    作为 VapourSynth 管道的替代方案。
    """

    # 信号（与 VideoProcessor 兼容）
    progress_updated = pyqtSignal(int, int, float)  # current_frame, total_frames, fps
    stage_changed = pyqtSignal(str)  # 阶段名称
    log_message = pyqtSignal(str)  # 日志消息
    finished = pyqtSignal(str)  # 输出路径
    error_occurred = pyqtSignal(str)  # 错误消息

    def __init__(
        self,
        video_path: str,
        config: Dict[str, Any],
        torch_config: Dict[str, Any],
        parent: Optional[QObject] = None,
    ):
        """
        Args:
            video_path: 输入视频路径
            config: 处理配置（插值、输出等）
            torch_config: PyTorch 配置（设备、模型目录等）
            parent: 父 QObject
        """
        super().__init__(parent)
        self._video_path = video_path
        self._config = config
        self._torch_config = torch_config
        self._paused = False
        self._cancelled = False
        self._output_path: Optional[str] = None
        self._model = None

    def run(self):
        """运行视频处理管道。"""
        try:
            self.stage_changed.emit("初始化...")
            self.log_message.emit(f"输入: {self._video_path}")

            # 设置路径
            input_path = Path(self._video_path)
            output_dir = Path(self._torch_config.get("output_dir", "output"))
            output_dir.mkdir(parents=True, exist_ok=True)

            # 生成输出文件名
            output_name = f"{input_path.stem}_torch_interpolated.mp4"
            self._output_path = str(output_dir / output_name)

            # 加载模型
            self._load_model()

            # 处理视频
            self._process_video()

            if not self._cancelled:
                self.stage_changed.emit("完成")
                self.log_message.emit(f"输出: {self._output_path}")
                self.finished.emit(self._output_path)

        except Exception as e:
            import traceback
            self.error_occurred.emit(f"{str(e)}\n{traceback.format_exc()}")

        finally:
            self._cleanup()

    def _load_model(self):
        """加载插值模型。"""
        interp_config = self._config.get("interpolation", {})
        model_type_str = interp_config.get("model_type", "rife").lower()
        model_version = interp_config.get("model_version", "")

        # 获取默认版本
        if not model_version and model_type_str in SUPPORTED_TORCH_MODELS:
            model_version = SUPPORTED_TORCH_MODELS[model_type_str]["default_version"]

        self.stage_changed.emit("加载模型...")
        self.log_message.emit(f"加载 {model_type_str} 模型: {model_version}")

        # 映射模型类型
        model_type_map = {
            "rife": ModelType.RIFE,
            "film": ModelType.FILM,
            "ifrnet": ModelType.IFRNET,
            "amt": ModelType.AMT,
        }

        model_type = model_type_map.get(model_type_str, ModelType.RIFE)

        # 创建配置
        vfi_config = VFIConfig(
            model_type=model_type,
            model_version=model_version,
            multiplier=interp_config.get("multi", 2),
            scale=interp_config.get("scale", 1.0),
            fp16=self._torch_config.get("fp16", True),
        )

        # 优先使用配置中的完整路径
        checkpoint_path = interp_config.get("checkpoint_path")
        
        if not checkpoint_path:
            # 回退到手动构建路径
            models_dir = self._torch_config.get("models_dir", "models")
            checkpoint_name = self._get_checkpoint_name(model_type_str, model_version)
            checkpoint_path = str(Path(models_dir) / model_type_str / checkpoint_name)

        # 创建模型
        self._model = get_model(model_type, vfi_config)
        self._model.load_model(checkpoint_path)

        self.log_message.emit(f"模型已加载: {model_type_str}")

    def _get_checkpoint_name(self, model_type: str, version: str) -> str:
        """获取检查点文件名。"""
        checkpoint_map = {
            "rife": {
                "4.0": "sudo_rife4_269.662_testV1_scale1.pth",
                "4.6": "flownet.pkl",
                "4.7": "rife47.pth",
                "4.17": "rife417.pth",
                "4.22": "rife49.pth",
                "4.26": "rife426.pth",
            },
            "film": {
                "fp32": "film_net_fp32.pt",
            },
            "ifrnet": {
                "S_Vimeo90K": "IFRNet_S_Vimeo90K.pth",
                "L_Vimeo90K": "IFRNet_L_Vimeo90K.pth",
            },
            "amt": {
                "s": "amt-s.pth",
                "l": "amt-l.pth",
                "g": "amt-g.pth",
            },
        }

        if model_type in checkpoint_map and version in checkpoint_map[model_type]:
            return checkpoint_map[model_type][version]
        return f"{model_type}_{version}.pth"

    def _process_video(self):
        """处理视频文件。"""
        import cv2

        # 打开视频
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self._video_path}")

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.log_message.emit(f"视频: {width}x{height} @ {fps:.2f} fps, {total_frames} 帧")

        # 读取所有帧
        self.stage_changed.emit("读取帧...")
        frames = []
        frame_idx = 0

        while True:
            if self._cancelled:
                break

            ret, frame = cap.read()
            if not ret:
                break

            # BGR 转 RGB，归一化到 [0, 1]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
            frames.append(frame_tensor)

            frame_idx += 1
            if frame_idx % 100 == 0:
                self.log_message.emit(f"已读取 {frame_idx}/{total_frames} 帧")

        cap.release()

        if self._cancelled:
            return

        # 堆叠帧 [N, H, W, C]
        frames_tensor = torch.stack(frames, dim=0)
        self.log_message.emit(f"读取完成: {len(frames)} 帧, 形状: {frames_tensor.shape}")

        # 插值
        self.stage_changed.emit("插值处理中...")

        interp_config = self._config.get("interpolation", {})
        multiplier = interp_config.get("multi", 2)

        # 处理帧
        output_frames = self._interpolate_frames(
            frames_tensor,
            multiplier,
            progress_callback=lambda c, t: self.progress_updated.emit(c, t, 0.0),
        )

        self.log_message.emit(f"插值完成: {len(output_frames)} 帧")

        if self._cancelled:
            return

        # 编码输出
        self.stage_changed.emit("编码中...")
        self._encode_video(output_frames, fps, width, height)

    def _interpolate_frames(
        self,
        frames: torch.Tensor,
        multiplier: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        """插值帧序列。

        Args:
            frames: 输入帧 [N, H, W, C]
            multiplier: 插值倍数
            progress_callback: 进度回调

        Returns:
            输出帧 [N * multiplier, H, W, C]
        """
        if self._model is None:
            raise RuntimeError("模型未加载")

        n = len(frames)
        output_frames = []
        clear_interval = self._torch_config.get("clear_cache_every", 10)

        # 预处理：NHWC -> NCHW
        frames_chw = frames.permute(0, 3, 1, 2)  # [N, C, H, W]

        for i in range(n - 1):
            if self._cancelled:
                break

            # 添加原始帧
            output_frames.append(frames[i])

            # 生成插值帧
            for j in range(1, multiplier):
                timestep = j / multiplier

                # 插值
                result: VFIResult = self._model.interpolate(
                    frames_chw[i],
                    frames_chw[i + 1],
                    timestep=timestep,
                )

                # 转换回 NHWC
                interp_frame = result.frame.permute(1, 2, 0)  # [H, W, C]
                output_frames.append(interp_frame)

            # 进度回调
            if progress_callback:
                progress_callback(i + 1, n - 1)

            # 清理缓存
            if (i + 1) % clear_interval == 0:
                clear_cache()

        # 添加最后一帧
        if not self._cancelled:
            output_frames.append(frames[-1])

        return torch.stack(output_frames)

    def _encode_video(
        self,
        frames: torch.Tensor,
        fps: float,
        width: int,
        height: int,
    ):
        """使用 FFmpeg 编码视频。

        Args:
            frames: 输出帧 [N, H, W, C]
            fps: 帧率
            width: 宽度
            height: 高度
        """
        # 计算输出帧率
        interp_config = self._config.get("interpolation", {})
        multiplier = interp_config.get("multi", 2)
        output_fps = fps * multiplier

        output_config = self._config.get("output", {})
        codec = output_config.get("codec", "libx265")
        quality = output_config.get("quality", 22)
        preset = output_config.get("preset", "medium")

        # 构建 FFmpeg 命令
        cmd = [
            "ffmpeg",
            "-y",  # 覆盖输出
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(output_fps),
            "-i", "pipe:",
        ]

        # 编码器设置
        if codec == "libx265":
            cmd.extend([
                "-c:v", "libx265",
                "-crf", str(quality),
                "-preset", preset,
            ])
        elif codec == "libx264":
            cmd.extend([
                "-c:v", "libx264",
                "-crf", str(quality),
                "-preset", preset,
            ])
        elif codec == "hevc_nvenc":
            cmd.extend([
                "-c:v", "hevc_nvenc",
                "-rc:v", "vbr",
                "-cq:v", str(quality),
                "-preset:v", preset,
            ])
        else:
            cmd.extend(["-c:v", codec])

        # 输出
        assert self._output_path is not None
        cmd.append(self._output_path)

        self.log_message.emit(f"运行: {' '.join(cmd)}")

        # 启动 FFmpeg
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # 确保 stdin 存在
        assert proc.stdin is not None

        # 写入帧
        for i, frame in enumerate(frames):
            if self._cancelled:
                proc.terminate()
                break

            # 转换为 uint8
            frame_uint8 = (frame.clamp(0, 1) * 255).byte().numpy()
            proc.stdin.write(frame_uint8.tobytes())

            if i % 100 == 0:
                self.progress_updated.emit(i, len(frames), 0.0)

        proc.stdin.close()
        proc.wait()

        if proc.returncode != 0 and not self._cancelled:
            stderr = proc.stderr.read().decode() if proc.stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg 错误: {stderr}")

    def _cleanup(self):
        """清理资源。"""
        if self._model is not None:
            self._model.unload()
            self._model = None
        clear_cache()
        gc.collect()

    def pause(self):
        """暂停处理。"""
        self._paused = True

    def resume(self):
        """恢复处理。"""
        self._paused = False

    def cancel(self):
        """取消处理。"""
        self._cancelled = True

    def is_paused(self) -> bool:
        """检查是否暂停。"""
        return self._paused

    def is_running(self) -> bool:
        """检查是否正在运行。"""
        return self.isRunning() and not self._cancelled
