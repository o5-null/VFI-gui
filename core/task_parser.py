"""TaskParser：将 TaskDescriptor 解析为 TaskDefinition

解析规则：
1. 参数验证：检查 video_path 存在、pipeline_config 完整
2. 后端解析：从 inference.backend 确定 BackendType
3. 后端配置解析：设备、精度、线程数等
4. 处理配置解析：插帧、超分、场景检测、输出配置
5. 输出路径解析：生成输出文件路径（基于输入文件名+时间戳+编解码器）
6. 子任务规划：预计算子任务数量（由 SubTaskGenerator 后续计算）

Architecture:
    TaskDescriptor → TaskParser → TaskDefinition → SubTaskGenerator → List[SubTask]

Usage:
    from core.task_parser import TaskParser
    from core.types import TaskDescriptor

    parser = TaskParser(config)
    task_def = parser.parse(descriptor)

CLI:
    python -m core.task_parser --video test.mp4 --config '{"interpolation":{"model_type":"rife","multi":2}}' --dry-run --json
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from loguru import logger

from core.types import (
    BackendConfig,
    BackendType,
    ProcessingConfig,
    SubTaskPlan,
    TaskDefinition,
    TaskDescriptor,
)

if TYPE_CHECKING:
    from core.config.config_facade import ConfigFacade


class TaskParser:
    """任务解析器：将 TaskDescriptor 解析为 TaskDefinition

    解析流程：
    1. 参数验证：检查 video_path 存在、pipeline_config 完整
    2. 后端解析：从 inference.backend 确定 BackendType
    3. 后端配置解析：设备、精度、线程数等
    4. 处理配置解析：插帧、超分、场景检测、输出配置
    5. 输出路径解析：生成输出文件路径
    6. 任务 ID 生成：使用 UUID

    Attributes:
        _config: ConfigFacade 实例（可选），用于获取全局配置作为 fallback
    """

    def __init__(self, config: Optional["ConfigFacade"] = None):
        """初始化 TaskParser

        Args:
            config: ConfigFacade 实例（可选），用于获取全局配置作为 fallback
        """
        self._config = config

    def parse(self, descriptor: TaskDescriptor) -> TaskDefinition:
        """解析 TaskDescriptor 为 TaskDefinition

        Args:
            descriptor: 任务描述符，包含 video_path 和 pipeline_config

        Returns:
            TaskDefinition 实例，包含完整的解析结果

        Raises:
            ValueError: video_path 或 pipeline_config 为空
            FileNotFoundError: video_path 指定的文件不存在（非图像序列）
        """
        # 1. 参数验证
        self._validate(descriptor)

        # 2. 解析配置
        pipeline_config = descriptor.pipeline_config
        backend_type = self._parse_backend_type(pipeline_config)
        backend_config = self._parse_backend_config(pipeline_config, backend_type)
        processing_config = self._parse_processing_config(pipeline_config)

        # 3. 生成输出路径
        output_path = self._generate_output_path(descriptor, processing_config)

        # 4. 生成任务 ID
        task_id = self._generate_task_id()

        # 5. 构造 SubTaskPlan（placeholder，由 SubTaskGenerator 后续计算）
        subtask_plan = SubTaskPlan(
            total_subtasks=0,
            input_frame_count=0,
            output_frame_count=0,
            multiplier=processing_config.interpolation.get("multi", 2),
            batch_size=1,
            requires_scene_detect=processing_config.scene_detection.get("enabled", False),
        )

        # 6. 构造 TaskDefinition
        return TaskDefinition(
            task_id=task_id,
            video_path=descriptor.video_path,
            image_sequence_frames=descriptor.image_sequence_frames,
            backend_type=backend_type,
            backend_config=backend_config,
            processing_config=processing_config,
            subtask_plan=subtask_plan,
            output_path=output_path,
        )

    def _validate(self, descriptor: TaskDescriptor) -> None:
        """验证参数完整性

        Args:
            descriptor: 任务描述符

        Raises:
            ValueError: video_path 或 pipeline_config 为空
            FileNotFoundError: video_path 指定的文件不存在（非图像序列）
        """
        if not descriptor.video_path:
            raise ValueError("video_path is required")

        # 图像序列模式：如果提供了 image_sequence_frames，跳过文件存在检查
        if descriptor.image_sequence_frames and len(descriptor.image_sequence_frames) > 0:
            # 图像序列模式，不需要单个视频文件
            pass
        else:
            # 视频文件模式，检查文件存在
            if not os.path.exists(descriptor.video_path):
                raise FileNotFoundError(f"Video file not found: {descriptor.video_path}")

        if not descriptor.pipeline_config:
            raise ValueError("pipeline_config is required")

    def _parse_backend_type(self, config: Dict[str, Any]) -> BackendType:
        """解析后端类型

        从 inference.backend 或 legacy pipeline_config.backend 获取后端类型。
        使用 BackendType 枚举构造函数，fallback 到 BackendType.TORCH。

        Args:
            config: pipeline_config 字典

        Returns:
            BackendType 实例
        """
        inference = config.get("inference", {})
        backend_str = inference.get("backend", config.get("backend", "torch"))

        try:
            return BackendType(backend_str)
        except ValueError:
            logger.warning(f"Unknown backend type: {backend_str}, falling back to TORCH")
            return BackendType.TORCH

    def _parse_backend_config(
        self,
        config: Dict[str, Any],
        backend_type: BackendType,
    ) -> BackendConfig:
        """解析后端配置

        从 pipeline_config.inference 获取后端配置，
        使用 ConfigFacade.runtime.get_all() 作为 fallback。

        Args:
            config: pipeline_config 字典
            backend_type: 已解析的后端类型

        Returns:
            BackendConfig 实例
        """
        inference = config.get("inference", {})

        # 获取全局运行时配置作为 fallback
        runtime_settings: Dict[str, Any] = {}
        if self._config:
            runtime_settings = self._config.runtime.get_all()

        # Resolve device: inference config > runtime settings
        device = inference.get("device", runtime_settings.get("device", "auto"))

        # Resolve precision: inference config > runtime settings > fp16 boolean compat
        precision = inference.get("precision", None)
        if precision is None:
            # Fallback: derive from runtime fp16 boolean
            fp16 = runtime_settings.get("fp16", True)
            precision = "fp16" if fp16 else "fp32"

        # Resolve torch.compile
        torch_compile = inference.get("torch_compile", False)

        return BackendConfig(
            backend_type=backend_type,
            models_dir=config.get("models_dir", "models"),
            temp_dir=config.get("temp_dir", "temp"),
            output_dir=config.get("output_dir", "output"),
            num_threads=runtime_settings.get("num_threads", 4),
            device=device,
            precision=precision,
            torch_compile=torch_compile,
            extra=config.get("backend_extra", {}),
        )

    def _parse_processing_config(self, config: Dict[str, Any]) -> ProcessingConfig:
        """解析处理配置

        Args:
            config: pipeline_config 字典

        Returns:
            ProcessingConfig 实例
        """
        return ProcessingConfig(
            interpolation=config.get("interpolation", {}),
            upscaling=config.get("upscaling", {}),
            scene_detection=config.get("scene_detection", {}),
            output=config.get("output", {}),
        )

    def _generate_output_path(
        self,
        descriptor: TaskDescriptor,
        processing_config: ProcessingConfig,
    ) -> Path:
        """生成输出文件路径

        根据输入文件名、时间戳、编解码器生成输出路径。
        支持视频和图像序列两种输出模式。

        Args:
            descriptor: 任务描述符
            processing_config: 处理配置

        Returns:
            输出文件路径
        """
        from core.codec_manager import get_codec_manager, CodecConfig

        # Get output config from pipeline config
        output_config = processing_config.output

        # Resolve output_dir
        output_dir = output_config.get("output_dir", "")
        if not output_dir:
            # Get from ConfigFacade.paths as fallback
            if self._config:
                output_dir = self._config.paths.get_output_dir()
            else:
                output_dir = "output"

        output_subdir = output_config.get("output_subdir", "")
        output_filename = output_config.get("output_filename", "")

        # Generate output filename if not specified
        if not output_filename:
            input_path = Path(descriptor.video_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{input_path.stem}_processed_{timestamp}"

        # Combine output_dir and output_subdir
        if output_subdir:
            output_dir = str(Path(output_dir) / output_subdir)

        # Get codec manager and config
        codec_manager = get_codec_manager()
        codec_config = CodecConfig.from_dict(output_config)
        codec_manager.set_config(codec_config)

        # Check output mode first
        output_mode = output_config.get("output_mode", "video")
        if output_mode == "images":
            # Image sequence output - use specified format
            image_format = output_config.get("image_format", "png")
            extension = f".{image_format}"
        else:
            # Video output - determine extension based on codec
            codec = codec_config.codec or "hevc_nvenc"
            if codec in ("hevc_nvenc", "h265_nvenc", "av1_nvenc"):
                extension = ".mkv"
            elif codec in ("h264_nvenc", "avc_nvenc"):
                extension = ".mp4"
            elif codec in ("vp9", "av1", "libaom-av1", "libvpx-vp9"):
                extension = ".mkv"
            elif codec == "libx265":
                extension = ".mkv"
            elif codec == "libx264":
                extension = ".mp4"
            elif codec == "gif":
                extension = ".gif"
            else:
                extension = ".mkv"  # Default

        output_path = Path(output_dir) / f"{output_filename}{extension}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        return output_path

    def _generate_task_id(self) -> str:
        """生成任务 ID

        使用 UUID 的前 8 位作为任务 ID（与 TaskOrchestrator.submit_task 一致）。

        Returns:
            任务 ID 字符串
        """
        return str(uuid.uuid4())[:8]


# ====================
# CLI Entry Point
# ====================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TaskParser CLI: 解析任务描述符为任务定义"
    )
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument(
        "--config",
        required=True,
        help="pipeline_config JSON 字符串",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只解析不执行（TaskParser 本身不执行，此参数仅作标识）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON 格式输出",
    )
    args = parser.parse_args()

    try:
        # 解析 JSON 配置
        pipeline_config = json.loads(args.config)

        # 构造 TaskDescriptor
        descriptor = TaskDescriptor(
            video_path=args.video,
            pipeline_config=pipeline_config,
        )

        # 解析为 TaskDefinition
        task_parser = TaskParser()
        task_def = task_parser.parse(descriptor)

        result = {
            "task_id": task_def.task_id,
            "video_path": task_def.video_path,
            "backend_type": task_def.backend_type.value,
            "backend_config": {
                "device": task_def.backend_config.device,
                "precision": task_def.backend_config.precision,
                "num_threads": task_def.backend_config.num_threads,
            },
            "processing_config": {
                "interpolation": task_def.processing_config.interpolation,
                "upscaling": task_def.processing_config.upscaling,
                "scene_detection": task_def.processing_config.scene_detection,
                "output": task_def.processing_config.output,
            },
            "subtask_plan": {
                "multiplier": task_def.subtask_plan.multiplier,
                "requires_scene_detect": task_def.subtask_plan.requires_scene_detect,
            },
            "output_path": str(task_def.output_path),
        }

        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"任务 ID: {task_def.task_id}")
            print(f"视频路径: {task_def.video_path}")
            print(f"后端类型: {task_def.backend_type.value}")
            print(f"设备: {task_def.backend_config.device}")
            print(f"精度: {task_def.backend_config.precision}")
            print(f"输出路径: {task_def.output_path}")

    except FileNotFoundError as e:
        logger.error(f"文件未找到: {e}")
        raise SystemExit(1)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析错误: {e}")
        raise SystemExit(1)
    except ValueError as e:
        logger.error(f"参数错误: {e}")
        raise SystemExit(1)
    except Exception as e:
        logger.error(f"解析失败: {e}")
        if args.json:
            print(json.dumps({"error": str(e)}, ensure_ascii=False))
        else:
            print(f"错误: {e}")
        raise SystemExit(1)


__all__ = ["TaskParser"]