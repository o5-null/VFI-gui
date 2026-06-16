"""子任务生成器：Task 层级生成最小调度单元

根据 TaskDefinition 和视频元数据生成子任务列表。
核心规则：对 (frame_i, frame_i+1) × (multiplier-1) 个子任务。

Architecture:
    TaskParser → TaskDefinition → SubTaskGenerator → List[SubTask]
    SubTaskGenerator 只负责规划，不持有帧数据、不执行推理。

Usage:
    from core.subtask_generator import SubTaskGenerator
    from core.types import TaskDefinition, VideoMetadata

    generator = SubTaskGenerator()
    subtasks = generator.generate(task_def, metadata)
    plan = generator.compute_plan(task_def, metadata)
"""

from __future__ import annotations

import argparse
import json
from typing import List

from loguru import logger

from core.types import (
    FrameRef,
    ProcessingConfig,
    SubTask,
    SubTaskPlan,
    SubTaskState,
    TaskDefinition,
    VideoMetadata,
)


class SubTaskGenerator:
    """根据 TaskDefinition 和视频元数据生成子任务列表

    核心规则：对 (frame_i, frame_i+1) × (multiplier-1) 个子任务

    每个 SubTask 只引用 FrameRef（不持有帧数据），
    cache_key 格式统一为 f"{source_path}:{frame_index}"，
    timestep 计算为 j / multiplier（均匀插帧）。
    """

    def generate(
        self,
        task_def: TaskDefinition,
        metadata: VideoMetadata,
    ) -> List[SubTask]:
        """生成子任务列表

        对每个相邻帧对 (frame_i, frame_i+1)，生成 (multiplier-1) 个子任务，
        每个子任务对应一个插帧位置 timestep = j / multiplier。

        Args:
            task_def: 解析后的任务定义
            metadata: 视频元数据（帧数、分辨率等）

        Returns:
            子任务列表，长度 = (total_frames - 1) × (multiplier - 1)
        """
        multiplier = task_def.processing_config.interpolation.get("multi", 2)
        frame_count = metadata.total_frames
        subtasks: List[SubTask] = []

        for i in range(frame_count - 1):
            for j in range(1, multiplier):
                subtask_id = f"st_{task_def.task_id}_{i}_{j}"
                timestep = j / multiplier

                subtask = SubTask(
                    subtask_id=subtask_id,
                    parent_task_id=task_def.task_id,
                    input_frames=[
                        FrameRef(
                            source_path=task_def.video_path,
                            frame_index=i,
                            cache_key=f"{task_def.video_path}:{i}",
                        ),
                        FrameRef(
                            source_path=task_def.video_path,
                            frame_index=i + 1,
                            cache_key=f"{task_def.video_path}:{i + 1}",
                        ),
                    ],
                    model_config={"timestep": timestep},
                    required_files=[task_def.video_path],
                    state=SubTaskState.PENDING,
                )
                subtasks.append(subtask)

        return subtasks

    def compute_plan(
        self,
        task_def: TaskDefinition,
        metadata: VideoMetadata,
    ) -> SubTaskPlan:
        """预计算子任务规划

        计算输出帧数公式：output = frame_count × multiplier - (multiplier - 1)
        即 N 帧输入 → N×M - (M-1) 帧输出，其中 M = multiplier

        Args:
            task_def: 解析后的任务定义
            metadata: 视频元数据

        Returns:
            SubTaskPlan 包含子任务数量、输入/输出帧数等规划信息
        """
        multiplier = task_def.processing_config.interpolation.get("multi", 2)
        frame_count = metadata.total_frames

        return SubTaskPlan(
            total_subtasks=(frame_count - 1) * (multiplier - 1),
            input_frame_count=frame_count,
            output_frame_count=frame_count * multiplier - (multiplier - 1),
            multiplier=multiplier,
            batch_size=1,
            requires_scene_detect=task_def.processing_config.scene_detection.get(
                "enabled", False
            ),
        )


# ====================
# CLI Entry Point
# ====================


def _get_video_metadata(video_path: str) -> VideoMetadata:
    """从视频文件获取元数据（用于 CLI 模式）

    Args:
        video_path: 视频文件路径

    Returns:
        VideoMetadata 实例
    """
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames or 0
    fps = float(stream.average_rate) if stream.average_rate else 30.0
    width = stream.width or 0
    height = stream.height or 0
    codec = stream.codec_context.name if stream.codec_context else ""
    container.close()

    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        codec=codec,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SubTaskGenerator CLI: 预计算子任务规划"
    )
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument(
        "--multi", type=int, default=2, help="帧倍率 (默认: 2)"
    )
    parser.add_argument(
        "--json", action="store_true", help="JSON 格式输出"
    )
    args = parser.parse_args()

    try:
        from core.types import (
            BackendType,
            BackendConfig,
            ProcessingConfig as PC,
            TaskDefinition as TD,
        )
        from pathlib import Path
        import uuid

        # 获取视频元数据
        metadata = _get_video_metadata(args.video)

        # 构造 TaskDefinition（CLI 模式使用默认配置）
        task_def = TD(
            task_id=str(uuid.uuid4())[:8],
            video_path=args.video,
            backend_type=BackendType.TORCH,
            backend_config=BackendConfig(),
            processing_config=PC(
                interpolation={
                    "enabled": True,
                    "model_type": "rife",
                    "multi": args.multi,
                    "scale": 1.0,
                    "scene_change": False,
                }
            ),
            subtask_plan=SubTaskPlan(
                total_subtasks=0,
                input_frame_count=0,
                output_frame_count=0,
                multiplier=args.multi,
                batch_size=1,
                requires_scene_detect=False,
            ),
            output_path=Path("output"),
        )

        # 生成子任务规划和样例子任务
        generator = SubTaskGenerator()
        plan = generator.compute_plan(task_def, metadata)
        subtasks = generator.generate(task_def, metadata)

        result = {
            "video": args.video,
            "total_frames": metadata.total_frames,
            "multiplier": args.multi,
            "plan": {
                "total_subtasks": plan.total_subtasks,
                "input_frame_count": plan.input_frame_count,
                "output_frame_count": plan.output_frame_count,
                "requires_scene_detect": plan.requires_scene_detect,
            },
            "sample_subtask": (
                {
                    "subtask_id": subtasks[0].subtask_id,
                    "input_frames": [
                        {
                            "source_path": f.source_path,
                            "frame_index": f.frame_index,
                            "cache_key": f.cache_key,
                        }
                        for f in subtasks[0].input_frames
                    ],
                    "model_config": subtasks[0].model_config,
                    "state": subtasks[0].state.value,
                }
                if subtasks
                else None
            ),
            "actual_subtask_count": len(subtasks),
        }

        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"视频: {args.video}")
            print(f"输入帧数: {metadata.total_frames}")
            print(f"倍率: {args.multi}x")
            print(f"子任务数: {plan.total_subtasks}")
            print(f"输出帧数: {plan.output_frame_count}")
            if subtasks:
                st = subtasks[0]
                print(f"样例子任务: {st.subtask_id}")
                print(f"  timestep: {st.model_config['timestep']}")

    except FileNotFoundError as e:
        logger.error(f"文件未找到: {e}")
        raise SystemExit(1)
    except Exception as e:
        logger.error(f"生成失败: {e}")
        if args.json:
            print(json.dumps({"error": str(e)}, ensure_ascii=False))
        else:
            print(f"错误: {e}")
        raise SystemExit(1)
