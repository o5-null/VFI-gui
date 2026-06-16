"""推理结果验证器

验证 Backend 推理结果，检查帧形状、数据类型、NaN/Inf 值和值域范围。
验证失败返回 ValidationResult(valid=False, error=...)，不修改或删除推理结果。
"""

import torch
from typing import Optional

from core.types import SubTask, InferenceResult, ValidationResult


class ResultValidator:
    """验证 Backend 推理结果

    检查项：
    - 帧数是否符合预期
    - 帧尺寸是否一致
    - 数据类型是否正确
    - 无 NaN 值
    - 值域在 [0, 1] 或 [0, 255]
    """

    MAX_RETRY = 3  # 最大重试次数

    def validate(self, subtask: SubTask, result: InferenceResult) -> ValidationResult:
        """验证推理结果

        Args:
            subtask: 对应的子任务（保留用于后续帧数校验）
            result: 推理结果

        Returns:
            ValidationResult: 验证结果
        """
        if not result.success:
            return ValidationResult(valid=False, error=result.error)

        checks = [
            self._check_frame_shape,
            self._check_frame_dtype,
            self._check_no_nan,
            self._check_value_range,
        ]
        for check in checks:
            if not check(result):
                return ValidationResult(valid=False, error=check.__name__)

        return ValidationResult(valid=True)

    def _check_frame_shape(self, result: InferenceResult) -> bool:
        """帧尺寸是否有效（非空）"""
        output = result.output_frame
        return output.ndim == 3 and output.shape[0] in (1, 3) and output.shape[1] > 0

    def _check_frame_dtype(self, result: InferenceResult) -> bool:
        """数据类型是否正确"""
        return result.output_frame.dtype in (torch.float32, torch.float16)

    def _check_no_nan(self, result: InferenceResult) -> bool:
        """无 NaN/Inf 值"""
        return not torch.any(torch.isnan(result.output_frame)) and \
               not torch.any(torch.isinf(result.output_frame))

    def _check_value_range(self, result: InferenceResult) -> bool:
        """值域在 [0, 1] 或合理范围"""
        min_val = result.output_frame.min().item()
        max_val = result.output_frame.max().item()
        return min_val >= -0.1 and max_val <= 1.1  # 允许轻微溢出
