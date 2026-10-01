"""多帧适配器的纯 numpy 单测（不依赖 onnxruntime / torch 运行时）。

覆盖 ``docs/ONNX_TRT_MULTI_MODEL_SPEC.md`` §5/§9：

- STMFNet ``expand_frames([a,b]) -> [a,a,b,b]`` 且 ``build_feeds`` 四输入
  ``I0..I3`` 的顺序与复制语义一致；
- FLAVR 单输入 ``[1,12,H,W]`` 与 2x/4x/8x 的输出选帧（``postprocess``）；
- ``get_adapter`` 对未知类型抛 :class:`AdapterError`（不再静默回落）；
- ``get_adapter("cain")`` 仍为 :class:`UnsupportedAdapter`。
"""

from __future__ import annotations

import numpy as np
import pytest

from core.backends.adapters import get_adapter
from core.backends.adapters.base import AdapterContext, AdapterError
from core.backends.adapters.multiframe import FLAVRAdapter, MultiframeAdapter
from core.backends.adapters.registry import GenericNCHWAdapter
from core.backends.adapters.unsupported import UnsupportedAdapter


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _const(value: float, h: int = 16, w: int = 16) -> np.ndarray:
    """生成常量帧，便于按数值识别来源。"""
    return np.full((3, h, w), value, dtype=np.float32)


def _ctx(h: int = 16, w: int = 16, **extra) -> AdapterContext:
    return AdapterContext(
        input_names=[],
        output_names=["output"],
        src_hw=(h, w),
        pad_hw=(h, w),
        extra=dict(extra),
    )


# ---------------------------------------------------------------------------
# STMFNet：2 -> 4 复制
# ---------------------------------------------------------------------------

class TestStmfnetExpandFrames:
    def test_two_frames_duplicated(self):
        a, b = _const(0.1), _const(0.2)
        frames = MultiframeAdapter().expand_frames([a, b])
        assert len(frames) == 4
        assert np.array_equal(frames[0], a)
        assert np.array_equal(frames[1], a)
        assert np.array_equal(frames[2], b)
        assert np.array_equal(frames[3], b)

    def test_four_frames_identity_order(self):
        frames_in = [_const(v) for v in (0.1, 0.2, 0.3, 0.4)]
        frames = MultiframeAdapter().expand_frames(frames_in)
        assert len(frames) == 4
        for got, expected in zip(frames, frames_in):
            assert np.array_equal(got, expected)

    def test_more_than_four_truncates(self):
        frames = MultiframeAdapter().expand_frames([_const(v) for v in range(6)])
        assert len(frames) == 4

    def test_other_count_raises(self):
        with pytest.raises(AdapterError):
            MultiframeAdapter().expand_frames([_const(0.1)])

    def test_build_feeds_i0_i1_i2_i3_mapping(self):
        a, b = _const(0.1), _const(0.2)
        adapter = MultiframeAdapter()
        frames = adapter.expand_frames([a, b])
        ctx = _ctx()
        feeds = adapter.build_feeds(frames, 0.5, ctx)

        # 默认输入名按位置绑定 I0..I3。
        assert list(feeds.keys()) == ["I0", "I1", "I2", "I3"]
        # I0=I1=a, I2=I3=b （与 PyTorch stmfnet/__init__.py:87 一致）。
        assert np.allclose(feeds["I0"], 0.1)
        assert np.allclose(feeds["I1"], 0.1)
        assert np.allclose(feeds["I2"], 0.2)
        assert np.allclose(feeds["I3"], 0.2)
        # 每路均为 [1,3,H,W]。
        for key in ("I0", "I1", "I2", "I3"):
            assert feeds[key].shape == (1, 3, 16, 16)

    def test_build_feeds_respects_session_names(self):
        a, b = _const(0.1), _const(0.2)
        adapter = MultiframeAdapter()
        frames = adapter.expand_frames([a, b])
        ctx = AdapterContext(
            input_names=["f0", "f1", "f2", "f3"],
            output_names=["output"],
            src_hw=(16, 16),
            pad_hw=(16, 16),
        )
        feeds = adapter.build_feeds(frames, 0.5, ctx)
        assert list(feeds.keys()) == ["f0", "f1", "f2", "f3"]


# ---------------------------------------------------------------------------
# FLAVR：单输入 [1,12,H,W] + 输出选帧
# ---------------------------------------------------------------------------

class TestFlavrAdapter:
    def test_expand_two_to_four(self):
        a, b = _const(0.1), _const(0.2)
        frames = FLAVRAdapter().expand_frames([a, b])
        assert len(frames) == 4
        assert np.array_equal(frames[0], a)
        assert np.array_equal(frames[1], a)
        assert np.array_equal(frames[2], b)
        assert np.array_equal(frames[3], b)

    def test_single_stacked_input(self):
        adapter = FLAVRAdapter()
        frames = adapter.expand_frames([_const(0.1), _const(0.2)])
        ctx = _ctx()
        feeds = adapter.build_feeds(frames, 0.25, ctx)
        assert list(feeds.keys()) == ["frames"]
        assert feeds["frames"].shape == (1, 12, 16, 16)
        # timestep 透传给 postprocess 用于选帧。
        assert ctx.extra["timestep"] == 0.25

    @staticmethod
    def _stacked_output(n_outputs: int, h: int = 16, w: int = 16) -> np.ndarray:
        """构造 [1, 3n, H, W]，第 k 帧块填 k+1（便于识别选中帧）。"""
        out = np.zeros((1, 3 * n_outputs, h, w), dtype=np.float32)
        for k in range(n_outputs):
            out[:, k * 3 : k * 3 + 3] = float(k + 1)
        return out

    def test_2x_single_output_selects_only_frame(self):
        adapter = FLAVRAdapter()
        adapter.build_feeds([_const(0.1)] * 4, 0.5, _ctx())
        frame = adapter.postprocess([self._stacked_output(1)], _ctx())
        assert frame.shape == (3, 16, 16)
        assert np.allclose(frame, 1.0)

    @pytest.mark.parametrize(
        "timestep,expected_index",
        [(0.0, 0), (0.25, 0), (0.5, 1), (0.75, 2), (1.0, 2)],
    )
    def test_4x_frame_selection(self, timestep, expected_index):
        adapter = FLAVRAdapter()
        ctx = _ctx()
        adapter.build_feeds([_const(0.1)] * 4, timestep, ctx)
        frame = adapter.postprocess([self._stacked_output(3)], ctx)
        assert np.allclose(frame, float(expected_index + 1))

    @pytest.mark.parametrize(
        "timestep,expected_index",
        [(0.0, 0), (0.5, 3), (1.0, 6)],
    )
    def test_8x_frame_selection(self, timestep, expected_index):
        adapter = FLAVRAdapter()
        ctx = _ctx()
        adapter.build_feeds([_const(0.1)] * 4, timestep, ctx)
        frame = adapter.postprocess([self._stacked_output(7)], ctx)
        assert np.allclose(frame, float(expected_index + 1))

    def test_indices_are_clamped_in_range(self):
        adapter = FLAVRAdapter()
        for t in (0.0, 0.5, 1.0):
            ctx = _ctx()
            adapter.build_feeds([_const(0.1)] * 4, t, ctx)
            idx = adapter._resolve_index(3, ctx)
            assert 0 <= idx <= 2

    def test_non_multiple_of_three_channels_raises(self):
        adapter = FLAVRAdapter()
        ctx = _ctx()
        adapter.build_feeds([_const(0.1)] * 4, 0.5, ctx)
        with pytest.raises(AdapterError):
            adapter.postprocess([np.zeros((1, 7, 16, 16), np.float32)], ctx)

    def test_postprocess_unpads_to_src(self):
        adapter = FLAVRAdapter()
        ctx = AdapterContext(
            input_names=[],
            output_names=["output"],
            src_hw=(10, 12),
            pad_hw=(16, 16),
            extra={},
        )
        adapter.build_feeds([_const(0.1, 16, 16)] * 4, 0.5, ctx)
        frame = adapter.postprocess([self._stacked_output(3, 16, 16)], ctx)
        assert frame.shape == (3, 10, 12)


# ---------------------------------------------------------------------------
# registry：未知类型拒绝 + cain/sepconv
# ---------------------------------------------------------------------------

class TestGetAdapterDispatch:
    def test_unknown_type_raises(self):
        with pytest.raises(AdapterError):
            get_adapter("unknown_xyz")

    def test_unknown_type_does_not_return_generic(self):
        # 确保静默回落已被移除（GenericNCHWAdapter 不再自动返回）。
        try:
            adapter = get_adapter("totally_unregistered_model")
        except AdapterError:
            adapter = None
        assert not isinstance(adapter, GenericNCHWAdapter)

    def test_cain_is_unsupported_adapter(self):
        adapter = get_adapter("cain")
        assert isinstance(adapter, UnsupportedAdapter)
        assert adapter.model_type == "cain"
        with pytest.raises(AdapterError):
            adapter.build_feeds([_const(0.1), _const(0.2)], 0.5, _ctx())

    def test_sepconv_is_unsupported_adapter(self):
        assert isinstance(get_adapter("sepconv"), UnsupportedAdapter)

    def test_known_type_returns_registered_adapter(self):
        assert get_adapter("stmfnet").__class__ is MultiframeAdapter
        assert get_adapter("flavr").__class__ is FLAVRAdapter
