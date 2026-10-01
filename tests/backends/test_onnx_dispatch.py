"""Adapter-dispatch tests for :class:`core.backends.onnx_backend.OnnxBackend`.

These tests use a **fake ONNX Runtime session** (monkeypatched into
``sys.modules``) so no ``.onnx`` asset or real ``onnxruntime`` install is
required. They verify the §6 pipeline of ``docs/ONNX_TRT_MULTI_MODEL_SPEC.md``:

* ``load_model`` reads EVERY input/output name and derives channels;
* ``SUPPORTED_MODELS`` is the adapter registry (no cain/sepconv);
* RIFE (7ch, single input) and FILM (multi-input) dispatch correctly;
* multi-output graphs collect ALL outputs before ``postprocess``.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

import core.backends.onnx_backend as onnx_mod
from core.backends.adapters import get_adapter, supported_models
from core.backends.onnx_backend import OnnxBackend
from core.types import BackendConfig, InferenceRequest


class _Tensor:
    """Minimal stand-in for an ORT NodeArg (name + shape)."""

    def __init__(self, name: str, shape: List[Any]):
        self.name = name
        self.shape = list(shape)


class _FakeSession:
    """Fake ``ort.InferenceSession`` recording the last run() call."""

    input_specs: List[Tuple[str, List[Any]]] = []
    output_specs: List[Tuple[str, List[Any]]] = []
    run_impl = None

    def __init__(self, path, sess_options=None, providers=None):
        self.path = path
        self.providers = list(providers or [])
        self.last_feeds: Dict[str, np.ndarray] = {}
        self.last_output_names: List[str] = []
        self.run_count = 0

    def get_inputs(self):
        return [_Tensor(n, s) for n, s in type(self).input_specs]

    def get_outputs(self):
        return [_Tensor(n, s) for n, s in type(self).output_specs]

    def get_providers(self):
        return self.providers

    def run(self, output_names, feeds):
        self.run_count += 1
        self.last_output_names = list(output_names)
        self.last_feeds = {k: np.asarray(v) for k, v in feeds.items()}
        return type(self).run_impl(output_names, feeds)


def _install_fake_ort(monkeypatch, inputs, outputs, run_impl):
    """Install a fake ``onnxruntime`` module and return the session class."""

    class Session(_FakeSession):
        input_specs = inputs
        output_specs = outputs

    Session.run_impl = staticmethod(run_impl)

    fake = types.ModuleType("onnxruntime")
    fake.__version__ = "0.0-fake"
    fake.InferenceSession = Session
    fake.SessionOptions = lambda: types.SimpleNamespace()
    fake.get_available_providers = lambda: ["CPUExecutionProvider"]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    return Session


@pytest.fixture
def patch_asset(monkeypatch, tmp_path):
    """Make resolve path return an existing dummy file; version passthrough."""

    dummy = tmp_path / "model.onnx"
    dummy.write_bytes(b"not-a-real-onnx")

    monkeypatch.setattr(onnx_mod, "resolve_onnx_path", lambda *a, **k: dummy)
    monkeypatch.setattr(onnx_mod, "checkpoint_to_version", lambda t, v: v or "")
    return dummy


def _make_backend(tmp_path) -> OnnxBackend:
    config = BackendConfig(models_dir=str(tmp_path))
    backend = OnnxBackend(config)
    assert backend.initialize()
    return backend


def _request(model_config: Dict[str, Any], h: int = 64, w: int = 64) -> InferenceRequest:
    return InferenceRequest(
        frame0=torch.rand(3, h, w, dtype=torch.float32),
        frame1=torch.rand(3, h, w, dtype=torch.float32),
        timestep=0.5,
        model_config=model_config,
    )


class TestSupportedModels:
    def test_derived_from_adapter_registry(self):
        assert OnnxBackend.SUPPORTED_MODELS == supported_models()

    def test_cain_sepconv_excluded(self):
        assert "cain" not in OnnxBackend.SUPPORTED_MODELS
        assert "sepconv" not in OnnxBackend.SUPPORTED_MODELS

    def test_expected_types_present(self):
        for t in ("rife", "film", "amt", "xvfi", "atm", "momo", "m2m"):
            assert t in OnnxBackend.SUPPORTED_MODELS, t


class TestRifeDispatch:
    def test_single_input_7ch_and_postprocess(self, monkeypatch, tmp_path, patch_asset):
        def run_impl(output_names, feeds):
            arr = next(iter(feeds.values()))
            return [np.zeros((1, 3, arr.shape[2], arr.shape[3]), np.float32)]

        Session = _install_fake_ort(
            monkeypatch,
            inputs=[("input", [1, 7, None, None])],
            outputs=[("output", [1, 3, None, None])],
            run_impl=run_impl,
        )
        backend = _make_backend(tmp_path)
        assert backend.load_model({"model_type": "rife", "model_version": ""})

        # All I/O names and channel count read from the session.
        assert backend._input_names == ["input"]
        assert backend._output_names == ["output"]
        assert backend._in_channels == 7
        assert backend._adapter is get_adapter("rife")

        result = backend.infer(_request({"model_type": "rife"}, h=64, w=64))
        assert result.success is True, result.error
        assert tuple(result.output_frame.shape) == (3, 64, 64)

        # The fake session saw the 7-channel RIFE feed and the declared output.
        feeds = backend._session.last_feeds
        assert list(feeds.keys()) == ["input"]
        assert feeds["input"].shape == (1, 7, 64, 64)
        assert backend._session.last_output_names == ["output"]
        assert isinstance(backend._session, Session)


class TestFilmDispatch:
    def test_multi_input_feeds_and_timestep(self, monkeypatch, tmp_path, patch_asset):
        def run_impl(output_names, feeds):
            img = feeds["img0"]
            return [np.zeros((1, 3, img.shape[2], img.shape[3]), np.float32)]

        _install_fake_ort(
            monkeypatch,
            inputs=[
                ("img0", [1, 3, None, None]),
                ("img1", [1, 3, None, None]),
                ("timestep", [1]),
            ],
            outputs=[("output", [1, 3, None, None])],
            run_impl=run_impl,
        )
        backend = _make_backend(tmp_path)
        assert backend.load_model({"model_type": "film", "model_version": ""})

        assert backend._input_names == ["img0", "img1", "timestep"]
        assert backend._in_channels == 3
        assert backend._adapter is get_adapter("film")

        result = backend.infer(_request({"model_type": "film"}, h=64, w=64))
        assert result.success is True, result.error
        assert tuple(result.output_frame.shape) == (3, 64, 64)

        feeds = backend._session.last_feeds
        assert set(feeds.keys()) == {"img0", "img1", "timestep"}
        assert feeds["img0"].shape == (1, 3, 64, 64)
        assert feeds["img1"].shape == (1, 3, 64, 64)
        # FILM's timestep feed is a 1-element vector [t].
        assert feeds["timestep"].shape == (1,)


    def test_static_256_rejects_other_resolution(self, monkeypatch, tmp_path, patch_asset):
        # Real FILM asset declares [1,3,256,256]; a non-256 request must be
        # rejected up front (M2) instead of raising INVALID_ARGUMENT in ORT.
        _install_fake_ort(
            monkeypatch,
            inputs=[
                ("img0", [1, 3, 256, 256]),
                ("img1", [1, 3, 256, 256]),
                ("timestep", [1]),
            ],
            outputs=[("output", [1, 3, 256, 256])],
            run_impl=lambda o, f: [
                np.zeros((1, 3, 256, 256), np.float32)
            ],
        )
        backend = _make_backend(tmp_path)
        assert backend.load_model({"model_type": "film", "model_version": ""})
        assert backend._fixed_hw == (256, 256)

        result = backend.infer(_request({"model_type": "film"}, h=64, w=64))
        assert result.success is False
        assert "256x256" in result.error
        assert "64x64" in result.error

    def test_static_256_accepts_matching_resolution(self, monkeypatch, tmp_path, patch_asset):
        def run_impl(output_names, feeds):
            img = feeds["img0"]
            return [np.zeros((1, 3, img.shape[2], img.shape[3]), np.float32)]

        _install_fake_ort(
            monkeypatch,
            inputs=[
                ("img0", [1, 3, 256, 256]),
                ("img1", [1, 3, 256, 256]),
                ("timestep", [1]),
            ],
            outputs=[("output", [1, 3, 256, 256])],
            run_impl=run_impl,
        )
        backend = _make_backend(tmp_path)
        assert backend.load_model({"model_type": "film", "model_version": ""})

        result = backend.infer(_request({"model_type": "film"}, h=256, w=256))
        assert result.success is True, result.error
        assert tuple(result.output_frame.shape) == (3, 256, 256)


class TestOnlyT05Guard:
    def test_non_half_timestep_rejected(self, monkeypatch, tmp_path, patch_asset):
        # ATM ignores timestep and always returns t=0.5; a t != 0.5 request must
        # fail loudly rather than silently returning the wrong frame (m1).
        _install_fake_ort(
            monkeypatch,
            inputs=[("img0", [1, 3, None, None]), ("img1", [1, 3, None, None])],
            outputs=[("output", [1, 3, None, None])],
            run_impl=lambda o, f: [np.zeros((1, 3, 64, 64), np.float32)],
        )
        backend = _make_backend(tmp_path)
        assert backend.load_model({"model_type": "atm", "model_version": ""})

        # t=0.5 is allowed.
        ok = backend.infer(
            InferenceRequest(
                frame0=torch.rand(3, 64, 64),
                frame1=torch.rand(3, 64, 64),
                timestep=0.5,
                model_config={"model_type": "atm"},
            )
        )
        assert ok.success is True, ok.error

        bad = backend.infer(
            InferenceRequest(
                frame0=torch.rand(3, 64, 64),
                frame1=torch.rand(3, 64, 64),
                timestep=0.25,
                model_config={"model_type": "atm"},
            )
        )
        assert bad.success is False
        assert "timestep=0.5" in bad.error


class TestMultiOutputCollection:
    def test_all_outputs_collected(self, monkeypatch, tmp_path, patch_asset):
        seen = {}

        def run_impl(output_names, feeds):
            arr = next(iter(feeds.values()))
            seen["n_outputs"] = len(output_names)
            return [
                np.zeros((1, 3, arr.shape[2], arr.shape[3]), np.float32),
                np.zeros((1, 3, arr.shape[2], arr.shape[3]), np.float32),
            ]

        _install_fake_ort(
            monkeypatch,
            inputs=[("input", [1, 7, None, None])],
            outputs=[("output", [1, 3, None, None]), ("aux", [1, 3, None, None])],
            run_impl=run_impl,
        )
        backend = _make_backend(tmp_path)
        assert backend.load_model({"model_type": "rife", "model_version": ""})

        # Both declared outputs are requested from the session.
        assert backend._output_names == ["output", "aux"]
        result = backend.infer(_request({"model_type": "rife"}))
        assert result.success is True, result.error
        assert backend._session.last_output_names == ["output", "aux"]
        assert seen["n_outputs"] == 2


class TestFlavrDispatch:
    def test_single_stacked_multiframe_input(self, monkeypatch, tmp_path, patch_asset):
        # Contract: FLAVR uses a SINGLE input "frames" carrying the stacked
        # multi-frame tensor [1, 12, H, W] (NOT multiple inputs); frame
        # selection happens inside adapter.postprocess.
        def run_impl(output_names, feeds):
            f = feeds["frames"]
            # FLAVR emits a channel-concatenated [1, 3n, H, W] tensor.
            return [np.zeros((1, 9, f.shape[2], f.shape[3]), np.float32)]

        _install_fake_ort(
            monkeypatch,
            inputs=[("frames", [1, 12, None, None])],
            outputs=[("output", [1, 3, None, None])],
            run_impl=run_impl,
        )
        backend = _make_backend(tmp_path)
        assert backend.load_model({"model_type": "flavr", "model_version": ""})

        assert backend._input_names == ["frames"]
        assert backend._in_channels == 12
        assert backend._adapter is get_adapter("flavr")

        result = backend.infer(_request({"model_type": "flavr"}, h=64, w=64))
        assert result.success is True, result.error
        assert tuple(result.output_frame.shape) == (3, 64, 64)

        feeds = backend._session.last_feeds
        # Single stacked input, 12 channels (= 4 frames x 3).
        assert set(feeds.keys()) == {"frames"}
        assert feeds["frames"].shape[1] == 12
        assert backend._session.last_output_names == ["output"]


class TestLoadFailure:
    def test_unsupported_model_returns_false(self, monkeypatch, tmp_path):
        # cain has no implementation -> adapter registry rejects it.
        _install_fake_ort(
            monkeypatch,
            inputs=[("input", [1, 7, None, None])],
            outputs=[("output", [1, 3, None, None])],
            run_impl=lambda o, f: [],
        )
        backend = _make_backend(tmp_path)
        assert backend.load_model({"model_type": "cain", "model_version": ""}) is False
        assert backend._adapter is None

    def test_missing_path_returns_false(self, monkeypatch, tmp_path):
        _install_fake_ort(
            monkeypatch,
            inputs=[("input", [1, 7, None, None])],
            outputs=[("output", [1, 3, None, None])],
            run_impl=lambda o, f: [],
        )
        monkeypatch.setattr(onnx_mod, "resolve_onnx_path", lambda *a, **k: None)
        monkeypatch.setattr(onnx_mod, "checkpoint_to_version", lambda t, v: v or "")
        backend = _make_backend(tmp_path)
        assert backend.load_model({"model_type": "rife", "model_version": ""}) is False
