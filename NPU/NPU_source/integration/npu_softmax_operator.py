from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np


def _cpu_softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


class AscendCMedicalSoftmax:
    """Persistent ctypes wrapper around the custom AscendC runtime."""

    def __init__(self, library: str | None = None, block_dim: int = 40):
        default = (
            Path(__file__).resolve().parents[1]
            / "ascendc" / "build"
            / "libmedical_npu_runtime.so"
        )
        path = Path(
            library
            or os.environ.get("MEDICAL_NPU_LIBRARY", str(default))
        )

        if not path.exists():
            raise FileNotFoundError(path)

        self.library_path = path
        self.block_dim = block_dim
        self.lib = ctypes.CDLL(
            str(path), mode=ctypes.RTLD_GLOBAL
        )

        self.lib.medical_npu_init.restype = ctypes.c_int
        self.lib.medical_npu_run_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        self.lib.medical_npu_run_f32.restype = ctypes.c_int

        status = self.lib.medical_npu_init()
        if status != 0:
            raise RuntimeError(
                f"medical_npu_init failed: {status}"
            )

    def softmax(self, scores: np.ndarray) -> np.ndarray:
        source = np.ascontiguousarray(
            scores, dtype=np.float32
        )
        if source.ndim == 1:
            source = source[None, :]
        if source.ndim != 2 or source.shape[1] != 32:
            raise ValueError(
                "AscendC optimized path requires shape [N,32]"
            )

        output = np.empty_like(source)

        status = self.lib.medical_npu_run_f32(
            source.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)
            ),
            output.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)
            ),
            source.shape[0],
            source.shape[1],
            self.block_dim,
        )
        if status != 0:
            raise RuntimeError(
                f"medical_npu_run_f32 failed: {status}"
            )

        return output


class NpuSoftmaxOperator:
    """Unified medical entity-score normalization operator."""

    def __init__(
        self,
        backend: str = "auto",
        block_dim: int = 40,
    ):
        self.requested_backend = backend
        self.backend = "cpu"
        self.ascendc = None
        self.torch = None

        if backend in {"auto", "ascendc"}:
            try:
                self.ascendc = AscendCMedicalSoftmax(
                    block_dim=block_dim
                )
                self.backend = "ascendc"
                return
            except Exception:
                if backend == "ascendc":
                    raise

        if backend in {"auto", "torch_npu"}:
            try:
                import torch
                import torch_npu  # noqa: F401
                if torch.npu.is_available():
                    self.torch = torch
                    self.backend = "torch_npu"
                    return
            except Exception:
                if backend == "torch_npu":
                    raise

        if backend not in {"auto", "cpu", "ascendc", "torch_npu"}:
            raise ValueError(f"unknown backend: {backend}")

    def run(self, inputs: dict, **kwargs) -> dict:
        scores = np.asarray(
            inputs["scores"], dtype=np.float32
        )
        if scores.ndim == 1:
            scores = scores[None, :]

        if self.backend == "ascendc":
            probabilities = self.ascendc.softmax(scores)
        elif self.backend == "torch_npu":
            tensor = self.torch.from_numpy(
                np.ascontiguousarray(scores)
            ).to("npu:0")
            probabilities = (
                self.torch.softmax(tensor, dim=-1)
                .cpu().numpy()
            )
        else:
            probabilities = _cpu_softmax(scores)

        return {
            "probs": probabilities.tolist(),
            "backend": self.backend,
        }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    scores = rng.normal(size=(257, 32)).astype(np.float32)

    operator = NpuSoftmaxOperator(backend="ascendc")
    result = operator.run({"scores": scores})

    actual = np.asarray(result["probs"], dtype=np.float32)
    expected = _cpu_softmax(scores)
    maximum_error = np.max(np.abs(actual - expected))

    print("backend:", result["backend"])
    print("shape:", actual.shape)
    print("max_abs_diff:", maximum_error)
    print("row_sum_error:",
          np.max(np.abs(actual.sum(axis=1) - 1.0)))

    assert maximum_error <= 1e-5
    assert np.all(np.isfinite(actual))
