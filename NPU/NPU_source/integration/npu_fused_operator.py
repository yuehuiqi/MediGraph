from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np


def additive_mask(mask) -> np.ndarray:
    values = np.asarray(mask)

    if values.shape != (32,):
        raise ValueError("label mask must have shape [32]")

    if values.dtype == np.bool_:
        return np.ascontiguousarray(
            np.where(values, 0.0, -10000.0),
            dtype=np.float32,
        )

    values = np.asarray(values, dtype=np.float32)

    if np.all((values == 0) | (values == 1)):
        values = np.where(values != 0, 0.0, -10000.0)

    return np.ascontiguousarray(values, dtype=np.float32)


class AscendCFusedMedicalSoftmax:
    def __init__(self, library=None, block_dim=40):
        default = (
            Path(__file__).resolve().parents[1]
            / "ascendc/build/libmedical_npu_runtime.so"
        )
        path = Path(
            library
            or os.environ.get("MEDICAL_NPU_LIBRARY", default)
        )

        self.block_dim = block_dim
        self.lib = ctypes.CDLL(
            str(path), mode=ctypes.RTLD_GLOBAL
        )

        float_pointer = ctypes.POINTER(ctypes.c_float)

        self.lib.medical_npu_fused_init.restype = ctypes.c_int

        self.lib.medical_npu_fused_run_f32.argtypes = [
            float_pointer,
            float_pointer,
            float_pointer,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_float,
        ]
        self.lib.medical_npu_fused_run_f32.restype = ctypes.c_int

        self.lib.medical_npu_fused_run_device_async_f32.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_float,
        ]
        self.lib.medical_npu_fused_run_device_async_f32.restype = (
            ctypes.c_int
        )

        self.lib.medical_npu_fused_synchronize.restype = ctypes.c_int

        status = self.lib.medical_npu_fused_init()
        if status != 0:
            raise RuntimeError(f"fused init failed: {status}")

    def softmax(self, scores, mask, temperature=1.35):
        source = np.ascontiguousarray(scores, dtype=np.float32)
        if source.ndim == 1:
            source = source[None, :]

        if source.ndim != 2 or source.shape[1] != 32:
            raise ValueError("scores must have shape [N,32]")

        mask = additive_mask(mask)
        output = np.empty_like(source)

        pointer = ctypes.POINTER(ctypes.c_float)

        status = self.lib.medical_npu_fused_run_f32(
            source.ctypes.data_as(pointer),
            mask.ctypes.data_as(pointer),
            output.ctypes.data_as(pointer),
            source.shape[0],
            32,
            self.block_dim,
            temperature,
        )
        if status != 0:
            raise RuntimeError(f"fused run failed: {status}")

        return output

    def enqueue_device(
        self,
        scores,
        mask,
        output,
        temperature=1.35,
    ):
        if scores.shape[-1] != 32:
            raise ValueError("scores must have shape [N,32]")
        if tuple(mask.shape) != (32,):
            raise ValueError("mask must have shape [32]")
        if not scores.is_contiguous():
            raise ValueError("scores must be contiguous")
        if not mask.is_contiguous() or not output.is_contiguous():
            raise ValueError("mask/output must be contiguous")

        status = (
            self.lib.medical_npu_fused_run_device_async_f32(
                scores.data_ptr(),
                mask.data_ptr(),
                output.data_ptr(),
                scores.shape[0],
                32,
                self.block_dim,
                temperature,
            )
        )
        if status != 0:
            raise RuntimeError(f"fused enqueue failed: {status}")

    def synchronize(self):
        status = self.lib.medical_npu_fused_synchronize()
        if status != 0:
            raise RuntimeError(f"fused synchronize failed: {status}")


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    scores = rng.normal(size=(257, 32)).astype(np.float32)

    valid = np.ones(32, dtype=bool)
    valid[::7] = False
    mask = additive_mask(valid)

    operator = AscendCFusedMedicalSoftmax()
    actual = operator.softmax(scores, valid, temperature=1.35)

    calibrated = scores / 1.35 + mask
    calibrated -= calibrated.max(axis=1, keepdims=True)
    expected = np.exp(calibrated)
    expected /= expected.sum(axis=1, keepdims=True)

    print("shape:", actual.shape)
    print("max_abs_diff:",
          np.max(np.abs(actual - expected)))
    print("row_sum_error:",
          np.max(np.abs(actual.sum(axis=1) - 1)))
    print("masked_probability_max:",
          np.max(actual[:, ~valid]))

    assert np.max(np.abs(actual - expected)) <= 1e-5
    assert np.max(actual[:, ~valid]) <= 1e-6
