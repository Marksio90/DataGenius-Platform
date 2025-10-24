from __future__ import annotations
import pytest

@pytest.mark.skip(reason="ONNX equivalence requires skl2onnx+onnxruntime and a pipeline — gated.")
def test_onnx_equivalence():
    pass