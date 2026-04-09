"""Generate a tiny reference ONNX classifier artifact for the hybrid demo."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    out_path = base / "artifacts" / "linear_state_classifier.onnx"

    weights = np.array(
        [
            [-0.02, 0.10, 0.30],
            [0.00, 0.15, -0.05],
            [0.25, -0.10, 0.05],
            [0.10, 0.05, 0.20],
        ],
        dtype=np.float32,
    )
    bias = np.array([0.2, -0.1, 0.0], dtype=np.float32)

    input_info = helper.make_tensor_value_info("state_vector", TensorProto.FLOAT, [1, 4])
    logits_info = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3])
    output_info = helper.make_tensor_value_info("state_probabilities", TensorProto.FLOAT, [1, 3])

    nodes = [
        helper.make_node("MatMul", ["state_vector", "W"], ["matmul_out"]),
        helper.make_node("Add", ["matmul_out", "B"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["state_probabilities"], axis=1),
    ]

    graph = helper.make_graph(
        nodes,
        "linear_state_classifier",
        [input_info],
        [output_info],
        initializer=[
            numpy_helper.from_array(weights, name="W"),
            numpy_helper.from_array(bias, name="B"),
        ],
        value_info=[logits_info],
    )

    model = helper.make_model(graph, producer_name="biosimulant-reference")
    model.opset_import[0].version = 13
    onnx.checker.check_model(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
