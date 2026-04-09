# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Reference ONNX-backed classifier for hybrid biosim spaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional

from src.compat_onnx import OnnxClassifierModule


class OnnxStateClassifier(OnnxClassifierModule):
    """Thin repo-specific wrapper around the shared biosim ONNX helper."""

    def __init__(
        self,
        model_path: str = "artifacts/linear_state_classifier.onnx",
        class_labels: Optional[List[str]] = None,
        min_dt: float = 0.001,
        session_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        super().__init__(
            model_path=model_path,
            class_labels=class_labels or ["quiescent", "subthreshold", "spiking"],
            input_port="state_vector",
            probabilities_port="state_probabilities",
            predicted_port="predicted_state",
            model_input_name="state_vector",
            model_output_name="state_probabilities",
            base_dir=str(Path(__file__).resolve().parents[1]),
            input_vector_length=4,
            min_dt=min_dt,
            session_factory=session_factory,
            probabilities_description="Classifier probabilities over the declared ONNX class labels",
            predicted_description="Most likely ONNX-predicted neural state",
        )
