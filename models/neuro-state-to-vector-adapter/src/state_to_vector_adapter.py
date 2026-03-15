# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Convert mechanistic state payloads into a compact feature vector."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

from biosim import BioModule
from biosim.signals import BioSignal, SignalMetadata


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class StateToVectorAdapter(BioModule):
    """Project a structured state payload into a fixed numeric vector.

    The default feature order is:
    - mean membrane potential
    - mean sodium activation gate (m)
    - mean sodium inactivation gate (h)
    - mean potassium activation gate (n_gate)
    """

    def __init__(
        self,
        feature_order: Optional[List[str]] = None,
        min_dt: float = 0.001,
    ) -> None:
        self.min_dt = min_dt
        self.feature_order = feature_order or ["v", "m", "h", "n_gate"]
        self._latest_vector: List[float] = [0.0] * len(self.feature_order)
        self._latest_time: float = 0.0
        self._outputs: Dict[str, BioSignal] = {}

    def inputs(self) -> Set[str]:
        return {"state"}

    def outputs(self) -> Set[str]:
        return {"state_vector"}

    def reset(self) -> None:
        self._latest_vector = [0.0] * len(self.feature_order)
        self._latest_time = 0.0
        self._outputs = {}

    def _coerce_series(self, payload: Dict[str, Any], key: str) -> List[float]:
        raw = payload.get(key, [])
        if isinstance(raw, (int, float)):
            return [float(raw)]
        if not isinstance(raw, list):
            return []
        return [float(item) for item in raw if isinstance(item, (int, float))]

    def set_inputs(self, signals: Dict[str, BioSignal]) -> None:
        signal = signals.get("state")
        if signal is None or not isinstance(signal.value, dict):
            return

        payload = signal.value
        vector: List[float] = []
        for key in self.feature_order:
            vector.append(_mean(self._coerce_series(payload, key)))

        self._latest_vector = vector
        self._latest_time = float(payload.get("t", signal.time))

    def advance_to(self, t: float) -> None:
        source = getattr(self, "_world_name", self.__class__.__name__)
        self._outputs = {
            "state_vector": BioSignal(
                source=source,
                name="state_vector",
                value=list(self._latest_vector),
                time=t,
                metadata=SignalMetadata(
                    description="Fixed feature vector derived from mechanistic neuron state",
                    dtype="float32",
                    shape=(len(self._latest_vector),),
                    kind="state",
                ),
            )
        }

    def get_outputs(self) -> Dict[str, BioSignal]:
        return dict(self._outputs)

    def visualize(self) -> Optional[Dict[str, Any]]:
        if not self._outputs:
            return None

        return {
            "render": "table",
            "data": {
                "columns": ["feature", "value"],
                "rows": [
                    [feature, round(value, 6)]
                    for feature, value in zip(self.feature_order, self._latest_vector)
                ],
            },
            "description": "Feature vector presented to the ONNX classifier.",
        }
