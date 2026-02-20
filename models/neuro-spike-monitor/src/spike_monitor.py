# SPDX-FileCopyrightText: 2025-present Demi <bjaiye1@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Spike monitor: collect spike events and produce raster plot visualizations."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING
import base64

if TYPE_CHECKING:  # pragma: no cover - typing only
    from biosim.visuals import VisualSpec

from biosim import BioModule
from biosim.signals import BioSignal, SignalMetadata


def _escape_svg_text(s: str) -> str:
    """Escape text for safe SVG embedding."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


class SpikeMonitor(BioModule):
    """Collect spike events and produce a raster plot visualization.

    Receives `spikes` signals and stores events for visualization.
    Produces an SVG raster plot as an `image` VisualSpec.

    Parameters:
        max_events: Maximum number of spike events to store (oldest dropped).
        max_neurons: Maximum neuron index to display in raster (for limiting plot size).
        width: SVG width in pixels.
        height: SVG height in pixels.
    """

    def __init__(
        self,
        max_events: int = 10000,
        max_neurons: Optional[int] = None,
        width: int = 600,
        height: int = 300,
        min_dt: float = 0.001,
    ) -> None:
        self.min_dt = min_dt
        self.max_events = max_events
        self.max_neurons = max_neurons
        self.width = width
        self.height = height

        self._events: List[tuple] = []  # [(t, neuron_id), ...]
        self._t_min: float = float("inf")
        self._t_max: float = float("-inf")
        self._neuron_max: int = 0
        self._outputs: Dict[str, BioSignal] = {}

    def inputs(self) -> Set[str]:
        return {"spikes"}

    def outputs(self) -> Set[str]:
        return {"spike_summary"}

    def reset(self) -> None:
        """Reset collected data for a new simulation run."""
        self._events = []
        self._t_min = float("inf")
        self._t_max = float("-inf")
        self._neuron_max = 0
        self._outputs = {}

    def set_inputs(self, signals: Dict[str, BioSignal]) -> None:
        signal = signals.get("spikes")
        if signal is None:
            return
        t = float(signal.time)
        ids = signal.value or []

        self._t_min = min(self._t_min, t)
        self._t_max = max(self._t_max, t)

        for nid in ids:
            nid = int(nid)
            self._neuron_max = max(self._neuron_max, nid)
            self._events.append((t, nid))

        # Trim if over limit
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events:]
            # Recompute t_min from remaining events
            if self._events:
                self._t_min = min(e[0] for e in self._events)

    def advance_to(self, t: float) -> None:
        # Emit a bounded summary signal so downstream modules/platform can persist
        # metrics even if they choose to ignore visualization specs.
        recent = self._events[-min(200, len(self._events)):] if self._events else []
        source = getattr(self, "_world_name", self.__class__.__name__)
        self._outputs = {
            "spike_summary": BioSignal(
                source=source,
                name="spike_summary",
                value={
                    "t": t,
                    "n_events": len(self._events),
                    "t_min": None if self._t_min == float("inf") else self._t_min,
                    "t_max": None if self._t_max == float("-inf") else self._t_max,
                    "neuron_max": self._neuron_max,
                    "recent_events": [[float(et), int(nid)] for et, nid in recent],
                },
                time=t,
                metadata=SignalMetadata(description="Spike monitor summary", kind="state"),
            )
        }

    def get_outputs(self) -> Dict[str, BioSignal]:
        return dict(self._outputs)

    def visualize(self) -> Optional["VisualSpec"]:
        """Generate an SVG raster plot of collected spikes."""
        description = (
            "Spike raster plot showing neural activity over time. "
            "Each row represents a neuron, and each vertical tick marks a spike event. "
            "Dense regions indicate high synchronous activity across the population."
        )

        if not self._events:
            # Return empty placeholder
            empty_svg = self._generate_empty_svg()
            empty_b64 = base64.b64encode(empty_svg.encode("utf-8")).decode("ascii")
            return {
                "render": "image",
                "data": {
                    "src": f"data:image/svg+xml;base64,{empty_b64}",
                    "alt": "Raster plot (no spikes)",
                    "width": self.width,
                    "height": self.height,
                },
                "description": description,
            }

        svg = self._generate_raster_svg()
        # Use data URI with base64 encoding for reliable rendering
        svg_b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        data_uri = f"data:image/svg+xml;base64,{svg_b64}"

        return {
            "render": "image",
            "data": {
                "src": data_uri,
                "alt": f"Raster plot ({len(self._events)} spikes)",
                "width": self.width,
                "height": self.height,
            },
            "description": description,
        }

    def _generate_empty_svg(self) -> str:
        """Generate an empty placeholder SVG."""
        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">
  <rect width="100%" height="100%" fill="#f8f8f8"/>
  <text x="50%" y="50%" text-anchor="middle" fill="#999" font-family="sans-serif" font-size="14">No spikes recorded</text>
</svg>"""

    def _generate_raster_svg(self) -> str:
        """Generate an SVG raster plot from collected spike events."""
        w = self.width
        h = self.height
        margin = {"top": 20, "right": 20, "bottom": 40, "left": 50}
        plot_w = w - margin["left"] - margin["right"]
        plot_h = h - margin["top"] - margin["bottom"]

        # Determine ranges
        t_min = self._t_min if self._t_min != float("inf") else 0.0
        t_max = self._t_max if self._t_max != float("-inf") else 1.0
        t_range = t_max - t_min if t_max > t_min else 1.0

        n_max = self.max_neurons if self.max_neurons is not None else self._neuron_max
        n_max = max(n_max, 1)

        # Scale functions
        def x_scale(t: float) -> float:
            return margin["left"] + ((t - t_min) / t_range) * plot_w

        def y_scale(nid: int) -> float:
            return margin["top"] + (1 - nid / n_max) * plot_h

        # Build SVG
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
            '  <style>',
            '    .axis { stroke: #333; stroke-width: 1; }',
            '    .tick { stroke: #333; stroke-width: 1; }',
            '    .label { font-family: sans-serif; font-size: 10px; fill: #333; }',
            '    .title { font-family: sans-serif; font-size: 12px; fill: #333; font-weight: bold; }',
            '    .spike { stroke: #2563eb; stroke-width: 1; }',
            '  </style>',
            f'  <rect width="{w}" height="{h}" fill="white"/>',
        ]

        # Draw axes
        x0 = margin["left"]
        x1 = margin["left"] + plot_w
        y0 = margin["top"]
        y1 = margin["top"] + plot_h

        lines.append(f'  <line class="axis" x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}"/>')  # x-axis
        lines.append(f'  <line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}"/>')  # y-axis

        # X-axis ticks and labels
        n_xticks = 5
        for i in range(n_xticks + 1):
            t_val = t_min + (i / n_xticks) * t_range
            x_pos = x_scale(t_val)
            lines.append(f'  <line class="tick" x1="{x_pos}" y1="{y1}" x2="{x_pos}" y2="{y1 + 5}"/>')
            lines.append(f'  <text class="label" x="{x_pos}" y="{y1 + 18}" text-anchor="middle">{t_val:.2f}</text>')

        # Y-axis ticks and labels
        n_yticks = min(5, n_max)
        for i in range(n_yticks + 1):
            n_val = int((i / n_yticks) * n_max)
            y_pos = y_scale(n_val)
            lines.append(f'  <line class="tick" x1="{x0 - 5}" y1="{y_pos}" x2="{x0}" y2="{y_pos}"/>')
            lines.append(f'  <text class="label" x="{x0 - 8}" y="{y_pos + 3}" text-anchor="end">{n_val}</text>')

        # Axis labels
        lines.append(f'  <text class="label" x="{x0 + plot_w / 2}" y="{h - 5}" text-anchor="middle">Time (s)</text>')
        lines.append(f'  <text class="label" x="12" y="{y0 + plot_h / 2}" text-anchor="middle" transform="rotate(-90, 12, {y0 + plot_h / 2})">Neuron</text>')

        # Draw spikes as short vertical lines
        for t, nid in self._events:
            if self.max_neurons is not None and nid > self.max_neurons:
                continue
            x_pos = x_scale(t)
            y_pos = y_scale(nid)
            # Small vertical tick for each spike
            lines.append(f'  <line class="spike" x1="{x_pos}" y1="{y_pos - 1}" x2="{x_pos}" y2="{y_pos + 1}"/>')

        lines.append('</svg>')
        return "\n".join(lines)
