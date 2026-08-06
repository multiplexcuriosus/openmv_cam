"""Lightweight rolling performance diagnostics for event publication."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DurationSamples:
    """Bounded monotonic-duration samples in milliseconds."""

    capacity: int = 300
    values: deque = field(default_factory=deque)

    def add_seconds(self, duration_s: float) -> None:
        """Add a nonnegative duration measured with a monotonic clock."""
        self.values.append(max(0.0, float(duration_s)) * 1_000.0)
        while len(self.values) > self.capacity:
            self.values.popleft()

    def summary_ms(self) -> tuple[float, float, float]:
        """Return p50, p95, and maximum milliseconds."""
        if not self.values:
            return 0.0, 0.0, 0.0
        values = np.asarray(self.values, dtype=np.float64)
        return (
            float(np.percentile(values, 50)),
            float(np.percentile(values, 95)),
            float(np.max(values)),
        )


@dataclass
class EventDiagnosticCounters:
    """Counters and rolling timings for the native activity publisher."""

    timer_ticks: int = 0
    messages_published: int = 0
    skipped_no_new_events: int = 0
    anchors_advanced: int = 0
    anchors_repeated: int = 0
    packets_dropped_by_cap: int = 0
    events_dropped_by_cap: int = 0
    buffer_processing: DurationSamples = field(default_factory=DurationSamples)
    voxel_build: DurationSamples = field(default_factory=DurationSamples)
    message_build: DurationSamples = field(default_factory=DurationSamples)
    publish_call: DurationSamples = field(default_factory=DurationSamples)
    timer_interval: DurationSamples = field(default_factory=DurationSamples)
    publish_interval: DurationSamples = field(default_factory=DurationSamples)
