"""Validated EVT1 packet decoding and sensor timestamp reconstruction."""

from dataclasses import dataclass

import numpy as np


EVENT_COLUMNS = 6
EVENT_BYTES = EVENT_COLUMNS * 2
MAX_EVENT_COUNT = 8192


def reconstruct_timestamps_us(events: np.ndarray) -> np.ndarray:
    """Return GENX320 timestamps for EVT1 ``type,sec,ms,us,x,y`` rows."""
    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] != EVENT_COLUMNS:
        raise ValueError(f"events must have shape (N, 6), got {events.shape}")
    return (
        events[:, 1].astype(np.int64) * 1_000_000
        + events[:, 2].astype(np.int64) * 1_000
        + events[:, 3].astype(np.int64)
    )


@dataclass(frozen=True)
class EventPacket:
    """One validated EVT1 payload with packet timestamp bounds."""

    events: np.ndarray
    packet_id: int
    event_count: int
    payload_length: int
    packet_ros_stamp_ns: int
    packet_monotonic_stamp_ns: int
    timestamps_us: np.ndarray
    first_event_timestamp_us: int
    last_event_timestamp_us: int

    @classmethod
    def decode(
        cls, payload: bytes, *, event_count: int, payload_length: int,
        packet_id: int, packet_ros_stamp_ns: int,
        packet_monotonic_stamp_ns: int,
    ) -> "EventPacket":
        if event_count < 0 or event_count > MAX_EVENT_COUNT:
            raise ValueError(
                f"event_count must be in [0, {MAX_EVENT_COUNT}], got {event_count}")
        expected = int(event_count) * EVENT_BYTES
        if payload_length != expected or len(payload) != expected:
            raise ValueError(
                f"invalid EVT1 payload length: header={payload_length}, "
                f"actual={len(payload)}, expected={expected}"
            )
        events = np.frombuffer(payload, dtype="<u2").reshape(event_count, 6).copy()
        timestamps = reconstruct_timestamps_us(events)
        if timestamps.size:
            # Preserve wire order for legacy preview/HDF5 behavior. Consumers use
            # explicit timestamps, while these bounds safely handle local disorder.
            first, last = int(timestamps.min()), int(timestamps.max())
        else:
            first = last = -1
        return cls(
            events=events, packet_id=int(packet_id), event_count=int(event_count),
            payload_length=int(payload_length),
            packet_ros_stamp_ns=int(packet_ros_stamp_ns),
            packet_monotonic_stamp_ns=int(packet_monotonic_stamp_ns),
            timestamps_us=timestamps, first_event_timestamp_us=first,
            last_event_timestamp_us=last,
        )
