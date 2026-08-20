"""Validated EVT1 packet decoding and sensor timestamp reconstruction."""

from dataclasses import dataclass

import numpy as np


EVT1_MAGIC = b"EVT1"
EVT1_HEADER_FORMAT = "<LL"
EVT1_HEADER_LENGTH = 8
EVENT_COLUMNS = 6
EVENT_BYTES = EVENT_COLUMNS * 2
MAX_EVENT_COUNT = 8192
MAX_EVT1_PAYLOAD_SIZE = MAX_EVENT_COUNT * EVENT_BYTES


def parse_evt1_header(header: bytes) -> tuple[int, int]:
    """Validate and return ``event_count, payload_length`` from an EVT1 header."""
    import struct

    if len(header) != EVT1_HEADER_LENGTH:
        raise ValueError(f"EVT1 header must be {EVT1_HEADER_LENGTH} bytes")
    event_count, payload_length = struct.unpack(EVT1_HEADER_FORMAT, header)
    if event_count > MAX_EVENT_COUNT:
        raise ValueError(
            f"event_count must be in [0, {MAX_EVENT_COUNT}], got {event_count}")
    expected = event_count * EVENT_BYTES
    if payload_length != expected:
        raise ValueError(
            f"invalid EVT1 payload length: got {payload_length}, expected {expected}")
    return event_count, payload_length


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
    """One decoded event packet with packet timestamp bounds."""

    events: np.ndarray
    packet_id: int
    event_count: int
    payload_length: int
    packet_ros_stamp_ns: int
    packet_monotonic_stamp_ns: int
    timestamps_us: np.ndarray
    first_event_timestamp_us: int
    last_event_timestamp_us: int
    source: str = "hardware"
    original_ros_stamp_ns: int = -1
    original_monotonic_stamp_ns: int = -1
    wire_format: str = "EVT1"
    wire_payload_length: int = 0
    wire_sequence: int = -1

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
            last_event_timestamp_us=last, wire_format="processed_evt1",
            wire_payload_length=int(payload_length),
        )

    @classmethod
    def from_decoded_events(
        cls, events: np.ndarray, *, packet_id: int, packet_ros_stamp_ns: int,
        packet_monotonic_stamp_ns: int, wire_format: str = "EVT1",
        wire_payload_length: int = 0, wire_sequence: int = -1,
        source: str = "hardware",
    ) -> "EventPacket":
        """Build a packet from decoded ``type,sec,ms,us,x,y`` event rows."""
        rows = np.asarray(events)
        if rows.ndim != 2 or rows.shape[1] != EVENT_COLUMNS:
            raise ValueError(f"events must have shape (N, 6), got {rows.shape}")
        count = int(rows.shape[0])
        if count > MAX_EVENT_COUNT:
            raise ValueError(
                f"event_count must be in [0, {MAX_EVENT_COUNT}], got {count}")
        if not np.issubdtype(rows.dtype, np.integer):
            raise ValueError("decoded events must use an integer dtype")
        if np.any(rows < 0) or np.any(rows > np.iinfo(np.uint16).max):
            raise ValueError("decoded event values must fit in uint16")
        rows = rows.astype(np.uint16, copy=True)
        timestamps = reconstruct_timestamps_us(rows)
        first = int(timestamps.min()) if count else -1
        last = int(timestamps.max()) if count else -1
        return cls(
            events=rows, packet_id=int(packet_id), event_count=count,
            payload_length=count * EVENT_BYTES,
            packet_ros_stamp_ns=int(packet_ros_stamp_ns),
            packet_monotonic_stamp_ns=int(packet_monotonic_stamp_ns),
            timestamps_us=timestamps, first_event_timestamp_us=first,
            last_event_timestamp_us=last, source=source,
            wire_format=str(wire_format),
            wire_payload_length=int(wire_payload_length),
            wire_sequence=int(wire_sequence),
        )

    @classmethod
    def from_recorded_arrays(
        cls, *, event_type: np.ndarray, event_x: np.ndarray,
        event_y: np.ndarray, timestamps_us: np.ndarray, packet_id: int,
        packet_ros_stamp_ns: int, packet_monotonic_stamp_ns: int,
        original_ros_stamp_ns: int, original_monotonic_stamp_ns: int,
    ) -> "EventPacket":
        """Reconstruct the live EVT1 row representation without changing data."""
        event_type = np.asarray(event_type)
        event_x = np.asarray(event_x)
        event_y = np.asarray(event_y)
        timestamps = np.asarray(timestamps_us, dtype=np.int64)
        count = int(timestamps.size)
        if not (event_type.size == event_x.size == event_y.size == count):
            raise ValueError("recorded event columns have different lengths")
        if count > MAX_EVENT_COUNT:
            raise ValueError(
                f"event_count must be in [0, {MAX_EVENT_COUNT}], got {count}")
        if np.any(timestamps < 0):
            raise ValueError("recorded sensor timestamps must be non-negative")

        events = np.empty((count, EVENT_COLUMNS), dtype=np.uint16)
        events[:, 0] = event_type
        seconds, remainder = np.divmod(timestamps, 1_000_000)
        milliseconds, microseconds = np.divmod(remainder, 1_000)
        if np.any(seconds > np.iinfo(np.uint16).max):
            raise ValueError("recorded sensor timestamp seconds exceed EVT1 uint16")
        events[:, 1] = seconds
        events[:, 2] = milliseconds
        events[:, 3] = microseconds
        events[:, 4] = event_x
        events[:, 5] = event_y
        first = int(timestamps.min()) if count else -1
        last = int(timestamps.max()) if count else -1
        return cls(
            events=events, packet_id=int(packet_id), event_count=count,
            payload_length=count * EVENT_BYTES,
            packet_ros_stamp_ns=int(packet_ros_stamp_ns),
            packet_monotonic_stamp_ns=int(packet_monotonic_stamp_ns),
            timestamps_us=timestamps.copy(), first_event_timestamp_us=first,
            last_event_timestamp_us=last, source="hdf5_replay",
            original_ros_stamp_ns=int(original_ros_stamp_ns),
            original_monotonic_stamp_ns=int(original_monotonic_stamp_ns),
            wire_format="processed_evt1",
            wire_payload_length=count * EVENT_BYTES,
        )
