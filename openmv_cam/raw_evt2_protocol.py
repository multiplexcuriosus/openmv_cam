"""Dependency-free GenX320 EVT2.0 decoding and EVR1 framing helpers."""

import struct


EVR1_MAGIC = b"EVR1"
EVR1_HEADER_FORMAT = "<LL"
EVR1_HEADER_LENGTH = struct.calcsize(EVR1_HEADER_FORMAT)
# Match the existing 8192-event hardware buffer while allowing control words.
MAX_RAW_PAYLOAD_SIZE = 8192 * 4

CD_TYPES = (0x0, 0x1)
TIME_HIGH_TYPE = 0x8
TRIGGER_TYPE = 0xA


def parse_evr1_header(header: bytes) -> tuple[int, int]:
    """Validate and return ``sequence, payload_length`` from an EVR1 header."""
    if len(header) != EVR1_HEADER_LENGTH:
        raise ValueError(f"EVR1 header must be {EVR1_HEADER_LENGTH} bytes")
    sequence, payload_length = struct.unpack(EVR1_HEADER_FORMAT, header)
    if payload_length > MAX_RAW_PAYLOAD_SIZE:
        raise ValueError(
            f"EVR1 payload exceeds {MAX_RAW_PAYLOAD_SIZE} bytes: {payload_length}")
    if payload_length % 4:
        raise ValueError("EVR1 payload length must be divisible by four")
    return sequence, payload_length


def sequence_gap(previous: int | None, current: int) -> int:
    """Return the number of missing uint32 sequence values, including wrap."""
    if previous is None:
        return 0
    return ((int(current) - int(previous)) & 0xFFFFFFFF) - 1


def decode_evt20(payload, time_high: int = 0):
    """
    Decode little-endian EVT2.0 words into six-column event rows.

    TIME_HIGH state is returned for use with the next payload. Trigger and
    unknown control words are ignored and never emitted as CD events.
    """
    try:
        raw = bytes(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("EVT2.0 payload must be bytes-like") from error
    if len(raw) % 4:
        raise ValueError("EVT2.0 payload length must be divisible by four")
    if not isinstance(time_high, int) or time_high < 0:
        raise ValueError("time_high must be a non-negative integer")

    events = []
    current_high = time_high
    for (word,) in struct.iter_unpack("<L", raw):
        event_type = word >> 28
        if event_type == TIME_HIGH_TYPE:
            current_high = (word & 0x0FFFFFFF) << 6
        elif event_type in CD_TYPES:
            timestamp_us = current_high + ((word >> 22) & 0x3F)
            seconds, remainder = divmod(timestamp_us, 1_000_000)
            milliseconds, microseconds = divmod(remainder, 1_000)
            if seconds > 0xFFFF:
                raise ValueError("EVT2.0 timestamp seconds exceed uint16")
            x = (word >> 11) & 0x7FF
            y = word & 0x7FF
            events.append(
                [event_type, seconds, milliseconds, microseconds, x, y])
        # TRIGGER and all other control/reserved words are consistently ignored.
    return events, current_high
