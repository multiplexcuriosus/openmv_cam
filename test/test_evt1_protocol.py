import struct

import numpy as np
import pytest

from openmv_cam.evt1_protocol import (
    EventPacket, parse_evt1_header, reconstruct_timestamps_us,
)


def test_evt1_timestamp_reconstruction_and_little_endian_decode():
    events = np.array([[1, 2, 3, 4, 100, 200], [0, 1, 999, 999, 4, 5]], dtype="<u2")
    assert reconstruct_timestamps_us(events).tolist() == [2_003_004, 1_999_999]
    packet = EventPacket.decode(events.tobytes(), event_count=2,
                                payload_length=24, packet_id=7,
                                packet_ros_stamp_ns=123, packet_monotonic_stamp_ns=456)
    assert packet.events.dtype == np.dtype("uint16")
    assert packet.timestamps_us.tolist() == [2_003_004, 1_999_999]
    assert packet.first_event_timestamp_us == 1_999_999
    assert packet.wire_format == "processed_evt1"
    assert packet.wire_payload_length == 24


def test_evt1_header_behavior_is_unchanged():
    assert parse_evt1_header(struct.pack("<LL", 2, 24)) == (2, 24)
    with pytest.raises(ValueError):
        parse_evt1_header(struct.pack("<LL", 2, 12))


def test_evt1_rejects_inconsistent_length():
    with pytest.raises(ValueError):
        EventPacket.decode(b"\0" * 12, event_count=2, payload_length=12,
                           packet_id=1, packet_ros_stamp_ns=2,
                           packet_monotonic_stamp_ns=3)
