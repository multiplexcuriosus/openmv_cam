import struct

import numpy as np
import pytest

from openmv_cam.evt1_protocol import EventPacket
from openmv_cam.raw_evt2_protocol import (
    MAX_RAW_PAYLOAD_SIZE,
    decode_evt20,
    parse_evr1_header,
    sequence_gap,
)


def word(event_type, body=0):
    return (event_type << 28) | body


def cd(polarity, timestamp_low, x, y):
    return word(polarity, (timestamp_low << 22) | (x << 11) | y)


def payload(*words):
    return struct.pack("<" + "L" * len(words), *words)


def test_time_high_followed_by_cd_events():
    events, state = decode_evt20(payload(
        word(0x8, 20), cd(1, 3, 100, 200), cd(0, 4, 5, 6)))
    assert state == 20 << 6
    assert events == [
        [1, 0, 1, 283, 100, 200],
        [0, 0, 1, 284, 5, 6],
    ]


def test_time_high_state_is_carried_across_payloads():
    _, state = decode_evt20(payload(word(0x8, 31)))
    events, next_state = decode_evt20(payload(cd(1, 7, 9, 10)), state)
    assert next_state == state
    assert events[0][1:4] == [0, 1, 991]


def test_evt20_is_little_endian():
    event_word = cd(1, 12, 0x123, 0x45)
    events, _ = decode_evt20(event_word.to_bytes(4, "little"))
    assert events == [[1, 0, 0, 12, 0x123, 0x45]]


def test_control_words_are_not_emitted():
    events, _ = decode_evt20(payload(
        word(0xA, 123), word(0xF, 456), cd(0, 1, 2, 3)))
    assert events == [[0, 0, 0, 1, 2, 3]]


def test_malformed_payload_length_is_rejected():
    with pytest.raises(ValueError, match="divisible by four"):
        decode_evt20(b"abc")


def test_evr1_header_validation():
    assert parse_evr1_header(struct.pack("<LL", 17, 8)) == (17, 8)
    with pytest.raises(ValueError, match="divisible by four"):
        parse_evr1_header(struct.pack("<LL", 17, 7))
    with pytest.raises(ValueError, match="exceeds"):
        parse_evr1_header(struct.pack("<LL", 17, MAX_RAW_PAYLOAD_SIZE + 4))


def test_sequence_gap_detection_including_wrap():
    assert sequence_gap(None, 4) == 0
    assert sequence_gap(4, 5) == 0
    assert sequence_gap(4, 7) == 2
    assert sequence_gap(0xFFFFFFFF, 0) == 0


def test_event_packet_creation_from_raw_decoded_events():
    decoded, _ = decode_evt20(payload(word(0x8, 100), cd(1, 2, 3, 4)))
    packet = EventPacket.from_decoded_events(
        np.asarray(decoded), packet_id=8, packet_ros_stamp_ns=10,
        packet_monotonic_stamp_ns=11, wire_format="raw_evt20",
        wire_payload_length=8, wire_sequence=123)
    assert packet.events.tolist() == [[1, 0, 6, 402, 3, 4]]
    assert packet.event_count == 1
    assert packet.payload_length == 12
    assert packet.wire_payload_length == 8
    assert packet.wire_sequence == 123
    assert packet.wire_format == "raw_evt20"
