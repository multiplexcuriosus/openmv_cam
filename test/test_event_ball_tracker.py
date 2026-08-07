import json

import numpy as np

from openmv_cam.event_ball_tracker import EventBallTracker, trace_detail_json
from openmv_cam.evt1_protocol import EventPacket


def packet(rows, packet_id=1, ros_ns=9_000_000_000):
    array = np.asarray(rows, dtype="<u2").reshape(-1, 6)
    return EventPacket.decode(array.tobytes(), event_count=len(array),
                              payload_length=array.nbytes, packet_id=packet_id,
                              packet_ros_stamp_ns=ros_ns,
                              packet_monotonic_stamp_ns=123456)


def event(t_us, x, y, polarity=1):
    sec, rem = divmod(t_us, 1_000_000)
    ms, us = divmod(rem, 1000)
    return [polarity, sec, ms, us, x, y]


def close_at(tracker, t_us, packet_id=2):
    return tracker.update(packet([event(t_us, 319, 319)], packet_id))


def tracker(**kwargs):
    defaults = dict(min_event_count=1, min_blob_area_px=1,
                    velocity_min_span_ms=1.0)
    defaults.update(kwargs)
    return EventBallTracker(**defaults)


def test_one_bin_activity_map_and_weighted_com():
    subject = tracker()
    activity = subject.build_activity_map(np.asarray([
        event(100, 10, 20), event(200, 10, 20), event(300, 11, 20)], dtype=np.uint16))
    assert activity[20, 10] == 2 and activity.sum() == 3
    subject.update(packet([event(100, 10, 20), event(200, 10, 20), event(300, 11, 20)]))
    result = close_at(subject, 1000)[0]
    assert result.valid
    assert abs(result.x_px - (31 / 3)) < 1e-6
    assert result.y_px == 20


def test_adjacent_bins_do_not_leak_and_empty_bins_are_safe():
    subject = tracker()
    subject.update(packet([event(100, 4, 5), event(1100, 20, 30)]))
    result = close_at(subject, 2000)
    assert [(d.bin_start_us, d.event_count) for d in result] == [(1000, 1)]
    result = close_at(subject, 4000, 3)
    assert any(d.event_count == 0 and not d.valid for d in result)


def test_synthetic_motion_velocity_and_proximity_selection():
    subject = tracker(max_jump_px=30, velocity_history_size=5)
    outputs = []
    for index in range(6):
        t = index * 1000 + 100
        rows = [event(t, 10 + index, 20), event(t, 100, 100)]
        outputs.extend(subject.update(packet(rows, index + 1)))
    assert outputs[-1].x_px < 20  # nearby blob wins despite deterministic tie
    assert outputs[-1].velocity_valid
    assert abs(outputs[-1].vx_px_s - 1000.0) < 100
    assert abs(outputs[-1].vy_px_s) < 1


def test_circularity_filter_is_optional():
    rows = [event(100, x, 10) for x in range(10, 16)]
    default = tracker(use_circularity=False, min_circularity=0.99)
    default.update(packet(rows))
    assert close_at(default, 1000)[0].valid
    filtered = tracker(use_circularity=True, min_circularity=0.99)
    filtered.update(packet(rows))
    assert not close_at(filtered, 1000)[0].valid


def test_late_events_are_ignored_deterministically():
    subject = tracker()
    subject.update(packet([event(100, 10, 10)]))
    close_at(subject, 2000)
    assert subject.update(packet([event(200, 20, 20)], 3)) == []
    assert subject.counters["late_events_or_bins"] == 1


def test_insufficient_velocity_history_and_bounded_statistics():
    subject = tracker(stats_history_size=3, velocity_min_span_ms=3.0)
    subject.update(packet([event(100, 10, 10)]))
    first = close_at(subject, 1000)[0]
    assert first.valid and not first.velocity_valid and first.speed_px_s == 0
    for index in range(10):
        subject.update(packet([event(2000 + index * 1000, 10, 10)], index + 3))
    assert all(len(values) <= 3 for values in subject.timings.values())
    stats = subject.statistics()
    assert "map_build" in stats and len(stats["map_build"]) == 3


def test_latency_detail_json_is_finite_and_keeps_sensor_time_out_of_ros_stamp():
    subject = tracker()
    source = packet([event(123, 10, 10)], ros_ns=8_000_000_321)
    subject.update(source)
    detection = close_at(subject, 1000)[0]
    detail = json.loads(trace_detail_json(detection))
    assert detail["sensor_timestamp_domain"] == "genx320_microseconds"
    assert detail["bin_start_us"] == 0
    assert source.packet_ros_stamp_ns == 8_000_000_321
    assert source.packet_ros_stamp_ns not in (detail["bin_start_us"], detail["bin_end_us"])
    assert all(np.isfinite(value) for value in detail.values()
               if isinstance(value, (int, float)) and not isinstance(value, bool))
