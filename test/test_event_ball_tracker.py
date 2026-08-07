import json

import numpy as np

from openmv_cam.event_ball_tracker import (
    EventBallTracker,
    TrackerDebugSnapshot,
    TrackerDetection,
    render_debug_image,
    render_debug_images,
    trace_detail_json,
)
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


def test_x_crop_uses_only_events_in_half_open_native_corridor():
    subject = tracker(x_crop=(100, 200))
    rows = [event(100, 99, 10), event(100, 100, 20),
            event(100, 199, 30), event(100, 200, 40)]
    activity = subject.build_activity_map(np.asarray(rows, dtype=np.uint16))
    assert activity.sum() == 2
    assert activity[20, 100] == 1
    assert activity[30, 199] == 1
    assert activity[10, 99] == 0
    assert activity[40, 200] == 0


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


def test_debug_snapshot_is_cached_and_rendering_uses_fixed_scale():
    subject = tracker()
    subject.update(packet([event(100, 100, 200)]))
    close_at(subject, 1000)
    cached = subject.latest_debug_snapshot()
    assert cached is not None
    assert cached.activity.shape == (320, 320)
    assert cached.detection.bin_start_us == 0

    activity = np.zeros((320, 320), dtype=np.uint16)
    activity[200, 200] = 1
    activity[201, 200] = 16
    selected = np.array([[[50, 50]], [[55, 50]], [[55, 55]], [[50, 55]]],
                        dtype=np.int32)
    rejected = selected + np.array([[[20, 0]]], dtype=np.int32)
    detection = TrackerDetection(
        10_000, 11_000, 1, 17, x_px=52.0, y_px=52.0,
        vx_px_s=100.0, vy_px_s=0.0, speed_px_s=100.0,
        valid=True, velocity_valid=True, candidate_count=2)
    snapshot = TrackerDebugSnapshot(
        activity, (activity > 0).astype(np.uint8) * 255, detection,
        (selected, rejected), (selected,), selected, (40.0, 40.0),
        ((45.0, 52.0), (52.0, 52.0)), (40, 280))
    image = render_debug_image(snapshot, clip_count=16, rotation_degrees=0)
    assert image.shape == (320, 320, 3)
    assert image.dtype == np.uint8
    assert image[200, 200].tolist() == [16, 16, 16]
    assert image[201, 200].tolist() == [255, 255, 255]
    assert image[50, 50].tolist() == [0, 255, 0]
    assert image[50, 70].tolist() == [0, 128, 255]

    rotated = render_debug_image(snapshot, clip_count=16, rotation_degrees=90)
    # CCW mapping for a 320-square image: (y, x) -> (319-x, y).
    assert rotated[119, 201].tolist() == [255, 255, 255]
    # Native vertical x-crop bounds become horizontal after CCW rotation.
    assert rotated[279, 300].tolist() == [255, 0, 255]

    stages = render_debug_images(snapshot, clip_count=16, rotation_degrees=0)
    assert set(stages) == {"activity", "threshold", "contours", "tracking"}
    assert all(image.shape == (320, 320, 3) for image in stages.values())
    assert stages["contours"][50, 50].tolist() == [0, 0, 255]
    assert stages["contours"][50, 70].tolist() == [0, 128, 255]
    assert stages["activity"][300, 40].tolist() == [255, 0, 255]
    assert stages["threshold"][300, 279].tolist() == [255, 0, 255]
