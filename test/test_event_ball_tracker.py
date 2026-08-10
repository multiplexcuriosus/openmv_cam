import inspect
import json

import cv2
import numpy as np
import pytest

from openmv_cam.event_ball_tracker import (
    EventBallTracker,
    TrackerDebugSnapshot,
    TrackerDetection,
    render_debug_image,
    render_debug_event_frame,
    render_debug_images,
    trace_detail_json,
)
from openmv_cam.evt1_protocol import EventPacket, reconstruct_timestamps_us


def packet(rows, packet_id=1, ros_ns=9_000_000_000):
    array = np.asarray(rows, dtype="<u2").reshape(-1, 6)
    return EventPacket.decode(
        array.tobytes(), event_count=len(array), payload_length=array.nbytes,
        packet_id=packet_id, packet_ros_stamp_ns=ros_ns,
        packet_monotonic_stamp_ns=123456)


def event(t_us, x, y, polarity=1):
    sec, rem = divmod(t_us, 1_000_000)
    ms, us = divmod(rem, 1000)
    return [polarity, sec, ms, us, x, y]


def tracker(**kwargs):
    defaults = dict(
        min_event_count=1, min_blob_area_px=1,
        morphology_operation="none", morphology_kernel=0,
        morphology_iterations=0, velocity_min_span_ms=1.0)
    defaults.update(kwargs)
    return EventBallTracker(**defaults)


def complete_window(subject, rows, packet_id=1):
    """Add rows plus a cropped-out timestamp watermark one bin later."""
    maximum = max(row[1] * 1_000_000 + row[2] * 1000 + row[3]
                  for row in rows)
    watermark = ((maximum // subject.bin_us) + 1) * subject.bin_us
    return subject.update(packet(rows + [event(watermark, 319, 319)], packet_id))


def test_ten_ms_window_combines_several_adjacent_bins():
    subject = tracker(x_crop=(0, 300))
    rows = [event(100, 10, 20), event(1100, 11, 20),
            event(5100, 12, 20), event(9100, 13, 20)]
    result = complete_window(subject, rows)[0]
    assert (result.window_start_us, result.window_end_us) == (0, 10_000)
    assert result.window_event_count == 4
    snapshot = subject.latest_debug_snapshot()
    assert snapshot.activity.sum() == 4
    assert all(snapshot.activity[20, x] == 1 for x in range(10, 14))
    assert snapshot.debug_events is None


def test_debug_event_frame_selects_exact_sensor_time_interval():
    subject = tracker(
        x_crop=(0, 300), debug_event_frame_window_ms=33.0)
    rows = [
        event(8999, 10, 10), event(9000, 11, 10),
        event(10_000, 15, 10),
        event(20_000, 12, 10), event(41_999, 13, 10),
        event(42_000, 14, 10),
    ]
    complete_window(subject, rows)
    snapshot = subject.latest_debug_snapshot()
    assert snapshot.detection.window_end_us == 43_000
    assert reconstruct_timestamps_us(snapshot.debug_events).tolist() == [
        10_000, 20_000, 41_999, 42_000]
    assert 319 not in snapshot.debug_events[:, 4]
    assert not snapshot.debug_events.flags.writeable


@pytest.mark.parametrize("source", ["hardware", "hdf5_replay"])
def test_debug_event_capture_supports_hardware_and_replay_packets(source):
    subject = tracker(
        x_crop=(0, 300), debug_event_frame_window_ms=33.0)
    if source == "hardware":
        source_packet = packet([
            event(10_000, 10, 20), event(11_000, 319, 319)])
    else:
        source_packet = EventPacket.from_recorded_arrays(
            event_type=np.asarray([1, 1]),
            event_x=np.asarray([10, 319]), event_y=np.asarray([20, 319]),
            timestamps_us=np.asarray([10_000, 11_000]), packet_id=1,
            packet_ros_stamp_ns=1, packet_monotonic_stamp_ns=2,
            original_ros_stamp_ns=3, original_monotonic_stamp_ns=4)
    assert subject.update(source_packet)
    snapshot = subject.latest_debug_snapshot()
    assert snapshot.debug_events.shape == (1, 6)
    assert snapshot.debug_events[0, 4:6].tolist() == [10, 20]


def test_events_older_than_window_are_removed():
    subject = tracker(x_crop=(0, 300))
    complete_window(subject, [event(100, 10, 20)], 1)
    result = complete_window(subject, [event(19_100, 50, 60)], 2)[0]
    assert result.window_start_us == 10_000
    snapshot = subject.latest_debug_snapshot()
    assert snapshot.activity[20, 10] == 0
    assert snapshot.activity[60, 50] == 1


def test_one_packet_produces_at_most_one_detection():
    subject = tracker(x_crop=(0, 300))
    rows = [event(offset + 100, 20 + offset // 1000, 30)
            for offset in range(0, 16_000, 1000)]
    result = complete_window(subject, rows)
    assert len(result) == 1
    assert subject.counters["processed_1ms_bins"] >= 16
    assert subject.counters["window_updates"] == 1


def test_sparse_elongated_trail_does_not_require_circularity():
    subject = tracker(x_crop=(0, 300), morphology_operation="dilate",
                      morphology_kernel=3, morphology_iterations=1,
                      use_circularity=False)
    rows = [event(100 + index * 100, 30 + index * 3, 80 + index)
            for index in range(12)]
    detection = complete_window(subject, rows)[0]
    assert detection.valid
    assert detection.blob_width_px > detection.blob_height_px
    assert detection.circularity < 0.8


def test_morphological_opening_is_never_used(monkeypatch):
    source = inspect.getsource(EventBallTracker._grouping_mask)
    assert "MORPH_OPEN" not in source
    operations = []
    original = cv2.morphologyEx

    def record_operation(*args, **kwargs):
        operations.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(cv2, "morphologyEx", record_operation)
    subject = tracker(morphology_operation="close", morphology_kernel=3,
                      morphology_iterations=1)
    subject._grouping_mask(np.ones((320, 320), dtype=np.uint16))
    assert operations == [cv2.MORPH_CLOSE]


def test_spatial_filter_removes_one_isolated_pixel():
    subject = tracker(spatial_filter_enabled=True,
                      spatial_filter_min_neighbors=1)
    activity = np.zeros((320, 320), dtype=np.uint16)
    activity[20, 10] = 1
    assert cv2.countNonZero(subject._grouping_mask(activity)) == 0
    assert subject.counters["threshold_foreground_pixels"] == 1
    assert subject.counters["spatial_filter_removed_pixels"] == 1


def test_spatial_filter_retains_two_adjacent_pixels():
    subject = tracker(spatial_filter_enabled=True,
                      spatial_filter_min_neighbors=1)
    activity = np.zeros((320, 320), dtype=np.uint16)
    activity[20, 10:12] = 1
    assert cv2.countNonZero(subject._grouping_mask(activity)) == 2


def test_spatial_filter_counts_diagonal_neighbors():
    subject = tracker(spatial_filter_enabled=True,
                      spatial_filter_min_neighbors=1)
    activity = np.zeros((320, 320), dtype=np.uint16)
    activity[20, 10] = 1
    activity[21, 11] = 1
    assert cv2.countNonZero(subject._grouping_mask(activity)) == 2


def test_spatial_filter_crop_boundary_excludes_outside_support():
    subject = tracker(x_crop=(10, 20), spatial_filter_enabled=True,
                      spatial_filter_min_neighbors=1)
    activity = np.zeros((320, 320), dtype=np.uint16)
    activity[20, 9:11] = 1
    assert cv2.countNonZero(subject._grouping_mask(activity)) == 0


def test_disabled_spatial_filter_preserves_isolated_foreground():
    subject = tracker(spatial_filter_enabled=False,
                      spatial_filter_min_neighbors=1)
    activity = np.zeros((320, 320), dtype=np.uint16)
    activity[20, 10] = 1
    assert subject._grouping_mask(activity)[20, 10] == 255
    assert subject.counters["spatial_filter_removed_pixels"] == 0


def test_spatial_filter_does_not_modify_activity_or_raw_com_inputs():
    subject = tracker(x_crop=(0, 300), spatial_filter_enabled=True,
                      spatial_filter_min_neighbors=1)
    activity = np.zeros((320, 320), dtype=np.uint16)
    activity[40, 10] = 2
    activity[40, 11] = 1
    activity[100, 100] = 1
    original = activity.copy()
    candidates, _, grouping_mask = subject._candidates(activity, None)
    assert np.array_equal(activity, original)
    assert grouping_mask[100, 100] == 0
    assert len(candidates) == 1
    assert candidates[0]["raw_event_count"] == 3
    assert candidates[0]["x"] == pytest.approx(31.0 / 3.0)
    assert candidates[0]["y"] == 40.0


def test_component_filter_removes_one_pixel_before_dilation():
    subject = tracker(
        spatial_filter_enabled=True, spatial_filter_min_neighbors=0,
        spatial_filter_min_component_area_px=2,
        morphology_operation="dilate", morphology_kernel=3,
        morphology_iterations=1)
    activity = np.zeros((320, 320), dtype=np.uint16)
    activity[20, 10] = 1
    assert cv2.countNonZero(subject._grouping_mask(activity)) == 0
    assert subject.counters["spatial_filter_removed_components"] == 1
    assert subject.counters["spatial_filter_removed_component_pixels"] == 1


def test_component_filter_removes_small_two_by_two_component():
    subject = tracker(
        spatial_filter_enabled=True, spatial_filter_min_neighbors=0,
        spatial_filter_min_component_area_px=5)
    activity = np.zeros((320, 320), dtype=np.uint16)
    activity[20:22, 10:12] = 1
    assert cv2.countNonZero(subject._grouping_mask(activity)) == 0
    assert subject.counters["spatial_filter_removed_components"] == 1
    assert subject.counters["spatial_filter_removed_component_pixels"] == 4


def test_component_filter_retains_component_at_area_cutoff():
    subject = tracker(
        spatial_filter_enabled=True, spatial_filter_min_neighbors=0,
        spatial_filter_min_component_area_px=4)
    activity = np.zeros((320, 320), dtype=np.uint16)
    activity[20:22, 10:12] = 1
    assert cv2.countNonZero(subject._grouping_mask(activity)) == 4


def test_component_filter_uses_diagonal_eight_connectivity():
    subject = tracker(
        spatial_filter_enabled=True, spatial_filter_min_neighbors=0,
        spatial_filter_min_component_area_px=2)
    activity = np.zeros((320, 320), dtype=np.uint16)
    activity[20, 10] = 1
    activity[21, 11] = 1
    assert cv2.countNonZero(subject._grouping_mask(activity)) == 2


def test_component_filter_crop_boundary_excludes_outside_pixels():
    subject = tracker(
        x_crop=(10, 20), spatial_filter_enabled=True,
        spatial_filter_min_neighbors=0,
        spatial_filter_min_component_area_px=2)
    activity = np.zeros((320, 320), dtype=np.uint16)
    activity[20, 9:11] = 1
    assert cv2.countNonZero(subject._grouping_mask(activity)) == 0


def test_disabled_component_filter_preserves_small_component():
    subject = tracker(
        spatial_filter_enabled=False,
        spatial_filter_min_component_area_px=10)
    activity = np.zeros((320, 320), dtype=np.uint16)
    activity[20, 10] = 1
    assert subject._grouping_mask(activity)[20, 10] == 255
    assert subject.counters["spatial_filter_removed_components"] == 0


def test_component_filter_preserves_activity_and_raw_com():
    subject = tracker(
        x_crop=(0, 300), spatial_filter_enabled=True,
        spatial_filter_min_neighbors=0,
        spatial_filter_min_component_area_px=2)
    activity = np.zeros((320, 320), dtype=np.uint16)
    activity[40, 10] = 2
    activity[40, 11] = 1
    activity[100, 100] = 7
    original = activity.copy()
    candidates, _, grouping_mask = subject._candidates(activity, None)
    assert np.array_equal(activity, original)
    assert grouping_mask[100, 100] == 0
    assert len(candidates) == 1
    assert candidates[0]["raw_event_count"] == 3
    assert candidates[0]["x"] == pytest.approx(31.0 / 3.0)
    assert candidates[0]["y"] == 40.0


def test_debug_distinguishes_removed_rejected_and_selected_contours():
    removed = np.asarray([[[20, 80]]], dtype=np.int32)
    rejected = np.asarray([[[30, 90]]], dtype=np.int32)
    selected = np.asarray([[[40, 100]]], dtype=np.int32)
    detection = TrackerDetection(0, 1000, 1, 0)
    snapshot = TrackerDebugSnapshot(
        activity=np.zeros((320, 320), dtype=np.uint16),
        threshold_mask=np.zeros((320, 320), dtype=np.uint8),
        detection=detection,
        candidate_contours=(rejected, selected),
        accepted_candidate_contours=(selected,),
        selected_contour=selected,
        predicted_position=None,
        trajectory=(),
        x_crop=(0, 320),
        removed_component_contours=(removed,))
    tracking = render_debug_images(
        snapshot, rotation_degrees=0)["tracking"]
    assert tracking[80, 20].tolist() == [255, 255, 0]
    assert tracking[90, 30].tolist() == [0, 128, 255]
    assert tracking[100, 40].tolist() == [0, 255, 0]


def test_contours_debug_labels_biggest_contour_area(monkeypatch):
    small = np.asarray([[[20, 80]], [[21, 80]], [[21, 81]], [[20, 81]]],
                       dtype=np.int32)
    large = np.asarray([[[50, 100]], [[54, 100]], [[54, 103]], [[50, 103]]],
                       dtype=np.int32)
    detection = TrackerDetection(0, 1000, 1, 0)
    snapshot = TrackerDebugSnapshot(
        activity=np.zeros((320, 320), dtype=np.uint16),
        threshold_mask=np.zeros((320, 320), dtype=np.uint8),
        detection=detection, candidate_contours=(small, large),
        accepted_candidate_contours=(), selected_contour=None,
        predicted_position=None, trajectory=(), x_crop=(0, 320),
        candidate_details=({"area": 4}, {"area": 20}))
    labels = []
    original = cv2.putText
    original_rotate = cv2.rotate
    rotations = []

    def record_label(image, text, *args, **kwargs):
        labels.append(text)
        return original(image, text, *args, **kwargs)

    def record_rotation(image, operation):
        rotations.append(operation)
        return original_rotate(image, operation)

    monkeypatch.setattr(cv2, "putText", record_label)
    monkeypatch.setattr(cv2, "rotate", record_rotation)
    render_debug_images(snapshot, rotation_degrees=0)
    assert labels.count("area=20") == 1
    assert "area=4" not in labels
    assert rotations == [cv2.ROTATE_90_CLOCKWISE]


def debug_event_snapshot(*, valid, x=0.0, y=0.0):
    detection = TrackerDetection(
        0, 33_000, 1, 1, window_start_us=0, window_end_us=33_000,
        valid=valid, x_px=x, y_px=y)
    events = np.asarray([event(32_000, 100, 120)], dtype=np.uint16)
    events.setflags(write=False)
    return TrackerDebugSnapshot(
        activity=np.zeros((320, 320), dtype=np.uint16),
        threshold_mask=np.zeros((320, 320), dtype=np.uint8),
        detection=detection, candidate_contours=(),
        accepted_candidate_contours=(), selected_contour=None,
        predicted_position=None, trajectory=(), x_crop=(0, 320),
        debug_events=events)


def test_debug_event_frame_uses_newest_valid_snapshot_com():
    older = debug_event_snapshot(valid=True, x=20.0, y=30.0)
    newest = debug_event_snapshot(valid=True, x=70.0, y=80.0)
    older_image = render_debug_event_frame(older, rotation_degrees=0)
    newest_image = render_debug_event_frame(newest, rotation_degrees=0)
    assert older_image[30, 20].tolist() == [0, 0, 255]
    assert newest_image[80, 70].tolist() == [0, 0, 255]
    assert newest_image[30, 20].tolist() != [0, 0, 255]


def test_invalid_debug_event_detection_has_no_red_origin_marker():
    snapshot = debug_event_snapshot(valid=False, x=0.0, y=0.0)
    image = render_debug_event_frame(snapshot, rotation_degrees=0)
    assert image[0, 0].tolist() == [128, 128, 128]
    assert not np.any(
        (image[:, :, 0] == 0) & (image[:, :, 1] == 0) &
        (image[:, :, 2] == 255))


def test_debug_event_frame_rotation_moves_com_with_background():
    snapshot = debug_event_snapshot(valid=True, x=70.0, y=80.0)
    native = render_debug_event_frame(snapshot, rotation_degrees=0)
    rotated = render_debug_event_frame(snapshot, rotation_degrees=90)
    assert native[80, 70].tolist() == [0, 0, 255]
    assert rotated[320 - 1 - 70, 80].tolist() == [0, 0, 255]
    assert np.array_equal(rotated, cv2.rotate(
        native, cv2.ROTATE_90_COUNTERCLOCKWISE))


def test_dilation_groups_but_cannot_bias_raw_event_com():
    subject = tracker(x_crop=(0, 300), morphology_operation="dilate",
                      morphology_kernel=5, morphology_iterations=1)
    rows = [event(100, 10, 40), event(200, 10, 40),
            event(300, 14, 40)]
    detection = complete_window(subject, rows)[0]
    assert detection.valid
    assert abs(detection.x_px - (34.0 / 3.0)) < 1e-6
    assert detection.y_px == 40.0
    assert detection.blob_event_count == 3


def test_predicted_near_candidate_beats_larger_distant_candidate():
    subject = tracker(x_crop=(0, 300), accumulation_window_ms=1.0,
                      max_jump_px=30.0)
    complete_window(subject, [event(100, 20, 50)], 1)
    near = [event(2100, 22, 50), event(2200, 22, 50)]
    distant = [event(2100 + i, 150 + i, 100) for i in range(10)]
    result = complete_window(subject, near + distant, 2)[0]
    assert result.valid
    assert result.x_px == 22.0
    assert result.blob_event_count == 2


def test_candidate_geometry_and_raw_count_are_reported():
    subject = tracker(x_crop=(0, 300))
    rows = [event(100, x, y) for x in range(40, 45) for y in range(70, 73)]
    detection = complete_window(subject, rows)[0]
    assert detection.valid
    assert detection.blob_event_count == 15
    assert detection.blob_area_px == 15
    assert detection.blob_width_px == 5
    assert detection.blob_height_px == 3
    assert detection.blob_perimeter_px > 0


def test_velocity_uses_sensor_timestamps():
    subject = tracker(x_crop=(0, 300), accumulation_window_ms=1.0,
                      velocity_history_size=5)
    outputs = []
    for index in range(6):
        timestamp = index * 2000 + 100
        outputs.extend(complete_window(
            subject, [event(timestamp, 20 + 2 * index, 40)], index + 1))
    assert outputs[-1].velocity_valid
    assert abs(outputs[-1].vx_px_s - 1000.0) < 100
    assert abs(outputs[-1].vy_px_s) < 1


def test_late_events_are_ignored_deterministically():
    subject = tracker(x_crop=(0, 300))
    complete_window(subject, [event(100, 10, 10)], 1)
    assert subject.update(packet([event(200, 20, 20)], 2)) == []
    assert subject.counters["late_events_or_bins"] == 1


def test_debug_snapshot_shows_accumulated_window_and_stages():
    subject = tracker(x_crop=(40, 280))
    rows = [event(100, 100, 200), event(5100, 101, 200)]
    detection = complete_window(subject, rows)[0]
    snapshot = subject.latest_debug_snapshot()
    assert snapshot.activity.sum() == 2
    assert snapshot.detection.window_event_count == 2
    images = render_debug_images(snapshot, clip_count=16, rotation_degrees=0)
    assert set(images) == {"activity", "threshold", "contours", "tracking"}
    assert all(image.shape == (320, 320, 3) for image in images.values())
    assert images["activity"][300, 40].tolist() == [255, 0, 255]
    assert detection.window_end_us - detection.window_start_us == 10_000


def test_trace_json_is_finite_and_sensor_time_stays_in_detail():
    subject = tracker(x_crop=(0, 300))
    source = packet([event(123, 10, 10), event(1000, 319, 319)],
                    ros_ns=8_000_000_321)
    detection = subject.update(source)[0]
    detail = json.loads(trace_detail_json(detection))
    assert detail["sensor_timestamp_domain"] == "genx320_microseconds"
    assert detail["window_start_us"] == -9000
    assert detail["window_end_us"] == 1000
    assert detail["selected_raw_event_count"] == 1
    assert source.packet_ros_stamp_ns == 8_000_000_321
    assert source.packet_ros_stamp_ns not in (
        detail["window_start_us"], detail["window_end_us"])
    assert all(np.isfinite(value) for value in detail.values()
               if isinstance(value, (int, float)) and not isinstance(value, bool))


def test_timing_and_history_are_bounded():
    subject = tracker(x_crop=(0, 300), stats_history_size=3,
                      history_limit_ms=10.0)
    for index in range(10):
        complete_window(
            subject, [event(index * 2000 + 100, 10, 10)], index + 1)
    assert all(len(values) <= 3 for values in subject.timings.values())
    assert len(subject.history_bins) <= 11
    stats = subject.statistics()
    assert stats["window_updates"] == 10
    assert stats["processed_1ms_bins"] >= stats["window_updates"]
    assert "window_update_rate_hz" in stats


def test_x_crop_remains_half_open_and_debug_rotation_is_compatible():
    subject = tracker(x_crop=(100, 200))
    rows = [event(100, 99, 10), event(100, 100, 20),
            event(100, 199, 30), event(100, 200, 40)]
    activity = subject.build_activity_map(np.asarray(rows, dtype=np.uint16))
    assert activity.sum() == 2
    detection = TrackerDetection(
        0, 10_000, 1, 2, window_start_us=0, window_end_us=10_000,
        window_event_count=2)
    snapshot = TrackerDebugSnapshot(
        activity, (activity > 0).astype(np.uint8) * 255, detection,
        (), (), None, None, (), (100, 200))
    rotated = render_debug_image(snapshot, rotation_degrees=90)
    assert rotated[219, 300].tolist() == [255, 0, 255]


def test_y_crop_interpolates_half_open_bounds_across_x_crop():
    subject = tracker(x_crop=(100, 200), y_crop=(20, 100, 40, 80))
    rows = [
        event(100, 100, 19), event(200, 100, 20),
        event(300, 150, 29), event(400, 150, 30),
        event(500, 150, 89), event(600, 150, 90),
        event(700, 199, 40), event(800, 199, 79),
    ]
    activity = subject.build_activity_map(np.asarray(rows, dtype=np.uint16))
    assert activity.sum() == 5
    assert activity[20, 100] == 1
    assert activity[30, 150] == 1
    assert activity[89, 150] == 1
    assert activity[40, 199] == 1
    assert activity[79, 199] == 1


def test_y_crop_validation_rejects_invalid_endpoint_ranges():
    with pytest.raises(ValueError, match="must contain"):
        tracker(y_crop=(0, 320))
    with pytest.raises(ValueError, match="endpoint pairs"):
        tracker(y_crop=(100, 100, 0, 320))
