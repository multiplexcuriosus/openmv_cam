import numpy as np

from openmv_cam.event_frame_contract import (
    build_event_activity_voxel,
    has_new_event_data,
)


def _events(types, xs, ys):
    events = np.zeros((len(types), 6), dtype=np.int64)
    events[:, 0] = types
    events[:, 4] = xs
    events[:, 5] = ys
    return events


def _build(events, timestamps, anchor=1999, **overrides):
    options = {
        "width": 4,
        "height": 3,
        "bin_ms": 1.0,
        "temporal_bins": 2,
        "activity_mode": "absolute_activity",
        "clip_count": 4.0,
    }
    options.update(overrides)
    return build_event_activity_voxel(
        events, np.asarray(timestamps, dtype=np.int64), anchor, **options
    )


def test_exact_one_ms_boundaries_and_oldest_to_newest_order():
    events = _events([1] * 4, [0, 1, 2, 3], [0] * 4)
    voxel, counts = _build(events, [0, 999, 1000, 1999])

    np.testing.assert_array_equal(counts, [2, 2])
    assert voxel[0, 0, 0] > 0
    assert voxel[0, 1, 0] > 0
    assert voxel[0, 2, 1] > 0
    assert voxel[0, 3, 1] > 0


def test_events_spanning_packets_are_binned_by_timestamp_not_packet():
    packet_a = _events([1, 1], [0, 1], [0, 0])
    packet_b = _events([1, 1], [2, 3], [0, 0])
    events = np.concatenate([packet_a, packet_b])
    voxel, counts = _build(events, [1000, 0, 1999, 999])

    np.testing.assert_array_equal(counts, [2, 2])
    assert np.count_nonzero(voxel[:, :, 0]) == 2
    assert np.count_nonzero(voxel[:, :, 1]) == 2


def test_absolute_activity_opposite_polarities_do_not_cancel():
    events = _events([0, 1], [2, 2], [1, 1])
    voxel, counts = _build(events, [1999, 1999])

    assert counts[1] == 2
    assert voxel[1, 2, 1] == 128


def test_signed_activity_is_available_for_debug_and_uses_magnitude():
    events = _events([0, 1], [2, 2], [1, 1])
    voxel, _ = _build(events, [1999, 1999], activity_mode="signed_activity")
    assert voxel[1, 2, 1] == 0


def test_empty_future_and_out_of_range_events_produce_zero_background():
    empty, counts = _build(np.zeros((0, 6), dtype=np.int64), [])
    assert not np.any(empty)
    assert not np.any(counts)

    events = _events([1, 1, 1], [0, 4, -1], [0, 0, 0])
    voxel, counts = _build(events, [2000, 1999, 1999])
    assert not np.any(voxel)
    assert not np.any(counts)


def test_one_bin_shape_fixed_scaling_and_native_coordinates():
    events = _events([1, 1, 1, 1, 1], [3] * 5, [2] * 5)
    voxel, counts = _build(
        events,
        [1000] * 5,
        anchor=1000,
        width=320,
        height=320,
        temporal_bins=1,
    )
    assert voxel.shape == (320, 320)
    assert voxel.dtype == np.uint8
    assert voxel.flags.c_contiguous
    assert voxel[2, 3] == 255
    assert counts.tolist() == [5]
    assert len(voxel.tobytes()) == 320 * 320


def test_deterministic_translating_blob_moves_positive_x_in_newer_bins():
    events = _events([1] * 6, [0, 0, 1, 1, 2, 2], [1] * 6)
    voxel, counts = _build(
        events,
        [100, 200, 1100, 1200, 2100, 2200],
        anchor=2200,
        temporal_bins=3,
    )
    np.testing.assert_array_equal(counts, [2, 2, 2])
    active_x = [int(np.argmax(voxel[1, :, channel])) for channel in range(3)]
    assert active_x == [0, 1, 2]


def test_repeated_tick_without_new_events_does_not_publish_fresh_activity():
    assert not has_new_event_data(7, 7)
    assert has_new_event_data(8, 7)
