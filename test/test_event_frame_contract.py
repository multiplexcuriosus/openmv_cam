import importlib.util
from pathlib import Path

import numpy as np

from openmv_cam.event_frame_contract import (
    build_event_frame_3ch,
    build_time_ranges_us,
    render_event_frame_from_arrays,
    retention_history_ms,
)


def _load_act_raw_event_hdf5_module():
    repo_root = Path(__file__).resolve().parents[2]
    act_mod_path = repo_root / "act" / "helpers" / "raw_event_hdf5.py"
    spec = importlib.util.spec_from_file_location("act_raw_event_hdf5", str(act_mod_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _single_pixel_event(tp: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([tp], dtype=np.uint8),
        np.asarray([0], dtype=np.int16),
        np.asarray([0], dtype=np.int16),
    )


def test_fixed_log_positive_and_negative_values_exact():
    clip = 16.0
    denom = np.log1p(clip)

    t_pos, x_pos, y_pos = _single_pixel_event(1)
    frame_pos = render_event_frame_from_arrays(
        t_pos,
        x_pos,
        y_pos,
        width=1,
        height=1,
        scaling_mode="signed_log1p_fixed_clip",
        event_clip_count=clip,
    )
    expected_pos = int(np.rint(128.0 + 127.0 * (np.log1p(1.0) / denom)))
    assert int(frame_pos[0, 0]) == expected_pos

    t_neg, x_neg, y_neg = _single_pixel_event(0)
    frame_neg = render_event_frame_from_arrays(
        t_neg,
        x_neg,
        y_neg,
        width=1,
        height=1,
        scaling_mode="signed_log1p_fixed_clip",
        event_clip_count=clip,
    )
    expected_neg = int(np.rint(128.0 - 127.0 * (np.log1p(1.0) / denom)))
    assert int(frame_neg[0, 0]) == expected_neg


def test_fixed_log_saturates_at_and_beyond_clip_count():
    clip = 16.0
    event_type = np.asarray([1] * 20 + [0] * 20, dtype=np.uint8)
    event_x = np.asarray([0] * 20 + [1] * 20, dtype=np.int16)
    event_y = np.asarray([0] * 20 + [0] * 20, dtype=np.int16)

    frame = render_event_frame_from_arrays(
        event_type,
        event_x,
        event_y,
        width=2,
        height=1,
        scaling_mode="signed_log1p_fixed_clip",
        event_clip_count=clip,
    )
    assert int(frame[0, 0]) == 255
    assert int(frame[0, 1]) == 1


def test_fixed_log_neutral_is_128_for_empty_and_unhit_pixels():
    frame_empty = render_event_frame_from_arrays(
        np.asarray([], dtype=np.uint8),
        np.asarray([], dtype=np.int16),
        np.asarray([], dtype=np.int16),
        width=3,
        height=2,
        scaling_mode="signed_log1p_fixed_clip",
        event_clip_count=16.0,
    )
    assert frame_empty.dtype == np.uint8
    assert frame_empty.shape == (2, 3)
    assert np.all(frame_empty == 128)

    frame_partial = render_event_frame_from_arrays(
        np.asarray([1], dtype=np.uint8),
        np.asarray([0], dtype=np.int16),
        np.asarray([0], dtype=np.int16),
        width=3,
        height=2,
        scaling_mode="signed_log1p_fixed_clip",
        event_clip_count=16.0,
    )
    assert int(frame_partial[1, 2]) == 128


def test_shifted_boundaries_are_exclusive_for_older_bins_and_no_duplication():
    now_us = 1_000_000
    event_ts_us = np.asarray(
        [
            now_us - 200_000,
            now_us - 100_000,
            now_us - 50_000,
            now_us,
        ],
        dtype=np.int64,
    )
    events = np.zeros((4, 6), dtype=np.uint16)
    events[:, 0] = 1
    events[:, 4] = np.asarray([0, 1, 2, 3], dtype=np.uint16)
    events[:, 5] = 0

    frame, counts = build_event_frame_3ch(
        events=events,
        event_ts_us=event_ts_us,
        now_event_t_us=now_us,
        width=4,
        height=1,
        windows_ms=(50.0, 100.0, 200.0),
        mode="shifted",
        scaling_mode="signed_log1p_fixed_clip",
        event_clip_count=16.0,
        contrast=4.0,
        step=1.0,
    )

    # shifted bins: [now-50, now], [now-100, now-50), [now-200, now-100)
    np.testing.assert_array_equal(counts, np.asarray([2, 1, 1], dtype=np.int32))
    assert frame.shape == (1, 4, 3)


def test_channel_order_recent_to_oldest():
    now_us = 1_000_000
    # One event per shifted channel range at distinct x positions.
    event_ts_us = np.asarray([now_us - 10_000, now_us - 60_000, now_us - 150_000], dtype=np.int64)
    events = np.zeros((3, 6), dtype=np.uint16)
    events[:, 0] = 1
    events[:, 4] = np.asarray([0, 1, 2], dtype=np.uint16)
    events[:, 5] = 0

    frame, counts = build_event_frame_3ch(
        events=events,
        event_ts_us=event_ts_us,
        now_event_t_us=now_us,
        width=3,
        height=1,
        windows_ms=(50.0, 100.0, 200.0),
        mode="shifted",
        scaling_mode="signed_log1p_fixed_clip",
        event_clip_count=16.0,
        contrast=4.0,
        step=1.0,
    )

    assert tuple(counts.tolist()) == (1, 1, 1)
    # Event in most recent channel should appear in channel 0 only.
    assert int(frame[0, 0, 0]) != 128
    assert int(frame[0, 0, 1]) == 128
    assert int(frame[0, 0, 2]) == 128


def test_packet_margin_changes_retention_not_bin_widths():
    # Retention increases when event channels are enabled.
    keep_no_margin = retention_history_ms(100.0, 200.0, 0.0, include_event_channels=True)
    keep_with_margin = retention_history_ms(100.0, 200.0, 50.0, include_event_channels=True)
    assert keep_no_margin == 200.0
    assert keep_with_margin == 250.0

    # Bin geometry is independent of packet margin.
    ranges = build_time_ranges_us(1_000_000, (50.0, 100.0, 200.0), "shifted")
    assert ranges == [
        (950_000, 1_000_000),
        (900_000, 950_000),
        (800_000, 900_000),
    ]


def test_output_shape_dtype_and_legacy_scaling_available():
    event_ts_us = np.asarray([1_000_000], dtype=np.int64)
    events = np.zeros((1, 6), dtype=np.uint16)
    events[:, 0] = 1
    events[:, 4] = 0
    events[:, 5] = 0

    frame, _counts = build_event_frame_3ch(
        events=events,
        event_ts_us=event_ts_us,
        now_event_t_us=1_000_000,
        width=2,
        height=2,
        windows_ms=(50.0, 100.0, 200.0),
        mode="cumulative",
        scaling_mode="legacy_per_frame_max",
        event_clip_count=None,
        contrast=4.0,
        step=1.0,
    )
    assert frame.dtype == np.uint8
    assert frame.shape == (2, 2, 3)


def test_equivalence_with_act_raw_renderer_for_identical_arrays():
    act_raw = _load_act_raw_event_hdf5_module()

    event_type = np.asarray([1, 1, 0, 3, 1], dtype=np.uint8)
    event_x = np.asarray([0, 0, 1, 1, 0], dtype=np.int16)
    event_y = np.asarray([0, 0, 0, 0, 1], dtype=np.int16)

    ours = render_event_frame_from_arrays(
        event_type,
        event_x,
        event_y,
        width=2,
        height=2,
        scaling_mode="signed_log1p_fixed_clip",
        event_clip_count=16.0,
    )
    theirs = act_raw.render_event_frame_from_raw_arrays(
        event_type=event_type,
        event_x=event_x,
        event_y=event_y,
        width=2,
        height=2,
        scaling_mode="signed_log1p_fixed_clip",
        event_clip_count=16.0,
    )
    np.testing.assert_array_equal(ours, theirs)

    # Also verify shifted masks match ACT helper conventions.
    now_us = 1_000_000
    event_ts_us = np.asarray([980_000, 930_000, 880_000], dtype=np.int64)
    events = np.zeros((3, 6), dtype=np.uint16)
    events[:, 0] = np.asarray([1, 1, 1], dtype=np.uint16)
    events[:, 4] = np.asarray([0, 1, 0], dtype=np.uint16)
    events[:, 5] = np.asarray([0, 0, 1], dtype=np.uint16)

    frame, _counts = build_event_frame_3ch(
        events=events,
        event_ts_us=event_ts_us,
        now_event_t_us=now_us,
        width=2,
        height=2,
        windows_ms=(50.0, 100.0, 200.0),
        mode="shifted",
        scaling_mode="signed_log1p_fixed_clip",
        event_clip_count=16.0,
        contrast=4.0,
        step=1.0,
    )

    ranges = [
        (now_us - 50_000, now_us),
        (now_us - 100_000, now_us - 50_000),
        (now_us - 200_000, now_us - 100_000),
    ]
    channels = []
    for lo, hi in ranges:
        if hi == now_us:
            mask = (event_ts_us >= lo) & (event_ts_us <= hi)
        else:
            mask = (event_ts_us >= lo) & (event_ts_us < hi)
        if not np.any(mask):
            channels.append(np.full((2, 2), 128, dtype=np.uint8))
        else:
            channels.append(
                act_raw.render_event_frame_from_raw_arrays(
                    event_type=events[mask, 0],
                    event_x=events[mask, 4],
                    event_y=events[mask, 5],
                    width=2,
                    height=2,
                    scaling_mode="signed_log1p_fixed_clip",
                    event_clip_count=16.0,
                )
            )
    expected = np.stack(channels, axis=2)
    np.testing.assert_array_equal(frame, expected)
