from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


SUPPORTED_EVENT_SCALING = (
    "legacy_per_frame_max",
    "signed_log1p_fixed_clip",
)
SUPPORTED_EVENT_FRAME_MODE = ("cumulative", "shifted")
SUPPORTED_ACTIVITY_MODE = ("absolute_activity", "signed_activity")


def has_new_event_data(generation: int, last_published_generation: int) -> bool:
    """Return whether a timer tick has newly buffered non-empty event data."""
    return int(generation) != int(last_published_generation)


@dataclass(frozen=True)
class EventContractConfig:
    windows_ms: Tuple[float, float, float]
    mode: str
    scaling: str
    clip_count: Optional[float]
    packet_margin_ms: float


def validate_windows_and_mode(
    windows_ms: Sequence[float],
    mode: str,
) -> Tuple[float, float, float]:
    if len(windows_ms) != 3:
        raise ValueError(f"event frame windows must have 3 values, got {windows_ms}")

    w0, w1, w2 = [float(window_ms) for window_ms in windows_ms]
    if not all(np.isfinite([w0, w1, w2])):
        raise ValueError(f"event frame windows must be finite, got {windows_ms}")
    if not (0.0 < w0 <= w1 <= w2):
        raise ValueError(
            "event frame windows must satisfy 0 < ch0 <= ch1 <= ch2, "
            f"got {windows_ms}"
        )

    mode = str(mode).strip().lower()
    if mode not in SUPPORTED_EVENT_FRAME_MODE:
        raise ValueError(
            f"Unsupported event_frame_mode {mode!r}; expected one of {SUPPORTED_EVENT_FRAME_MODE}"
        )

    return w0, w1, w2


def validate_event_scaling(
    event_scaling: str,
    event_clip_count: Optional[float],
) -> Tuple[str, Optional[float]]:
    scaling = str(event_scaling).strip().lower()
    if scaling not in SUPPORTED_EVENT_SCALING:
        raise ValueError(
            f"Unsupported event_scaling {scaling!r}; expected one of {SUPPORTED_EVENT_SCALING}"
        )

    if scaling == "signed_log1p_fixed_clip":
        if event_clip_count is None:
            raise ValueError(
                "event_clip_count must be set when event_scaling=signed_log1p_fixed_clip"
            )
        clip_count = float(event_clip_count)
        if not np.isfinite(clip_count) or clip_count <= 0.0:
            raise ValueError(
                "event_clip_count must be positive and finite when "
                "event_scaling=signed_log1p_fixed_clip"
            )
        return scaling, clip_count

    return scaling, None


def validate_packet_margin_ms(packet_margin_ms: float) -> float:
    margin = float(packet_margin_ms)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError(
            f"event_packet_margin_ms must be finite and >= 0, got {packet_margin_ms}"
        )
    return margin


def validate_event_contract_config(
    windows_ms: Sequence[float],
    mode: str,
    event_scaling: str,
    event_clip_count: Optional[float],
    packet_margin_ms: float,
) -> EventContractConfig:
    w0, w1, w2 = validate_windows_and_mode(windows_ms, mode)
    scaling, clip_count = validate_event_scaling(event_scaling, event_clip_count)
    margin = validate_packet_margin_ms(packet_margin_ms)
    return EventContractConfig(
        windows_ms=(w0, w1, w2),
        mode=str(mode).strip().lower(),
        scaling=scaling,
        clip_count=clip_count,
        packet_margin_ms=margin,
    )


def retention_history_ms(
    mono_window_ms: float,
    max_event_window_ms: float,
    event_packet_margin_ms: float,
    include_event_channels: bool,
    event_voxel_horizon_ms: float = 0.0,
    include_event_voxel: bool = False,
) -> float:
    mono_ms = float(mono_window_ms)
    histories_ms = [mono_ms]
    if include_event_channels:
        histories_ms.append(
            float(max_event_window_ms) + float(event_packet_margin_ms)
        )
    if include_event_voxel:
        histories_ms.append(
            float(event_voxel_horizon_ms) + float(event_packet_margin_ms)
        )
    return max(histories_ms)


def validate_xyt_voxel_config(
    *,
    sensor_width: int,
    sensor_height: int,
    output_width: int,
    output_height: int,
    horizon_ms: float,
    temporal_bins: int,
    scaling_mode: str,
    event_clip_count: Optional[float],
) -> Tuple[int, int, int, int, float, int, str, float]:
    dimensions = (
        int(sensor_width),
        int(sensor_height),
        int(output_width),
        int(output_height),
    )
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError(
            "sensor and output dimensions must be positive, got "
            f"sensor=({sensor_width}, {sensor_height}), "
            f"output=({output_width}, {output_height})"
        )

    horizon = float(horizon_ms)
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError(f"horizon_ms must be positive and finite, got {horizon_ms}")

    bins = int(temporal_bins)
    if bins <= 0 or bins != temporal_bins:
        raise ValueError(f"temporal_bins must be a positive integer, got {temporal_bins}")

    scaling, clip_count = validate_event_scaling(scaling_mode, event_clip_count)
    if scaling != "signed_log1p_fixed_clip":
        raise ValueError(
            "XYT voxel scaling must be 'signed_log1p_fixed_clip', "
            f"got {scaling_mode!r}"
        )
    assert clip_count is not None
    return (*dimensions, horizon, bins, scaling, clip_count)


def build_xyt_signed_voxel(
    events: np.ndarray,
    event_ts_us: np.ndarray,
    anchor_t_us: int,
    *,
    sensor_width: int,
    sensor_height: int,
    output_width: int,
    output_height: int,
    horizon_ms: float,
    temporal_bins: int,
    scaling_mode: str,
    event_clip_count: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a causal signed XYT volume in oldest-to-newest HWC order."""
    (
        sensor_width,
        sensor_height,
        output_width,
        output_height,
        horizon_ms,
        temporal_bins,
        _scaling,
        clip_count,
    ) = validate_xyt_voxel_config(
        sensor_width=sensor_width,
        sensor_height=sensor_height,
        output_width=output_width,
        output_height=output_height,
        horizon_ms=horizon_ms,
        temporal_bins=temporal_bins,
        scaling_mode=scaling_mode,
        event_clip_count=event_clip_count,
    )

    shape = (output_height, output_width, temporal_bins)
    empty = np.full(shape, 128, dtype=np.uint8)
    counts = np.zeros((temporal_bins,), dtype=np.int32)
    if events.size == 0 or event_ts_us.size == 0:
        return empty, counts

    events_array = np.asarray(events)
    timestamps = np.asarray(event_ts_us, dtype=np.int64)
    if events_array.ndim != 2 or events_array.shape[1] < 6:
        raise ValueError(f"events must have shape (N, >=6), got {events_array.shape}")
    if timestamps.ndim != 1 or timestamps.shape[0] != events_array.shape[0]:
        raise ValueError(
            "event_ts_us must be one-dimensional and match the event count, "
            f"got events={events_array.shape[0]}, timestamps={timestamps.shape}"
        )

    horizon_us = int(round(horizon_ms * 1_000.0))
    if horizon_us <= 0:
        raise ValueError(f"horizon_ms is below timestamp resolution, got {horizon_ms}")
    anchor = int(anchor_t_us)
    start = anchor - horizon_us

    xs = events_array[:, 4].astype(np.int64, copy=False)
    ys = events_array[:, 5].astype(np.int64, copy=False)
    valid = (
        (timestamps >= start)
        & (timestamps <= anchor)
        & (xs >= 0)
        & (xs < sensor_width)
        & (ys >= 0)
        & (ys < sensor_height)
    )
    if not np.any(valid):
        return empty, counts

    selected_ts = timestamps[valid]
    selected_x = xs[valid]
    selected_y = ys[valid]
    selected_type = events_array[valid, 0]

    temporal_index = ((selected_ts - start) * temporal_bins) // horizon_us
    temporal_index = np.minimum(temporal_index, temporal_bins - 1)
    output_x = (selected_x * output_width) // sensor_width
    output_y = (selected_y * output_height) // sensor_height

    counts = np.bincount(
        temporal_index, minlength=temporal_bins
    ).astype(np.int32, copy=False)
    flat_index = (
        (output_y * output_width + output_x) * temporal_bins + temporal_index
    )
    polarity = np.where(selected_type == 1, 1, -1).astype(np.int32)
    accumulation = np.zeros(shape, dtype=np.int32)
    np.add.at(accumulation.ravel(), flat_index, polarity)

    normalized = (
        np.sign(accumulation)
        * np.log1p(np.abs(accumulation))
        / np.log1p(float(clip_count))
    )
    normalized = np.clip(normalized, -1.0, 1.0)
    voxel = np.rint(128.0 + 127.0 * normalized).astype(np.uint8)
    return np.ascontiguousarray(voxel), counts


def validate_activity_voxel_config(
    *,
    width: int,
    height: int,
    bin_ms: float,
    temporal_bins: int,
    activity_mode: str,
    clip_count: float,
) -> Tuple[int, int, int, int, str, float]:
    """Validate the native event-camera activity voxel configuration."""
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got ({width}, {height})")

    bin_us_float = float(bin_ms) * 1_000.0
    bin_us = int(round(bin_us_float))
    if not np.isfinite(bin_us_float) or bin_us <= 0 or not np.isclose(
        bin_us_float, bin_us
    ):
        raise ValueError(
            "event_voxel_bin_ms must be a positive whole number of microseconds, "
            f"got {bin_ms}"
        )

    bins = int(temporal_bins)
    if bins <= 0 or bins != temporal_bins:
        raise ValueError(f"temporal_bins must be a positive integer, got {temporal_bins}")

    mode = str(activity_mode).strip().lower()
    if mode not in SUPPORTED_ACTIVITY_MODE:
        raise ValueError(
            f"Unsupported activity_mode {mode!r}; expected one of {SUPPORTED_ACTIVITY_MODE}"
        )

    clip = float(clip_count)
    if not np.isfinite(clip) or clip <= 0.0:
        raise ValueError(f"clip_count must be positive and finite, got {clip_count}")
    return width, height, bin_us, bins, mode, clip


def build_event_activity_voxel(
    events: np.ndarray,
    event_ts_us: np.ndarray,
    anchor_t_us: int,
    *,
    width: int,
    height: int,
    bin_ms: float,
    temporal_bins: int,
    activity_mode: str,
    clip_count: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a causal native-coordinate activity voxel, oldest bin first.

    Bin intervals are right-closed: for depth N the retained interval is
    ``(anchor - N * bin_us, anchor]``. Internal boundary events enter the
    older bin. A one-bin result is returned as HW; multiple bins use HWC.
    """
    width, height, bin_us, bins, mode, clip = validate_activity_voxel_config(
        width=width,
        height=height,
        bin_ms=bin_ms,
        temporal_bins=temporal_bins,
        activity_mode=activity_mode,
        clip_count=clip_count,
    )
    shape = (height, width, bins)
    accumulation = np.zeros(shape, dtype=np.int32)
    bin_counts = np.zeros((bins,), dtype=np.int32)
    if events.size == 0 or event_ts_us.size == 0:
        result = accumulation[:, :, 0] if bins == 1 else accumulation
        return np.ascontiguousarray(result.astype(np.uint8)), bin_counts

    events_array = np.asarray(events)
    timestamps = np.asarray(event_ts_us, dtype=np.int64)
    if events_array.ndim != 2 or events_array.shape[1] < 6:
        raise ValueError(f"events must have shape (N, >=6), got {events_array.shape}")
    if timestamps.ndim != 1 or timestamps.shape[0] != events_array.shape[0]:
        raise ValueError(
            "event_ts_us must be one-dimensional and match the event count, "
            f"got events={events_array.shape[0]}, timestamps={timestamps.shape}"
        )

    anchor = int(anchor_t_us)
    start = anchor - bins * bin_us
    xs = events_array[:, 4].astype(np.int64, copy=False)
    ys = events_array[:, 5].astype(np.int64, copy=False)
    valid = (
        (timestamps > start)
        & (timestamps <= anchor)
        & (xs >= 0)
        & (xs < width)
        & (ys >= 0)
        & (ys < height)
    )
    if not np.any(valid):
        result = accumulation[:, :, 0] if bins == 1 else accumulation
        return np.ascontiguousarray(result.astype(np.uint8)), bin_counts

    selected_ts = timestamps[valid]
    selected_x = xs[valid]
    selected_y = ys[valid]
    # Right-closed bins: anchor itself is in the newest bin and an event
    # exactly one bin before anchor remains in the preceding bin.
    temporal_index = bins - 1 - ((anchor - selected_ts) // bin_us)
    bin_counts = np.bincount(temporal_index, minlength=bins).astype(np.int32)
    flat_index = (selected_y * width + selected_x) * bins + temporal_index
    if mode == "absolute_activity":
        contribution = np.ones(selected_ts.shape, dtype=np.int32)
    else:
        contribution = np.where(events_array[valid, 0] == 1, 1, -1).astype(np.int32)
    np.add.at(accumulation.ravel(), flat_index, contribution)

    if mode == "absolute_activity":
        magnitude = accumulation.astype(np.float64)
    else:
        magnitude = np.abs(accumulation).astype(np.float64)
    output = np.rint(255.0 * np.minimum(magnitude, clip) / clip).astype(np.uint8)
    result = output[:, :, 0] if bins == 1 else output
    return np.ascontiguousarray(result), bin_counts


def build_time_ranges_us(
    now_event_t_us: int,
    windows_ms: Sequence[float],
    mode: str,
) -> list[Tuple[int, int]]:
    w0_ms, w1_ms, w2_ms = validate_windows_and_mode(windows_ms, mode)
    w0_us = int(w0_ms * 1e3)
    w1_us = int(w1_ms * 1e3)
    w2_us = int(w2_ms * 1e3)

    if mode == "shifted":
        # Channel order is fixed: recent, middle, oldest.
        return [
            (int(now_event_t_us - w0_us), int(now_event_t_us)),
            (int(now_event_t_us - w1_us), int(now_event_t_us - w0_us)),
            (int(now_event_t_us - w2_us), int(now_event_t_us - w1_us)),
        ]

    return [
        (int(now_event_t_us - w0_us), int(now_event_t_us)),
        (int(now_event_t_us - w1_us), int(now_event_t_us)),
        (int(now_event_t_us - w2_us), int(now_event_t_us)),
    ]


def render_event_frame_from_arrays(
    event_type: np.ndarray,
    event_x: np.ndarray,
    event_y: np.ndarray,
    *,
    width: int,
    height: int,
    scaling_mode: str,
    contrast: float = 4.0,
    step: float = 1.0,
    event_clip_count: Optional[float] = None,
) -> np.ndarray:
    scaling_mode, resolved_clip = validate_event_scaling(scaling_mode, event_clip_count)

    if event_type.size == 0:
        return np.full((height, width), 128, dtype=np.uint8)

    xs = np.asarray(event_x, dtype=np.int64)
    ys = np.asarray(event_y, dtype=np.int64)
    et = np.asarray(event_type)

    valid = (xs >= 0) & (xs < int(width)) & (ys >= 0) & (ys < int(height))
    if not np.any(valid):
        return np.full((height, width), 128, dtype=np.uint8)

    xs = xs[valid]
    ys = ys[valid]
    et = et[valid]

    acc = np.zeros((height, width), dtype=np.float32)
    values = np.where(et == 1, float(step), -float(step)).astype(np.float32)
    np.add.at(acc, (ys, xs), values)

    if scaling_mode == "legacy_per_frame_max":
        max_abs = float(np.max(np.abs(acc)))
        if max_abs > 0.0:
            acc = acc / max_abs
        img = 128.0 + (acc * float(contrast) * 127.0)
        img = np.clip(img, 0.0, 255.0)
        return img.astype(np.uint8)

    # signed_log1p_fixed_clip mode (exact ACT/offline contract)
    assert resolved_clip is not None
    denom = np.log1p(float(resolved_clip))
    z = np.sign(acc) * np.log1p(np.abs(acc)) / denom
    z = np.clip(z, -1.0, 1.0)
    return np.rint(128.0 + 127.0 * z).astype(np.uint8)


def build_event_frame_3ch(
    events: np.ndarray,
    event_ts_us: np.ndarray,
    now_event_t_us: int,
    *,
    width: int,
    height: int,
    windows_ms: Sequence[float],
    mode: str,
    scaling_mode: str,
    event_clip_count: Optional[float],
    contrast: float,
    step: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if events.size == 0 or event_ts_us.size == 0:
        return np.full((height, width, 3), 128, dtype=np.uint8), np.zeros((3,), dtype=np.int32)

    ranges = build_time_ranges_us(now_event_t_us, windows_ms, mode)

    channels = []
    counts = []
    for lo_us, hi_us in ranges:
        if hi_us == int(now_event_t_us):
            # Newest interval includes upper boundary.
            mask = (event_ts_us >= lo_us) & (event_ts_us <= hi_us)
        else:
            # Older shifted intervals have exclusive upper boundaries.
            mask = (event_ts_us >= lo_us) & (event_ts_us < hi_us)

        count = int(np.count_nonzero(mask))
        counts.append(count)
        if count == 0:
            channels.append(np.full((height, width), 128, dtype=np.uint8))
            continue

        sliced = events[mask]
        channels.append(
            render_event_frame_from_arrays(
                event_type=sliced[:, 0],
                event_x=sliced[:, 4],
                event_y=sliced[:, 5],
                width=width,
                height=height,
                scaling_mode=scaling_mode,
                contrast=contrast,
                step=step,
                event_clip_count=event_clip_count,
            )
        )

    return np.stack(channels, axis=2).astype(np.uint8), np.asarray(counts, dtype=np.int32)
