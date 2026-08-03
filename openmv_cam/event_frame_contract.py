from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


SUPPORTED_EVENT_SCALING = (
    "legacy_per_frame_max",
    "signed_log1p_fixed_clip",
)
SUPPORTED_EVENT_FRAME_MODE = ("cumulative", "shifted")


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
) -> float:
    mono_ms = float(mono_window_ms)
    if not include_event_channels:
        return mono_ms
    return max(mono_ms, float(max_event_window_ms) + float(event_packet_margin_ms))


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
