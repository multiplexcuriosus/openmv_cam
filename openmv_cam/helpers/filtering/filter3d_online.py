#!/usr/bin/env python3
import argparse
from pathlib import Path
import struct
import time

import cv2
import numpy as np
import serial

MAGIC = b"EVT1"
HEADER_FMT = "<LL"

W = 320
H = 320
WINDOW_US = 33_000


def read_exactly(ser, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        chunk = ser.read(n - len(data))
        if not chunk:
            raise RuntimeError("Serial read timeout.")
        data.extend(chunk)
    return bytes(data)


def read_until_magic(ser):
    window = bytearray()
    while True:
        b = ser.read(1)
        if not b:
            raise RuntimeError("Timeout while waiting for magic.")
        window += b
        if len(window) > len(MAGIC):
            window = window[-len(MAGIC):]
        if bytes(window) == MAGIC:
            return


def event_timestamps_us(events: np.ndarray) -> np.ndarray:
    return (
        events[:, 1].astype(np.int64) * 1_000_000
        + events[:, 2].astype(np.int64) * 1_000
        + events[:, 3].astype(np.int64)
    )


def apply_pixel_cooldown(
    events: np.ndarray,
    last_accept_ts_us: np.ndarray,
    cooldown_us: int,
) -> np.ndarray:
    if events.size == 0 or cooldown_us <= 0:
        return events

    ts_us = event_timestamps_us(events).astype(np.int64)
    xs = events[:, 4].astype(np.int32)
    ys = events[:, 5].astype(np.int32)

    height, width = last_accept_ts_us.shape
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)

    keep = np.zeros(len(events), dtype=bool)
    order = np.argsort(ts_us, kind="stable")
    for idx in order:
        if not valid[idx]:
            continue

        x = xs[idx]
        y = ys[idx]
        t = ts_us[idx]
        if t - last_accept_ts_us[y, x] >= cooldown_us:
            keep[idx] = True
            last_accept_ts_us[y, x] = t

    return events[keep]


def extract_xyt_points(
    events: np.ndarray, width: int, height: int, origin_us: int | None
) -> tuple[np.ndarray, int | None]:
    if events.size == 0:
        return np.zeros((0, 3), dtype=np.int64), origin_us

    xs = events[:, 4].astype(np.int64)
    ys = events[:, 5].astype(np.int64)
    ts_us = event_timestamps_us(events).astype(np.int64)

    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs = xs[valid]
    ys = ys[valid]
    ts_us = ts_us[valid]

    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.int64), origin_us

    order = np.argsort(ts_us, kind="stable")
    xs = xs[order]
    ys = ys[order]
    ts_us = ts_us[order]

    if origin_us is None:
        origin_us = int(ts_us.min())

    rel_ts_us = ts_us - origin_us
    points = np.column_stack((xs, ys, rel_ts_us)).astype(np.int64)
    return points, origin_us


def save_xyt_points(points: np.ndarray, output_path: str):
    with open(output_path, "wt", encoding="utf-8") as f:
        np.savetxt(f, points, fmt="%d", delimiter="\t")


def find_temporal_neighbor_points(points: np.ndarray, radius: int) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=bool)

    times = points[:, 2].astype(np.int64)
    order = np.argsort(times, kind="stable")
    sorted_times = times[order]

    left = np.searchsorted(sorted_times, sorted_times - radius, side="left")
    right = np.searchsorted(sorted_times, sorted_times + radius, side="right")

    has_neighbor_sorted = (right - left) > 1
    has_neighbor = np.zeros(len(points), dtype=bool)
    has_neighbor[order] = has_neighbor_sorted
    return has_neighbor


def find_spatial_neighbor_points(
    points: np.ndarray, radius: int, width: int, height: int
) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=bool)

    xs = points[:, 0].astype(np.int64)
    ys = points[:, 1].astype(np.int64)

    counts = np.zeros((width, height), dtype=np.int32)
    np.add.at(counts, (xs, ys), 1)

    neighbor_offsets = []
    radius_sq = radius * radius
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            if dx * dx + dy * dy <= radius_sq:
                neighbor_offsets.append((dx, dy))

    has_neighbor = np.zeros(len(points), dtype=bool)
    for idx, (x, y) in enumerate(zip(xs, ys)):
        for dx, dy in neighbor_offsets:
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if counts[nx, ny] > 0:
                has_neighbor[idx] = True
                break

    return has_neighbor


def apply_filters(
    points: np.ndarray,
    filter_time: int | None,
    filter_spatial: int | None,
    hide_noise: bool,
    width: int,
    height: int,
) -> tuple[np.ndarray, bool, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if points.size == 0:
        return points, False, None, None, None

    if filter_time is None and filter_spatial is None:
        return points, False, None, None, None

    temporal_mask = None
    spatial_mask = None
    matched_mask = np.zeros(len(points), dtype=bool)

    if filter_time is not None:
        temporal_mask = find_temporal_neighbor_points(points, filter_time)
        matched_mask |= temporal_mask

    if filter_spatial is not None:
        spatial_mask = find_spatial_neighbor_points(
            points, filter_spatial, width=width, height=height
        )
        matched_mask |= spatial_mask

    render_points = points
    render_temporal_mask = temporal_mask
    render_spatial_mask = spatial_mask
    if hide_noise:
        keep_mask = np.ones(len(points), dtype=bool)
        if temporal_mask is not None:
            keep_mask &= temporal_mask
        if spatial_mask is not None:
            keep_mask &= spatial_mask

        render_points = points[keep_mask]
        if temporal_mask is not None:
            render_temporal_mask = temporal_mask[keep_mask]
        if spatial_mask is not None:
            render_spatial_mask = spatial_mask[keep_mask]
        matched_mask = keep_mask

    return render_points, True, render_temporal_mask, render_spatial_mask, matched_mask


def apply_filters_by_slices(
    points: np.ndarray,
    filter_time: int | None,
    filter_spatial: int | None,
    hide_noise: bool,
    width: int,
    height: int,
    slice_us: int,
) -> np.ndarray:
    if points.size == 0:
        return points

    if filter_time is None and filter_spatial is None:
        return points

    timestamps = points[:, 2].astype(np.int64)
    slice_ids = timestamps // slice_us

    render_chunks = []
    boundaries = np.flatnonzero(np.diff(slice_ids)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(points)]))

    for start, end in zip(starts, ends):
        slice_points = points[start:end]
        filtered_points, _, _, _, _ = apply_filters(
            slice_points,
            filter_time=filter_time,
            filter_spatial=filter_spatial,
            hide_noise=hide_noise,
            width=width,
            height=height,
        )
        if len(filtered_points) > 0:
            render_chunks.append(filtered_points)

    if not render_chunks:
        return np.zeros((0, 3), dtype=np.int64)

    return np.concatenate(render_chunks, axis=0)


def points_to_preview_frame(
    points: np.ndarray,
    filter_mode: bool,
    temporal_mask: np.ndarray | None,
    spatial_mask: np.ndarray | None,
    width: int,
    height: int,
    resize: int,
) -> np.ndarray:
    frame = np.full((height, width, 3), 255, dtype=np.uint8)

    if points.size == 0:
        return cv2.resize(
            frame, (width * resize, height * resize), interpolation=cv2.INTER_NEAREST
        )

    xs = points[:, 0].astype(np.int32)
    ys = points[:, 1].astype(np.int32)

    if not filter_mode:
        frame[ys, xs] = np.array([255, 120, 60], dtype=np.uint8)
    else:
        frame[ys, xs] = np.array([140, 140, 140], dtype=np.uint8)
        if temporal_mask is not None:
            frame[ys[temporal_mask], xs[temporal_mask]] = np.array(
                [40, 40, 255], dtype=np.uint8
            )
        if spatial_mask is not None:
            frame[ys[spatial_mask], xs[spatial_mask]] = np.array(
                [255, 40, 40], dtype=np.uint8
            )
        if temporal_mask is not None and spatial_mask is not None:
            both_mask = temporal_mask & spatial_mask
            frame[ys[both_mask], xs[both_mask]] = np.array(
                [255, 40, 255], dtype=np.uint8
            )

    frame = cv2.blur(frame, (2, 2))
    return cv2.resize(
        frame, (width * resize, height * resize), interpolation=cv2.INTER_NEAREST
    )


def points_to_frequency_frame(
    points: np.ndarray,
    width: int,
    height: int,
    resize: int,
    freq_window_us: int,
    max_hz: float,
) -> np.ndarray:
    frame = np.full((height, width), 255, dtype=np.uint8)

    if points.size == 0:
        return cv2.resize(
            frame, (width * resize, height * resize), interpolation=cv2.INTER_NEAREST
        )

    safe_window_us = max(int(freq_window_us), 1)
    safe_max_hz = max(float(max_hz), 1e-9)

    latest_t = int(points[:, 2].max())
    recent = points[points[:, 2] >= (latest_t - safe_window_us)]
    if recent.size == 0:
        return cv2.resize(
            frame, (width * resize, height * resize), interpolation=cv2.INTER_NEAREST
        )

    xs = recent[:, 0].astype(np.int32)
    ys = recent[:, 1].astype(np.int32)

    counts = np.zeros((height, width), dtype=np.float32)
    np.add.at(counts, (ys, xs), 1.0)

    window_sec = safe_window_us / 1_000_000.0
    freq_hz = counts / max(window_sec, 1e-9)
    freq_hz = np.clip(freq_hz, 0.0, safe_max_hz)

    intensity = 255.0 * (1.0 - freq_hz / safe_max_hz)
    frame = intensity.astype(np.uint8)

    return cv2.resize(
        frame, (width * resize, height * resize), interpolation=cv2.INTER_NEAREST
    )


def frequency_stats(points: np.ndarray, freq_window_us: int) -> tuple[float, float]:
    if points.size == 0:
        return 0.0, 0.0

    safe_window_us = max(int(freq_window_us), 1)
    latest_t = int(points[:, 2].max())
    recent = points[points[:, 2] >= (latest_t - safe_window_us)]
    if recent.size == 0:
        return 0.0, 0.0

    xs = recent[:, 0].astype(np.int32)
    ys = recent[:, 1].astype(np.int32)

    counts = np.zeros((H, W), dtype=np.float32)
    np.add.at(counts, (ys, xs), 1.0)

    window_sec = safe_window_us / 1_000_000.0
    freq_hz = counts / max(window_sec, 1e-9)

    max_freq = float(freq_hz.max()) if freq_hz.size > 0 else 0.0
    active = freq_hz[freq_hz > 0]
    mean_active = float(active.mean()) if active.size > 0 else 0.0
    return max_freq, mean_active


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(
        description=(
            "Collect GenX320 events, apply plot3d-style filtering to a live 33ms "
            "x/y/t window, and preview with OpenCV"
        )
    )
    ap.add_argument("--port", default="/dev/openmvcam")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=3.0)
    ap.add_argument(
        "--point-size",
        type=float,
        default=4.0,
        help="Unused in OpenCV preview mode; kept for CLI compatibility.",
    )
    ap.add_argument(
        "--time-scale",
        type=float,
        default=0.05,
        help="Unused in OpenCV preview mode; kept for CLI compatibility.",
    )
    ap.add_argument(
        "--filter-time",
        type=int,
        default=None,
        metavar="R",
        help="Match points with a temporal neighbor in [t-R, t+R] within the live 33ms window.",
    )
    ap.add_argument(
        "--filter-spatial",
        type=int,
        default=None,
        metavar="R",
        help="Match points with a spatial neighbor within XY radius R inside the live 33ms window.",
    )
    ap.add_argument(
        "--hide-noise",
        action="store_true",
        help="Render only points that survive all active filters.",
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Preview FPS for the OpenCV window.",
    )
    ap.add_argument(
        "--resize",
        type=int,
        default=2,
        help="Integer upscale factor for display only.",
    )
    ap.add_argument(
        "--window-ms",
        type=float,
        default=33.0,
        help="Live accumulation window in milliseconds.",
    )
    ap.add_argument(
        "--pixel-cooldown-ms",
        type=float,
        default=0.0,
        help="Reject same-pixel events occurring within this cooldown (ms); 0 disables.",
    )
    ap.add_argument(
        "--show-frequency-map",
        action="store_true",
        help="Render per-pixel firing frequency map instead of filter-color preview.",
    )
    ap.add_argument(
        "--freq-map-window-ms",
        type=float,
        default=5.0,
        help="Temporal window in ms for per-pixel frequency estimation.",
    )
    ap.add_argument(
        "--freq-map-max-hz",
        type=float,
        default=1000.0,
        help="Frequency mapped to black in the grayscale frequency view.",
    )
    ap.add_argument(
        "--save",
        nargs="?",
        const=str(script_dir / "filtered_events_xyt.tsv"),
        metavar="PATH",
        help="Save filtered x/y/t points after capture using fixed-size time slices.",
    )
    args = ap.parse_args()

    if args.filter_time is not None and args.filter_time < 0:
        raise ValueError("--filter-time must be >= 0")
    if args.filter_spatial is not None and args.filter_spatial < 0:
        raise ValueError("--filter-spatial must be >= 0")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.resize < 1:
        raise ValueError("--resize must be >= 1")
    if args.window_ms <= 0:
        raise ValueError("--window-ms must be > 0")
    if args.pixel_cooldown_ms < 0:
        raise ValueError("--pixel-cooldown-ms must be >= 0")
    if args.freq_map_window_ms <= 0:
        raise ValueError("--freq-map-window-ms must be > 0")
    if args.freq_map_max_hz <= 0:
        raise ValueError("--freq-map-max-hz must be > 0")

    window_us = int(args.window_ms * 1000.0)
    cooldown_us = int(round(args.pixel_cooldown_ms * 1000.0))
    preview_dt = 1.0 / args.fps

    ser = serial.Serial(
        args.port,
        baudrate=args.baud,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
        timeout=args.timeout,
    )
    ser.reset_input_buffer()

    total_packets = 0
    total_events = 0
    origin_us = None
    preview_buffer: list[tuple[int, np.ndarray]] = []
    all_points: list[np.ndarray] = []
    last_accept_ts_us = np.full((H, W), np.iinfo(np.int64).min // 4, dtype=np.int64)
    cooldown_input_total = 0
    cooldown_kept_total = 0

    t0 = time.monotonic()
    last_print = t0
    last_preview_time = t0

    print(f"[INFO] Listening on {args.port} @ {args.baud}")
    print("[INFO] Collecting continuously (press q/Esc to stop)...")
    print(f"[INFO] Live accumulation window: {args.window_ms:.1f} ms")
    if cooldown_us > 0:
        print(f"[INFO] Pixel cooldown enabled: {args.pixel_cooldown_ms:.3f} ms")

    try:
        while True:
            now = time.monotonic()
            read_until_magic(ser)
            header_rest = read_exactly(ser, struct.calcsize(HEADER_FMT))
            event_count, payload_len = struct.unpack(HEADER_FMT, header_rest)

            expected_len = event_count * 6 * 2
            if payload_len != expected_len:
                raise RuntimeError(
                    f"Invalid payload length: got {payload_len}, expected {expected_len}"
                )

            payload = read_exactly(ser, payload_len)
            events = np.frombuffer(payload, dtype=np.uint16).reshape((event_count, 6)).copy()

            cooldown_input_total += len(events)
            if cooldown_us > 0:
                events = apply_pixel_cooldown(events, last_accept_ts_us, cooldown_us)
            cooldown_kept_total += len(events)

            total_packets += 1
            total_events += event_count

            points_chunk, origin_us = extract_xyt_points(events, W, H, origin_us)
            if len(points_chunk) > 0:
                chunk_max_t = int(points_chunk[:, 2].max())
                preview_buffer.append((chunk_max_t, points_chunk))
                all_points.append(points_chunk)

                cutoff = chunk_max_t - window_us
                preview_buffer = [
                    (max_t, chunk)
                    for (max_t, chunk) in preview_buffer
                    if max_t >= cutoff
                ]

            now = time.monotonic()
            if now - last_print >= 1.0:
                elapsed = now - t0
                if cooldown_input_total > 0:
                    cooldown_keep_frac = cooldown_kept_total / cooldown_input_total
                else:
                    cooldown_keep_frac = 1.0

                if cooldown_us > 0:
                    print(
                        f"[INFO] elapsed={elapsed:.1f}s  packets={total_packets}  "
                        f"events={total_events}  cooldown_kept={cooldown_keep_frac:.3f}"
                    )
                else:
                    print(
                        f"[INFO] elapsed={elapsed:.1f}s  packets={total_packets}  "
                        f"events={total_events}"
                    )
                last_print = now

            if now - last_preview_time >= preview_dt:
                if preview_buffer:
                    preview_points = np.concatenate(
                        [chunk for (_, chunk) in preview_buffer], axis=0
                    )
                    latest_t = int(preview_points[:, 2].max())
                    preview_points = preview_points[
                        preview_points[:, 2] >= latest_t - window_us
                    ]
                else:
                    preview_points = np.zeros((0, 3), dtype=np.int64)

                render_points, filter_mode, temporal_mask, spatial_mask, _ = apply_filters(
                    preview_points,
                    filter_time=args.filter_time,
                    filter_spatial=args.filter_spatial,
                    hide_noise=args.hide_noise,
                    width=W,
                    height=H,
                )

                if args.show_frequency_map:
                    freq_window_us = int(args.freq_map_window_ms * 1000.0)
                    frame = points_to_frequency_frame(
                        preview_points,
                        width=W,
                        height=H,
                        resize=args.resize,
                        freq_window_us=freq_window_us,
                        max_hz=args.freq_map_max_hz,
                    )
                    max_pixel_hz, mean_active_hz = frequency_stats(
                        preview_points, freq_window_us
                    )
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    cv2.putText(
                        frame,
                        f"freq window: {args.freq_map_window_ms:.2f} ms",
                        (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (32, 32, 32),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"max pixel freq: {max_pixel_hz:.1f} Hz",
                        (8, 44),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (32, 32, 32),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"mean active freq: {mean_active_hz:.1f} Hz",
                        (8, 66),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (32, 32, 32),
                        1,
                        cv2.LINE_AA,
                    )
                else:
                    frame = points_to_preview_frame(
                        render_points,
                        filter_mode,
                        temporal_mask,
                        spatial_mask,
                        width=W,
                        height=H,
                        resize=args.resize,
                    )
                cv2.imshow("GenX320 filter3d preview", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

                last_preview_time = now

    finally:
        ser.close()
        cv2.destroyAllWindows()

    if args.save and all_points:
        points = np.concatenate(all_points, axis=0)
        render_points = apply_filters_by_slices(
            points,
            filter_time=args.filter_time,
            filter_spatial=args.filter_spatial,
            hide_noise=args.hide_noise,
            width=W,
            height=H,
            slice_us=window_us,
        )
        save_path = Path(args.save)
        if not save_path.is_absolute() and save_path.parent == Path("."):
            save_path = script_dir / save_path
        save_xyt_points(render_points, str(save_path))
        print(f"[INFO] Saved {len(render_points)} x/y/t points to {save_path}")


if __name__ == "__main__":
    main()