#!/usr/bin/env python3
"""Autoplay viewer for one accumulated event frame from a raw event HDF5 file.

The viewer advances automatically and wraps from the end back to the start.
Press Space to pause/resume. While paused, A/D step backward/forward.

Requires Python 3 with NumPy, h5py, and OpenCV (cv2), as did the reference
viewer.

Examples:
    python3 view_raw_h5_event_frame.py recording_raw_events.h5
    python3 view_raw_h5_event_frame.py recording_dir --window-ms 50 --step-ms 20
    python3 view_raw_h5_event_frame.py recording_raw_events.h5 --mode polarity
    python3 view_raw_h5_event_frame.py recording_dir --video
    python3 view_raw_h5_event_frame.py recording_dir --video --out_dir /tmp/videos
    python3 view_raw_h5_event_frame.py --top_dir /path/to/top --video
    python3 view_raw_h5_event_frame.py recording_raw_events.h5 \
        --spatial-radius 2 --spatial-min-neighbors 1

Spatial filtering is performed independently for every accumulation window.
All events at a pixel are removed unless that pixel has at least
``--spatial-min-neighbors`` other active pixels within a Chebyshev radius of
``--spatial-radius``. The center pixel itself is not counted as a neighbor.

Controls:
    Space       pause/resume
    A / D       previous/next frame (and pause)
    R           restart from the first frame
    F           toggle spatial filtering (radius 1 if not configured)
    [ / ]       decrease/increase the spatial-filter radius
    Q / Esc     quit
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np


X_KEYS = ["/events/x", "events/x", "/event/x", "event/x", "/x", "x"]
Y_KEYS = ["/events/y", "events/y", "/event/y", "event/y", "/y", "y"]
T_KEYS = [
    "/events/t_us", "events/t_us",
    "/events/t", "events/t",
    "/events/ts", "events/ts",
    "/events/timestamp", "events/timestamp",
    "/events/timestamps", "events/timestamps",
    "/event/t_us", "event/t_us",
    "/event/t", "event/t",
    "/event/ts", "event/ts",
    "/t_us", "t_us",
    "/t", "t",
    "/ts", "ts",
    "/timestamp", "timestamp",
    "/timestamps", "timestamps",
]
P_KEYS = [
    "/events/p", "events/p",
    "/events/polarity", "events/polarity",
    "/event/p", "event/p",
    "/event/polarity", "event/polarity",
    "/p", "p",
    "/polarity", "polarity",
]


def resolve_h5_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(path.glob("*_raw_events.h5"))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            choices = "\n".join(f"  {candidate}" for candidate in candidates)
            raise SystemExit(
                f"Multiple *_raw_events.h5 files found:\n{choices}\n"
                "Pass the desired file explicitly."
            )
        raise SystemExit(f"No *_raw_events.h5 file found in: {path}")
    raise SystemExit(f"Path does not exist: {path}")


def print_h5_tree(h5_path: Path) -> None:
    with h5py.File(h5_path, "r") as h5_file:
        print(f"\nH5 tree: {h5_path}\n")

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"DATASET /{name}  shape={obj.shape}  dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"GROUP   /{name}")

        h5_file.visititems(visitor)
        print()


def get_dataset(h5_file: h5py.File, key: str):
    try:
        obj = h5_file[key]
    except KeyError:
        return None
    return obj if isinstance(obj, h5py.Dataset) else None


def find_dataset(
    h5_file: h5py.File,
    keys: Sequence[str],
    aliases: set[str],
    label: str,
    required: bool = True,
):
    for key in keys:
        dataset = get_dataset(h5_file, key)
        if dataset is not None:
            return key, dataset

    matches = []

    def visitor(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        basename = name.split("/")[-1].lower()
        if basename not in aliases:
            return
        lower_name = name.lower()
        score = 0.0
        if lower_name.startswith("events/") or "/events/" in f"/{lower_name}/":
            score += 10.0
        if lower_name.startswith("event/") or "/event/" in f"/{lower_name}/":
            score += 5.0
        score -= 0.001 * len(name)
        matches.append((score, name, obj))

    h5_file.visititems(visitor)
    if matches:
        matches.sort(reverse=True, key=lambda match: match[0])
        _, name, dataset = matches[0]
        return "/" + name, dataset

    if required:
        raise KeyError(
            f"Could not find an H5 dataset for {label!r}. "
            "Run with --list to inspect the file."
        )
    return None, None


def infer_time_unit(t_key: str, t_raw: np.ndarray, requested_unit: str) -> str:
    if requested_unit != "auto":
        return requested_unit

    lower_key = t_key.lower()
    if "t_us" in lower_key or "usec" in lower_key or "micro" in lower_key:
        return "us"
    if "t_ns" in lower_key or "nsec" in lower_key or "nano" in lower_key:
        return "ns"
    if "t_ms" in lower_key or "msec" in lower_key or "milli" in lower_key:
        return "ms"

    if np.issubdtype(t_raw.dtype, np.floating):
        duration = float(t_raw[-1] - t_raw[0]) if len(t_raw) > 1 else 0.0
        if duration < 10_000.0:
            return "s"
    return "us"


def convert_timestamps_to_us(t_raw: np.ndarray, unit: str) -> np.ndarray:
    timestamps = np.asarray(t_raw).reshape(-1).astype(np.float64)
    factors = {"ns": 0.001, "us": 1.0, "ms": 1_000.0, "s": 1_000_000.0}
    return np.rint(timestamps * factors[unit]).astype(np.int64)


def infer_sensor_size(
    h5_file: h5py.File,
    x_data,
    y_data,
    width_arg: Optional[int],
    height_arg: Optional[int],
) -> Tuple[int, int]:
    width = width_arg
    height = height_arg

    if width is None:
        for key in ["width", "image_width", "sensor_width", "W"]:
            if key in h5_file.attrs:
                width = int(h5_file.attrs[key])
                break
    if height is None:
        for key in ["height", "image_height", "sensor_height", "H"]:
            if key in h5_file.attrs:
                height = int(h5_file.attrs[key])
                break

    if width is None:
        width = int(np.max(x_data)) + 1
    if height is None:
        height = int(np.max(y_data)) + 1
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid sensor size: width={width}, height={height}")
    return width, height


def select_polarity(x, y, p, mode: str):
    """Apply ON/OFF selection before spatial filtering and accumulation."""
    if p is None or mode in {"count", "polarity"}:
        return x, y, p

    polarity = np.asarray(p)
    keep = polarity > 0 if mode == "on" else polarity <= 0
    return x[keep], y[keep], polarity[keep]


def spatial_filter_events(
    x: np.ndarray,
    y: np.ndarray,
    p: Optional[np.ndarray],
    height: int,
    width: int,
    radius: int,
    min_neighbors: int,
):
    """Remove events whose pixel lacks nearby active pixels in this window.

    Multiple events at the same coordinate still form one active pixel, and the
    center pixel is excluded from its own neighbor count. A square/Chebyshev
    neighborhood is used, so radius 1 examines the surrounding 8 pixels.
    """
    if radius <= 0 or len(x) == 0:
        return x, y, p

    occupancy = np.zeros((height, width), dtype=np.float32)
    occupancy[y, x] = 1.0
    kernel_size = 2 * radius + 1
    neighbor_counts = cv2.boxFilter(
        occupancy,
        ddepth=-1,
        ksize=(kernel_size, kernel_size),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    neighbor_counts -= occupancy  # Do not count the active pixel itself.
    active_pixel_survives = neighbor_counts >= float(min_neighbors)
    keep = active_pixel_survives[y, x]
    return x[keep], y[keep], p[keep] if p is not None else None


def percentile_scale(values: np.ndarray, percentile: float) -> float:
    nonzero = values[values > 0]
    if len(nonzero) == 0:
        return 1.0
    return max(float(np.percentile(nonzero, percentile)), 1.0)


def render_accumulation(
    x,
    y,
    p,
    height: int,
    width: int,
    mode: str,
    percentile: float,
) -> np.ndarray:
    """Render count/ON/OFF/polarity data as a BGR image."""
    if len(x) == 0:
        return np.full((height, width, 3), 255, dtype=np.uint8)

    if mode == "polarity" and p is not None:
        polarity = np.asarray(p)
        on_counts = np.zeros((height, width), dtype=np.uint32)
        off_counts = np.zeros((height, width), dtype=np.uint32)
        np.add.at(on_counts, (y[polarity > 0], x[polarity > 0]), 1)
        np.add.at(off_counts, (y[polarity <= 0], x[polarity <= 0]), 1)
        scale = percentile_scale(
            np.concatenate((on_counts[on_counts > 0], off_counts[off_counts > 0])),
            percentile,
        )
        on_strength = np.clip(on_counts / scale, 0.0, 1.0)
        off_strength = np.clip(off_counts / scale, 0.0, 1.0)

        # ON is red, OFF is blue, and equal overlap is magenta.
        strongest = np.maximum(on_strength, off_strength)
        background = 255.0 * (1.0 - strongest)
        image = np.empty((height, width, 3), dtype=np.float32)
        image[:, :, 0] = background + 255.0 * off_strength  # blue
        image[:, :, 1] = background
        image[:, :, 2] = background + 255.0 * on_strength   # red
        return np.clip(image, 0, 255).astype(np.uint8)

    counts = np.zeros((height, width), dtype=np.uint32)
    np.add.at(counts, (y, x), 1)
    scale = percentile_scale(counts, percentile)
    strength = np.clip(counts.astype(np.float32) / scale, 0.0, 1.0)
    image = np.full((height, width, 3), 255.0, dtype=np.float32)

    if mode == "on" and p is not None:
        image[:, :, 0] -= 255.0 * strength
        image[:, :, 1] -= 255.0 * strength  # red
    elif mode == "off" and p is not None:
        image[:, :, 1] -= 255.0 * strength
        image[:, :, 2] -= 255.0 * strength  # blue
    else:
        image -= (255.0 * strength)[:, :, None]  # black count image
    return np.clip(image, 0, 255).astype(np.uint8)


def make_frame(
    frame_idx: int,
    num_frames: int,
    t_us: int,
    t0_us: int,
    event_t_us: np.ndarray,
    x_data,
    y_data,
    p_data,
    height: int,
    width: int,
    window_ms: float,
    step_ms: float,
    mode: str,
    percentile: float,
    display_scale: float,
    playing: bool,
    spatial_radius: int,
    spatial_min_neighbors: int,
    include_info_overlay: bool,
) -> np.ndarray:
    window_us = int(round(window_ms * 1_000.0))
    i0 = int(np.searchsorted(event_t_us, t_us - window_us, side="left"))
    i1 = int(np.searchsorted(event_t_us, t_us, side="right"))

    x = np.asarray(x_data[i0:i1], dtype=np.int64)
    y = np.asarray(y_data[i0:i1], dtype=np.int64)
    p = np.asarray(p_data[i0:i1]) if p_data is not None else None

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x[valid]
    y = y[valid]
    p = p[valid] if p is not None else None
    x, y, p = select_polarity(x, y, p, mode)
    raw_count = len(x)
    x, y, p = spatial_filter_events(
        x=x,
        y=y,
        p=p,
        height=height,
        width=width,
        radius=spatial_radius,
        min_neighbors=spatial_min_neighbors,
    )

    image = render_accumulation(
        x=x,
        y=y,
        p=p,
        height=height,
        width=width,
        mode=mode,
        percentile=percentile,
    )
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if display_scale != 1.0:
        image = cv2.resize(
            image,
            None,
            fx=display_scale,
            fy=display_scale,
            interpolation=cv2.INTER_NEAREST,
        )

    elapsed_s = (t_us - t0_us) / 1_000_000.0
    state = "PLAYING" if playing else "PAUSED"
    filter_text = (
        f"r={spatial_radius}, min={spatial_min_neighbors}"
        if spatial_radius > 0
        else "off"
    )
    if not include_info_overlay:
        return image

    lines = [
        f"{state}  {frame_idx + 1}/{num_frames}  t={elapsed_s:.3f}s",
        f"win={window_ms:g}ms  step={step_ms:g}ms  mode={mode}",
        f"spatial={filter_text}  events={len(x)}/{raw_count}",
        "Space play/pause  A/D step  F filter  [/] radius  Q quit",
    ]
    bar = np.zeros((86, image.shape[1], 3), dtype=np.uint8)
    for line_idx, line in enumerate(lines):
        cv2.putText(
            bar,
            line,
            (6, 18 + line_idx * 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return np.vstack((bar, image))


def make_display_frame(include_info_overlay: bool, **kwargs) -> np.ndarray:
    """Build a display frame and apply the final nearest-neighbor upscaling."""
    frame = make_frame(include_info_overlay=include_info_overlay, **kwargs)
    return cv2.resize(
        frame,
        (frame.shape[1] * 2, frame.shape[0] * 2),
        interpolation=cv2.INTER_NEAREST,
    )


def default_video_output_path(
    input_path: Path,
    h5_path: Path,
    out_dir: Optional[Path] = None,
) -> Path:
    if input_path.is_file():
        output_name = f"{h5_path.stem}_event_frames.mp4"
    elif input_path.is_dir():
        output_name = f"{input_path.name}_event_frames.mp4"
    else:
        output_name = f"{h5_path.stem}_event_frames.mp4"

    if out_dir is not None:
        return out_dir / output_name
    if input_path.is_dir():
        return input_path / output_name
    return h5_path.with_name(output_name)


def find_top_level_recordings(top_dir: Path) -> list[Path]:
    """Find only direct child recording directories (no recursion)."""
    children = [
        child
        for child in sorted(top_dir.iterdir())
        if child.is_dir() and child.name.startswith("recording_")
    ]
    if children:
        return children
    return [child for child in sorted(top_dir.iterdir()) if child.is_dir()]


def write_video(
    output_path: Path,
    frame_times: np.ndarray,
    t0_us: int,
    event_t_us: np.ndarray,
    x_data,
    y_data,
    p_data,
    height: int,
    width: int,
    args,
) -> None:
    fps = args.video_fps if args.video_fps is not None else (1000.0 / args.step_ms)
    if fps <= 0:
        raise RuntimeError("Video FPS must be positive.")

    first_frame = make_display_frame(
        include_info_overlay=False,
        frame_idx=0,
        num_frames=len(frame_times),
        t_us=int(frame_times[0]),
        t0_us=t0_us,
        event_t_us=event_t_us,
        x_data=x_data,
        y_data=y_data,
        p_data=p_data,
        height=height,
        width=width,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        mode=args.mode,
        percentile=args.percentile,
        display_scale=args.display_scale,
        playing=True,
        spatial_radius=args.spatial_radius,
        spatial_min_neighbors=args.spatial_min_neighbors,
    )
    frame_h, frame_w = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open video writer for: {output_path}. "
            "Check OpenCV/FFmpeg codec support."
        )

    print(f"[INFO] writing video: {output_path}")
    print(f"[INFO] video fps: {fps:.6g}")
    writer.write(first_frame)
    for frame_idx in range(1, len(frame_times)):
        frame = make_display_frame(
            include_info_overlay=False,
            frame_idx=frame_idx,
            num_frames=len(frame_times),
            t_us=int(frame_times[frame_idx]),
            t0_us=t0_us,
            event_t_us=event_t_us,
            x_data=x_data,
            y_data=y_data,
            p_data=p_data,
            height=height,
            width=width,
            window_ms=args.window_ms,
            step_ms=args.step_ms,
            mode=args.mode,
            percentile=args.percentile,
            display_scale=args.display_scale,
            playing=True,
            spatial_radius=args.spatial_radius,
            spatial_min_neighbors=args.spatial_min_neighbors,
        )
        writer.write(frame)
    writer.release()
    print(f"[INFO] wrote {len(frame_times)} frames")


def parse_args():
    parser = argparse.ArgumentParser(
        description="View accumulated event frames interactively or export them to MP4."
    )
    parser.add_argument(
        "h5_or_session_dir",
        nargs="?",
        default=None,
        help="Raw event H5 file or directory containing one *_raw_events.h5 file.",
    )
    parser.add_argument(
        "--top_dir",
        type=str,
        default=None,
        help=(
            "Batch mode input. Generate one video per direct child recording "
            "directory (non-recursive)."
        ),
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Export MP4 video instead of opening the interactive viewer.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Optional video FPS override. Default: 1000 / --step-ms.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help=(
            "Directory for generated MP4 files (video mode only). "
            "Default: next to each input recording."
        ),
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument(
        "--window-ms",
        type=float,
        default=33.0,
        help="Event accumulation window in milliseconds. Default: 33.",
    )
    parser.add_argument(
        "--step-ms",
        type=float,
        default=33.333,
        help="Recording-time step between displayed frames. Default: 33.333.",
    )
    parser.add_argument(
        "--mode",
        choices=["count", "polarity", "signed", "on", "off"],
        default="count",
        help=(
            "count=all events in black; polarity/signed=ON red and OFF blue; "
            "on/off=one polarity only. Default: count."
        ),
    )
    parser.add_argument(
        "--spatial-radius",
        type=int,
        default=0,
        help="Neighbor radius in pixels; 0 disables filtering. Default: 0.",
    )
    parser.add_argument(
        "--spatial-min-neighbors",
        type=int,
        default=1,
        help="Required other active pixels inside the radius. Default: 1.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.5,
        help="Count normalization percentile. Default: 99.5.",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=1.5,
        help="Nearest-neighbor display scale. Default: 1.5.",
    )
    parser.add_argument(
        "--playback-delay-ms",
        type=int,
        default=None,
        help=(
            "Wall-clock delay per frame. Default: rounded --step-ms, which is "
            "approximately real-time playback."
        ),
    )
    parser.add_argument(
        "--time-unit",
        choices=["auto", "ns", "us", "ms", "s"],
        default="auto",
        help="Raw timestamp unit. Default: auto.",
    )
    parser.add_argument(
        "--start-at-zero",
        action="store_true",
        help="Start at the first event instead of waiting for one full window.",
    )
    parser.add_argument("--list", action="store_true", help="Print the H5 tree and exit.")
    args = parser.parse_args()

    if args.window_ms <= 0:
        parser.error("--window-ms must be positive")
    if args.step_ms <= 0:
        parser.error("--step-ms must be positive")
    if args.spatial_radius < 0:
        parser.error("--spatial-radius cannot be negative")
    if args.spatial_min_neighbors < 1:
        parser.error("--spatial-min-neighbors must be at least 1")
    if not 0 < args.percentile <= 100:
        parser.error("--percentile must be in (0, 100]")
    if args.display_scale <= 0:
        parser.error("--display-scale must be positive")
    if args.playback_delay_ms is not None and args.playback_delay_ms < 1:
        parser.error("--playback-delay-ms must be at least 1")
    if args.video_fps is not None and args.video_fps <= 0:
        parser.error("--video-fps must be positive")

    has_single_input = args.h5_or_session_dir is not None
    has_top_input = args.top_dir is not None
    if has_single_input == has_top_input:
        parser.error("Provide exactly one of positional h5_or_session_dir or --top_dir")

    if args.top_dir is not None and not args.video:
        parser.error("--top_dir requires --video")

    if args.out_dir is not None and not args.video:
        parser.error("--out_dir is only supported with --video")

    if args.top_dir is not None and args.list:
        parser.error("--list is only supported with a single input path")

    if args.mode == "signed":
        args.mode = "polarity"
    return args


def run_for_recording(h5_path: Path, args, video_output_path: Optional[Path]) -> None:
    with h5py.File(h5_path, "r") as h5_file:
        x_key, x_dataset = find_dataset(h5_file, X_KEYS, {"x"}, "x")
        y_key, y_dataset = find_dataset(h5_file, Y_KEYS, {"y"}, "y")
        t_key, t_dataset = find_dataset(
            h5_file,
            T_KEYS,
            {"t", "ts", "t_us", "timestamp", "timestamps"},
            "timestamp",
        )
        p_key, p_dataset = find_dataset(
            h5_file, P_KEYS, {"p", "polarity"}, "polarity", required=False
        )

        lengths = [len(x_dataset), len(y_dataset), len(t_dataset)]
        if p_dataset is not None:
            lengths.append(len(p_dataset))
        if len(set(lengths)) != 1:
            raise RuntimeError(f"Event dataset lengths differ: {lengths}")
        if len(t_dataset) == 0:
            raise RuntimeError("The H5 file contains no events.")

        print(f"[INFO] H5: {h5_path}")
        print(f"[INFO] x: {x_key} shape={x_dataset.shape} dtype={x_dataset.dtype}")
        print(f"[INFO] y: {y_key} shape={y_dataset.shape} dtype={y_dataset.dtype}")
        print(f"[INFO] t: {t_key} shape={t_dataset.shape} dtype={t_dataset.dtype}")
        if p_dataset is None:
            print("[INFO] p: not found; polarity/on/off modes fall back to count")
        else:
            print(f"[INFO] p: {p_key} shape={p_dataset.shape} dtype={p_dataset.dtype}")

        t_raw = t_dataset[:]
        time_unit = infer_time_unit(t_key, t_raw, args.time_unit)
        event_t_us = convert_timestamps_to_us(t_raw, time_unit)
        backward_jumps = np.flatnonzero(np.diff(event_t_us) < 0)
        if len(backward_jumps) > 0:
            print(
                f"[WARN] Found {len(backward_jumps)} backward timestamp jumps; "
                "sorting all events for visualization."
            )
            order = np.argsort(event_t_us, kind="stable")
            event_t_us = event_t_us[order]
            x_data = np.asarray(x_dataset[:])[order]
            y_data = np.asarray(y_dataset[:])[order]
            p_data = np.asarray(p_dataset[:])[order] if p_dataset is not None else None
        else:
            x_data = x_dataset
            y_data = y_dataset
            p_data = p_dataset

        width, height = infer_sensor_size(
            h5_file,
            x_data=x_data,
            y_data=y_data,
            width_arg=args.width,
            height_arg=args.height,
        )
        t0_us = int(event_t_us[0])
        t_end_us = int(event_t_us[-1])
        window_us = int(round(args.window_ms * 1_000.0))
        step_us = int(round(args.step_ms * 1_000.0))
        first_t_us = t0_us if args.start_at_zero else min(t0_us + window_us, t_end_us)
        frame_times = np.arange(first_t_us, t_end_us + 1, step_us, dtype=np.int64)
        if len(frame_times) == 0:
            frame_times = np.array([t_end_us], dtype=np.int64)

        playback_delay_ms = (
            args.playback_delay_ms
            if args.playback_delay_ms is not None
            else max(1, int(round(args.step_ms)))
        )
        duration_s = (t_end_us - t0_us) / 1_000_000.0
        print(f"[INFO] timestamp unit: {time_unit}")
        print(f"[INFO] sensor size: {width}x{height}")
        print(f"[INFO] duration: {duration_s:.3f}s; frames: {len(frame_times)}")
        print(
            f"[INFO] window={args.window_ms:g}ms; step={args.step_ms:g}ms; "
            f"mode={args.mode}; playback delay={playback_delay_ms}ms"
        )
        print(
            f"[INFO] spatial filter: radius={args.spatial_radius}, "
            f"minimum neighbors={args.spatial_min_neighbors}"
        )

        if args.video:
            if video_output_path is None:
                raise RuntimeError("video output path was not provided")
            write_video(
                output_path=video_output_path,
                frame_times=frame_times,
                t0_us=t0_us,
                event_t_us=event_t_us,
                x_data=x_data,
                y_data=y_data,
                p_data=p_data,
                height=height,
                width=width,
                args=args,
            )
            return

        print("[INFO] controls: Space pause/resume; A/D step; F filter; [/] radius; Q quit")
        window_name = "raw H5 single event-frame viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        frame_idx = 0
        playing = True
        spatial_radius = args.spatial_radius
        previous_nonzero_radius = max(1, spatial_radius)

        while True:
            frame = make_display_frame(
                include_info_overlay=True,
                frame_idx=frame_idx,
                num_frames=len(frame_times),
                t_us=int(frame_times[frame_idx]),
                t0_us=t0_us,
                event_t_us=event_t_us,
                x_data=x_data,
                y_data=y_data,
                p_data=p_data,
                height=height,
                width=width,
                window_ms=args.window_ms,
                step_ms=args.step_ms,
                mode=args.mode,
                percentile=args.percentile,
                display_scale=args.display_scale,
                playing=playing,
                spatial_radius=spatial_radius,
                spatial_min_neighbors=args.spatial_min_neighbors,
            )
            cv2.imshow(window_name, frame)
            key = cv2.waitKeyEx(playback_delay_ms if playing else 30)
            advance_automatically = playing

            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" "):
                playing = not playing
                advance_automatically = False
            elif key in (ord("a"), ord("A")):
                playing = False
                frame_idx = (frame_idx - 1) % len(frame_times)
                advance_automatically = False
            elif key in (ord("d"), ord("D")):
                playing = False
                frame_idx = (frame_idx + 1) % len(frame_times)
                advance_automatically = False
            elif key in (ord("r"), ord("R")):
                frame_idx = 0
                advance_automatically = False
            elif key in (ord("f"), ord("F")):
                if spatial_radius > 0:
                    previous_nonzero_radius = spatial_radius
                    spatial_radius = 0
                else:
                    spatial_radius = previous_nonzero_radius
                advance_automatically = False
            elif key in (ord("["), ord("{")):
                spatial_radius = max(0, spatial_radius - 1)
                if spatial_radius > 0:
                    previous_nonzero_radius = spatial_radius
                advance_automatically = False
            elif key in (ord("]"), ord("}")):
                spatial_radius += 1
                previous_nonzero_radius = spatial_radius
                advance_automatically = False

            if advance_automatically:
                frame_idx = (frame_idx + 1) % len(frame_times)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None

    if args.top_dir is not None:
        top_dir = Path(args.top_dir).expanduser().resolve()
        if not top_dir.is_dir():
            raise SystemExit(f"Top directory does not exist or is not a directory: {top_dir}")

        recording_dirs = find_top_level_recordings(top_dir)
        if len(recording_dirs) == 0:
            raise SystemExit(f"No direct child recording directories found in: {top_dir}")

        print(f"[INFO] batch top_dir: {top_dir}")
        print(f"[INFO] found {len(recording_dirs)} direct child recording directories")
        failures: list[tuple[Path, str]] = []

        for idx, recording_dir in enumerate(recording_dirs, start=1):
            print()
            print("=" * 80)
            print(
                f"[INFO] processing recording {idx}/{len(recording_dirs)}: "
                f"{recording_dir.name}"
            )
            try:
                h5_path = resolve_h5_path(str(recording_dir))
            except SystemExit as exc:
                message = str(exc)
                print(f"[WARN] skipping {recording_dir}: {message}")
                failures.append((recording_dir, message))
                continue

            if out_dir is not None:
                output_path = out_dir / f"{recording_dir.name}_event_frames.mp4"
            else:
                output_path = recording_dir / f"{recording_dir.name}_event_frames.mp4"
            try:
                run_for_recording(
                    h5_path=h5_path,
                    args=args,
                    video_output_path=output_path,
                )
            except Exception as exc:  # pylint: disable=broad-except
                message = repr(exc)
                print(f"[ERROR] failed for {recording_dir}: {message}")
                failures.append((recording_dir, message))

        if failures:
            print()
            print("[ERROR] Some recordings failed:")
            for directory, reason in failures:
                print(f"  {directory}: {reason}")
            raise SystemExit(1)

        print()
        print("[INFO] Batch video export complete.")
        return

    input_path = Path(args.h5_or_session_dir).expanduser().resolve()
    h5_path = resolve_h5_path(str(input_path))

    if args.list:
        print_h5_tree(h5_path)
        return

    video_output_path = (
        default_video_output_path(input_path, h5_path, out_dir=out_dir)
        if args.video
        else None
    )
    run_for_recording(h5_path=h5_path, args=args, video_output_path=video_output_path)


if __name__ == "__main__":
    main()
