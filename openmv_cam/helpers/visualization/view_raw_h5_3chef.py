#!/usr/bin/env python3
"""
view_raw_h5_3chef.py

Interactive viewer for raw OpenMV/Event-camera H5 files.

Shows 3 event accumulation images next to each other:

    [t - 50 ms, t]    [t - 250 ms, t]    [t - 1500 ms, t]

Controls:
    d       next frame
    a       previous frame
    q/esc   quit

No loop-over:
    - pressing d at the last frame stays at the last frame
    - pressing a at the first frame stays at the first frame

Examples:
    python3 view_raw_h5_3chef.py recording_20260626_xxx_raw_events.h5

    # You can also pass the session directory;
    # it will search for *_raw_events.h5 inside:
    python3 view_raw_h5_3chef.py /home/dyros/Data/jg_data/bags/recording_xxx/

    # Inspect H5 structure:
    python3 view_raw_h5_3chef.py recording_xxx_raw_events.h5 --list

    # Faster/slower traversal granularity:
    python3 view_raw_h5_3chef.py recording_xxx_raw_events.h5 --step-ms 20
    python3 view_raw_h5_3chef.py recording_xxx_raw_events.h5 --step-ms 100
"""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


# ---------------------------------------------------------------------
# H5 discovery
# ---------------------------------------------------------------------

X_KEYS = [
    "/events/x", "events/x",
    "/event/x", "event/x",
    "/x", "x",
]

Y_KEYS = [
    "/events/y", "events/y",
    "/event/y", "event/y",
    "/y", "y",
]

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
            print("[ERROR] Multiple *_raw_events.h5 files found:")
            for c in candidates:
                print(f"  {c}")
            raise SystemExit("Pass the desired H5 file explicitly.")

        raise SystemExit(f"No *_raw_events.h5 file found in directory: {path}")

    raise SystemExit(f"Path does not exist: {path}")


def print_h5_tree(h5_path: Path) -> None:
    with h5py.File(h5_path, "r") as f:
        print(f"\nH5 tree: {h5_path}\n")

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"DATASET /{name}  shape={obj.shape}  dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"GROUP   /{name}")

        f.visititems(visitor)
        print("")


def get_dataset(h5_file: h5py.File, key: str):
    try:
        obj = h5_file[key]
    except KeyError:
        return None
    return obj if isinstance(obj, h5py.Dataset) else None


def find_dataset(h5_file: h5py.File, keys, aliases, label: str, required: bool = True):
    """
    Find a normal dataset, first by explicit path, then recursively by basename.
    """
    for key in keys:
        ds = get_dataset(h5_file, key)
        if ds is not None:
            return key, ds

    matches = []

    def visitor(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return

        base = name.split("/")[-1].lower()
        if base in aliases:
            score = 0
            lname = name.lower()

            if lname.startswith("events/") or "/events/" in f"/{lname}/":
                score += 10
            if lname.startswith("event/") or "/event/" in f"/{lname}/":
                score += 5

            score -= 0.001 * len(name)
            matches.append((score, name, obj))

    h5_file.visititems(visitor)

    if matches:
        matches.sort(reverse=True, key=lambda x: x[0])
        _, name, ds = matches[0]
        return "/" + name, ds

    if required:
        raise KeyError(
            f"Could not find H5 dataset for '{label}'. "
            f"Run this script with --list to inspect the file."
        )

    return None, None


def infer_time_unit(t_key: str, t_raw: np.ndarray, user_unit: str) -> str:
    """
    Infer timestamp unit. Your pipeline likely uses t_us, but this keeps it robust.
    """
    if user_unit != "auto":
        return user_unit

    low = t_key.lower()

    if "t_us" in low or "usec" in low or "micro" in low:
        return "us"
    if "t_ns" in low or "nsec" in low or "nano" in low:
        return "ns"
    if "t_ms" in low or "msec" in low or "milli" in low:
        return "ms"

    # Heuristic fallback.
    # If float timestamps span a plausible number of seconds, treat as seconds.
    if np.issubdtype(t_raw.dtype, np.floating):
        duration = float(t_raw[-1] - t_raw[0]) if len(t_raw) > 1 else 0.0
        if duration < 10_000.0:
            return "s"

    # Most likely for your raw event recorder.
    return "us"


def convert_timestamps_to_us(t_raw: np.ndarray, unit: str) -> np.ndarray:
    t = np.asarray(t_raw).reshape(-1)

    if unit == "ns":
        return np.round(t.astype(np.float64) / 1_000.0).astype(np.int64)
    if unit == "us":
        return np.round(t.astype(np.float64)).astype(np.int64)
    if unit == "ms":
        return np.round(t.astype(np.float64) * 1_000.0).astype(np.int64)
    if unit == "s":
        return np.round(t.astype(np.float64) * 1_000_000.0).astype(np.int64)

    raise ValueError(f"Unsupported timestamp unit: {unit}")


def infer_sensor_size(h5_file: h5py.File, x_ds, y_ds, width_arg, height_arg):
    """
    Prefer explicit CLI args. Then H5 attrs. Then max x/y + 1.
    """
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
        # For your GenX320 this should become 320.
        width = int(np.max(x_ds[:])) + 1

    if height is None:
        height = int(np.max(y_ds[:])) + 1

    return width, height


# ---------------------------------------------------------------------
# Event rendering
# ---------------------------------------------------------------------

def normalize_count_image(acc: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """
    Event-count image: white background, events dark.
    """
    if acc.size == 0 or np.max(acc) <= 0:
        return np.full(acc.shape, 255, dtype=np.uint8)

    nonzero = acc[acc > 0]
    scale = np.percentile(nonzero, percentile) if len(nonzero) > 0 else 1.0
    scale = max(float(scale), 1.0)

    img = np.clip(acc.astype(np.float32), 0, scale)
    img = img / scale * 255.0

    # Invert: background becomes white, high event count becomes black.
    img = 255.0 - img

    return img.astype(np.uint8)


def normalize_signed_image(acc: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """
    Signed polarity image:
        zero      -> gray
        positive  -> bright
        negative  -> dark
    """
    if acc.size == 0 or np.max(np.abs(acc)) <= 0:
        return np.full(acc.shape, 127, dtype=np.uint8)

    nonzero = np.abs(acc[acc != 0])
    scale = np.percentile(nonzero, percentile) if len(nonzero) > 0 else 1.0
    scale = max(float(scale), 1.0)

    img = np.clip(acc.astype(np.float32), -scale, scale)
    img = (img / scale * 127.0) + 127.0
    return img.astype(np.uint8)


def tint_dark_events_on_white(gray: np.ndarray, color_name: str) -> np.ndarray:
    """
    Convert grayscale event image to colored events on white background.

    Assumes:
        gray = 255 for background
        gray = 0   for strongest events

    Returns BGR image for OpenCV display.
    """
    event_strength = 255 - gray  # 0 background, 255 strongest events

    img_bgr = np.full((*gray.shape, 3), 255, dtype=np.uint8)

    if color_name == "red":
        img_bgr[:, :, 0] = 255 - event_strength  # B
        img_bgr[:, :, 1] = 255 - event_strength  # G
        img_bgr[:, :, 2] = 255  # R
    elif color_name == "green":
        img_bgr[:, :, 0] = 255 - event_strength  # B
        img_bgr[:, :, 1] = 255  # G
        img_bgr[:, :, 2] = 255 - event_strength  # R
    elif color_name == "blue":
        img_bgr[:, :, 0] = 255  # B
        img_bgr[:, :, 1] = 255 - event_strength  # G
        img_bgr[:, :, 2] = 255 - event_strength  # R
    else:
        raise ValueError(f"Unknown color_name: {color_name}")

    return img_bgr


def render_event_window(
    x,
    y,
    p,
    height: int,
    width: int,
    mode: str,
    percentile: float,
):
    """
    Accumulate one time window into one 2D image.
    """
    x = np.asarray(x).astype(np.int64, copy=False)
    y = np.asarray(y).astype(np.int64, copy=False)

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        if mode == "signed":
            return np.full((height, width), 127, dtype=np.uint8)
        return np.full((height, width), 255, dtype=np.uint8)

    if mode == "signed" and p is not None:
        p = np.asarray(p)[valid]
        values = np.where(p > 0, 1, -1).astype(np.int16)
        acc = np.zeros((height, width), dtype=np.int16)
        np.add.at(acc, (y, x), values)
        return normalize_signed_image(acc, percentile=percentile)

    if mode == "on" and p is not None:
        p = np.asarray(p)[valid]
        keep = p > 0
        x = x[keep]
        y = y[keep]

    elif mode == "off" and p is not None:
        p = np.asarray(p)[valid]
        keep = p <= 0
        x = x[keep]
        y = y[keep]

    acc = np.zeros((height, width), dtype=np.uint16)
    np.add.at(acc, (y, x), 1)
    return normalize_count_image(acc, percentile=percentile)


def make_panel(
    frame_idx: int,
    num_frames: int,
    t_us: int,
    t0_us: int,
    x_ds,
    y_ds,
    p_ds,
    event_t_us: np.ndarray,
    height: int,
    width: int,
    windows_ms,
    mode: str,
    percentile: float,
    scale: float,
):
    images = []
    panel_colors = ["red", "green", "blue"]

    for panel_idx, window_ms in enumerate(windows_ms):
        window_us = int(round(window_ms * 1000.0))
        start_us = t_us - window_us

        i0 = int(np.searchsorted(event_t_us, start_us, side="left"))
        i1 = int(np.searchsorted(event_t_us, t_us, side="right"))

        x = x_ds[i0:i1]
        y = y_ds[i0:i1]
        p = p_ds[i0:i1] if p_ds is not None else None

        img = render_event_window(
            x=x,
            y=y,
            p=p,
            height=height,
            width=width,
            mode=mode,
            percentile=percentile,
        )

        color_name = panel_colors[min(panel_idx, len(panel_colors) - 1)]
        img_bgr = tint_dark_events_on_white(img, color_name=color_name)
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

        title = f"t-{int(window_ms)}ms"
        cv2.putText(
            img_bgr,
            title,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        images.append(img_bgr)

    panel = np.hstack(images)

    if scale != 1.0:
        panel = cv2.resize(
            panel,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )

    elapsed_s = (t_us - t0_us) / 1_000_000.0
    status = f"frame {frame_idx + 1}/{num_frames}   t={elapsed_s:.3f}s   mode={mode}   a=prev d=next q=quit"

    if frame_idx == 0:
        status += "   [START]"
    if frame_idx == num_frames - 1:
        status += "   [END]"

    bar_h = 34
    bar = np.zeros((bar_h, panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        bar,
        status,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return np.vstack([bar, panel])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "h5_or_session_dir",
        help="Raw event H5 file, or session directory containing *_raw_events.h5",
    )

    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)

    parser.add_argument(
        "--windows-ms",
        type=float,
        nargs=3,
        default=[33.0,33.0,33.0],
        help="Three accumulation windows in ms. Default: 50 250 1500",
    )

    parser.add_argument(
        "--step-ms",
        type=float,
        default=33.333,
        help="Time step between frames. Default: 33.333 ms, roughly 30 FPS timeline.",
    )

    parser.add_argument(
        "--time-unit",
        choices=["auto", "ns", "us", "ms", "s"],
        default="auto",
        help="Timestamp unit. Default: auto.",
    )

    parser.add_argument(
        "--mode",
        choices=["count", "signed", "on", "off"],
        default="count",
        help=(
            "Rendering mode. "
            "count ignores polarity; signed uses polarity if available; "
            "on/off show only one polarity."
        ),
    )

    parser.add_argument(
        "--percentile",
        type=float,
        default=99.5,
        help="Display normalization percentile. Default: 99.5",
    )

    parser.add_argument(
        "--display-scale",
        type=float,
        default=1.5,
        help="Display scale factor. Default: 1.5",
    )

    parser.add_argument(
        "--delay-ms",
        type=int,
        default=1,
        help="cv2 waitKey delay. Default: 1. Low value helps rapid key repeat.",
    )

    parser.add_argument(
        "--start-at-zero",
        action="store_true",
        help=(
            "Start at first event timestamp. "
            "Default starts after the largest window, so t-1500ms is full."
        ),
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="Print H5 tree and exit.",
    )

    args = parser.parse_args()

    h5_path = resolve_h5_path(args.h5_or_session_dir)

    if args.list:
        print_h5_tree(h5_path)
        return

    with h5py.File(h5_path, "r") as f:
        x_key, x_ds = find_dataset(
            f,
            X_KEYS,
            aliases={"x"},
            label="x",
            required=True,
        )
        y_key, y_ds = find_dataset(
            f,
            Y_KEYS,
            aliases={"y"},
            label="y",
            required=True,
        )
        t_key, t_ds = find_dataset(
            f,
            T_KEYS,
            aliases={"t", "ts", "t_us", "timestamp", "timestamps"},
            label="timestamp",
            required=True,
        )
        p_key, p_ds = find_dataset(
            f,
            P_KEYS,
            aliases={"p", "polarity"},
            label="polarity",
            required=False,
        )

        print(f"[INFO] H5: {h5_path}")
        print(f"[INFO] x: {x_key} shape={x_ds.shape} dtype={x_ds.dtype}")
        print(f"[INFO] y: {y_key} shape={y_ds.shape} dtype={y_ds.dtype}")
        print(f"[INFO] t: {t_key} shape={t_ds.shape} dtype={t_ds.dtype}")
        if p_ds is not None:
            print(f"[INFO] p: {p_key} shape={p_ds.shape} dtype={p_ds.dtype}")
        else:
            print("[INFO] p: not found; using polarity-agnostic count rendering")

        if not (len(x_ds) == len(y_ds) == len(t_ds)):
            raise RuntimeError(
                f"Dataset lengths mismatch: len(x)={len(x_ds)}, len(y)={len(y_ds)}, len(t)={len(t_ds)}"
            )

        if len(t_ds) == 0:
            raise RuntimeError("No events in H5 file.")

        t_raw = t_ds[:]
        time_unit = infer_time_unit(t_key, t_raw, args.time_unit)
        event_t_us = convert_timestamps_to_us(t_raw, time_unit)

        time_jumps = np.flatnonzero(np.diff(event_t_us) < 0)

        if len(time_jumps) > 0:
            print(
                f"[WARN] Found {len(time_jumps)} backward timestamp jumps. "
                "Sorting events by timestamp for visualization."
            )

            # Show a few examples for sanity/debugging.
            for j in time_jumps[:5]:
                print(
                    f"[WARN] jump at index {j}: "
                    f"{event_t_us[j]} -> {event_t_us[j + 1]} "
                    f"delta={event_t_us[j + 1] - event_t_us[j]} us"
                )

            order = np.argsort(event_t_us, kind="stable")

            # Important: after sorting, use in-memory arrays, not h5py fancy indexing.
            event_t_us = event_t_us[order]
            x_data = np.asarray(x_ds[:])[order]
            y_data = np.asarray(y_ds[:])[order]
            p_data = np.asarray(p_ds[:])[order] if p_ds is not None else None

            print("[INFO] Finished timestamp sorting.")
        else:
            # Keep using H5 datasets directly if already sorted.
            x_data = x_ds
            y_data = y_ds
            p_data = p_ds

        width, height = infer_sensor_size(
            f,
            x_ds=x_ds,
            y_ds=y_ds,
            width_arg=args.width,
            height_arg=args.height,
        )

        print(f"[INFO] timestamp unit: {time_unit}")
        print(f"[INFO] sensor size: width={width}, height={height}")

        t0_us = int(event_t_us[0])
        t_end_us = int(event_t_us[-1])
        max_window_us = int(round(max(args.windows_ms) * 1000.0))
        step_us = int(round(args.step_ms * 1000.0))

        if step_us <= 0:
            raise ValueError("--step-ms must be positive")

        if args.start_at_zero:
            first_frame_t_us = t0_us
        else:
            first_frame_t_us = min(t0_us + max_window_us, t_end_us)

        frame_times = np.arange(first_frame_t_us, t_end_us + 1, step_us, dtype=np.int64)

        if len(frame_times) == 0:
            frame_times = np.array([t_end_us], dtype=np.int64)

        num_frames = len(frame_times)

        duration_s = (t_end_us - t0_us) / 1_000_000.0
        print(f"[INFO] duration: {duration_s:.3f} s")
        print(f"[INFO] frames: {num_frames}, step={args.step_ms:.3f} ms")
        print("[INFO] controls: a=previous, d=next, q/esc=quit")
        print("[INFO] no wrap-around at start/end")

        window_name = "raw H5 3chef viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_idx = 0

        while True:
            t_us = int(frame_times[frame_idx])

            panel = make_panel(
                frame_idx=frame_idx,
                num_frames=num_frames,
                t_us=t_us,
                t0_us=t0_us,
                x_ds=x_ds,
                y_ds=y_ds,
                p_ds=p_ds,
                event_t_us=event_t_us,
                height=height,
                width=width,
                windows_ms=args.windows_ms,
                mode=args.mode,
                percentile=args.percentile,
                scale=args.display_scale,
            )

            cv2.imshow(window_name, panel)
            key = cv2.waitKeyEx(args.delay_ms)

            if key in [27, ord("q"), ord("Q")]:
                break

            elif key in [ord("d"), ord("D")]:
                # No loop-over.
                if frame_idx < num_frames - 1:
                    frame_idx += 1

            elif key in [ord("a"), ord("A")]:
                # No loop-over.
                if frame_idx > 0:
                    frame_idx -= 1

            # If window was closed manually.
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()