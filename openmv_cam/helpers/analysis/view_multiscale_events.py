#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np


W_DEFAULT = 320
H_DEFAULT = 320

LEFT_KEYS = {81, 2424832, 65361}
RIGHT_KEYS = {83, 2555904, 65363}


def load_xyt_points(file_path: Path) -> np.ndarray:
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if file_path.stat().st_size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    points = np.loadtxt(file_path, dtype=np.int64, delimiter="\t")

    if points.ndim == 1:
        if points.size != 3:
            raise RuntimeError(
                f"Invalid x/y/t input shape from {file_path}: expected 3 columns"
            )
        points = points.reshape(1, 3)

    if points.shape[1] != 3:
        raise RuntimeError(
            f"Invalid x/y/t input shape from {file_path}: expected 3 columns"
        )

    return points


def filter_valid_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    xs = points[:, 0]
    ys = points[:, 1]

    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    return points[valid]


def normalize_to_uint8(img: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    img = img.astype(np.float32)

    hi = np.percentile(img, percentile)
    if hi <= 0:
        return np.full(img.shape, 128, dtype=np.uint8)

    img = np.clip(img / hi, 0.0, 1.0)
    return (128 + img * 127).astype(np.uint8)


def build_event_image_for_window(
    points: np.ndarray,
    start_us: int,
    end_us: int,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Naive event accumulation:
      image[y, x] += 1 for events with start_us <= t < end_us
    """
    ts = points[:, 2]
    lo = np.searchsorted(ts, start_us, side="left")
    hi = np.searchsorted(ts, end_us, side="left")

    window_points = points[lo:hi]

    img = np.zeros((height, width), dtype=np.float32)

    if len(window_points) == 0:
        return img

    xs = window_points[:, 0].astype(np.int64)
    ys = window_points[:, 1].astype(np.int64)

    np.add.at(img, (ys, xs), 1.0)
    return img


def make_tile(img: np.ndarray, label: str, scale: int = 2) -> np.ndarray:
    img_u8 = normalize_to_uint8(img)
    bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

    if scale != 1:
        bgr = cv2.resize(
            bgr,
            (bgr.shape[1] * scale, bgr.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )

    cv2.putText(
        bgr,
        label,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    bgr = cv2.copyMakeBorder(
        bgr,
        2,
        2,
        2,
        2,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    return bgr


def build_triplet_view(
    points: np.ndarray,
    tick_idx: int,
    tick_times_us: np.ndarray,
    width: int,
    height: int,
    scale: int,
) -> np.ndarray:
    t_us = int(tick_times_us[tick_idx])

    windows_ms = [50, 250, 1000]
    tiles = []

    for ms in windows_ms:
        start_us = t_us - ms * 1000
        end_us = t_us

        img = build_event_image_for_window(
            points=points,
            start_us=start_us,
            end_us=end_us,
            width=width,
            height=height,
        )

        label = f"tick {tick_idx} | last {ms} ms"
        tiles.append(make_tile(img, label, scale=scale))

    view = np.hstack(tiles)

    cv2.putText(
        view,
        f"t = {t_us / 1e6:.3f}s | channels: 50ms / 250ms / 1000ms",
        (20, view.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to x/y/t TSV file")
    ap.add_argument("--width", type=int, default=W_DEFAULT)
    ap.add_argument("--height", type=int, default=H_DEFAULT)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--window", default="multi_timescale_event_view")
    ap.add_argument(
        "--start-time-us",
        type=int,
        default=None,
        help="Optional absolute start timestamp in microseconds",
    )
    ap.add_argument(
        "--end-time-us",
        type=int,
        default=None,
        help="Optional absolute end timestamp in microseconds",
    )
    args = ap.parse_args()

    points = load_xyt_points(Path(args.input))
    points = filter_valid_points(points, args.width, args.height)

    if len(points) == 0:
        print("[WARN] No valid events.")
        return

    # Important: searchsorted requires sorted timestamps.
    order = np.argsort(points[:, 2], kind="stable")
    points = points[order]

    min_t = int(points[0, 2])
    max_t = int(points[-1, 2])

    start_t = args.start_time_us if args.start_time_us is not None else min_t
    end_t = args.end_time_us if args.end_time_us is not None else max_t

    if end_t <= start_t:
        raise ValueError("end time must be greater than start time")

    tick_period_us = int(round(1e6 / args.fps))
    tick_times_us = np.arange(start_t, end_t + 1, tick_period_us, dtype=np.int64)

    if len(tick_times_us) == 0:
        print("[WARN] No ticks generated.")
        return

    print(f"[INFO] Loaded {len(points)} valid events")
    print(f"[INFO] Event time range: {min_t} -> {max_t} us")
    print(f"[INFO] Tick range:       {start_t} -> {end_t} us")
    print(f"[INFO] FPS: {args.fps}, ticks: {len(tick_times_us)}")
    print("[INFO] Controls: LEFT/RIGHT browse, q/ESC quit")

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

    i = 0

    while True:
        view = build_triplet_view(
            points=points,
            tick_idx=i,
            tick_times_us=tick_times_us,
            width=args.width,
            height=args.height,
            scale=args.scale,
        )

        cv2.imshow(args.window, view)

        key = cv2.waitKeyEx(0)
        key8 = key & 0xFF

        if key8 in [ord("q"), 27]:
            break
        elif key in LEFT_KEYS:
            i = max(0, i - 1)
        elif key in RIGHT_KEYS:
            i = min(len(tick_times_us) - 1, i + 1)
        elif key8 == ord("a"):
            i = max(0, i - 10)
        elif key8 == ord("d"):
            i = min(len(tick_times_us) - 1, i + 10)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()