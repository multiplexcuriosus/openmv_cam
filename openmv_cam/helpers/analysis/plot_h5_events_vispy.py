#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import numpy as np
from vispy import app, scene
from vispy.scene import visuals

W = 320
H = 320


def load_xyt_points_from_h5(
    file_path: Path,
    x_path: str,
    y_path: str,
    t_path: str,
    max_events: int | None = None,
    start_event: int = 0,
) -> np.ndarray:
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with h5py.File(file_path, "r") as f:
        for ds_path in [x_path, y_path, t_path]:
            if ds_path not in f:
                raise KeyError(f"Dataset not found in HDF5 file: {ds_path}")

        x_ds = f[x_path]
        y_ds = f[y_path]
        t_ds = f[t_path]

        n = len(x_ds)
        if len(y_ds) != n or len(t_ds) != n:
            raise RuntimeError(
                f"Dataset length mismatch: "
                f"x={len(x_ds)}, y={len(y_ds)}, t={len(t_ds)}"
            )

        if n == 0:
            return np.zeros((0, 3), dtype=np.int64)

        start = max(0, int(start_event))
        if start >= n:
            return np.zeros((0, 3), dtype=np.int64)

        if max_events is None or max_events <= 0:
            end = n
        else:
            end = min(n, start + int(max_events))

        xs = x_ds[start:end].astype(np.int64)
        ys = y_ds[start:end].astype(np.int64)
        ts = t_ds[start:end].astype(np.int64)

    return np.column_stack((xs, ys, ts)).astype(np.int64)


def filter_valid_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    xs = points[:, 0]
    ys = points[:, 1]
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    return points[valid]


def build_plot_positions(
    points: np.ndarray,
    width: int,
    height: int,
    time_scale: float,
    normalize_time: bool,
) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    xs = points[:, 0].astype(np.float32)
    ys = points[:, 1].astype(np.float32)
    ts_us = points[:, 2].astype(np.float64)

    if normalize_time:
        ts_us = ts_us - ts_us[0]

    x_plot = xs - width / 2.0
    y_plot = (height - 1 - ys) - height / 2.0
    t_plot = (ts_us / 1000.0) * time_scale

    return np.column_stack((x_plot, y_plot, t_plot)).astype(np.float32)


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
    points: np.ndarray,
    radius: int,
    width: int,
    height: int,
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


def build_colors(
    num_points: int,
    filter_mode: bool,
    temporal_mask: np.ndarray | None = None,
    spatial_mask: np.ndarray | None = None,
) -> np.ndarray:
    if num_points == 0:
        return np.zeros((0, 4), dtype=np.float32)

    if not filter_mode:
        colors = np.zeros((num_points, 4), dtype=np.float32)
        colors[:] = np.array([0.2, 0.4, 1.0, 0.6], dtype=np.float32)
        return colors

    colors = np.zeros((num_points, 4), dtype=np.float32)
    colors[:] = np.array([0.55, 0.55, 0.55, 0.35], dtype=np.float32)

    if temporal_mask is not None:
        colors[temporal_mask] = np.array([1.0, 0.1, 0.1, 0.95], dtype=np.float32)

    if spatial_mask is not None:
        colors[spatial_mask] = np.array([0.1, 0.25, 1.0, 0.95], dtype=np.float32)

    if temporal_mask is not None and spatial_mask is not None:
        both_mask = temporal_mask & spatial_mask
        colors[both_mask] = np.array([0.75, 0.1, 0.75, 0.95], dtype=np.float32)

    return colors


def show_vispy_cloud(pos: np.ndarray, colors: np.ndarray, point_size: float) -> None:
    canvas = scene.SceneCanvas(
        keys="interactive",
        show=True,
        size=(900, 900),
        bgcolor="white",
    )

    view = canvas.central_widget.add_view()
    cam = scene.cameras.TurntableCamera(
        fov=45,
        azimuth=35,
        elevation=25,
        distance=700,
    )

    cam.center = (0, 120, 0)
    view.camera = cam

    scatter = visuals.Markers()
    scatter.set_data(
        pos,
        face_color=colors,
        edge_color=None,
        size=point_size,
    )
    view.add(scatter)

    axis = visuals.XYZAxis(parent=view.scene)
    axis.transform = scene.transforms.STTransform(scale=(80, 80, 80))

    app.run()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot raw event points from raw_events.h5 with VisPy"
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Path to raw_events.h5",
    )
    ap.add_argument("--x-path", default="/events/x")
    ap.add_argument("--y-path", default="/events/y")
    ap.add_argument("--t-path", default="/events/t_us")
    ap.add_argument("--width", type=int, default=W)
    ap.add_argument("--height", type=int, default=H)
    ap.add_argument("--point-size", type=float, default=4.0)
    ap.add_argument(
        "--time-scale",
        type=float,
        default=0.05,
        help="Scale factor applied to time axis in ms for visualization",
    )
    ap.add_argument(
        "--no-normalize-time",
        action="store_true",
        help="Use absolute t_us values on plot axis instead of shifting first timestamp to 0",
    )
    ap.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Load at most this many events. Default: all events",
    )
    ap.add_argument(
        "--start-event",
        type=int,
        default=0,
        help="Start loading from this event index",
    )
    ap.add_argument(
        "--filter-time",
        type=int,
        default=None,
        metavar="R_US",
        help=(
            "Color a point red if it has a temporal neighbor in [t-R, t+R] microseconds. "
            "Set R >= 0."
        ),
    )
    ap.add_argument(
        "--filter-spatial",
        type=int,
        default=None,
        metavar="R_PX",
        help=(
            "Color a point blue if it has a spatial neighbor within XY radius R pixels. "
            "Set R >= 0."
        ),
    )
    ap.add_argument(
        "--hide-noise",
        action="store_true",
        help=(
            "Render only points that survive all active filters. "
            "If not set, filter mode uses gray base, red temporal matches, and blue spatial matches."
        ),
    )
    args = ap.parse_args()

    if args.filter_time is not None and args.filter_time < 0:
        raise ValueError("--filter-time must be >= 0")
    if args.filter_spatial is not None and args.filter_spatial < 0:
        raise ValueError("--filter-spatial must be >= 0")

    input_path = Path(args.input).expanduser().resolve()

    print(f"[INFO] Loading points from {input_path}")
    points = load_xyt_points_from_h5(
        input_path,
        x_path=args.x_path,
        y_path=args.y_path,
        t_path=args.t_path,
        max_events=args.max_events,
        start_event=args.start_event,
    )

    points = filter_valid_points(points, args.width, args.height)

    print(f"[INFO] Loaded {len(points)} valid points")

    if len(points) == 0:
        print("[WARN] No valid points to display.")
        return

    t0 = int(points[0, 2])
    t1 = int(points[-1, 2])
    print(f"[INFO] Event time: {t0} -> {t1} us, duration={(t1 - t0) / 1e6:.3f} s")

    matched_mask = None
    temporal_mask = None
    spatial_mask = None

    if args.filter_time is not None or args.filter_spatial is not None:
        matched_mask = np.zeros(len(points), dtype=bool)

        if args.filter_time is not None:
            print(f"[INFO] Applying temporal filter with radius {args.filter_time} us...")
            temporal_mask = find_temporal_neighbor_points(points, args.filter_time)
            matched_mask |= temporal_mask
            print(f"[INFO] Temporal matches: {int(temporal_mask.sum())} / {len(points)}")

        if args.filter_spatial is not None:
            print(f"[INFO] Applying spatial filter with XY radius {args.filter_spatial}px...")
            spatial_mask = find_spatial_neighbor_points(
                points,
                args.filter_spatial,
                width=args.width,
                height=args.height,
            )
            matched_mask |= spatial_mask
            print(f"[INFO] Spatial matches: {int(spatial_mask.sum())} / {len(points)}")

        print(f"[INFO] Combined matches: {int(matched_mask.sum())} / {len(points)}")

    render_points = points
    filter_mode = matched_mask is not None
    render_temporal_mask = temporal_mask
    render_spatial_mask = spatial_mask

    if filter_mode:
        if args.hide_noise:
            keep_mask = np.ones(len(points), dtype=bool)

            if temporal_mask is not None:
                keep_mask &= temporal_mask

            if spatial_mask is not None:
                keep_mask &= spatial_mask

            kept = int(keep_mask.sum())
            render_points = points[keep_mask]

            if temporal_mask is not None:
                render_temporal_mask = temporal_mask[keep_mask]

            if spatial_mask is not None:
                render_spatial_mask = spatial_mask[keep_mask]

            print(
                f"[INFO] Keeping only points that satisfy all active filters: "
                f"{kept} / {len(points)}"
            )
        else:
            print("[INFO] Filter colors: base=gray, temporal=red, spatial=blue.")

    if len(render_points) == 0:
        print("[WARN] No points left to display after filtering.")
        return

    pos = build_plot_positions(
        render_points,
        width=args.width,
        height=args.height,
        time_scale=args.time_scale,
        normalize_time=not args.no_normalize_time,
    )

    colors = build_colors(
        len(pos),
        filter_mode,
        render_temporal_mask,
        render_spatial_mask,
    )

    print(f"[INFO] Displaying {len(pos)} points.")
    print("[INFO] Mouse:")
    print("       left-drag  = rotate")
    print("       right-drag = zoom")
    print("       middle-drag / shift-drag = pan")

    show_vispy_cloud(pos, colors, point_size=args.point_size)


if __name__ == "__main__":
    main()