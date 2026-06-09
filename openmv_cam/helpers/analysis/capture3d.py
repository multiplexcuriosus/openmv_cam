#!/usr/bin/env python3
import argparse
import gzip
from pathlib import Path
import struct
import time
from typing import Optional

import numpy as np
import serial
from vispy import app, scene
from vispy.scene import visuals

MAGIC = b"EVT1"
HEADER_FMT = "<LL"   # event_count, payload_len

W = 320
H = 320


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


def sort_events_by_timestamp(events: np.ndarray) -> np.ndarray:
    if events.size == 0:
        return events
    ts = event_timestamps_us(events)
    order = np.argsort(ts, kind="stable")
    return events[order]


def collect_events(ser, duration_s: Optional[float]) -> np.ndarray:
    all_chunks = []
    total_packets = 0
    total_events = 0

    t0 = time.monotonic()
    last_print = t0

    try:
        while True:
            now = time.monotonic()
            if duration_s is not None and now - t0 >= duration_s:
                break

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

            all_chunks.append(events)
            total_packets += 1
            total_events += event_count

            if now - last_print >= 1.0:
                elapsed = now - t0
                print(
                    f"[INFO] elapsed={elapsed:.1f}s  "
                    f"packets={total_packets}  events={total_events}"
                )
                last_print = now
    except KeyboardInterrupt:
        elapsed = time.monotonic() - t0
        print(f"\n[INFO] Ctrl+C received after {elapsed:.1f}s. Stopping capture.")

    if not all_chunks:
        return np.zeros((0, 6), dtype=np.uint16)

    return np.concatenate(all_chunks, axis=0)


def prepare_vispy_cloud(events: np.ndarray, width: int, height: int, time_scale: float = 0.05):
    if events.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    events = sort_events_by_timestamp(events)

    xs = events[:, 4].astype(np.float32)
    ys = events[:, 5].astype(np.float32)
    tp = events[:, 0].astype(np.int32)
    ts_us = event_timestamps_us(events)

    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs = xs[valid]
    ys = ys[valid]
    tp = tp[valid]
    ts_us = ts_us[valid]

    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    # relative time in ms
    ts_ms = (ts_us - ts_us.min()).astype(np.float32) / 1000.0

    # center spatial coordinates a bit for nicer camera behavior
    x_plot = xs - width / 2.0
    y_plot = (height - 1 - ys) - height / 2.0
    t_plot = ts_ms * time_scale

    pos = np.column_stack((x_plot, y_plot, t_plot)).astype(np.float32)

    colors = np.zeros((len(pos), 4), dtype=np.float32)
    pos_mask = (tp == 1)
    neg_mask = ~pos_mask

    colors[pos_mask] = np.array([1.0, 0.2, 0.2, 0.6], dtype=np.float32)  # reddish
    colors[neg_mask] = np.array([0.2, 0.4, 1.0, 0.6], dtype=np.float32)  # bluish

    return pos, colors


def extract_xyt_points(events: np.ndarray, width: int, height: int) -> np.ndarray:
    if events.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    events = sort_events_by_timestamp(events)

    xs = events[:, 4].astype(np.int64)
    ys = events[:, 5].astype(np.int64)
    ts_us = event_timestamps_us(events).astype(np.int64)

    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs = xs[valid]
    ys = ys[valid]
    ts_us = ts_us[valid]

    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    # Save time relative to first valid event to keep values compact.
    ts_rel_us = ts_us - ts_us.min()
    return np.column_stack((xs, ys, ts_rel_us)).astype(np.int64)


def save_xyt_points(points: np.ndarray, output_path: str):
    opener = gzip.open if output_path.endswith(".gz") else open
    with opener(output_path, "wt", encoding="utf-8") as f:
        np.savetxt(f, points, fmt="%d", delimiter="\t")


def show_vispy_cloud(pos: np.ndarray, colors: np.ndarray, point_size: float = 4.0):
    canvas = scene.SceneCanvas(
        keys="interactive",
        show=True,
        size=(900, 900),
        bgcolor="white",
    )

    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(
        fov=45,
        azimuth=35,
        elevation=25,
        distance=700,
    )

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


def main():
    script_dir = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(description="Collect GenX320 events and show 3D x-y-t cloud with VisPy")
    ap.add_argument("--port", default="/dev/openmvcam")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=3.0)
    ap.add_argument(
        "--mode",
        choices=["timed", "continuous"],
        default="timed",
        help="Capture mode: 'timed' uses --duration, 'continuous' runs until Ctrl+C",
    )
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--point-size", type=float, default=2.0)
    ap.add_argument("--time-scale", type=float, default=0.05,
                    help="Scale factor applied to time axis in ms for visualization")
    ap.add_argument(
        "--save",
        nargs="?",
        const=str(script_dir / "events_xyt.tsv"),
        metavar="PATH",
        help="Save extracted x/y/t points as compact TSV text. If PATH ends with .gz, output is gzipped.",
    )
    args = ap.parse_args()

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
    print(f"[INFO] Listening on {args.port} @ {args.baud}")
    if args.mode == "continuous":
        print("[INFO] Collecting continuously. Press Ctrl+C to stop.")
    else:
        print(f"[INFO] Collecting for {args.duration:.1f} seconds...")

    try:
        duration_s = None if args.mode == "continuous" else args.duration
        events = collect_events(ser, duration_s)
    finally:
        ser.close()

    print(f"[INFO] Done. Collected {len(events)} events.")

    if args.save:
        save_path = Path(args.save)
        if not save_path.is_absolute() and save_path.parent == Path("."):
            save_path = script_dir / save_path

        xyt_points = extract_xyt_points(events, W, H)
        save_xyt_points(xyt_points, str(save_path))
        print(f"[INFO] Saved {len(xyt_points)} x/y/t points to {save_path}")

    pos, colors = prepare_vispy_cloud(events, W, H, time_scale=args.time_scale)

    if len(pos) == 0:
        print("[WARN] No valid events to display.")
        return

    print(f"[INFO] Displaying {len(pos)} points.")
    print("[INFO] Mouse:")
    print("       left-drag  = rotate")
    print("       right-drag = zoom")
    print("       middle-drag / shift-drag = pan")

    show_vispy_cloud(pos, colors, point_size=args.point_size)


if __name__ == "__main__":
    main()