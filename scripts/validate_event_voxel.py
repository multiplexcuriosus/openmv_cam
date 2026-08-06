#!/usr/bin/env python3
"""Headlessly validate and summarize an OpenMV activity voxel topic."""

import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


def interval_stats(values):
    """Return count, mean, p50, p95, maximum, and derived rate."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean_s": None, "p50_s": None,
                "p95_s": None, "max_s": None, "rate_hz": None}
    mean = float(np.mean(array))
    return {
        "count": int(array.size),
        "mean_s": mean,
        "p50_s": float(np.percentile(array, 50)),
        "p95_s": float(np.percentile(array, 95)),
        "max_s": float(np.max(array)),
        "rate_hz": 1.0 / mean if mean > 0.0 else None,
    }


class VoxelValidator(Node):
    """Collect validation data from sensor_msgs/Image activity voxels."""

    def __init__(self, args):
        super().__init__("validate_event_voxel")
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.arrival_intervals = []
        self.header_intervals = []
        self.delays = []
        self.previous_arrival_ns = None
        self.previous_header_ns = None
        self.previous_payload = None
        self.saved_arrays = []
        self.layout_errors = 0
        self.nonmonotonic_headers = 0
        self.repeated_headers = 0
        self.repeated_empty = 0
        self.repeated_nonempty = 0
        self.subscription = self.create_subscription(
            Image, args.topic, self._callback, 10
        )

    def _callback(self, msg):
        arrival_ns = time.time_ns()
        monotonic_ns = time.monotonic_ns()
        header_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(
            msg.header.stamp.nanosec
        )
        channels = 1
        if msg.encoding.startswith("8UC"):
            channels = int(msg.encoding[3:])
        expected_step = int(msg.width) * channels
        expected_length = int(msg.height) * expected_step
        layout_ok = (
            msg.height == 320
            and msg.width == 320
            and msg.encoding == f"8UC{channels}"
            and msg.step == expected_step
            and len(msg.data) == expected_length
            and msg.header.frame_id == "openmv_cam"
        )
        if not layout_ok:
            self.layout_errors += 1
            self.get_logger().error(
                "Invalid layout: "
                f"{msg.height}x{msg.width} {msg.encoding} step={msg.step} "
                f"bytes={len(msg.data)} frame_id={msg.header.frame_id!r}"
            )
            return

        array = np.frombuffer(msg.data, dtype=np.uint8)
        shape = (msg.height, msg.width) if channels == 1 else (
            msg.height, msg.width, channels
        )
        array = array.reshape(shape)
        nonzero = int(np.count_nonzero(array))
        empty = nonzero == 0
        payload = bytes(msg.data)
        repeated = self.previous_payload == payload
        if repeated:
            if empty:
                self.repeated_empty += 1
            else:
                self.repeated_nonempty += 1

        if self.previous_arrival_ns is not None:
            self.arrival_intervals.append(
                (monotonic_ns - self.previous_arrival_ns) / 1e9
            )
        if self.previous_header_ns is not None:
            delta = (header_ns - self.previous_header_ns) / 1e9
            self.header_intervals.append(delta)
            if delta == 0.0:
                self.repeated_headers += 1
            elif delta < 0.0:
                self.nonmonotonic_headers += 1
        delay_s = (arrival_ns - header_ns) / 1e9 if header_ns > 0 else None
        if delay_s is not None:
            self.delays.append(delay_s)

        active = np.any(array > 0, axis=2) if channels > 1 else array > 0
        ys, xs = np.nonzero(active)
        bbox = None if xs.size == 0 else [
            int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        ]
        row = {
            "index": len(self.rows),
            "arrival_wall_ns": arrival_ns,
            "header_ns": header_ns,
            "header_to_arrival_s": delay_s,
            "height": int(msg.height),
            "width": int(msg.width),
            "channels": channels,
            "encoding": msg.encoding,
            "step": int(msg.step),
            "payload_length": len(msg.data),
            "nonzero_pixels": nonzero,
            "total_activity": int(np.sum(array, dtype=np.int64)),
            "maximum": int(np.max(array)),
            "bbox": bbox,
            "empty": empty,
            "repeated_payload": repeated,
        }
        self.rows.append(row)
        if self.args.save_every > 0 and row["index"] % self.args.save_every == 0:
            self._save_frame(array, row["index"])
        if self.args.save_npz:
            self.saved_arrays.append(array.copy())
        self.previous_arrival_ns = monotonic_ns
        self.previous_header_ns = header_ns
        self.previous_payload = payload

    def _save_frame(self, array, index):
        if array.ndim == 2:
            cv2.imwrite(str(self.output_dir / f"frame_{index:06d}.png"), array)
            return
        cv2.imwrite(
            str(self.output_dir / f"frame_{index:06d}_max.png"),
            np.max(array, axis=2),
        )
        for channel in range(array.shape[2]):
            cv2.imwrite(
                str(self.output_dir / f"frame_{index:06d}_ch{channel:02d}.png"),
                array[:, :, channel],
            )
        summary = np.sum(array, axis=(0, 1), dtype=np.int64)
        np.savetxt(
            self.output_dir / f"frame_{index:06d}_channels.csv",
            summary,
            delimiter=",",
            fmt="%d",
        )

    def write_results(self):
        csv_path = self.output_dir / "frames.csv"
        if self.rows:
            with csv_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.rows[0].keys())
                writer.writeheader()
                writer.writerows(self.rows)
        summary = {
            "topic": self.args.topic,
            "duration_s": self.args.duration,
            "messages": len(self.rows),
            "layout_errors": self.layout_errors,
            "arrival_intervals": interval_stats(self.arrival_intervals),
            "header_intervals": interval_stats(self.header_intervals),
            "header_to_arrival": interval_stats(self.delays),
            "empty_frames": sum(row["empty"] for row in self.rows),
            "repeated_empty_frames": self.repeated_empty,
            "repeated_nonempty_stale_frames": self.repeated_nonempty,
            "repeated_header_timestamps": self.repeated_headers,
            "nonmonotonic_header_timestamps": self.nonmonotonic_headers,
        }
        with (self.output_dir / "summary.json").open("w") as stream:
            json.dump(summary, stream, indent=2)
        if self.args.save_npz and self.saved_arrays:
            np.savez_compressed(
                self.output_dir / "frames.npz",
                frames=np.stack(self.saved_arrays),
            )
        print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/openmv_cam/event_voxel_1ms")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--output-dir", default="/tmp/event_voxel_validation")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--save-npz", action="store_true")
    args = parser.parse_args()
    rclpy.init()
    node = VoxelValidator(args)
    deadline = time.monotonic() + args.duration
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.write_results()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
