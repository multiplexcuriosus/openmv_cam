#!/usr/bin/env python3
import argparse
import os
import struct
import time

import cv2
import numpy as np
import serial

MAGIC = b"EVT1"
HEADER_FMT = "<LL"   # event_count, payload_len
HEADER_SIZE = 4 + struct.calcsize(HEADER_FMT)

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


class EventPreviewRenderer:
    def __init__(
        self,
        width: int,
        height: int,
        decay: float = 0.94,
        step: float = 1.0,
        contrast: float = 10.0,
        blur_ksize: int = 0,
    ):
        self.width = width
        self.height = height
        self.decay = float(decay)
        self.step = float(step)
        self.contrast = float(contrast)
        self.blur_ksize = int(blur_ksize)
        self.acc = np.zeros((height, width), dtype=np.float32)

    def reset(self):
        self.acc.fill(0.0)

    def render(self, events: np.ndarray) -> np.ndarray:
        self.acc *= self.decay

        if events.size != 0:
            xs = events[:, 4].astype(np.int32)
            ys = events[:, 5].astype(np.int32)
            tp = events[:, 0].astype(np.int32)

            valid = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height)
            xs = xs[valid]
            ys = ys[valid]
            tp = tp[valid]

            if xs.size != 0:
                pos = (tp == 1)
                neg = ~pos
                np.add.at(self.acc, (ys[pos], xs[pos]), +self.step)
                np.add.at(self.acc, (ys[neg], xs[neg]), -self.step)

        frame = 128.0 + self.acc * self.contrast
        np.clip(frame, 0, 255, out=frame)
        img = frame.astype(np.uint8)

        if self.blur_ksize and self.blur_ksize >= 2:
            k = self.blur_ksize
            if k % 2 == 0:
                k += 1
            img = cv2.GaussianBlur(img, (k, k), 0)

        return img


def make_param_tag(args: argparse.Namespace) -> str:
    return (
        f"d{args.decay:.3f}"
        f"_s{args.step:.3f}"
        f"_c{args.contrast:.3f}"
        f"_w{args.window_ms:.1f}"
        f"_b{args.blur}"
        f"_fps{args.fps:.1f}"
        f"_sort{int(args.sort_ts)}"
    ).replace("/", "_")


def save_image(path: str, frame: np.ndarray, resize: int):
    out = frame
    if resize > 1:
        out = cv2.resize(
            frame,
            (frame.shape[1] * resize, frame.shape[0] * resize),
            interpolation=cv2.INTER_NEAREST,
        )
    ok = cv2.imwrite(path, out)
    if not ok:
        raise RuntimeError(f"Failed to save image to {path}")


def main():
    ap = argparse.ArgumentParser(description="GenX320 raw event stream receiver")
    ap.add_argument("--port", default="/dev/openmvcam")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=3.0)

    ap.add_argument("--show", action="store_true", help="Show event preview")
    ap.add_argument("--fps", type=float, default=30.0, help="Preview FPS when --show is used")
    ap.add_argument(
        "--max-preview-packets",
        type=int,
        default=10,
        help="Cap number of buffered packets for preview",
    )
    ap.add_argument(
        "--window-ms",
        type=float,
        default=100.0,
        help="Sliding preview window in milliseconds",
    )
    ap.add_argument(
        "--resize",
        type=int,
        default=2,
        help="Integer upscale factor for display/saved image",
    )

    ap.add_argument("--decay", type=float, default=0.94)
    ap.add_argument("--step", type=float, default=0.75)
    ap.add_argument("--contrast", type=float, default=10.0)
    ap.add_argument("--blur", type=int, default=0)
    ap.add_argument("--sort-ts", action="store_true")

    # tuning / save mode
    ap.add_argument(
        "--tune-save-dir",
        type=str,
        default="",
        help="If set, run headless, render for --tune-run-seconds, save one image, then exit",
    )
    ap.add_argument(
        "--tune-run-seconds",
        type=float,
        default=3.0,
        help="How long to accumulate before saving in tuning mode",
    )
    ap.add_argument(
        "--tune-prefix",
        type=str,
        default="preview",
        help="Filename prefix in tuning mode",
    )

    args = ap.parse_args()

    preview_dt = 1.0 / args.fps
    window_s = args.window_ms / 1000.0
    tune_mode = bool(args.tune_save_dir)

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
    total_payload_bytes = 0
    total_protocol_bytes = 0

    t0 = time.monotonic()
    last_print = t0
    last_preview_time = t0

    preview_buffer = []

    renderer = EventPreviewRenderer(
        width=W,
        height=H,
        decay=args.decay,
        step=args.step,
        contrast=args.contrast,
        blur_ksize=args.blur,
    )

    print(f"[INFO] listening on {args.port} @ {args.baud}")
    print(
        "[INFO] settings: "
        f"fps={args.fps}, window_ms={args.window_ms}, resize={args.resize}, "
        f"decay={args.decay}, step={args.step}, contrast={args.contrast}, "
        f"blur={args.blur}, sort_ts={args.sort_ts}, tune_mode={tune_mode}"
    )

    if tune_mode:
        os.makedirs(args.tune_save_dir, exist_ok=True)
        tune_start = time.monotonic()
        last_frame = np.full((H, W), 128, dtype=np.uint8)

    try:
        while True:
            read_until_magic(ser)
            header_rest = read_exactly(ser, struct.calcsize(HEADER_FMT))
            event_count, payload_len = struct.unpack(HEADER_FMT, header_rest)

            expected_len = event_count * 6 * 2
            if payload_len != expected_len:
                raise RuntimeError(
                    f"Invalid payload length: got {payload_len}, expected {expected_len}"
                )

            payload = read_exactly(ser, payload_len)
            events = np.frombuffer(payload, dtype=np.uint16).reshape((event_count, 6))

            total_packets += 1
            total_events += event_count
            total_payload_bytes += payload_len
            total_protocol_bytes += payload_len + HEADER_SIZE

            now = time.monotonic()

            if now - last_print >= 2.0:
                elapsed = now - t0
                events_per_s = total_events / elapsed if elapsed > 0 else 0.0
                payload_MBps = total_payload_bytes / elapsed / 1e6 if elapsed > 0 else 0.0
                protocol_MBps = total_protocol_bytes / elapsed / 1e6 if elapsed > 0 else 0.0
                packets_per_s = total_packets / elapsed if elapsed > 0 else 0.0

                print("\n===== EVENT STREAM STATS =====")
                print(f"packets_received    : {total_packets}")
                print(f"events_received     : {total_events}")
                print(f"elapsed_time        : {elapsed:.2f} s")
                print(f"packets_per_s       : {packets_per_s:.1f}")
                print(f"events_per_s        : {events_per_s:.1f}")
                print(f"payload_MBps        : {payload_MBps:.3f}")
                print(f"protocol_MBps       : {protocol_MBps:.3f}")
                print("==============================")
                last_print = now

            # keep sliding preview buffer
            preview_buffer.append((now, events.copy()))
            cutoff = now - window_s
            preview_buffer = [(t, ev) for (t, ev) in preview_buffer if t >= cutoff]

            if len(preview_buffer) > args.max_preview_packets:
                preview_buffer = preview_buffer[-args.max_preview_packets:]

            # render at fixed rate
            if now - last_preview_time >= preview_dt:
                if preview_buffer:
                    chunk = np.concatenate([ev for (_, ev) in preview_buffer], axis=0)
                    if args.sort_ts:
                        chunk = sort_events_by_timestamp(chunk)
                    last_frame = renderer.render(chunk)

                    if args.show and not tune_mode:
                        frame_show = last_frame
                        if args.resize > 1:
                            frame_show = cv2.resize(
                                frame_show,
                                (W * args.resize, H * args.resize),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        cv2.imshow("GenX320 event preview", frame_show)

                        key = cv2.waitKey(1) & 0xFF
                        if key == 27 or key == ord("q"):
                            break
                        elif key == ord("r"):
                            renderer.reset()
                            print("[INFO] preview accumulator reset")

                last_preview_time = now

            if tune_mode and (now - tune_start >= args.tune_run_seconds):
                tag = make_param_tag(args)
                out_path = os.path.join(
                    args.tune_save_dir,
                    f"{args.tune_prefix}_{tag}.png",
                )
                save_image(out_path, last_frame, args.resize)
                print(f"[INFO] saved {out_path}")
                break

    finally:
        ser.close()
        if args.show and not tune_mode:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()