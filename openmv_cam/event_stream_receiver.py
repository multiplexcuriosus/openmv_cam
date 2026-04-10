#!/usr/bin/env python3
import argparse
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
    """
    Reconstruct absolute timestamps in microseconds from columns:
      1: sec
      2: ms
      3: us
    """
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

def events_to_preview_frame(
    events: np.ndarray,
    width: int,
    height: int,
    contrast: float = 4.0,
    step: float = 1.0,
) -> np.ndarray:
    frame = np.full((height, width), 128.0, dtype=np.float32)

    if events.size == 0:
        return frame.astype(np.uint8)

    xs = events[:, 4].astype(np.int32)
    ys = events[:, 5].astype(np.int32)
    tp = events[:, 0].astype(np.int32)

    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs = xs[valid]
    ys = ys[valid]
    tp = tp[valid]

    if xs.size == 0:
        return frame.astype(np.uint8)

    pos = (tp == 1)
    neg = ~pos

    acc = np.zeros((height, width), dtype=np.float32)
    np.add.at(acc, (ys[pos], xs[pos]), +step)
    np.add.at(acc, (ys[neg], xs[neg]), -step)

    m = np.max(np.abs(acc))
    if m > 0:
        acc /= m

    frame = 128.0 + acc * (contrast * 127.0)
    np.clip(frame, 0, 255, out=frame)
    return frame.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="GenX320 raw event stream receiver")
    ap.add_argument("--port", default="/dev/openmvcam")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=3.0)
    ap.add_argument("--show", action="store_true", help="Show simple event preview")
    ap.add_argument("--fps", type=float, default=30.0, help="Preview FPS when --show is used")
    ap.add_argument("--max-preview-packets", type=int, default=10,
                    help="Cap number of buffered packets for preview to avoid memory blow-up")
    ap.add_argument("--window-ms", type=float, default=100.0,
                    help="Sliding preview window in milliseconds")
    ap.add_argument("--resize", type=int, default=2,
                    help="Integer upscale factor for display only")
    args = ap.parse_args()

    preview_dt = 1.0 / args.fps
    window_s = args.window_ms / 1000.0

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

    # list of (host_arrival_time, events_array)
    preview_buffer = []
    last_preview_time = t0

    print(f"[INFO] listening on {args.port} @ {args.baud}")

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

            if total_packets <= 3:
                print(f"\n[DEBUG] packet {total_packets}")
                print("  type unique:", np.unique(events[:, 0])[:10])
                print("  sec range  :", int(events[:, 1].min()), int(events[:, 1].max()))
                print("  ms range   :", int(events[:, 2].min()), int(events[:, 2].max()))
                print("  us range   :", int(events[:, 3].min()), int(events[:, 3].max()))
                print("  x range    :", int(events[:, 4].min()), int(events[:, 4].max()))
                print("  y range    :", int(events[:, 5].min()), int(events[:, 5].max()))

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

            if args.show:
                # store events with host arrival time for sliding window retention
                preview_buffer.append((now, events.copy()))

                # remove old packets from sliding preview window
                cutoff = now - window_s
                preview_buffer = [(t, ev) for (t, ev) in preview_buffer if t >= cutoff]

                # cap preview memory
                if len(preview_buffer) > args.max_preview_packets:
                    preview_buffer = preview_buffer[-args.max_preview_packets:]

                # render at fixed fps
                if now - last_preview_time >= preview_dt:
                    if preview_buffer:
                        chunk = np.concatenate([ev for (_, ev) in preview_buffer], axis=0)

                        
                        frame = events_to_preview_frame(chunk, W, H, contrast=4.0, step=1.0)
                        frame = cv2.blur(frame, (2, 2))
                        frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_NEAREST)
                        cv2.imshow("GenX320 event preview", frame)

                        key = cv2.waitKey(1) & 0xFF
                        if key == 27 or key == ord('q'):
                            break

                        last_preview_time = now

    finally:
        ser.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()