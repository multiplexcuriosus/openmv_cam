#!/usr/bin/env python3
import struct
import time
import serial
import numpy as np
import cv2

PORT = "/dev/openmvcam"
BAUD = 115200

MAGIC = b"EVT1"
HEADER_FMT = "<LL"   # event_count, payload_len

W = 320
H = 320
FRAME_DT_US = int(1e6 / 30.0)   # 33333 us


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
            raise RuntimeError("Timeout while searching for magic.")
        window += b
        if len(window) > len(MAGIC):
            window = window[-len(MAGIC):]
        if bytes(window) == MAGIC:
            return


def event_time_us(events: np.ndarray) -> np.ndarray:
    # cols:
    # 0 type
    # 1 sec
    # 2 ms
    # 3 us
    # 4 x
    # 5 y
    return (
        events[:, 1].astype(np.int64) * 1_000_000
        + events[:, 2].astype(np.int64) * 1_000
        + events[:, 3].astype(np.int64)
    )


def render_frame(accum: np.ndarray, clip_val: int = 8) -> np.ndarray:
    vis = np.clip(accum, -clip_val, clip_val).astype(np.float32)
    vis = 128.0 + vis * (127.0 / clip_val)
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    return vis


def main():
    ser = serial.Serial(PORT, baudrate=BAUD, timeout=1.0)

    accum = np.zeros((H, W), dtype=np.int16)
    current_bin_start_us = None

    packets_acc = 0
    events_acc = 0
    t_stats = time.time()

    cv2.namedWindow("event_frame_30fps", cv2.WINDOW_NORMAL)

    while True:
        read_until_magic(ser)

        header_rest = read_exactly(ser, struct.calcsize(HEADER_FMT))
        event_count, payload_len = struct.unpack(HEADER_FMT, header_rest)

        if payload_len != event_count * 12:
            raise RuntimeError(
                f"Invalid payload_len={payload_len}, expected={event_count * 12}"
            )

        payload = read_exactly(ser, payload_len)
        events = np.frombuffer(payload, dtype=np.uint16).reshape(event_count, 6)

        packets_acc += 1
        events_acc += event_count

        if event_count == 0:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue

        ts_us = event_time_us(events)
        xs = events[:, 4].astype(np.int32)
        ys = events[:, 5].astype(np.int32)
        types = events[:, 0].astype(np.int32)

        # assume type 1 = ON, type 0 = OFF
        vals = np.where(types == 1, 1, -1).astype(np.int16)

        if current_bin_start_us is None:
            current_bin_start_us = int(ts_us[0])

        for t_us, x, y, v in zip(ts_us, xs, ys, vals):
            # flush finished bins until event fits
            while t_us >= current_bin_start_us + FRAME_DT_US:
                
                vis = render_frame(accum, clip_val=2)
                vis = cv2.GaussianBlur(vis, (3, 3), 0)
                vis = cv2.resize(vis, (640, 640))
                cv2.imshow("event_frame_30fps", vis)
                accum.fill(0)
                current_bin_start_us += FRAME_DT_US

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    ser.close()
                    cv2.destroyAllWindows()
                    return

            if 0 <= x < W and 0 <= y < H:
                accum[y, x] += v

        now = time.time()
        if now - t_stats >= 1.0:
            print(f"pps={packets_acc:.0f} eps={events_acc:.0f}")
            packets_acc = 0
            events_acc = 0
            t_stats = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    ser.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()