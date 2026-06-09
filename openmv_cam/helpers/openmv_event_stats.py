#!/usr/bin/env python3
import struct
import time
import serial
import numpy as np

MAGIC = b"EVT1"
HEADER_FMT = "<LL"   # event_count, payload_len

ser = serial.Serial("/dev/openmvcam", baudrate=115200, timeout=1.0)


def read_exactly(n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        chunk = ser.read(n - len(data))
        if not chunk:
            raise RuntimeError("Serial read timeout.")
        data.extend(chunk)
    return bytes(data)


def read_until_magic():
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


# 1-second accumulators
t0 = time.time()
packets_acc = 0
events_acc = 0
max_count_acc = 0
sum_x_acc = 0
sum_y_acc = 0
on_acc = 0
off_acc = 0

while True:
    read_until_magic()

    header_rest = read_exactly(struct.calcsize(HEADER_FMT))
    event_count, payload_len = struct.unpack(HEADER_FMT, header_rest)

    if payload_len != event_count * 12:
        raise RuntimeError(
            f"Invalid payload_len={payload_len}, expected={event_count * 12}"
        )

    payload = read_exactly(payload_len)
    events = np.frombuffer(payload, dtype=np.uint16).reshape(event_count, 6)

    packets_acc += 1
    events_acc += event_count
    if event_count > max_count_acc:
        max_count_acc = event_count

    if event_count > 0:
        xs = events[:, 4]
        ys = events[:, 5]
        types = events[:, 0]

        sum_x_acc += int(xs.sum())
        sum_y_acc += int(ys.sum())

        # assuming 0/1 event type
        on_acc += int((types == 1).sum())
        off_acc += int((types == 0).sum())

    now = time.time()
    dt = now - t0
    if dt >= 1.0:
        eps = events_acc / dt
        pps = packets_acc / dt
        avg_count = events_acc / packets_acc if packets_acc > 0 else 0.0

        if events_acc > 0:
            mean_x = sum_x_acc / events_acc
            mean_y = sum_y_acc / events_acc
        else:
            mean_x = -1
            mean_y = -1

        print(
            f"pps={pps:6.1f}  "
            f"eps={eps:8.0f}  "
            f"avg_count={avg_count:6.1f}  "
            f"max_count={max_count_acc:4d}  "
            f"mean_xy=({mean_x:6.1f},{mean_y:6.1f})  "
            f"off={off_acc:6d}  on={on_acc:6d}"
        )

        t0 = now
        packets_acc = 0
        events_acc = 0
        max_count_acc = 0
        sum_x_acc = 0
        sum_y_acc = 0
        on_acc = 0
        off_acc = 0