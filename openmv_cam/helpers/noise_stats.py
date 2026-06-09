#!/usr/bin/env python3
import struct
import time
import serial
import numpy as np

MAGIC = b"EVT1"
HEADER_FMT = "<LL"   # event_count, payload_len

PORT = "/dev/openmvcam"
BAUD = 115200

W = 320
H = 320

# Temporal coherence parameters
COH_RADIUS = 1       # 1 => 3x3 neighborhood
COH_DT_US = 5000     # event is "coherent" if a neighbor fired within last 5 ms

# Hot pixel thresholds (events per 1-second window)
HOT_THRESHOLDS = [5, 10, 20]

ser = serial.Serial(PORT, baudrate=BAUD, timeout=1.0)


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


def reconstruct_t_us(events_u16: np.ndarray) -> np.ndarray:
    """
    events_u16 shape: (N, 6)
    columns:
      0: type
      1: seconds
      2: milliseconds
      3: microseconds
      4: x
      5: y
    """
    sec = events_u16[:, 1].astype(np.int64)
    ms = events_u16[:, 2].astype(np.int64)
    us = events_u16[:, 3].astype(np.int64)
    return sec * 1_000_000 + ms * 1_000 + us


# 1-second accumulators
t0 = time.time()
packets_acc = 0
events_acc = 0
max_count_acc = 0
sum_x_acc = 0
sum_y_acc = 0
on_acc = 0
off_acc = 0

# New accumulators
spatial_counts = np.zeros((H, W), dtype=np.int32)

# last event timestamp seen at each pixel (for coherence metric)
last_t_us = np.full((H, W), -10**15, dtype=np.int64)
coherent_events_acc = 0

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
        xs = events[:, 4].astype(np.int32)
        ys = events[:, 5].astype(np.int32)
        types = events[:, 0]
        ts_us = reconstruct_t_us(events)

        # basic stats
        sum_x_acc += int(xs.sum())
        sum_y_acc += int(ys.sum())
        on_acc += int((types == 1).sum())
        off_acc += int((types == 0).sum())

        # spatial histogram
        # assumes xs in [0, W-1], ys in [0, H-1]
        np.add.at(spatial_counts, (ys, xs), 1)

        # temporal local coherence
        # event is coherent if any pixel in radius neighborhood has fired within COH_DT_US
        for i in range(event_count):
            x = xs[i]
            y = ys[i]
            t_us = ts_us[i]

            x0 = max(0, x - COH_RADIUS)
            x1 = min(W, x + COH_RADIUS + 1)
            y0 = max(0, y - COH_RADIUS)
            y1 = min(H, y + COH_RADIUS + 1)

            neighborhood_last = last_t_us[y0:y1, x0:x1]
            if np.any((t_us - neighborhood_last) <= COH_DT_US):
                coherent_events_acc += 1

            last_t_us[y, x] = t_us

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
            mean_x = -1.0
            mean_y = -1.0

        # B) Spatial entropy / uniformity
        flat = spatial_counts.ravel()
        active_mask = flat > 0
        active_px = int(active_mask.sum())
        active_frac = active_px / (W * H)

        if events_acc > 0:
            p = flat[active_mask].astype(np.float64) / events_acc
            entropy = float(-(p * np.log2(p)).sum())
            entropy_norm = entropy / np.log2(W * H)  # normalized to full sensor support
        else:
            entropy = 0.0
            entropy_norm = 0.0

        # D) Per-pixel firing stats
        if active_px > 0:
            mean_epp = float(flat[active_mask].mean())   # mean events per active pixel
            max_epp = int(flat.max())                    # hottest pixel count
        else:
            mean_epp = 0.0
            max_epp = 0

        hot_counts = {thr: int((flat >= thr).sum()) for thr in HOT_THRESHOLDS}

        # C) Temporal local coherence
        coh_frac = coherent_events_acc / events_acc if events_acc > 0 else 0.0

        hot_str = "  ".join([f"hot_px>={thr}:{hot_counts[thr]:5d}" for thr in HOT_THRESHOLDS])

        print(
            f"pps={pps:6.1f}  "
            f"eps={eps:8.0f}  "
            f"avg_count={avg_count:6.1f}  "
            f"max_count={max_count_acc:4d}  "
            f"mean_xy=({mean_x:6.1f},{mean_y:6.1f})  "
            f"off={off_acc:6d}  on={on_acc:6d}  "
            f"active_px={active_px:5d}  "
            f"active_frac={active_frac:6.3f}  "
            f"entropy={entropy:6.2f}  "
            f"entropy_norm={entropy_norm:5.3f}  "
            f"coh_frac={coh_frac:5.3f}  "
            f"mean_epp={mean_epp:5.2f}  "
            f"max_epp={max_epp:4d}  "
            f"{hot_str}"
        )

        # reset 1-second accumulators
        t0 = now
        packets_acc = 0
        events_acc = 0
        max_count_acc = 0
        sum_x_acc = 0
        sum_y_acc = 0
        on_acc = 0
        off_acc = 0

        spatial_counts.fill(0)
        coherent_events_acc = 0

        # Important:
        # Keep last_t_us across windows so coherence does not get artificially broken
        # at each 1-second boundary. If you want strict per-window behavior, uncomment:
        #
        # last_t_us.fill(-10**15)