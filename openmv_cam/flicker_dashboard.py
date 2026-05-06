#!/usr/bin/env python3
import argparse
import struct
import time
from collections import deque

import numpy as np
import serial
import matplotlib.pyplot as plt


MAGIC = b"EVT1"
HEADER_FMT = "<LL"
EVENT_WORDS = 6
EVENT_DTYPE = np.uint16
EVENT_SIZE_BYTES = EVENT_WORDS * np.dtype(EVENT_DTYPE).itemsize

W = 320
H = 320


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/openmvcam")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--timeout", type=float, default=3.0)

    p.add_argument("--width", type=int, default=W)
    p.add_argument("--height", type=int, default=H)

    p.add_argument("--window-sec", type=float, default=4.0)
    p.add_argument("--bin-us", type=int, default=1000)
    p.add_argument("--grid", type=int, default=3)

    p.add_argument("--flicker-center-hz", type=float, default=240.0)
    p.add_argument("--flicker-band-hz", type=float, default=20.0)
    p.add_argument("--fft-min-hz", type=float, default=40.0)
    p.add_argument("--fft-max-hz", type=float, default=300.0)

    p.add_argument("--update-hz", type=float, default=2.0)

    # Optional burst filter visualization
    p.add_argument("--burst-k", type=float, default=2.0)
    p.add_argument("--burst-min-thresh", type=float, default=30.0)

    return p.parse_args()


def open_serial(args):
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
    time.sleep(0.2)
    return ser


def read_exactly(ser, n):
    data = bytearray()
    while len(data) < n:
        chunk = ser.read(n - len(data))
        if not chunk:
            raise RuntimeError("Serial read timeout")
        data.extend(chunk)
    return bytes(data)


def read_until_magic(ser):
    window = bytearray()
    while True:
        b = ser.read(1)
        if not b:
            raise RuntimeError("Timeout while searching for magic")
        window += b
        if len(window) > len(MAGIC):
            window = window[-len(MAGIC):]
        if bytes(window) == MAGIC:
            return


def read_packet(ser):
    read_until_magic(ser)

    header = read_exactly(ser, struct.calcsize(HEADER_FMT))
    event_count, payload_len = struct.unpack(HEADER_FMT, header)

    expected_len = event_count * EVENT_SIZE_BYTES
    if payload_len != expected_len:
        raise RuntimeError(f"Bad payload_len={payload_len}, expected={expected_len}")

    payload = read_exactly(ser, payload_len)

    if event_count == 0:
        return np.empty((0, EVENT_WORDS), dtype=np.uint16)

    return np.frombuffer(payload, dtype=np.uint16).reshape(event_count, EVENT_WORDS).copy()


def event_timestamps_us(events):
    return (
        events[:, 1].astype(np.int64) * 1_000_000
        + events[:, 2].astype(np.int64) * 1_000
        + events[:, 3].astype(np.int64)
    )


def valid_events(events, width, height):
    if events.size == 0:
        return events

    xs = events[:, 4].astype(np.int32)
    ys = events[:, 5].astype(np.int32)

    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    return events[valid]


def make_count_signal(ts_us, bin_us, window_sec):
    n_bins = int(round(window_sec * 1_000_000 / bin_us))
    counts = np.zeros(n_bins, dtype=np.float64)

    if ts_us.size == 0:
        return counts

    t_end = int(ts_us.max())
    t_start = t_end - n_bins * bin_us

    idx = ((ts_us - t_start) // bin_us).astype(np.int64)
    valid = (idx >= 0) & (idx < n_bins)

    np.add.at(counts, idx[valid], 1)
    return counts


def fft_power(counts, bin_us):
    x = counts.astype(np.float64)
    x -= x.mean()

    if len(x) < 16 or np.allclose(x, 0):
        fs = 1_000_000.0 / bin_us
        freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
        return freqs, np.zeros_like(freqs)

    x *= np.hanning(len(x))
    vals = np.fft.rfft(x)
    power = np.abs(vals) ** 2

    fs = 1_000_000.0 / bin_us
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    return freqs, power


def flicker_ratio_from_counts(counts, bin_us, f_center, f_band, f_min, f_max):
    freqs, power = fft_power(counts, bin_us)

    total_mask = (freqs >= f_min) & (freqs <= f_max)
    flicker_mask = (freqs >= f_center - f_band) & (freqs <= f_center + f_band)

    total = float(power[total_mask].sum())
    flicker = float(power[flicker_mask].sum())

    if total <= 1e-12:
        return 0.0

    return flicker / total


def event_count_heatmap(events, width, height):
    heat = np.zeros((height, width), dtype=np.float32)
    if events.size == 0:
        return heat

    xs = events[:, 4].astype(np.int32)
    ys = events[:, 5].astype(np.int32)
    np.add.at(heat, (ys, xs), 1)
    return heat


def grid_flicker_heatmap(events, width, height, grid, bin_us, window_sec,
                         f_center, f_band, f_min, f_max):
    out = np.zeros((grid, grid), dtype=np.float32)

    if events.size == 0:
        return out

    ts_us = event_timestamps_us(events)
    xs = events[:, 4].astype(np.int32)
    ys = events[:, 5].astype(np.int32)

    x_edges = np.linspace(0, width, grid + 1, dtype=np.int32)
    y_edges = np.linspace(0, height, grid + 1, dtype=np.int32)

    for gy in range(grid):
        for gx in range(grid):
            mask = (
                (xs >= x_edges[gx]) & (xs < x_edges[gx + 1]) &
                (ys >= y_edges[gy]) & (ys < y_edges[gy + 1])
            )

            counts = make_count_signal(ts_us[mask], bin_us, window_sec)
            out[gy, gx] = flicker_ratio_from_counts(
                counts, bin_us, f_center, f_band, f_min, f_max
            )

    return out


def burst_filter_counts(counts, k, min_thresh):
    threshold = max(float(counts.mean() + k * counts.std()), float(min_thresh))
    filt = counts.copy()
    filt[filt > threshold] = threshold
    return filt, threshold


def main():
    args = parse_args()
    ser = open_serial(args)

    packet_buffer = deque()
    last_update = 0.0
    t0_wall = time.time()

    plt.ion()
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("OpenMV GenX320 Flicker Dashboard")

    ax_signal = axs[0, 0]
    ax_fft = axs[0, 1]
    ax_grid = axs[0, 2]
    ax_heat = axs[1, 0]
    ax_raw_filt = axs[1, 1]
    ax_info = axs[1, 2]

    print("Reading EVT1 stream. Stop with Ctrl+C.")

    try:
        while True:
            try:
                events = read_packet(ser)
                events = valid_events(events, args.width, args.height)
            except Exception as e:
                print(f"[WARN] read failed: {e}")
                time.sleep(0.05)
                continue

            if events.size > 0:
                ts_us = event_timestamps_us(events)
                packet_buffer.append((int(ts_us.max()), events))

                t_cut = int(ts_us.max() - args.window_sec * 1_000_000)
                while packet_buffer and packet_buffer[0][0] < t_cut:
                    packet_buffer.popleft()

            now = time.time()
            if now - last_update < 1.0 / args.update_hz:
                continue
            last_update = now

            if not packet_buffer:
                continue

            all_events = np.concatenate([ev for _, ev in packet_buffer], axis=0)
            ts_all = event_timestamps_us(all_events)

            counts = make_count_signal(ts_all, args.bin_us, args.window_sec)
            freqs, power = fft_power(counts, args.bin_us)

            filt_counts, burst_thr = burst_filter_counts(
                counts,
                k=args.burst_k,
                min_thresh=args.burst_min_thresh,
            )
            freqs_filt, power_filt = fft_power(filt_counts, args.bin_us)

            heat = event_count_heatmap(all_events, args.width, args.height)
            grid_heat = grid_flicker_heatmap(
                all_events,
                args.width,
                args.height,
                args.grid,
                args.bin_us,
                args.window_sec,
                args.flicker_center_hz,
                args.flicker_band_hz,
                args.fft_min_hz,
                args.fft_max_hz,
            )

            flicker_ratio_global = flicker_ratio_from_counts(
                counts,
                args.bin_us,
                args.flicker_center_hz,
                args.flicker_band_hz,
                args.fft_min_hz,
                args.fft_max_hz,
            )

            flicker_ratio_filtered = flicker_ratio_from_counts(
                filt_counts,
                args.bin_us,
                args.flicker_center_hz,
                args.flicker_band_hz,
                args.fft_min_hz,
                args.fft_max_hz,
            )

            duration = args.window_sec
            eps = len(all_events) / duration
            active_px = int((heat > 0).sum())
            active_frac = active_px / (args.width * args.height)

            # Clear axes
            for ax in axs.ravel():
                ax.clear()

            # 1. event count signal
            t_axis = np.arange(len(counts)) * args.bin_us / 1_000_000.0
            ax_signal.plot(t_axis, counts, linewidth=1)
            ax_signal.set_title("Global event count, 1 ms bins")
            ax_signal.set_xlabel("time in rolling window [s]")
            ax_signal.set_ylabel("events / bin")
            ax_signal.grid(True)

            # 2. FFT global
            band = (freqs >= args.fft_min_hz) & (freqs <= args.fft_max_hz)
            ax_fft.plot(freqs[band], power[band], linewidth=1)
            ax_fft.axvline(args.flicker_center_hz, linestyle="--", linewidth=1)
            ax_fft.set_title("FFT of global count")
            ax_fft.set_xlabel("frequency [Hz]")
            ax_fft.set_ylabel("power")
            ax_fft.grid(True)

            # 3. 3x3 flicker heatmap
            im_grid = ax_grid.imshow(grid_heat, vmin=0.0, vmax=1.0,aspect='equal')
            ax_grid.set_title(
                f"{args.grid}x{args.grid} flicker ratio\n"
                f"power {args.flicker_center_hz-args.flicker_band_hz:.0f}–"
                f"{args.flicker_center_hz+args.flicker_band_hz:.0f} Hz / "
                f"{args.fft_min_hz:.0f}–{args.fft_max_hz:.0f} Hz"
            )
            for gy in range(args.grid):
                for gx in range(args.grid):
                    ax_grid.text(
                        gx, gy, f"{grid_heat[gy, gx]:.2f}",
                        ha="center", va="center"
                    )
            #fig.colorbar(im_grid, ax=ax_grid, fraction=0.046, pad=0.04)

            # 4. per-pixel event-count heatmap
            # log scale for readability
            heat_log = np.log1p(heat)
            im_heat = ax_heat.imshow(heat_log,aspect='equal')
            ax_heat.set_title("Per-pixel event count heatmap, log(1+count)")
            ax_heat.set_xlabel("x")
            ax_heat.set_ylabel("y")
            #fig.colorbar(im_heat, ax=ax_heat, fraction=0.046, pad=0.04)

            # 5. raw vs burst-clipped count signal
            ax_raw_filt.plot(t_axis, counts, label="raw", linewidth=1)
            ax_raw_filt.plot(t_axis, filt_counts, label="burst-clipped", linewidth=1)
            ax_raw_filt.axhline(burst_thr, linestyle="--", linewidth=1)
            ax_raw_filt.set_title("Raw vs simple burst-clipped count")
            ax_raw_filt.set_xlabel("time [s]")
            ax_raw_filt.set_ylabel("events / bin")
            ax_raw_filt.legend()
            ax_raw_filt.grid(True)

            # Info panel
            ax_info.axis("off")

            # top peaks
            band_power = power[band]
            band_freqs = freqs[band]
            if band_power.size > 0 and band_power.max() > 0:
                top_idx = np.argsort(band_power)[::-1][:5]
                peaks_txt = "\n".join(
                    f"{band_freqs[i]:7.1f} Hz   rel={band_power[i] / band_power.max():.2f}"
                    for i in top_idx
                )
            else:
                peaks_txt = "no peaks"

            info = (
                f"window: {args.window_sec:.1f} s\n"
                f"events in window: {len(all_events)}\n"
                f"eps: {eps:.0f}\n"
                f"active_px: {active_px} ({active_frac:.3f})\n"
                f"global flicker_ratio: {flicker_ratio_global:.3f}\n"
                f"burst-clipped flicker_ratio: {flicker_ratio_filtered:.3f}\n"
                f"burst threshold: {burst_thr:.1f}\n\n"
                f"Top FFT peaks:\n{peaks_txt}"
            )
            ax_info.text(0.02, 0.98, info, va="top", family="monospace")
            ax_info.set_title("Stats")

            
        

            #fig.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()