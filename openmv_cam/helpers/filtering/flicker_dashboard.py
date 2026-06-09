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
    p.add_argument("--tile", type=int, default=16, help="tile size in pixels for spatial extent metric")

    p.add_argument("--flicker-center-hz", type=float, default=240.0)
    p.add_argument("--flicker-band-hz", type=float, default=20.0)
    p.add_argument("--fft-min-hz", type=float, default=40.0)
    p.add_argument("--fft-max-hz", type=float, default=300.0)
    p.add_argument("--top-pixels", type=int, default=6, help="number of most active pixels used for per-pixel FFT")

    p.add_argument("--update-hz", type=float, default=2.0)

    # Optional burst filter visualization
    p.add_argument("--burst-k", type=float, default=2.0)
    p.add_argument("--burst-min-thresh", type=float, default=30.0)

    # Phase histogram
    p.add_argument("--phase-hz", type=float, default=None,
                   help="frequency for phase histogram; defaults to --flicker-center-hz")
    p.add_argument("--phase-bins", type=int, default=48)
    p.add_argument("--adaptive-phase-freq", action="store_true")
    p.add_argument("--phase-search-min-hz", type=float, default=220.0)
    p.add_argument("--phase-search-max-hz", type=float, default=260.0)
    p.add_argument("--phase-freq-alpha", type=float, default=0.2)

    args = p.parse_args()
    if args.phase_hz is None:
        args.phase_hz = args.flicker_center_hz
    return args


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


def make_count_signal(ts_us, bin_us, window_sec, t_end=None):
    n_bins = int(round(window_sec * 1_000_000 / bin_us))
    counts = np.zeros(n_bins, dtype=np.float64)

    if ts_us.size == 0:
        return counts

    if t_end is None:
        t_end = int(ts_us.max())
    t_start = int(t_end - n_bins * bin_us)

    idx = ((ts_us - t_start) // bin_us).astype(np.int64)
    valid = (idx >= 0) & (idx < n_bins)

    np.add.at(counts, idx[valid], 1)
    return counts


def bin_indices(ts_us, bin_us, window_sec, t_end):
    n_bins = int(round(window_sec * 1_000_000 / bin_us))
    t_start = int(t_end - n_bins * bin_us)
    idx = ((ts_us - t_start) // bin_us).astype(np.int64)
    valid = (idx >= 0) & (idx < n_bins)
    return idx, valid, n_bins


def fft_power(counts, bin_us):
    x = counts.astype(np.float64)
    x -= x.mean()

    fs = 1_000_000.0 / bin_us
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)

    if len(x) < 16 or np.allclose(x, 0):
        return freqs, np.zeros_like(freqs)

    x *= np.hanning(len(x))
    vals = np.fft.rfft(x)
    power = np.abs(vals) ** 2
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


def accumulated_event_frame(events, width, height):
    frame = np.zeros((height, width), dtype=np.float32)
    if events.size == 0:
        return frame

    xs = events[:, 4].astype(np.int32)
    ys = events[:, 5].astype(np.int32)
    pol = events[:, 0].astype(np.int32)
    delta = np.where(pol == 1, 1.0, -1.0)
    np.add.at(frame, (ys, xs), delta)
    return frame


def grid_flicker_heatmap(events, width, height, grid, bin_us, window_sec,
                         f_center, f_band, f_min, f_max, t_end):
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

            counts = make_count_signal(ts_us[mask], bin_us, window_sec, t_end=t_end)
            out[gy, gx] = flicker_ratio_from_counts(
                counts, bin_us, f_center, f_band, f_min, f_max
            )

    return out


def burst_filter_counts(counts, k, min_thresh):
    threshold = max(float(counts.mean() + k * counts.std()), float(min_thresh))
    filt = counts.copy()
    filt[filt > threshold] = threshold
    return filt, threshold


def polarity_counts(events, bin_us, window_sec, t_end):
    n_bins = int(round(window_sec * 1_000_000 / bin_us))
    on = np.zeros(n_bins, dtype=np.float64)
    off = np.zeros(n_bins, dtype=np.float64)
    if events.size == 0:
        return on, off

    ts = event_timestamps_us(events)
    idx, valid, n_bins = bin_indices(ts, bin_us, window_sec, t_end)
    pol = events[:, 0].astype(np.int32)

    # Assumes type 1 = ON and type 0 = OFF. If your firmware uses the opposite,
    # the labels are swapped but the diagnostic remains useful.
    np.add.at(on, idx[valid & (pol == 1)], 1)
    np.add.at(off, idx[valid & (pol == 0)], 1)
    return on, off


def active_pixel_and_tile_counts(events, width, height, bin_us, window_sec, t_end, tile):
    n_bins = int(round(window_sec * 1_000_000 / bin_us))
    active_px = np.zeros(n_bins, dtype=np.float64)
    active_tiles = np.zeros(n_bins, dtype=np.float64)

    if events.size == 0:
        return active_px, active_tiles

    ts = event_timestamps_us(events)
    xs = events[:, 4].astype(np.int32)
    ys = events[:, 5].astype(np.int32)
    idx, valid, n_bins = bin_indices(ts, bin_us, window_sec, t_end)

    pix_id = ys * width + xs
    tiles_x = int(np.ceil(width / tile))
    tx = np.clip(xs // tile, 0, tiles_x - 1)
    ty = np.clip(ys // tile, 0, int(np.ceil(height / tile)) - 1)
    tile_id = ty * tiles_x + tx

    for b in np.unique(idx[valid]):
        m = valid & (idx == b)
        active_px[b] = np.unique(pix_id[m]).size
        active_tiles[b] = np.unique(tile_id[m]).size

    return active_px, active_tiles


def phase_histogram(events, phase_hz, phase_bins):
    """Return (bin_centers, hist_on_norm, hist_off_norm, hist_on_raw, hist_off_raw).

    hist_on_norm / hist_off_norm are each normalized by their own max (if > 0).
    hist_on_raw / hist_off_raw are the raw integer counts (needed for stats).
    """
    bin_centers = (np.arange(phase_bins) + 0.5) / phase_bins
    hist_on_raw = np.zeros(phase_bins, dtype=np.float64)
    hist_off_raw = np.zeros(phase_bins, dtype=np.float64)

    if events.size == 0:
        return bin_centers, hist_on_raw, hist_off_raw, hist_on_raw, hist_off_raw

    ts_us = event_timestamps_us(events)
    phase = (ts_us * 1e-6 * phase_hz) % 1.0
    pol = events[:, 0].astype(np.int32)

    on_mask = pol == 1
    off_mask = pol == 0

    hist_on_raw, _ = np.histogram(phase[on_mask], bins=phase_bins, range=(0.0, 1.0))
    hist_off_raw, _ = np.histogram(phase[off_mask], bins=phase_bins, range=(0.0, 1.0))

    hist_on_raw = hist_on_raw.astype(np.float64)
    hist_off_raw = hist_off_raw.astype(np.float64)

    hist_on = hist_on_raw / hist_on_raw.max() if hist_on_raw.max() > 0 else hist_on_raw.copy()
    hist_off = hist_off_raw / hist_off_raw.max() if hist_off_raw.max() > 0 else hist_off_raw.copy()

    return bin_centers, hist_on, hist_off, hist_on_raw, hist_off_raw


def top_pixel_fft_curves(events, width, height, bin_us, window_sec, t_end, top_n, f_min, f_max):
    if events.size == 0:
        return [], None, None

    heat = event_count_heatmap(events, width, height).ravel()
    active = np.flatnonzero(heat > 0)
    if active.size == 0:
        return [], None, None

    top_ids = active[np.argsort(heat[active])[::-1][:top_n]]

    ts = event_timestamps_us(events)
    xs = events[:, 4].astype(np.int32)
    ys = events[:, 5].astype(np.int32)
    pix_id = ys * width + xs

    curves = []
    for pid in top_ids:
        counts = make_count_signal(ts[pix_id == pid], bin_us, window_sec, t_end=t_end)
        freqs, power = fft_power(counts, bin_us)
        band = (freqs >= f_min) & (freqs <= f_max)
        peak_hz = float(freqs[band][np.argmax(power[band])]) if np.any(band) and power[band].max() > 0 else 0.0
        y = pid // width
        x = pid % width
        curves.append((x, y, counts.sum(), peak_hz, freqs, power))

    return curves, freqs, None


def main():
    args = parse_args()
    ser = open_serial(args)

    packet_buffer = deque()
    last_update = 0.0
    smoothed_phase_freq = float(args.phase_hz)

    plt.ion()
    fig, axs = plt.subplots(3, 3, figsize=(17, 10))
    fig.suptitle("OpenMV GenX320 Flicker Dashboard + Noise Metrics")

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
            t_end = int(ts_all.max())

            counts = make_count_signal(ts_all, args.bin_us, args.window_sec, t_end=t_end)
            freqs, power = fft_power(counts, args.bin_us)

            phase_band_mask = (
                (freqs >= args.phase_search_min_hz) &
                (freqs <= args.phase_search_max_hz)
            )
            phase_band_freqs = freqs[phase_band_mask]
            phase_band_power = power[phase_band_mask]
            if phase_band_power.size > 0 and phase_band_power.max() > 1e-12:
                dominant_freq = float(phase_band_freqs[np.argmax(phase_band_power)])
            else:
                dominant_freq = float(args.phase_hz)

            alpha = float(np.clip(args.phase_freq_alpha, 0.0, 1.0))
            smoothed_phase_freq = alpha * dominant_freq + (1.0 - alpha) * smoothed_phase_freq
            phase_ref_hz = smoothed_phase_freq if args.adaptive_phase_freq else float(args.phase_hz)

            filt_counts, burst_thr = burst_filter_counts(counts, args.burst_k, args.burst_min_thresh)
            on_counts, off_counts = polarity_counts(all_events, args.bin_us, args.window_sec, t_end)
            active_px_bin, active_tiles_bin = active_pixel_and_tile_counts(
                all_events, args.width, args.height, args.bin_us, args.window_sec, t_end, args.tile
            )

            heat = event_count_heatmap(all_events, args.width, args.height)
            event_frame = accumulated_event_frame(all_events, args.width, args.height)
            grid_heat = grid_flicker_heatmap(
                all_events, args.width, args.height, args.grid, args.bin_us, args.window_sec,
                args.flicker_center_hz, args.flicker_band_hz, args.fft_min_hz, args.fft_max_hz, t_end
            )

            flicker_ratio_global = flicker_ratio_from_counts(
                counts, args.bin_us, args.flicker_center_hz, args.flicker_band_hz,
                args.fft_min_hz, args.fft_max_hz
            )
            flicker_ratio_filtered = flicker_ratio_from_counts(
                filt_counts, args.bin_us, args.flicker_center_hz, args.flicker_band_hz,
                args.fft_min_hz, args.fft_max_hz
            )

            ph_centers, ph_on, ph_off, ph_on_raw, ph_off_raw = phase_histogram(
                all_events, phase_ref_hz, args.phase_bins
            )

            eps = len(all_events) / args.window_sec
            active_px_total = int((heat > 0).sum())
            active_frac = active_px_total / (args.width * args.height)
            t_axis = np.arange(len(counts)) * args.bin_us / 1_000_000.0

            for ax in axs.ravel():
                ax.clear()

            # 1. Global count
            ax = axs[0, 0]
            ax.plot(t_axis, counts, linewidth=1)
            ax.set_title("Global event count")
            ax.set_xlabel("time [s]")
            ax.set_ylabel("events / bin")
            ax.grid(True)

            # 2. FFT global
            ax = axs[0, 1]
            band = (freqs >= args.fft_min_hz) & (freqs <= args.fft_max_hz)
            ax.plot(freqs[band], power[band], linewidth=1)
            ax.axvline(args.flicker_center_hz, linestyle="--", linewidth=1)
            ax.set_title("FFT of global count")
            ax.set_xlabel("frequency [Hz]")
            ax.set_ylabel("power")
            ax.set_xlim(args.fft_min_hz, args.fft_max_hz)
            ax.grid(True)

            # 3. Grid flicker heatmap
            ax = axs[0, 2]
            ax.imshow(grid_heat, vmin=0.0, vmax=1.0, aspect="equal")
            ax.set_title(f"{args.grid}x{args.grid} flicker ratio")
            for gy in range(args.grid):
                for gx in range(args.grid):
                    ax.text(gx, gy, f"{grid_heat[gy, gx]:.2f}", ha="center", va="center")

            # 4. Polarity balance
            ax = axs[1, 0]
            ax.plot(t_axis, on_counts, label="ON/type=1", linewidth=1)
            ax.plot(t_axis, off_counts, label="OFF/type=0", linewidth=1)
            ax.set_title("Polarity counts")
            ax.set_xlabel("time [s]")
            ax.set_ylabel("events / bin")
            ax.legend()
            ax.grid(True)

            # 5. Spatial coherence: active pixels and events per active pixel
            ax = axs[1, 1]
            events_per_active_px = counts / np.maximum(active_px_bin, 1)
            ax.plot(t_axis, active_px_bin, label="active pixels/bin", linewidth=1)
            ax.plot(t_axis, events_per_active_px, label="events/active pixel", linewidth=1)
            ax.set_title("Spatial coherence")
            ax.set_xlabel("time [s]")
            ax.legend()
            ax.grid(True)

            # 6. Event frame (ON brightens, OFF darkens)
            ax = axs[1, 2]
            frame_abs_max = float(np.max(np.abs(event_frame)))
            if frame_abs_max <= 0:
                frame_abs_max = 1.0
            ax.imshow(event_frame, cmap="gray", vmin=-frame_abs_max, vmax=frame_abs_max, aspect="equal")
            ax.set_title("Accumulated event frame (ON+/OFF-)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            # 7. Per-pixel heatmap
            ax = axs[2, 0]
            ax.imshow(np.log1p(heat), aspect="equal")
            ax.set_title("Per-pixel event count, log(1+count)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            # 8. Phase histogram (replaces "FFT of top active pixels")
            ax = axs[2, 1]
            ax.step(ph_centers, ph_on, where="mid", label="ON", linewidth=1)
            ax.step(ph_centers, ph_off, where="mid", label="OFF", linewidth=1)
            phase_mode = "adaptive" if args.adaptive_phase_freq else "fixed"
            ax.set_title(f"Event phase histogram @ {phase_mode} {phase_ref_hz:.1f} Hz")
            ax.set_xlabel("phase cycles [0, 1)")
            ax.set_ylabel("normalized count")
            ax.set_xlim(0.0, 1.0)
            ax.legend(fontsize=8)
            ax.grid(True, axis="x")

            # 9. Stats
            ax = axs[2, 2]
            ax.axis("off")
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

            on_total = float(on_counts.sum())
            off_total = float(off_counts.sum())
            pol_ratio = on_total / max(off_total, 1.0)
            high_bins = counts > burst_thr
            mean_extent_high = float(active_tiles_bin[high_bins].mean()) if np.any(high_bins) else 0.0
            mean_px_high = float(active_px_bin[high_bins].mean()) if np.any(high_bins) else 0.0

            on_peak_phase = float(ph_centers[np.argmax(ph_on_raw)]) if ph_on_raw.max() > 0 else 0.0
            off_peak_phase = float(ph_centers[np.argmax(ph_off_raw)]) if ph_off_raw.max() > 0 else 0.0
            on_concentration = float(ph_on_raw.max()) / float(ph_on_raw.sum()) if ph_on_raw.sum() > 0 else 0.0
            off_concentration = float(ph_off_raw.max()) / float(ph_off_raw.sum()) if ph_off_raw.sum() > 0 else 0.0

            info = (
                f"window: {args.window_sec:.1f} s\n"
                f"events: {len(all_events)}\n"
                f"eps: {eps:.0f}\n"
                f"active_px total: {active_px_total} ({active_frac:.3f})\n"
                f"ON/OFF total ratio: {pol_ratio:.2f}\n"
                f"global flicker_ratio: {flicker_ratio_global:.3f}\n"
                f"burst-clipped flicker_ratio: {flicker_ratio_filtered:.3f}\n"
                f"burst threshold: {burst_thr:.1f}\n"
                f"burst bins: {int(high_bins.sum())}/{len(counts)}\n"
                f"mean active px in burst bins: {mean_px_high:.1f}\n"
                f"mean active tiles in burst bins: {mean_extent_high:.1f}\n\n"
                f"dominant flicker freq: {phase_ref_hz:.1f} Hz\n"
                f"Phase histogram @ {phase_ref_hz:.1f} Hz\n"
                f"  ON  peak phase:  {on_peak_phase:.3f}\n"
                f"  OFF peak phase:  {off_peak_phase:.3f}\n"
                f"  ON  concentration: {on_concentration:.3f}\n"
                f"  OFF concentration: {off_concentration:.3f}\n\n"
                f"Top global FFT peaks:\n{peaks_txt}"
            )
            ax.text(0.02, 0.98, info, va="top", family="monospace")
            ax.set_title("Stats")

            fig.tight_layout()
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