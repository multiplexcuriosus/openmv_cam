#!/usr/bin/env python3
"""
Capture an EVT1 event stream for a fixed duration, then compute and plot the
same global event-count FFT used in the live dashboard.

Expected event packet format, inherited from the dashboard script:
  MAGIC: b"EVT1"
  HEADER: <LL = event_count, payload_len
  EVENT:  6 uint16 words = polarity/type, seconds, milliseconds, microseconds, x, y

Example:
  python3 capture_event_fft.py --duration-sec 10 --port /dev/openmvcam --out-dir fft_report
"""

import argparse
import os
import struct
import time
from pathlib import Path

import numpy as np
import serial
import matplotlib.pyplot as plt

MAGIC = b"EVT1"
HEADER_FMT = "<LL"
EVENT_WORDS = 6
EVENT_DTYPE = np.uint16
EVENT_SIZE_BYTES = EVENT_WORDS * np.dtype(EVENT_DTYPE).itemsize

DEFAULT_W = 320
DEFAULT_H = 320


def parse_args():
    p = argparse.ArgumentParser(
        description="Capture EVT1 events for a fixed duration and save a report-ready FFT plot."
    )

    # Capture / serial settings
    p.add_argument("--port", default="/dev/openmvcam", help="Serial device, e.g. /dev/openmvcam")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--timeout", type=float, default=1.0, help="Serial read timeout in seconds")
    p.add_argument("--duration-sec", type=float, default=None,
                   help="Capture/analysis duration. Default: 10 s for live capture, saved value for --load-npz.")
    p.add_argument("--warmup-sec", type=float, default=0.0,
                   help="Optional delay after opening serial before recording starts")
    p.add_argument("--max-read-errors", type=int, default=50,
                   help="Abort after this many serial packet read failures")

    # Event format / image geometry
    p.add_argument("--width", type=int, default=DEFAULT_W)
    p.add_argument("--height", type=int, default=DEFAULT_H)

    # FFT settings: intentionally match the dashboard defaults/logic
    p.add_argument("--bin-us", type=int, default=1000,
                   help="Temporal bin width for global event count signal")
    p.add_argument("--fft-min-hz", type=float, default=40.0)
    p.add_argument("--fft-max-hz", type=float, default=300.0)
    p.add_argument("--top-peaks", type=int, default=5)
    p.add_argument("--min-peak-separation-hz", type=float, default=5.0)
    p.add_argument("--reference-hz", type=float, nargs="*", default=[60.0, 120.0, 240.0],
                   help="Reference vertical lines to draw on FFT plot")

    # Plot/output settings
    p.add_argument("--out-dir", default="event_fft_out")
    p.add_argument("--name", default=None,
                   help="Output stem. Default: event_fft_<timestamp>")
    p.add_argument("--title", default="Event stream flicker spectrum")
    p.add_argument("--plot-format", choices=["png", "pdf", "both"], default="both")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--normalize-power", action="store_true",
                   help="Plot FFT power normalized by the max power in the displayed band")
    p.add_argument("--log-power", action="store_true",
                   help="Use logarithmic y-axis for FFT power")
    p.add_argument("--show", action="store_true", help="Show the figure interactively after saving")

    # Offline replotting
    p.add_argument("--load-npz", default=None,
                   help="Load a previous .npz event capture instead of reading serial")
    p.add_argument("--no-save-npz", action="store_true",
                   help="Do not save raw events/counts/frequency data")

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


def make_count_signal(ts_us, bin_us, window_sec, t_end=None):
    """Same global count signal logic as the dashboard script."""
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


def fft_power(counts, bin_us):
    """Same FFT as the dashboard: demean, Hanning window, rFFT, squared magnitude."""
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


def capture_events(args, duration_sec):
    print(f"Opening {args.port} at {args.baud} baud")
    ser = open_serial(args)
    events_list = []
    read_errors = 0
    packets = 0
    last_status = 0.0

    try:
        if args.warmup_sec > 0:
            print(f"Warmup: {args.warmup_sec:.2f} s")
            time.sleep(args.warmup_sec)
            ser.reset_input_buffer()

        print(f"Recording for {duration_sec:.2f} s ...")
        t0 = time.monotonic()

        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= duration_sec:
                break

            try:
                events = read_packet(ser)
                events = valid_events(events, args.width, args.height)
                packets += 1
                read_errors = 0
                if events.size > 0:
                    events_list.append(events)
            except Exception as e:
                read_errors += 1
                if read_errors >= args.max_read_errors:
                    raise RuntimeError(
                        f"Too many serial read errors ({read_errors}). Last error: {e}"
                    ) from e

            now = time.monotonic()
            if now - last_status >= 1.0:
                n_events = sum(len(ev) for ev in events_list)
                print(f"  {min(now - t0, duration_sec):5.1f}/{duration_sec:.1f} s | "
                      f"packets={packets} | events={n_events:,}")
                last_status = now

    finally:
        ser.close()

    if not events_list:
        return np.empty((0, EVENT_WORDS), dtype=np.uint16)

    return np.concatenate(events_list, axis=0)


def load_events_npz(path):
    data = np.load(path, allow_pickle=False)
    if "events" not in data:
        raise ValueError(f"{path} does not contain an 'events' array")
    events = data["events"]
    saved_duration = float(data["duration_sec"]) if "duration_sec" in data else None
    return events, saved_duration


def top_fft_peaks(freqs, power, f_min, f_max, n=5, min_separation_hz=5.0):
    band = (freqs >= f_min) & (freqs <= f_max)
    band_idxs = np.flatnonzero(band)
    if band_idxs.size == 0 or np.max(power[band]) <= 0:
        return []

    sorted_idxs = band_idxs[np.argsort(power[band_idxs])[::-1]]
    peaks = []
    for idx in sorted_idxs:
        f = float(freqs[idx])
        p = float(power[idx])
        if all(abs(f - existing_f) >= min_separation_hz for existing_f, _ in peaks):
            peaks.append((f, p))
        if len(peaks) >= n:
            break
    return peaks


def make_report_plot(counts, freqs, power, events, args, duration_sec, out_paths):
    band = (freqs >= args.fft_min_hz) & (freqs <= args.fft_max_hz)
    if not np.any(band):
        raise ValueError("No FFT frequencies inside requested display band")

    plot_power = power.copy()
    y_label = "FFT power [a.u.]"
    if args.normalize_power:
        denom = float(np.max(power[band]))
        if denom > 0:
            plot_power = plot_power / denom
        y_label = "Relative FFT power [max in band = 1]"

    t_axis = np.arange(len(counts)) * args.bin_us / 1_000_000.0
    bin_ms = args.bin_us / 1000.0
    freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else float("nan")

    peaks = top_fft_peaks(
        freqs, power,
        args.fft_min_hz, args.fft_max_hz,
        n=args.top_peaks,
        min_separation_hz=args.min_peak_separation_hz,
    )
    peak_text = "No clear peak"
    if peaks:
        peak_text = ", ".join(f"{f:.1f} Hz" for f, _ in peaks[:3])

    fig, (ax_time, ax_fft) = plt.subplots(
        2, 1,
        figsize=(7.2, 5.2),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.45]},
    )

    fig.suptitle(args.title, fontsize=13, fontweight="bold")

    # Time-domain global count signal
    ax_time.plot(t_axis, counts, linewidth=0.9)
    ax_time.set_title("Global event count signal", fontsize=10)
    ax_time.set_xlabel("Time [s]")
    ax_time.set_ylabel(f"Events / {bin_ms:g} ms bin")
    ax_time.set_xlim(0, duration_sec)
    ax_time.grid(True, linewidth=0.5, alpha=0.45)

    # Frequency-domain signal
    ax_fft.plot(freqs[band], plot_power[band], linewidth=1.4)
    ax_fft.set_title("FFT of global event count", fontsize=10)
    ax_fft.set_xlabel("Frequency [Hz]")
    ax_fft.set_ylabel(y_label)
    ax_fft.set_xlim(args.fft_min_hz, args.fft_max_hz)
    ax_fft.grid(True, linewidth=0.5, alpha=0.45)

    if args.log_power:
        # Avoid log(0) problems. This only affects plotting, not FFT computation.
        positive = plot_power[band][plot_power[band] > 0]
        if positive.size > 0:
            ax_fft.set_yscale("log")

    # Reference frequencies, e.g. 60/120/240 Hz
    y_top = ax_fft.get_ylim()[1]
    for ref in args.reference_hz:
        if args.fft_min_hz <= ref <= args.fft_max_hz:
            ax_fft.axvline(ref, linestyle="--", linewidth=0.9, alpha=0.65)
            ax_fft.text(ref + 5.0, y_top * 0.98, f" {ref:g} Hz", rotation=90,
                        va="top", ha="left", fontsize=8, alpha=0.8)

    # Mark selected local/global dominant peaks
    # if peaks:
    #     peak_max_power = max(p for _, p in peaks)
    #     for f, p_raw in peaks:
    #         idx = int(np.argmin(np.abs(freqs - f)))
    #         rel = p_raw / peak_max_power if peak_max_power > 0 else 0.0
    #         if rel > 0.2:
    #             p_plot = float(plot_power[idx])
    #             ax_fft.plot([f], [p_plot], marker="o", markersize=3.5)
    #             ax_fft.annotate(
    #                 f"{f:.1f} Hz",
    #                 xy=(f, p_plot),
    #                 xytext=(-40, -5), #4,1 for robot-room fft report plot
    #                 textcoords="offset points",
    #                 fontsize=8,
    #                 rotation=3,
    #             )

    events_per_sec = len(events) / duration_sec if duration_sec > 0 else float("nan")
    info = (
        f"Duration: {duration_sec:.2f} s   "
        f"Events: {len(events):,}   "
        f"Rate: {events_per_sec:,.0f} events/s   "
        f"Bin: {args.bin_us} µs   "
        f"Frequency resolution: {freq_resolution:.2f} Hz   "
        f"Top peaks: {peak_text}"
    )
    #fig.text(0.01, 0.005, info, fontsize=8, va="bottom")

    for path in out_paths:
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved plot: {path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.name = f"event_fft_{timestamp}"

    if args.load_npz:
        events, saved_duration = load_events_npz(args.load_npz)
        duration_sec = args.duration_sec if args.duration_sec is not None else saved_duration
        if duration_sec is None:
            raise ValueError("--duration-sec is required when the NPZ file has no saved duration_sec")
        print(f"Loaded {len(events):,} events from {args.load_npz}")
    else:
        duration_sec = args.duration_sec if args.duration_sec is not None else 10.0
        events = capture_events(args, duration_sec)

    events = valid_events(events, args.width, args.height)
    if len(events) == 0:
        raise RuntimeError("No valid events available for FFT")

    ts_us = event_timestamps_us(events)
    t_end = int(ts_us.max())

    # Same count signal and FFT pipeline as the live dashboard.
    counts = make_count_signal(ts_us, args.bin_us, duration_sec, t_end=t_end)
    freqs, power = fft_power(counts, args.bin_us)

    png_path = out_dir / f"{args.name}.png"
    pdf_path = out_dir / f"{args.name}.pdf"
    if args.plot_format == "png":
        out_paths = [png_path]
    elif args.plot_format == "pdf":
        out_paths = [pdf_path]
    else:
        out_paths = [png_path, pdf_path]

    make_report_plot(counts, freqs, power, events, args, duration_sec, out_paths)

    if not args.no_save_npz:
        npz_path = out_dir / f"{args.name}.npz"
        np.savez_compressed(
            npz_path,
            events=events,
            event_columns=np.array(["polarity_or_type", "sec", "ms", "us", "x", "y"]),
            counts=counts,
            freqs=freqs,
            power=power,
            duration_sec=np.array(duration_sec, dtype=np.float64),
            bin_us=np.array(args.bin_us, dtype=np.int64),
            fft_min_hz=np.array(args.fft_min_hz, dtype=np.float64),
            fft_max_hz=np.array(args.fft_max_hz, dtype=np.float64),
        )
        print(f"Saved data: {npz_path}")

    peaks = top_fft_peaks(
        freqs, power,
        args.fft_min_hz, args.fft_max_hz,
        n=args.top_peaks,
        min_separation_hz=args.min_peak_separation_hz,
    )
    print("\nTop FFT peaks:")
    if peaks:
        max_p = max(p for _, p in peaks)
        for f, p_raw in peaks:
            rel = p_raw / max_p if max_p > 0 else 0.0
            print(f"  {f:8.2f} Hz   rel={rel:.3f}")
    else:
        print("  none")


if __name__ == "__main__":
    main()
