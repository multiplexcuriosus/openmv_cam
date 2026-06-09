#!/usr/bin/env python3
import argparse
import struct
import time

import numpy as np
import serial

MAGIC = b"EVT1"
HEADER_FMT = "<LL"
EVENT_WORDS = 6
EVENT_DTYPE = np.uint16
EVENT_SIZE_BYTES = EVENT_WORDS * np.dtype(EVENT_DTYPE).itemsize

W = 320
H = 320

GRID_N = 3
ROI_COL = 1
ROI_ROW = 1

PHASE_PERIOD_US = 4170        # ~240 Hz
PHASE_BIN_US = 50             # coarse binning
PHASE_TOL_BINS = 4            # width of suppression window
PHASE_MIN_SAMPLES = 2000      # wait before estimating

FFT_WINDOW_SEC = 4.0
FREQ_MIN_HZ = 40.0
FREQ_MAX_HZ = 300.0
TOP_K = 5

COH_RADIUS = 1
COH_DT_US = 5000
HOT_THRESHOLDS = [5, 10, 20]

def phase_filter(events, ts_us,
                 period_us=4170,
                 bin_us=50,
                 tol_bins=4):
    """
    Removes events occurring in dominant phase of periodic flicker.
    """

    if events.size == 0:
        return events, ts_us, 1.0, -1

    phase = ts_us % period_us
    phase_bins = (phase // bin_us).astype(np.int32)

    num_bins = int(period_us // bin_us)
    hist = np.bincount(phase_bins, minlength=num_bins)

    bad_bin = int(np.argmax(hist))

    # mask: keep everything NOT near bad phase
    diff = np.abs(phase_bins - bad_bin)
    diff = np.minimum(diff, num_bins - diff)  # circular distance

    keep = diff > tol_bins

    keep_ratio = float(keep.mean())

    return events[keep], ts_us[keep], keep_ratio, bad_bin

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/openmvcam")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--timeout", type=float, default=3.0)
    p.add_argument("--width", type=int, default=W)
    p.add_argument("--height", type=int, default=H)

    # burst suppression
    p.add_argument("--burst-bin-us", type=int, default=1000)
    p.add_argument("--burst-k", type=float, default=2.0)
    p.add_argument("--burst-min-thresh", type=float, default=30.0)
    p.add_argument("--disable-burst-filter", action="store_true")

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


def reconstruct_t_us(events_u16: np.ndarray) -> np.ndarray:
    sec = events_u16[:, 1].astype(np.int64)
    ms = events_u16[:, 2].astype(np.int64)
    us = events_u16[:, 3].astype(np.int64)
    return sec * 1_000_000 + ms * 1_000 + us


def roi_bounds_3x3(w: int, h: int, row: int, col: int):
    x_edges = np.linspace(0, w, GRID_N + 1, dtype=np.int32)
    y_edges = np.linspace(0, h, GRID_N + 1, dtype=np.int32)
    return int(x_edges[col]), int(x_edges[col + 1]), int(y_edges[row]), int(y_edges[row + 1])


def suppress_global_bursts(events, ts_us, bin_us=1000, k=2.0, min_thresh=30.0):
    """
    Drops events that occur in globally over-active time bins.

    Flicker pattern:
      many pixels fire in the same 1 ms bin

    Real motion pattern:
      usually fewer, more localized events
    """
    if events.size == 0:
        return events, ts_us, 1.0, 0, 0.0

    t0 = int(ts_us.min())
    bin_idx = ((ts_us - t0) // bin_us).astype(np.int64)

    if bin_idx.size == 0:
        return events, ts_us, 1.0, 0, 0.0

    counts = np.bincount(bin_idx)
    threshold = max(float(counts.mean() + k * counts.std()), float(min_thresh))

    event_bin_counts = counts[bin_idx]
    keep = event_bin_counts <= threshold

    keep_ratio = float(keep.mean())
    dropped_bins = int((counts > threshold).sum())

    return events[keep], ts_us[keep], keep_ratio, dropped_bins, threshold


def compute_fft_peaks(counts: np.ndarray, bin_us: int, fmin_hz: float, fmax_hz: float, top_k: int):
    n = len(counts)
    if n < 16:
        return []

    x = counts.astype(np.float64)
    x -= x.mean()

    if np.allclose(x, 0):
        return []

    x *= np.hanning(n)

    fft_vals = np.fft.rfft(x)
    power = np.abs(fft_vals) ** 2

    fs = 1_000_000.0 / bin_us
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    band_mask = (freqs >= fmin_hz) & (freqs <= fmax_hz)
    freqs_band = freqs[band_mask]
    power_band = power[band_mask]

    if power_band.size == 0 or np.max(power_band) <= 0:
        return []

    idx_sorted = np.argsort(power_band)[::-1]
    peaks = []
    used_freqs = []

    for idx in idx_sorted:
        f = float(freqs_band[idx])
        p = float(power_band[idx])

        if any(abs(f - uf) < 5.0 for uf in used_freqs):
            continue

        peaks.append((f, p / np.max(power_band)))
        used_freqs.append(f)

        if len(peaks) >= top_k:
            break

    return peaks


def main():
    args = parse_args()
    width = args.width
    height = args.height

    ser = open_serial(args)

    ROI_X0, ROI_X1, ROI_Y0, ROI_Y1 = roi_bounds_3x3(width, height, ROI_ROW, ROI_COL)
    print(
        f"Using center ROI of 3x3 grid: "
        f"x=[{ROI_X0},{ROI_X1}) y=[{ROI_Y0},{ROI_Y1}) "
        f"size={ROI_X1 - ROI_X0}x{ROI_Y1 - ROI_Y0}"
    )
    print(
        f"Burst filter: {'OFF' if args.disable_burst_filter else 'ON'}  "
        f"bin={args.burst_bin_us}us  k={args.burst_k}  "
        f"min_thresh={args.burst_min_thresh}"
    )


    phase_buffer_ts = []
    phase_buffer_ev = []

    phase_keep_sum = 0.0
    phase_keep_n = 0
    phase_bad_bin_last = -1

    t0_wall = time.time()

    packets_acc = 0
    raw_events_acc = 0
    filt_events_acc = 0
    max_count_acc = 0

    sum_x_acc = 0
    sum_y_acc = 0
    on_acc = 0
    off_acc = 0

    spatial_counts = np.zeros((height, width), dtype=np.int32)
    last_t_us = np.full((height, width), -10**15, dtype=np.int64)
    coherent_events_acc = 0

    keep_ratio_sum = 0.0
    keep_ratio_n = 0
    dropped_bins_acc = 0
    thresh_sum = 0.0
    thresh_n = 0

    fft_counts = None
    fft_t0_us = None
    fft_num_bins = int(round((FFT_WINDOW_SEC * 1_000_000) / args.burst_bin_us))

    while True:
        try:
            read_until_magic(ser)

            header_rest = read_exactly(ser, struct.calcsize(HEADER_FMT))
            event_count, payload_len = struct.unpack(HEADER_FMT, header_rest)

            expected_len = event_count * EVENT_SIZE_BYTES
            if payload_len != expected_len:
                raise RuntimeError(f"Invalid payload_len={payload_len}, expected={expected_len}")

            payload = read_exactly(ser, payload_len)

        except Exception as e:
            print(f"[WARN] serial read failed: {e}")
            time.sleep(0.05)
            continue

        if event_count == 0:
            continue

        events = np.frombuffer(payload, dtype=np.uint16).reshape(event_count, EVENT_WORDS)

        xs = events[:, 4].astype(np.int32)
        ys = events[:, 5].astype(np.int32)
        types = events[:, 0].astype(np.int32)
        ts_us = reconstruct_t_us(events)

        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        events = events[valid]
        xs = xs[valid]
        ys = ys[valid]
        types = types[valid]
        ts_us = ts_us[valid]

        if events.size == 0:
            continue

        raw_n = len(events)

        if args.disable_burst_filter:
            filt_events = events
            filt_ts_us = ts_us
            keep_ratio = 1.0
            dropped_bins = 0
            threshold = 0.0
        else:
            # accumulate for phase estimation
            phase_buffer_ts.append(ts_us)
            phase_buffer_ev.append(events)

            # Keep stats variables defined even when using phase filter path.
            dropped_bins = 0
            threshold = 0.0

            # flatten buffer
            if len(phase_buffer_ts) > 10:
                phase_buffer_ts.pop(0)
                phase_buffer_ev.pop(0)

            ts_all = np.concatenate(phase_buffer_ts)
            ev_all = np.concatenate(phase_buffer_ev)

            if ts_all.size > PHASE_MIN_SAMPLES:
                filt_events, filt_ts_us, keep_ratio, bad_bin = phase_filter(
                    events,
                    ts_us,
                    period_us=PHASE_PERIOD_US,
                    bin_us=PHASE_BIN_US,
                    tol_bins=PHASE_TOL_BINS
                )
                phase_bad_bin_last = bad_bin
            else:
                filt_events = events
                filt_ts_us = ts_us
                keep_ratio = 1.0
                bad_bin = -1

            filt_n = len(filt_events)

        filt_n = len(filt_events)

        packets_acc += 1
        raw_events_acc += raw_n
        filt_events_acc += filt_n
        max_count_acc = max(max_count_acc, raw_n)

        keep_ratio_sum += keep_ratio
        keep_ratio_n += 1
        dropped_bins_acc += dropped_bins
        thresh_sum += threshold
        thresh_n += 1

        if filt_n > 0:
            fxs = filt_events[:, 4].astype(np.int32)
            fys = filt_events[:, 5].astype(np.int32)
            ftypes = filt_events[:, 0].astype(np.int32)

            sum_x_acc += int(fxs.sum())
            sum_y_acc += int(fys.sum())
            on_acc += int((ftypes == 1).sum())
            off_acc += int((ftypes == 0).sum())

            np.add.at(spatial_counts, (fys, fxs), 1)

            for i in range(fxs.size):
                x = fxs[i]
                y = fys[i]
                t_us = filt_ts_us[i]

                x0 = max(0, x - COH_RADIUS)
                x1 = min(width, x + COH_RADIUS + 1)
                y0 = max(0, y - COH_RADIUS)
                y1 = min(height, y + COH_RADIUS + 1)

                neighborhood_last = last_t_us[y0:y1, x0:x1]
                if np.any((t_us - neighborhood_last) <= COH_DT_US):
                    coherent_events_acc += 1

                last_t_us[y, x] = t_us

            # FFT on filtered center ROI
            roi_mask = (
                (fxs >= ROI_X0) & (fxs < ROI_X1) &
                (fys >= ROI_Y0) & (fys < ROI_Y1)
            )

            if np.any(roi_mask):
                ts_roi = filt_ts_us[roi_mask]

                if fft_t0_us is None:
                    fft_t0_us = int(ts_roi.min())
                    fft_counts = np.zeros(fft_num_bins, dtype=np.int32)

                bin_idx = ((ts_roi - fft_t0_us) // args.burst_bin_us).astype(np.int64)
                valid_bins = (bin_idx >= 0) & (bin_idx < fft_num_bins)
                if np.any(valid_bins):
                    np.add.at(fft_counts, bin_idx[valid_bins], 1)

                max_t_us = int(filt_ts_us.max())
                while max_t_us >= fft_t0_us + fft_num_bins * args.burst_bin_us:
                    peaks = compute_fft_peaks(
                        fft_counts,
                        bin_us=args.burst_bin_us,
                        fmin_hz=FREQ_MIN_HZ,
                        fmax_hz=FREQ_MAX_HZ,
                        top_k=TOP_K,
                    )

                    peak_str = "  ".join([f"{f:6.1f}Hz({rp:4.2f})" for f, rp in peaks]) if peaks else "no_peak"

                    print(
                        f"[FFT FILTERED center ROI {FFT_WINDOW_SEC:.1f}s, bin={args.burst_bin_us}us]  "
                        f"roi_events={int(fft_counts.sum()):6d}  "
                        f"mean_bin={float(fft_counts.mean()):6.2f}  "
                        f"std_bin={float(fft_counts.std()):6.2f}  "
                        f"top_peaks={peak_str}"
                    )

                    fft_t0_us += fft_num_bins * args.burst_bin_us
                    fft_counts.fill(0)

        now = time.time()
        dt = now - t0_wall

        if dt >= 1.0:
            raw_eps = raw_events_acc / dt
            filt_eps = filt_events_acc / dt
            pps = packets_acc / dt
            avg_raw_count = raw_events_acc / packets_acc if packets_acc > 0 else 0.0
            keep_avg = keep_ratio_sum / keep_ratio_n if keep_ratio_n > 0 else 1.0
            thresh_avg = thresh_sum / thresh_n if thresh_n > 0 else 0.0

            if filt_events_acc > 0:
                mean_x = sum_x_acc / filt_events_acc
                mean_y = sum_y_acc / filt_events_acc
            else:
                mean_x = -1.0
                mean_y = -1.0

            flat = spatial_counts.ravel()
            active_mask = flat > 0
            active_px = int(active_mask.sum())
            active_frac = active_px / (width * height)

            if filt_events_acc > 0 and active_px > 0:
                p = flat[active_mask].astype(np.float64) / filt_events_acc
                entropy = float(-(p * np.log2(p)).sum())
                entropy_norm = entropy / np.log2(width * height)
                mean_epp = float(flat[active_mask].mean())
                max_epp = int(flat.max())
            else:
                entropy = 0.0
                entropy_norm = 0.0
                mean_epp = 0.0
                max_epp = 0

            hot_counts = {thr: int((flat >= thr).sum()) for thr in HOT_THRESHOLDS}
            coh_frac = coherent_events_acc / filt_events_acc if filt_events_acc > 0 else 0.0

            hot_str = "  ".join([f"hot_px>={thr}:{hot_counts[thr]:5d}" for thr in HOT_THRESHOLDS])

            print(
                f"pps={pps:6.1f}  "
                f"raw_eps={raw_eps:8.0f}  "
                f"filt_eps={filt_eps:8.0f}  "
                f"keep={keep_avg:5.3f}  "
                f"phase_bin={phase_bad_bin_last:3d}"
                f"drop_bins={dropped_bins_acc:4d}  "
                f"thr={thresh_avg:6.1f}  "
                f"avg_raw_count={avg_raw_count:6.1f}  "
                f"max_raw_count={max_count_acc:4d}  "
                f"mean_xy=({mean_x:6.1f},{mean_y:6.1f})  "
                f"off={off_acc:6d}  on={on_acc:6d}  "
                f"active_px={active_px:5d}  "
                f"active_frac={active_frac:6.3f}  "
                f"entropy_norm={entropy_norm:5.3f}  "
                f"coh_frac={coh_frac:5.3f}  "
                f"mean_epp={mean_epp:5.2f}  "
                f"max_epp={max_epp:4d}  "
                f"{hot_str}"
            )

            t0_wall = now
            packets_acc = 0
            raw_events_acc = 0
            filt_events_acc = 0
            max_count_acc = 0
            sum_x_acc = 0
            sum_y_acc = 0
            on_acc = 0
            off_acc = 0
            spatial_counts.fill(0)
            coherent_events_acc = 0
            keep_ratio_sum = 0.0
            keep_ratio_n = 0
            dropped_bins_acc = 0
            thresh_sum = 0.0
            thresh_n = 0


if __name__ == "__main__":
    main()