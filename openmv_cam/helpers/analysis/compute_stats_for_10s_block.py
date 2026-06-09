#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

W = 320
H = 320


def load_xyt_points(file_path: Path) -> np.ndarray:
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if file_path.stat().st_size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    points = np.loadtxt(file_path, dtype=np.int64, delimiter="\t")

    if points.ndim == 1:
        if points.size != 3:
            raise RuntimeError("Expected 3 columns: x, y, t")
        points = points.reshape(1, 3)

    if points.shape[1] != 3:
        raise RuntimeError("Expected 3 columns: x, y, t")

    return points


def filter_valid_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    xs = points[:, 0]
    ys = points[:, 1]
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    return points[valid]


def filter_points_by_time_interval(
    points: np.ndarray, start_sec: float = None, end_sec: float = None
) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    if start_sec is None and end_sec is None:
        return points

    ts_us = points[:, 2].astype(np.float64)
    ts_rel_s = (ts_us - ts_us.min()) * 1e-6

    start_s = 0.0 if start_sec is None else start_sec
    end_s = np.inf if end_sec is None else end_sec

    mask = (ts_rel_s >= start_s) & (ts_rel_s <= end_s)
    return points[mask]


def compute_global_fft(points: np.ndarray, bin_ms: float):
    ts_us = points[:, 2].astype(np.float64)
    ts_s = (ts_us - ts_us.min()) * 1e-6

    duration_s = ts_s.max()
    bin_s = bin_ms * 1e-3
    num_bins = int(np.ceil(duration_s / bin_s)) + 1

    counts, edges = np.histogram(ts_s, bins=num_bins, range=(0, num_bins * bin_s))
    rate_hz = counts / bin_s
    time_axis_s = edges[:-1] + 0.5 * bin_s

    rate_centered = rate_hz - rate_hz.mean()

    freqs = np.fft.rfftfreq(len(rate_centered), d=bin_s)
    fft_mag = np.abs(np.fft.rfft(rate_centered))

    return freqs, fft_mag, rate_hz, time_axis_s, bin_s


def compute_pixel_stats(points: np.ndarray, width: int, height: int):
    xs = points[:, 0].astype(np.int64)
    ys = points[:, 1].astype(np.int64)
    ts = points[:, 2].astype(np.int64)

    duration_s = (ts.max() - ts.min()) * 1e-6

    count_img = np.zeros((height, width), dtype=np.int32)
    np.add.at(count_img, (ys, xs), 1)

    firing_rate = count_img / duration_s

    return count_img, firing_rate


def compute_phase_lock_map(
    points: np.ndarray, width: int, height: int, freq_hz: float, min_count: int = 5
) -> np.ndarray:
    phase_lock = np.full((height, width), np.nan, dtype=np.float32)

    if len(points) == 0 or freq_hz <= 0:
        return phase_lock

    xs = points[:, 0].astype(np.int64)
    ys = points[:, 1].astype(np.int64)
    ts_us = points[:, 2].astype(np.float64)
    ts_s = (ts_us - ts_us.min()) * 1e-6

    theta = 2.0 * np.pi * ((ts_s * freq_hz) % 1.0)
    z = np.exp(1j * theta)

    pixel_id = ys * width + xs
    num_pixels = width * height

    counts = np.bincount(pixel_id, minlength=num_pixels)
    sum_real = np.bincount(pixel_id, weights=z.real, minlength=num_pixels)
    sum_imag = np.bincount(pixel_id, weights=z.imag, minlength=num_pixels)

    valid = counts >= min_count
    r = np.zeros(num_pixels, dtype=np.float64)
    r[valid] = np.sqrt(sum_real[valid] ** 2 + sum_imag[valid] ** 2) / counts[valid]

    phase_lock.flat[valid] = r[valid].astype(np.float32)
    return phase_lock


def compute_tile_fft_stats(
    points: np.ndarray,
    width: int,
    height: int,
    tile_size: int,
    bin_ms: float,
    max_freq: float,
    min_events: int,
):
    n_tiles_x = (width + tile_size - 1) // tile_size
    n_tiles_y = (height + tile_size - 1) // tile_size
    num_tiles = n_tiles_x * n_tiles_y

    tile_event_count = np.zeros((n_tiles_y, n_tiles_x), dtype=np.int32)
    tile_dominant_freq = np.full((n_tiles_y, n_tiles_x), np.nan, dtype=np.float32)
    tile_dominant_mag = np.full((n_tiles_y, n_tiles_x), np.nan, dtype=np.float32)

    if len(points) == 0:
        return tile_event_count, tile_dominant_freq, tile_dominant_mag

    xs = points[:, 0].astype(np.int64)
    ys = points[:, 1].astype(np.int64)
    ts_us = points[:, 2].astype(np.float64)
    ts_s = (ts_us - ts_us.min()) * 1e-6

    bin_s = bin_ms * 1e-3
    duration_s = ts_s.max()
    num_bins = int(np.ceil(duration_s / bin_s)) + 1

    tile_x = xs // tile_size
    tile_y = ys // tile_size
    tile_id = tile_y * n_tiles_x + tile_x

    counts = np.bincount(tile_id, minlength=num_tiles)
    tile_event_count = counts.reshape(n_tiles_y, n_tiles_x).astype(np.int32)

    bin_idx = np.floor(ts_s / bin_s).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    flat_idx = tile_id * num_bins + bin_idx
    tile_bin_counts = np.bincount(flat_idx, minlength=num_tiles * num_bins).reshape(
        num_tiles, num_bins
    )

    rate_hz = tile_bin_counts / bin_s
    rate_centered = rate_hz - rate_hz.mean(axis=1, keepdims=True)

    freqs = np.fft.rfftfreq(num_bins, d=bin_s)
    fft_mag = np.abs(np.fft.rfft(rate_centered, axis=1))

    valid_band = (freqs >= 1.0) & (freqs <= max_freq)
    valid_tiles = counts >= min_events

    if np.any(valid_band):
        band_freqs = freqs[valid_band]
        band_mag = fft_mag[:, valid_band]
        best_idx = np.argmax(band_mag, axis=1)

        dominant_freq = band_freqs[best_idx]
        dominant_mag = band_mag[np.arange(num_tiles), best_idx]

        dom_freq_flat = np.full(num_tiles, np.nan, dtype=np.float32)
        dom_mag_flat = np.full(num_tiles, np.nan, dtype=np.float32)
        dom_freq_flat[valid_tiles] = dominant_freq[valid_tiles].astype(np.float32)
        dom_mag_flat[valid_tiles] = dominant_mag[valid_tiles].astype(np.float32)

        tile_dominant_freq = dom_freq_flat.reshape(n_tiles_y, n_tiles_x)
        tile_dominant_mag = dom_mag_flat.reshape(n_tiles_y, n_tiles_x)

    return tile_event_count, tile_dominant_freq, tile_dominant_mag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="events_xyt.tsv",
        help="Path to x/y/t TSV file.",
    )
    ap.add_argument(
        "--bin-ms",
        type=float,
        default=1.0,
        help="Temporal bin size for global event-rate FFT.",
    )
    ap.add_argument(
        "--max-freq",
        type=float,
        default=500.0,
        help="Maximum frequency shown in FFT plot.",
    )
    ap.add_argument(
        "--phase-min-count",
        type=int,
        default=10,
        help="Minimum events per pixel to include phase-lock value.",
    )
    ap.add_argument(
        "--phase-freq",
        type=float,
        default=None,
        help="Optional fixed frequency [Hz] for phase-lock map.",
    )
    ap.add_argument(
        "--tile-size",
        type=int,
        default=16,
        help="Spatial resolution of square tiles in pixels.",
    )
    ap.add_argument(
        "--tile-max-freq",
        type=float,
        default=None,
        help="Optional max frequency [Hz] for tile FFT analysis.",
    )
    ap.add_argument(
        "--tile-min-events",
        type=int,
        default=50,
        help="Minimum events per tile required for dominant FFT stats.",
    )
    ap.add_argument(
        "--time-window-sec",
        type=float,
        default=None,
        help="Optional time window [s] shown in global event-rate time plot.",
    )
    ap.add_argument(
        "--analysis-start-sec",
        type=float,
        default=None,
        help="Optional analysis start time [s] relative to first timestamp.",
    )
    ap.add_argument(
        "--analysis-end-sec",
        type=float,
        default=None,
        help="Optional analysis end time [s] relative to first timestamp.",
    )
    ap.add_argument(
        "--save",
        default=None,
        help="Optional path to save plot, e.g. stats.png",
    )
    args = ap.parse_args()

    input_path = Path(args.input)

    print(f"[INFO] Loading {input_path}")
    points = load_xyt_points(input_path)
    points = filter_valid_points(points, W, H)

    if args.analysis_start_sec is not None and args.analysis_start_sec < 0:
        raise ValueError("--analysis-start-sec must be >= 0")
    if args.analysis_end_sec is not None and args.analysis_end_sec < 0:
        raise ValueError("--analysis-end-sec must be >= 0")
    if (
        args.analysis_start_sec is not None
        and args.analysis_end_sec is not None
        and args.analysis_end_sec < args.analysis_start_sec
    ):
        raise ValueError("--analysis-end-sec must be >= --analysis-start-sec")

    if args.analysis_start_sec is not None or args.analysis_end_sec is not None:
        print(
            "[INFO] Analysis interval: "
            f"{args.analysis_start_sec if args.analysis_start_sec is not None else 0.0:.3f} "
            "to "
            f"{args.analysis_end_sec if args.analysis_end_sec is not None else np.inf} s "
            "(relative to first timestamp)"
        )
    points = filter_points_by_time_interval(
        points,
        start_sec=args.analysis_start_sec,
        end_sec=args.analysis_end_sec,
    )

    if len(points) == 0:
        print("[WARN] No valid points in selected analysis interval.")
        return

    ts_us = points[:, 2]
    duration_s = (ts_us.max() - ts_us.min()) * 1e-6
    tile_max_freq = args.tile_max_freq if args.tile_max_freq is not None else args.max_freq

    print(f"[INFO] Valid events: {len(points)}")
    print(f"[INFO] Duration: {duration_s:.3f} s")
    print(f"[INFO] Global mean event rate: {len(points) / duration_s:.1f} events/s")

    freqs, fft_mag, rate_hz, time_axis_s, bin_s = compute_global_fft(points, args.bin_ms)
    count_img, firing_rate = compute_pixel_stats(points, W, H)

    print(f"[INFO] Temporal bin size: {bin_s * 1e3:.3f} ms")
    print(f"[INFO] Number of temporal bins: {len(rate_hz)}")
    print(f"[INFO] Max event-rate spike: {np.max(rate_hz):.2f} events/s")
    print(f"[INFO] Median event-rate: {np.median(rate_hz):.2f} events/s")

    fft_band = (freqs >= 1.0) & (freqs <= args.max_freq)

    if np.any(fft_band):
        band_freqs = freqs[fft_band]
        band_mags = fft_mag[fft_band]
        n_peaks = min(10, len(band_mags))
        top_idx = np.argsort(band_mags)[-n_peaks:][::-1]

        print(f"[INFO] Strongest {n_peaks} FFT peaks in [1 Hz, {args.max_freq:.2f} Hz]:")
        for idx in top_idx:
            print(f"  {band_freqs[idx]:8.2f} Hz | mag {band_mags[idx]:.3e}")
    else:
        print("[WARN] No FFT bins available in [1 Hz, max_freq].")

    if args.phase_freq is not None:
        phase_lock_freq_hz = args.phase_freq
    else:
        # Select strongest non-DC FFT peak in [1 Hz, max_freq] for phase-lock analysis.
        if np.any(fft_band):
            best_idx = np.argmax(fft_mag[fft_band])
            phase_lock_freq_hz = freqs[fft_band][best_idx]
        else:
            phase_lock_freq_hz = 1.0
            print("[WARN] No FFT bins in [1 Hz, max_freq]; using 1.00 Hz for phase-lock map.")

    phase_lock_map = compute_phase_lock_map(
        points, W, H, phase_lock_freq_hz, min_count=args.phase_min_count
    )
    valid_phase_pixels = (count_img >= args.phase_min_count)
    surviving_pixels = int(valid_phase_pixels.sum())

    print(f"[INFO] Selected phase-lock frequency: {phase_lock_freq_hz:.2f} Hz")
    print(
        f"[INFO] Pixels with >= {args.phase_min_count} events: "
        f"{surviving_pixels} / {W * H}"
    )

    print("[INFO] Pixel stats:")
    print(f"  active pixels: {(count_img > 0).sum()} / {W * H}")
    print(f"  max pixel count: {count_img.max()}")
    print(f"  max pixel firing rate: {np.nanmax(firing_rate):.2f} Hz")
    print(f"  median active pixel firing rate: {np.median(firing_rate[count_img > 0]):.2f} Hz")

    tile_event_count, tile_dominant_freq, tile_dominant_mag = compute_tile_fft_stats(
        points,
        W,
        H,
        args.tile_size,
        args.bin_ms,
        tile_max_freq,
        args.tile_min_events,
    )
    n_tiles_y, n_tiles_x = tile_event_count.shape
    valid_tile_mask = np.isfinite(tile_dominant_freq)
    n_valid_tiles = int(valid_tile_mask.sum())

    print(f"[INFO] Tile grid size: {n_tiles_x} x {n_tiles_y}")
    print(
        f"[INFO] Valid tiles with >= {args.tile_min_events} events: "
        f"{n_valid_tiles} / {n_tiles_x * n_tiles_y}"
    )

    if n_valid_tiles > 0:
        median_tile_freq = float(np.nanmedian(tile_dominant_freq))
        print(f"[INFO] Median tile dominant frequency: {median_tile_freq:.2f} Hz")

        tile_flat_mag = tile_dominant_mag.ravel()
        tile_flat_freq = tile_dominant_freq.ravel()
        top_k = min(10, n_valid_tiles)
        top_idx = np.argsort(tile_flat_mag[np.isfinite(tile_flat_mag)])[-top_k:][::-1]
        valid_indices = np.flatnonzero(np.isfinite(tile_flat_mag))

        print(f"[INFO] Top {top_k} tile dominant frequencies by magnitude:")
        for idx in top_idx:
            flat_id = valid_indices[idx]
            ty, tx = np.unravel_index(flat_id, tile_dominant_mag.shape)
            print(
                f"  tile ({tx:2d}, {ty:2d}) | "
                f"f={tile_flat_freq[flat_id]:7.2f} Hz | "
                f"mag={tile_flat_mag[flat_id]:.3e}"
            )
    else:
        print("[WARN] No valid tiles for dominant tile FFT statistics.")

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # 1. Global FFT
    ax = axes[0, 0]
    show = freqs <= args.max_freq
    ax.plot(freqs[show], fft_mag[show])
    ax.set_title("Global event-rate FFT")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude")
    ax.grid(True, alpha=0.3)

    # 2. Global event rate over time
    ax = axes[0, 1]
    if args.time_window_sec is not None:
        show_time = time_axis_s <= args.time_window_sec
    else:
        show_time = np.ones_like(time_axis_s, dtype=bool)
    ax.plot(time_axis_s[show_time], rate_hz[show_time], linewidth=0.9)
    ax.set_title("Global event rate over time")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("events/s")
    ax.grid(True, alpha=0.3)

    # 3. Per-pixel firing rate
    ax = axes[0, 2]
    im = ax.imshow(
        firing_rate,
        origin="upper",
        cmap="inferno",
        vmin=0,
        vmax=np.percentile(firing_rate[firing_rate > 0], 99) if np.any(firing_rate > 0) else 1,
    )
    ax.set_title("Per-pixel firing rate [Hz]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # 4. Phase-lock map
    ax = axes[1, 0]
    phase_cmap = plt.get_cmap("magma").copy()
    phase_cmap.set_bad(color="white")
    im = ax.imshow(
        phase_lock_map,
        origin="upper",
        cmap=phase_cmap,
        vmin=0,
        vmax=1,
    )
    ax.set_title(f"Phase-lock strength @ {phase_lock_freq_hz:.2f} Hz")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # 5. Valid phase-lock mask by event count threshold
    ax = axes[1, 1]
    im = ax.imshow(
        valid_phase_pixels,
        origin="upper",
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    ax.set_title(f"Pixels with >= {args.phase_min_count} events")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # 6. Tile event count
    ax = axes[1, 2]
    im = ax.imshow(
        tile_event_count,
        origin="upper",
        cmap="inferno",
    )
    ax.set_title("Tile event count")
    ax.set_xlabel("tile x")
    ax.set_ylabel("tile y")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # 7. Tile dominant FFT frequency
    ax = axes[2, 0]
    tile_freq_cmap_name = "turbo" if "turbo" in plt.colormaps() else "viridis"
    tile_freq_cmap = plt.get_cmap(tile_freq_cmap_name).copy()
    tile_freq_cmap.set_bad(color="white")
    im = ax.imshow(
        tile_dominant_freq,
        origin="upper",
        cmap=tile_freq_cmap,
        vmin=1,
        vmax=tile_max_freq,
    )
    ax.set_title("Tile dominant FFT frequency [Hz]")
    ax.set_xlabel("tile x")
    ax.set_ylabel("tile y")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # 8. Tile dominant FFT magnitude
    ax = axes[2, 1]
    tile_mag_cmap = plt.get_cmap("magma").copy()
    tile_mag_cmap.set_bad(color="white")
    im = ax.imshow(
        tile_dominant_mag,
        origin="upper",
        cmap=tile_mag_cmap,
    )
    ax.set_title("Tile dominant FFT magnitude")
    ax.set_xlabel("tile x")
    ax.set_ylabel("tile y")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # 9. Summary panel
    ax = axes[2, 2]
    ax.axis("off")
    summary_lines = [
        f"Tile size: {args.tile_size}px",
        f"Tile max freq: {tile_max_freq:.2f} Hz",
        f"Tile min events: {args.tile_min_events}",
        f"Bin size: {bin_s * 1e3:.3f} ms",
        f"Temporal bins: {len(rate_hz)}",
        f"Valid tiles: {n_valid_tiles}/{n_tiles_x * n_tiles_y}",
        f"Phase freq: {phase_lock_freq_hz:.2f} Hz",
        f"Phase min count: {args.phase_min_count}",
    ]
    if n_valid_tiles > 0:
        summary_lines.append(
            f"Median tile dom freq: {float(np.nanmedian(tile_dominant_freq)):.2f} Hz"
        )
    ax.text(0.02, 0.98, "\n".join(summary_lines), va="top", ha="left", fontsize=11)

    fig.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=180)
        print(f"[INFO] Saved plot to {args.save}")

    plt.show()


if __name__ == "__main__":
    main()