"""Build causal event-tracking datasets without ROS."""

import argparse
import json
import os
import re
import shutil
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .event_ball_tracker import EventBallTracker
from .evt1_protocol import EventPacket
from .offline_raw_events import RawEventHDF5Reader


SCHEMA_VERSION = "sparse_tracking_v1"
STRING_DTYPE = h5py.string_dtype("utf-8", length=256)
TRACKER_FIELDS = {
    "available_ros_t_ns": "i8", "packet_id": "i8",
    "sensor_window_start_us": "i8", "sensor_window_end_us": "i8",
    "x_px": "f4", "y_px": "f4", "vx_px_s": "f4", "vy_px_s": "f4",
    "speed_px_s": "f4", "confidence": "f4", "valid": "u1",
    "velocity_valid": "u1", "window_event_count": "i4",
    "candidate_count": "i4", "blob_area_px": "i4",
    "blob_event_count": "i4", "blob_width_px": "i4",
    "blob_height_px": "i4", "circularity": "f4",
    "rejection_reason": STRING_DTYPE,
}


@dataclass(frozen=True)
class Episode:
    """Validated source episode metadata and exact observation grid."""

    path: Path
    name: str
    episode_index: int
    source_episode_index: int
    start: float
    end: float
    timestamps: np.ndarray


def discover_episodes(directory):
    """Discover numeric episode files and validate their time intervals."""
    directory = Path(directory)
    found = []
    for path in directory.glob("episode_*.hdf5"):
        match = re.fullmatch(r"episode_(\d+)\.hdf5", path.name)
        if match:
            found.append((int(match.group(1)), path))
    found.sort(key=lambda item: item[0])
    if not found:
        raise ValueError(f"no episode_#.hdf5 files found in {directory}")
    episodes = []
    for filename_index, path in found:
        with h5py.File(path, "r") as source:
            for attr in ("episode_start", "episode_end"):
                if attr not in source.attrs:
                    raise ValueError(f"{path}: missing root attribute {attr}")
            if "observations/timestamps" not in source:
                raise ValueError(f"{path}: missing /observations/timestamps")
            timestamps = np.asarray(source["observations/timestamps"][:])
            episode_index = int(source.attrs.get("episode_index", filename_index))
            source_index = int(source.attrs.get("source_episode_index", episode_index))
            start, end = float(source.attrs["episode_start"]), float(source.attrs["episode_end"])
        if timestamps.ndim != 1:
            raise ValueError(f"{path}: /observations/timestamps must be one-dimensional")
        if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) < 0):
            raise ValueError(f"{path}: observation timestamps must be finite and monotonic")
        if not np.isfinite(start) or not np.isfinite(end) or start > end:
            raise ValueError(f"{path}: invalid finite episode interval [{start}, {end}]")
        episodes.append(Episode(path, f"episode_{episode_index}", episode_index,
                                source_index, start, end, timestamps))
    for previous, current in zip(episodes, episodes[1:]):
        if current.start < previous.end:
            raise ValueError(
                f"overlapping episode intervals: {previous.path} (source episode "
                f"{previous.source_episode_index}) and {current.path} (source episode "
                f"{current.source_episode_index})")
    return episodes


def load_tracker_config(path):
    """Load JSON and validate every key against EventBallTracker."""
    with open(path, "r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("tracker configuration must be a JSON object")
    try:
        EventBallTracker(**config)
    except TypeError as error:
        raise ValueError(f"invalid tracker configuration: {error}") from error
    return config


def tracker_factory(config):
    """Return a new tracker; callers use one instance per episode."""
    return EventBallTracker(**dict(config))


def _create_extendable(group):
    return {name: group.create_dataset(name, shape=(0,), maxshape=(None,),
                                       chunks=True, dtype=dtype)
            for name, dtype in TRACKER_FIELDS.items()}


def _append(datasets, packet, detection):
    values = {
        "available_ros_t_ns": packet.packet_ros_t_ns,
        "packet_id": packet.packet_id,
        "sensor_window_start_us": detection.window_start_us,
        "sensor_window_end_us": detection.window_end_us,
        "x_px": detection.x_px, "y_px": detection.y_px,
        "vx_px_s": detection.vx_px_s, "vy_px_s": detection.vy_px_s,
        "speed_px_s": detection.speed_px_s, "confidence": detection.confidence,
        "valid": detection.valid, "velocity_valid": detection.velocity_valid,
        "window_event_count": detection.window_event_count,
        "candidate_count": detection.candidate_count,
        "blob_area_px": detection.blob_area_px,
        "blob_event_count": detection.blob_event_count,
        "blob_width_px": detection.blob_width_px,
        "blob_height_px": detection.blob_height_px,
        "circularity": detection.circularity,
        "rejection_reason": detection.rejection_reason,
    }
    for name, dataset in datasets.items():
        size = len(dataset)
        dataset.resize((size + 1,))
        dataset[size] = values[name]


def run_tracker(raw_events, episodes_dir, output, tracker_config, *, pre_roll_ms=0.0):
    """Run a fresh tracker for each timestamp-defined episode."""
    episodes = discover_episodes(episodes_dir)
    config = load_tracker_config(tracker_config)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    totals = dict(packets=0, events=0, updates=0, valid=0)
    summaries = []
    with RawEventHDF5Reader(
            raw_events, width=config.get("width"), height=config.get("height")) as reader:
        with h5py.File(output, "w") as sidecar:
            metadata = sidecar.create_group("metadata")
            metadata.attrs.update({
                "raw_events_h5": str(Path(raw_events).resolve()),
                "tracker_config_json": json.dumps(config, sort_keys=True),
                "tracker_code_version": "EventBallTracker/offline_dataset_v1",
                "sensor_width": reader.width, "sensor_height": reader.height,
                "availability_timestamp_domain": "packet_ros_t_ns",
                "sensor_timestamp_domain": "genx320_microseconds",
                "pre_roll_ms": float(pre_roll_ms),
            })
            groups = sidecar.create_group("episodes")
            for number, episode in enumerate(episodes, 1):
                tracker = tracker_factory(config)
                group = groups.create_group(episode.name)
                group.attrs.update({"episode_index": episode.episode_index,
                                    "source_episode_index": episode.source_episode_index,
                                    "episode_start": episode.start, "episode_end": episode.end})
                datasets = _create_extendable(group)
                start_ns = round((episode.start - pre_roll_ms / 1000.0) * 1e9)
                episode_start_ns = round(episode.start * 1e9)
                end_ns = round(episode.end * 1e9)
                local = dict(packets=0, events=0, updates=0, valid=0)
                for recorded in reader.iter_packets(start_ros_t_ns=start_ns,
                                                    end_ros_t_ns=end_ns):
                    packet = EventPacket.from_recorded_arrays(
                        event_type=recorded.event_type, event_x=recorded.event_x,
                        event_y=recorded.event_y, timestamps_us=recorded.event_t_us,
                        packet_id=recorded.packet_id,
                        packet_ros_stamp_ns=recorded.packet_ros_t_ns,
                        packet_monotonic_stamp_ns=recorded.packet_monotonic_t_ns,
                        original_ros_stamp_ns=recorded.packet_ros_t_ns,
                        original_monotonic_stamp_ns=recorded.packet_monotonic_t_ns)
                    local["packets"] += 1
                    local["events"] += packet.event_count
                    detections = tracker.update(packet)
                    if recorded.packet_ros_t_ns < episode_start_ns:
                        continue
                    for detection in detections:
                        _append(datasets, recorded, detection)
                        local["updates"] += 1
                        local["valid"] += int(detection.valid)
                for key in totals:
                    totals[key] += local[key]
                summaries.append((episode, local))
                elapsed = max(time.monotonic() - started, 1e-9)
                remaining = elapsed / number * (len(episodes) - number)
                print(f"[{number}/{len(episodes)}] {episode.name}: raw packets "
                      f"{local['packets']}, events {local['events']}, tracker updates "
                      f"{local['updates']}, valid {local['valid']}, "
                      f"{totals['packets']/elapsed:.1f} packets/s, ETA {remaining:.1f}s",
                      flush=True)
    elapsed = max(time.monotonic() - started, 1e-9)
    invalid = totals["updates"] - totals["valid"]
    rate = totals["valid"] / totals["updates"] if totals["updates"] else 0.0
    print(f"Completed {len(episodes)} episodes: {totals['packets']} raw packets, "
          f"{totals['events']} raw events, {totals['updates']} updates, "
          f"{totals['valid']} valid, {invalid} invalid, valid-detection rate "
          f"{rate:.1%}, processing speed {totals['packets']/elapsed:.1f} packets/s")
    return summaries


def _read_rgb_track(directory, episode):
    if directory is None:
        return None
    candidates = [Path(directory) / episode.path.name,
                  Path(directory) / f"{episode.name}.h5"]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return None
    with h5py.File(path, "r") as source:
        required = ("timestamps", "rgb_2d_px", "valid")
        if any(name not in source for name in required):
            raise ValueError(f"{path}: RGB track requires {required}")
        timestamps = np.asarray(source["timestamps"][:], dtype=np.float64)
        points = np.asarray(source["rgb_2d_px"][:], dtype=np.float32)
        valid = np.asarray(source["valid"][:], dtype=np.uint8)
    if (timestamps.ndim != 1 or points.shape != (len(timestamps), 2) or
            valid.shape != timestamps.shape or not np.all(np.isfinite(timestamps)) or
            np.any(np.diff(timestamps) < 0)):
        raise ValueError(f"{path}: malformed RGB track arrays")
    return timestamps, points, valid


def _sample_sidecar(group, timestamps):
    available = np.asarray(group["available_ros_t_ns"][:], dtype=np.int64)
    sample_ns = np.rint(timestamps * 1e9).astype(np.int64)
    indices = np.searchsorted(available, sample_ns, side="right") - 1
    has = indices >= 0
    safe = np.maximum(indices, 0)

    def take(name, dtype, fill=0):
        result = np.full(len(timestamps), fill, dtype=dtype)
        if np.any(has):
            # h5py fancy indexing rejects repeated indices. Tracker outputs are
            # compact sidecar rows (not raw events), so read this one column and
            # let NumPy perform the repeated causal selection.
            result[has] = np.asarray(group[name][:])[safe[has]]
        return result
    output = {
        "event_2d_px": np.column_stack((take("x_px", "f4"), take("y_px", "f4"))),
        "event_velocity_px_s": np.column_stack(
            (take("vx_px_s", "f4"), take("vy_px_s", "f4"))),
        "event_valid": take("valid", "u1"),
        "event_velocity_valid": take("velocity_valid", "u1"),
        "event_has_update": has.astype("u1"),
        "event_source_timestamps": np.full(len(timestamps), np.nan, dtype="f8"),
        "event_source_age_sec": np.full(len(timestamps), np.nan, dtype="f4"),
        "event_sensor_window_start_us": take("sensor_window_start_us", "i8", -1),
        "event_sensor_window_end_us": take("sensor_window_end_us", "i8", -1),
        "event_window_event_count": take("window_event_count", "i4"),
        "event_candidate_count": take("candidate_count", "i4"),
        "event_blob_area_px": take("blob_area_px", "i4"),
        "event_confidence": take("confidence", "f4"),
        "event_rejection_reason": take("rejection_reason", STRING_DTYPE, b""),
    }
    if np.any(has):
        source_seconds = available[safe[has]].astype(np.float64) / 1e9
        output["event_source_timestamps"][has] = source_seconds
        output["event_source_age_sec"][has] = timestamps[has] - source_seconds
    return output


def enrich(episodes_dir, tracker_output, output_dir, *, overwrite=False,
           rgb_tracks_dir=None, write_empty_rgb_track=False, require_rgb_2d=False):
    """Copy and causally enrich episodes, atomically, one file at a time."""
    episodes = discover_episodes(episodes_dir)
    output_dir = Path(output_dir)
    if output_dir.resolve() == Path(episodes_dir).resolve():
        raise ValueError("output directory must differ from the source episode directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    warned = False
    with h5py.File(tracker_output, "r") as sidecar:
        metadata = sidecar["metadata"].attrs
        for number, episode in enumerate(episodes, 1):
            destination = output_dir / episode.path.name
            if destination.exists() and not overwrite:
                raise FileExistsError(f"refusing to overwrite {destination}; use --overwrite")
            if f"episodes/{episode.name}" not in sidecar:
                raise ValueError(f"tracker sidecar has no /episodes/{episode.name}")
            rgb = _read_rgb_track(rgb_tracks_dir, episode)
            if rgb is None and require_rgb_2d:
                raise ValueError(f"RGB 2D track unavailable for {episode.path}")
            if rgb is None and not write_empty_rgb_track and not warned:
                warnings.warn("RGB tracks were not supplied; writing event-only sparse tracking")
                warned = True
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{episode.path.stem}.", suffix=".tmp", dir=output_dir)
            os.close(fd)
            temporary = Path(temporary_name)
            try:
                shutil.copy2(episode.path, temporary)
                with h5py.File(temporary, "r+") as target:
                    observations = target["observations"]
                    if "sparse_tracking" in observations:
                        del observations["sparse_tracking"]
                    sparse = observations.create_group("sparse_tracking")
                    sampled = _sample_sidecar(
                        sidecar[f"episodes/{episode.name}"], episode.timestamps)
                    for name, values in sampled.items():
                        sparse.create_dataset(name, data=values)
                    sparse.attrs.update({
                        "schema_version": SCHEMA_VERSION,
                        "event_coordinate_system": "openmv_native_unrotated_px",
                        "event_width_px": int(metadata["sensor_width"]),
                        "event_height_px": int(metadata["sensor_height"]),
                        "sampling_policy": (
                            "latest_tracker_update_at_or_before_"
                            "observation_timestamp"),
                        "invalid_coordinate_fill": 0.0,
                    })
                    if rgb is not None:
                        rgb_ts, rgb_points, rgb_valid = rgb
                        indices = np.searchsorted(rgb_ts, episode.timestamps,
                                                  side="right") - 1
                        present = indices >= 0
                        points = np.zeros((len(indices), 2), dtype="f4")
                        valid = np.zeros(len(indices), dtype="u1")
                        points[present] = rgb_points[indices[present]]
                        valid[present] = rgb_valid[indices[present]]
                        sparse.create_dataset("rgb_2d_px", data=points)
                        sparse.create_dataset("rgb_valid", data=valid)
                    elif write_empty_rgb_track:
                        sparse.create_dataset("rgb_2d_px", data=np.zeros(
                            (len(episode.timestamps), 2), dtype="f4"))
                        sparse.create_dataset("rgb_valid", data=np.zeros(
                            len(episode.timestamps), dtype="u1"))
                    target.attrs["sparse_tracking_source_raw_events_h5"] = (
                        metadata["raw_events_h5"])
                    target.attrs["sparse_tracking_tracker_config_json"] = (
                        metadata["tracker_config_json"])
                    target.attrs["sparse_tracking_sidecar_h5"] = str(
                        Path(tracker_output).resolve())
                    target.flush()
                os.replace(temporary, destination)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
            print(f"[{number}/{len(episodes)}] enriched {destination}", flush=True)


def _add_rgb_options(parser):
    parser.add_argument("--rgb-tracks-dir")
    parser.add_argument("--write-empty-rgb-track", action="store_true")
    parser.add_argument("--require-rgb-2d", action="store_true")


def make_parser():
    """Construct the offline dataset command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-tracker")
    run.add_argument("--raw-events", required=True)
    run.add_argument("--episodes", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--tracker-config", required=True)
    run.add_argument("--pre-roll-ms", type=float, default=0.0,
                     help="warm-up before episode start; pre-start outputs are discarded")
    enrich_parser = commands.add_parser("enrich")
    enrich_parser.add_argument("--episodes", required=True)
    enrich_parser.add_argument("--tracker-output", required=True)
    enrich_parser.add_argument("--output", required=True)
    enrich_parser.add_argument("--overwrite", action="store_true")
    _add_rgb_options(enrich_parser)
    build = commands.add_parser("build")
    build.add_argument("--raw-events", required=True)
    build.add_argument("--episodes", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--tracker-config", required=True)
    build.add_argument("--tracker-output", required=True)
    build.add_argument("--pre-roll-ms", type=float, default=0.0)
    build.add_argument("--overwrite", action="store_true")
    _add_rgb_options(build)
    return parser


def main(argv=None):
    """Run the selected offline dataset command."""
    args = make_parser().parse_args(argv)
    if args.command in ("run-tracker", "build"):
        if args.pre_roll_ms < 0:
            raise ValueError("--pre-roll-ms must be non-negative")
        sidecar = args.output if args.command == "run-tracker" else args.tracker_output
        run_tracker(args.raw_events, args.episodes, sidecar,
                    args.tracker_config, pre_roll_ms=args.pre_roll_ms)
    if args.command in ("enrich", "build"):
        enrich(args.episodes, args.tracker_output, args.output,
               overwrite=args.overwrite, rgb_tracks_dir=args.rgb_tracks_dir,
               write_empty_rgb_track=args.write_empty_rgb_track,
               require_rgb_2d=args.require_rgb_2d)


if __name__ == "__main__":
    main()
