"""Build causal event-tracking datasets without ROS."""

import argparse
import hashlib
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


SCHEMA_VERSION = "sparse_tracking_v2"
TRACKER_SCHEMA_VERSION = "event_tracker_updates_v2"
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


@dataclass(frozen=True)
class TrackerUpdate:
    """One native tracker update with explicit availability and sensor times."""

    episode_name: str
    episode_index: int
    source_episode_index: int
    available_ros_t_ns: int
    packet_id: int
    sensor_window_start_us: int
    sensor_window_end_us: int
    x_px: float
    y_px: float
    vx_px_s: float
    vy_px_s: float
    speed_px_s: float
    confidence: float
    valid: bool
    velocity_valid: bool
    window_event_count: int
    candidate_count: int
    blob_area_px: int
    blob_event_count: int
    blob_width_px: int
    blob_height_px: int
    circularity: float
    rejection_reason: str


def _config_dict(config):
    if isinstance(config, (str, os.PathLike)):
        return load_tracker_config(config)
    config = dict(config)
    try:
        EventBallTracker(**config)
    except TypeError as error:
        raise ValueError(f"invalid tracker configuration: {error}") from error
    return config


def _event_packet(recorded):
    return EventPacket.from_recorded_arrays(
        event_type=recorded.event_type, event_x=recorded.event_x,
        event_y=recorded.event_y, timestamps_us=recorded.event_t_us,
        packet_id=recorded.packet_id,
        packet_ros_stamp_ns=recorded.packet_ros_t_ns,
        packet_monotonic_stamp_ns=recorded.packet_monotonic_t_ns,
        original_ros_stamp_ns=recorded.packet_ros_t_ns,
        original_monotonic_stamp_ns=recorded.packet_monotonic_t_ns)


def _tracker_update(episode, recorded, detection):
    return TrackerUpdate(
        episode.name, episode.episode_index, episode.source_episode_index,
        int(recorded.packet_ros_t_ns), int(recorded.packet_id),
        int(detection.window_start_us), int(detection.window_end_us),
        float(detection.x_px), float(detection.y_px),
        float(detection.vx_px_s), float(detection.vy_px_s),
        float(detection.speed_px_s), float(detection.confidence),
        bool(detection.valid), bool(detection.velocity_valid),
        int(detection.window_event_count), int(detection.candidate_count),
        int(detection.blob_area_px), int(detection.blob_event_count),
        int(detection.blob_width_px), int(detection.blob_height_px),
        float(detection.circularity), str(detection.rejection_reason))


def stream_tracker_updates(raw_events, episodes, tracker_config, *, pre_roll_ms=0.0):
    """Yield every update, in episode order, while streaming raw packets."""
    if pre_roll_ms < 0:
        raise ValueError("pre_roll_ms must be non-negative")
    episodes = (discover_episodes(episodes) if isinstance(episodes, (str, os.PathLike))
                else list(episodes))
    config = _config_dict(tracker_config)
    with RawEventHDF5Reader(
            raw_events, width=config.get("width"), height=config.get("height")) as reader:
        for episode in episodes:
            tracker = tracker_factory(config)
            start_ns = int(round((episode.start - pre_roll_ms / 1000.0) * 1e9))
            episode_start_ns = int(round(episode.start * 1e9))
            end_ns = int(round(episode.end * 1e9))
            for recorded in reader.iter_packets(start_ros_t_ns=start_ns,
                                                end_ros_t_ns=end_ns):
                detections = tracker.update(_event_packet(recorded))
                if recorded.packet_ros_t_ns < episode_start_ns:
                    continue
                for detection in detections:
                    yield _tracker_update(episode, recorded, detection)


def run_tracker_updates(raw_events, episodes, tracker_config, *, pre_roll_ms=0.0):
    """Return native updates grouped by episode; each episode uses a fresh tracker."""
    episode_list = (discover_episodes(episodes)
                    if isinstance(episodes, (str, os.PathLike)) else list(episodes))
    result = {episode.name: [] for episode in episode_list}
    for update in stream_tracker_updates(
            raw_events, episode_list, tracker_config, pre_roll_ms=pre_roll_ms):
        result[update.episode_name].append(update)
    return result


def align_tracker_updates_to_policy_grid(updates, policy_timestamps_ns,
                                         max_observation_age_sec=np.inf):
    """Causally align update diagnostics and the independently held valid signal."""
    raw_policy_ns = np.asarray(policy_timestamps_ns)
    if not np.issubdtype(raw_policy_ns.dtype, np.integer):
        raise TypeError("policy_timestamps_ns must contain integer nanoseconds")
    policy_ns = raw_policy_ns.astype(np.int64, copy=False)
    if policy_ns.ndim != 1:
        raise ValueError("policy_timestamps_ns must be one-dimensional")
    max_age = float(max_observation_age_sec)
    if max_age < 0 or np.isnan(max_age):
        raise ValueError("max_observation_age_sec must be non-negative")
    max_age_ns = (np.iinfo(np.int64).max if np.isinf(max_age) else
                  int(round(max_age * 1e9)))
    rows = list(updates)
    # Stable sorting makes the last input update win for duplicate availability times.
    rows.sort(key=lambda row: row.available_ros_t_ns)
    available = np.asarray([row.available_ros_t_ns for row in rows], dtype=np.int64)
    latest = np.searchsorted(available, policy_ns, side="right") - 1
    has = latest >= 0
    valid_rows = [row for row in rows if row.valid]
    valid_available = np.asarray(
        [row.available_ros_t_ns for row in valid_rows], dtype=np.int64)
    latest_valid = np.searchsorted(valid_available, policy_ns, side="right") - 1
    has_valid = latest_valid >= 0
    source_ns = np.full(len(policy_ns), -1, dtype=np.int64)
    points = np.zeros((len(policy_ns), 2), dtype=np.float32)
    velocity = np.zeros((len(policy_ns), 2), dtype=np.float32)
    velocity_valid = np.zeros(len(policy_ns), dtype=np.uint8)
    sensor_start = np.full(len(policy_ns), -1, dtype=np.int64)
    sensor_end = np.full(len(policy_ns), -1, dtype=np.int64)
    packet_id = np.full(len(policy_ns), -1, dtype=np.int64)
    window_events = np.zeros(len(policy_ns), dtype=np.int32)
    candidates = np.zeros(len(policy_ns), dtype=np.int32)
    blob_area = np.zeros(len(policy_ns), dtype=np.int32)
    confidence = np.zeros(len(policy_ns), dtype=np.float32)
    if np.any(has_valid):
        positions = np.flatnonzero(has_valid)
        selected = [valid_rows[latest_valid[index]] for index in positions]
        source_ns[positions] = [row.available_ros_t_ns for row in selected]
        points[positions] = [(row.x_px, row.y_px) for row in selected]
        velocity[positions] = [(row.vx_px_s, row.vy_px_s) for row in selected]
        velocity_valid[positions] = [row.velocity_valid for row in selected]
        sensor_start[positions] = [row.sensor_window_start_us for row in selected]
        sensor_end[positions] = [row.sensor_window_end_us for row in selected]
        packet_id[positions] = [row.packet_id for row in selected]
        window_events[positions] = [row.window_event_count for row in selected]
        candidates[positions] = [row.candidate_count for row in selected]
        blob_area[positions] = [row.blob_area_px for row in selected]
        confidence[positions] = [row.confidence for row in selected]
    age_ns = np.zeros(len(policy_ns), dtype=np.int64)
    age_ns[has_valid] = policy_ns[has_valid] - source_ns[has_valid]
    fresh = has_valid & (age_ns <= max_age_ns)
    age_sec = np.full(len(policy_ns), np.nan, dtype=np.float64)
    age_sec[has_valid] = age_ns[has_valid].astype(np.float64) / 1e9
    latest_timestamp = np.full(len(policy_ns), -1, dtype=np.int64)
    latest_is_valid = np.zeros(len(policy_ns), dtype=np.uint8)
    reasons = np.full(len(policy_ns), "", dtype=object)
    if np.any(has):
        positions = np.flatnonzero(has)
        selected = [rows[latest[index]] for index in positions]
        latest_timestamp[positions] = [row.available_ros_t_ns for row in selected]
        latest_is_valid[positions] = [row.valid for row in selected]
        reasons[positions] = [row.rejection_reason for row in selected]
    return {
        "event_2d_px": points,
        "event_velocity_px_s": velocity,
        "event_valid": fresh.astype(np.uint8),
        "event_velocity_valid": velocity_valid,
        "event_source_timestamps_ns": source_ns,
        "event_source_timestamps": np.where(
            has_valid, source_ns.astype(np.float64) / 1e9, np.nan),
        "event_source_age_sec": age_sec,
        "event_source_packet_id": packet_id,
        "event_sensor_window_start_us": sensor_start,
        "event_sensor_window_end_us": sensor_end,
        "event_window_event_count": window_events,
        "event_candidate_count": candidates,
        "event_blob_area_px": blob_area,
        "event_confidence": confidence,
        "event_has_update": has.astype(np.uint8),
        "event_latest_update_valid": latest_is_valid,
        "event_latest_update_timestamp_ns": latest_timestamp,
        "event_latest_rejection_reason": reasons,
        # Backward-compatible name; explicitly describes the latest update,
        # not the independently held model-facing valid detection.
        "event_rejection_reason": reasons.copy(),
    }


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
    config = _config_dict(tracker_config)
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
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
                "tracker_config_json": config_json,
                "tracker_config_hash": hashlib.sha256(config_json.encode()).hexdigest(),
                "tracker_code_version": "EventBallTracker/offline_dataset_v2",
                "sensor_width": reader.width, "sensor_height": reader.height,
                "availability_timestamp_domain": "packet_ros_t_ns",
                "sensor_timestamp_domain": "genx320_microseconds",
                "pre_roll_ms": float(pre_roll_ms),
                "schema_version": TRACKER_SCHEMA_VERSION,
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


def _sidecar_updates(group, episode):
    columns = {name: np.asarray(group[name][:]) for name in TRACKER_FIELDS}
    reasons = group["rejection_reason"].asstr()[:]
    updates = []
    for index in range(len(columns["available_ros_t_ns"])):
        updates.append(TrackerUpdate(
            episode.name, episode.episode_index, episode.source_episode_index,
            int(columns["available_ros_t_ns"][index]), int(columns["packet_id"][index]),
            int(columns["sensor_window_start_us"][index]),
            int(columns["sensor_window_end_us"][index]),
            float(columns["x_px"][index]), float(columns["y_px"][index]),
            float(columns["vx_px_s"][index]), float(columns["vy_px_s"][index]),
            float(columns["speed_px_s"][index]), float(columns["confidence"][index]),
            bool(columns["valid"][index]), bool(columns["velocity_valid"][index]),
            int(columns["window_event_count"][index]),
            int(columns["candidate_count"][index]), int(columns["blob_area_px"][index]),
            int(columns["blob_event_count"][index]), int(columns["blob_width_px"][index]),
            int(columns["blob_height_px"][index]), float(columns["circularity"][index]),
            str(reasons[index])))
    return updates


def _sample_sidecar(group, episode, max_observation_age_sec):
    sample_ns = np.rint(episode.timestamps * 1e9).astype(np.int64)
    return align_tracker_updates_to_policy_grid(
        _sidecar_updates(group, episode), sample_ns, max_observation_age_sec)


def _put_dataset(group, name, values, *, overwrite):
    """Create a dataset, accept an identical rerun, or require explicit overwrite."""
    values = np.asarray(values)
    if name in group:
        existing = group[name]
        if name.endswith("rejection_reason"):
            same = existing.shape == values.shape and np.array_equal(
                existing.asstr()[:], values.astype(str))
        else:
            same = existing.shape == values.shape and np.array_equal(
                existing[:], values, equal_nan=True)
        if same:
            return
        if not overwrite:
            raise ValueError(f"conflicting /{group.name.strip('/')}/{name}; "
                             "use --overwrite-event-fields")
        del group[name]
    if name.endswith("rejection_reason"):
        group.create_dataset(name, data=np.asarray(values, dtype=STRING_DTYPE))
    else:
        group.create_dataset(name, data=values)


def enrich(episodes_dir, tracker_output, output_dir, *, overwrite=False,
           rgb_tracks_dir=None, write_empty_rgb_track=False, require_rgb_2d=False,
           overwrite_event_fields=False, max_observation_age_sec=np.inf):
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
                    sparse = observations.require_group("sparse_tracking")
                    sampled = _sample_sidecar(
                        sidecar[f"episodes/{episode.name}"], episode,
                        max_observation_age_sec)
                    for name, values in sampled.items():
                        _put_dataset(sparse, name, values,
                                     overwrite=overwrite_event_fields)
                    sparse.attrs.update({
                        "schema_version": SCHEMA_VERSION,
                        "event_coordinate_system": "openmv_native_unrotated_px",
                        "event_width_px": int(metadata["sensor_width"]),
                        "event_height_px": int(metadata["sensor_height"]),
                        "sampling_policy": (
                            "latest_valid_tracker_update_at_or_before_observation_timestamp"),
                        "max_observation_age_sec": float(max_observation_age_sec),
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
                        rgb_values = (("rgb_2d_px", points),
                                      ("rgb_valid", valid))
                        for name, values in rgb_values:
                            if (name in sparse and
                                    not np.array_equal(sparse[name][:], values)):
                                raise ValueError(
                                    "supplied RGB data conflicts with existing "
                                    f"{name}")
                            if name not in sparse:
                                sparse.create_dataset(name, data=values)
                    elif write_empty_rgb_track:
                        if "rgb_2d_px" not in sparse:
                            sparse.create_dataset("rgb_2d_px", data=np.zeros(
                                (len(episode.timestamps), 2), dtype="f4"))
                        if "rgb_valid" not in sparse:
                            sparse.create_dataset("rgb_valid", data=np.zeros(
                                len(episode.timestamps), dtype="u1"))
                    target.attrs["sparse_tracking_source_raw_events_h5"] = (
                        metadata["raw_events_h5"])
                    target.attrs["sparse_tracking_tracker_config_json"] = (
                        metadata["tracker_config_json"])
                    for key in ("tracker_config_hash", "tracker_code_version",
                                "availability_timestamp_domain", "sensor_timestamp_domain",
                                "pre_roll_ms", "schema_version"):
                        if key in metadata:
                            target.attrs[f"sparse_tracking_{key}"] = metadata[key]
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
    enrich_parser.add_argument("--overwrite-event-fields", action="store_true")
    enrich_parser.add_argument("--max-observation-age-sec", type=float,
                               default=np.inf)
    _add_rgb_options(enrich_parser)
    build = commands.add_parser("build")
    build.add_argument("--raw-events", required=True)
    build.add_argument("--episodes", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--tracker-config", required=True)
    build.add_argument("--tracker-output", "--save-tracker-output",
                       dest="tracker_output")
    build.add_argument("--reuse-tracker-output", action="store_true")
    build.add_argument("--pre-roll-ms", type=float, default=0.0)
    build.add_argument("--max-observation-age-sec", type=float, default=np.inf)
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--overwrite-event-fields", action="store_true")
    _add_rgb_options(build)
    return parser


def main(argv=None):
    """Run the selected offline dataset command."""
    args = make_parser().parse_args(argv)
    temporary_sidecar = None
    if args.command in ("run-tracker", "build"):
        if args.pre_roll_ms < 0:
            raise ValueError("--pre-roll-ms must be non-negative")
        if (args.command == "build" and args.reuse_tracker_output and
                (not args.tracker_output or
                 not Path(args.tracker_output).is_file())):
            raise ValueError(
                "--reuse-tracker-output requires an existing --tracker-output")
        if args.command == "run-tracker":
            sidecar = args.output
        elif args.tracker_output:
            sidecar = args.tracker_output
        else:
            fd, temporary_sidecar = tempfile.mkstemp(
                prefix="openmv-tracker-", suffix=".h5")
            os.close(fd)
            sidecar = temporary_sidecar
        if not (args.command == "build" and args.reuse_tracker_output):
            run_tracker(args.raw_events, args.episodes, sidecar,
                        args.tracker_config, pre_roll_ms=args.pre_roll_ms)
    try:
        if args.command in ("enrich", "build"):
            tracker_output = args.tracker_output if args.command == "enrich" else sidecar
            enrich(args.episodes, tracker_output, args.output,
                   overwrite=args.overwrite, rgb_tracks_dir=args.rgb_tracks_dir,
                   write_empty_rgb_track=args.write_empty_rgb_track,
                   require_rgb_2d=args.require_rgb_2d,
                   overwrite_event_fields=args.overwrite_event_fields,
                   max_observation_age_sec=args.max_observation_age_sec)
    finally:
        if temporary_sidecar is not None:
            Path(temporary_sidecar).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
