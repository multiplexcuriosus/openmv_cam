import hashlib
import json

import h5py
import numpy as np
import pytest

from openmv_cam.event_ball_tracker import TrackerDetection
from openmv_cam.offline_dataset import (
    discover_episodes, enrich, load_tracker_config, run_tracker,
)
from openmv_cam.offline_raw_events import RawEventHDF5Reader


def raw_file(path, *, bad=None):
    with h5py.File(path, "w") as target:
        target.attrs["width"] = 320
        target.attrs["height"] = 320
        events = target.create_group("events")
        events.create_dataset("type", data=np.ones(6, dtype="u1"))
        events.create_dataset("x", data=np.asarray([10, 11, 12, 20, 21, 22], "u2"))
        events.create_dataset("y", data=np.asarray([10, 10, 10, 20, 20, 20], "u2"))
        events.create_dataset("t_us", data=np.asarray(
            [100, 1100, 3100, 10100, 11100, 13100], "i8"))
        packets = target.create_group("packets")
        packets.create_dataset("ros_t_ns", data=np.asarray(
            [900_000_000, 1_000_000_000, 2_000_000_000], "i8"))
        packets.create_dataset("start_event_idx", data=np.asarray([0, 2, 4], "i8"))
        packets.create_dataset("end_event_idx", data=np.asarray([2, 4, 6], "i8"))
        packets.create_dataset("monotonic_t_ns", data=np.asarray([1, 2, 3], "i8"))
        if bad == "event_length":
            del events["x"]
            events.create_dataset("x", data=np.asarray([1], "u2"))
        elif bad == "packet_order":
            packets["ros_t_ns"][:] = [2, 1, 3]
        elif bad == "indices":
            packets["end_event_idx"][-1] = 7


def episode_file(path, index, start, end, timestamps):
    with h5py.File(path, "w") as target:
        target.attrs.update({"episode_index": index, "source_episode_index": index + 10,
                             "episode_start": start, "episode_end": end,
                             "preserved": "yes"})
        observations = target.create_group("observations")
        observations.create_dataset("timestamps", data=np.asarray(timestamps, "f8"))
        observations.create_dataset("images", data=np.zeros((len(timestamps), 2, 2), "u1"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("bad, text", [
    ("event_length", "different lengths"), ("packet_order", "non-decreasing"),
    ("indices", "0 <= start_event_idx")])
def test_raw_schema_validation(tmp_path, bad, text):
    path = tmp_path / "raw.h5"
    raw_file(path, bad=bad)
    with pytest.raises(ValueError, match=text):
        RawEventHDF5Reader(path)


def test_half_open_packet_streaming_and_default_packet_id(tmp_path):
    path = tmp_path / "raw.h5"
    raw_file(path)
    with RawEventHDF5Reader(path) as reader:
        iterator = reader.iter_packets()
        first = next(iterator)
        assert first.event_x.tolist() == [10, 11]
        assert first.packet_id == 0
        assert first.event_x.base is None or not isinstance(first.event_x.base, h5py.Dataset)
        second = next(iterator)
        assert second.event_x.tolist() == [12, 20]


def test_episode_numeric_sort_and_source_indices(tmp_path):
    episode_file(tmp_path / "episode_10.hdf5", 10, 2, 3, [2.0])
    episode_file(tmp_path / "episode_2.hdf5", 2, 0, 1, [0.0])
    episodes = discover_episodes(tmp_path)
    assert [episode.episode_index for episode in episodes] == [2, 10]
    assert [episode.source_episode_index for episode in episodes] == [12, 20]


def test_tracker_config_and_end_to_end_reset_assignment(tmp_path, monkeypatch):
    raw = tmp_path / "raw.h5"
    raw_file(raw)
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    episode_file(episodes / "episode_0.hdf5", 0, 0.8, 1.2, [0.8, 1.0, 1.2])
    episode_file(episodes / "episode_1.hdf5", 1, 1.8, 2.2, [1.8, 2.0, 2.2])
    config = tmp_path / "tracker.json"
    config.write_text(json.dumps({"min_event_count": 1, "min_blob_area_px": 1,
                                  "morphology_operation": "none"}))
    assert load_tracker_config(config)["min_event_count"] == 1

    instances = []

    class FakeTracker:
        def __init__(self, **kwargs):
            self.ids = []
            instances.append(self)

        def update(self, packet):
            self.ids.append(packet.packet_id)
            return [TrackerDetection(0, 1, packet.packet_id, packet.event_count,
                    window_start_us=packet.first_event_timestamp_us,
                    window_end_us=packet.last_event_timestamp_us + 1,
                    window_event_count=packet.event_count, valid=True)]
    monkeypatch.setattr("openmv_cam.offline_dataset.EventBallTracker", FakeTracker)
    sidecar = tmp_path / "tracker.h5"
    run_tracker(raw, episodes, sidecar, config)
    assert len(instances) == 3  # one config validation plus one fresh tracker/episode
    assert instances[1].ids == [0, 1]
    assert instances[2].ids == [2]
    with h5py.File(sidecar, "r") as result:
        assert result[
            "episodes/episode_0/available_ros_t_ns"][:].tolist() == [
                900_000_000, 1_000_000_000]
        assert result["episodes/episode_1/available_ros_t_ns"][:].tolist() == [2_000_000_000]


def sidecar_file(path):
    with h5py.File(path, "w") as target:
        metadata = target.create_group("metadata")
        metadata.attrs.update({"raw_events_h5": "/raw.h5", "tracker_config_json": "{}",
                               "sensor_width": 320, "sensor_height": 320})
        group = target.create_group("episodes/episode_0")
        values = {
            "available_ros_t_ns": np.asarray([1_000_000_000, 2_000_000_000], "i8"),
            "packet_id": np.asarray([1, 2], "i8"),
            "sensor_window_start_us": np.asarray([10, 20], "i8"),
            "sensor_window_end_us": np.asarray([11, 21], "i8"),
            "x_px": np.asarray([4, 9], "f4"), "y_px": np.asarray([5, 10], "f4"),
            "vx_px_s": np.asarray([1, 2], "f4"), "vy_px_s": np.asarray([2, 3], "f4"),
            "speed_px_s": np.zeros(2, "f4"), "confidence": np.asarray([.2, .3], "f4"),
            "valid": np.asarray([0, 1], "u1"), "velocity_valid": np.asarray([0, 1], "u1"),
            "window_event_count": np.asarray([3, 4], "i4"),
            "candidate_count": np.asarray([0, 1], "i4"),
            "blob_area_px": np.asarray([0, 6], "i4"), "blob_event_count": np.zeros(2, "i4"),
            "blob_width_px": np.zeros(2, "i4"), "blob_height_px": np.zeros(2, "i4"),
            "circularity": np.zeros(2, "f4"),
            "rejection_reason": np.asarray([b"no_blob", b""], dtype="S256"),
        }
        for name, value in values.items():
            group.create_dataset(name, data=value)


def test_enrichment_is_causal_atomic_and_preserves_source(tmp_path):
    episodes = tmp_path / "episodes"
    output = tmp_path / "output"
    episodes.mkdir()
    source = episodes / "episode_0.hdf5"
    episode_file(source, 0, 0, 3, [0.5, 1.0, 1.5, 2.0])
    before = digest(source)
    sidecar = tmp_path / "tracker.h5"
    sidecar_file(sidecar)
    with pytest.warns(UserWarning, match="event-only"):
        enrich(episodes, sidecar, output)
    assert digest(source) == before
    assert not list(output.glob("*.tmp"))
    with h5py.File(output / source.name, "r") as result:
        sparse = result["observations/sparse_tracking"]
        assert sparse["event_2d_px"].shape == (4, 2)
        assert sparse["event_2d_px"].dtype == np.dtype("f4")
        assert sparse["event_2d_px"][:].tolist() == [[0, 0], [4, 5], [4, 5], [9, 10]]
        assert sparse["event_has_update"][:].tolist() == [0, 1, 1, 1]
        assert sparse["event_valid"][:].tolist() == [0, 0, 0, 1]
        assert sparse["event_rejection_reason"].asstr()[:].tolist()[1] == "no_blob"
        assert np.isnan(sparse["event_source_timestamps"][0])
        assert result.attrs["preserved"] == "yes"
        assert result["observations/images"].dtype == np.dtype("u1")
        assert sparse.attrs["schema_version"] == "sparse_tracking_v1"
        assert result.attrs["sparse_tracking_source_raw_events_h5"] == "/raw.h5"
    with pytest.raises(FileExistsError):
        enrich(episodes, sidecar, output)


def test_rgb_causal_alignment_and_missing_requirement(tmp_path):
    episodes = tmp_path / "episodes"
    rgb_dir = tmp_path / "rgb"
    episodes.mkdir()
    rgb_dir.mkdir()
    episode_file(episodes / "episode_0.hdf5", 0, 0, 3, [0.5, 1.0, 1.5])
    sidecar = tmp_path / "tracker.h5"
    sidecar_file(sidecar)
    with h5py.File(rgb_dir / "episode_0.hdf5", "w") as rgb:
        rgb.create_dataset("timestamps", data=[1.0, 2.0])
        rgb.create_dataset("rgb_2d_px", data=np.asarray([[3, 4], [8, 9]], "f4"))
        rgb.create_dataset("valid", data=np.asarray([1, 1], "u1"))
    output = tmp_path / "output"
    enrich(episodes, sidecar, output, rgb_tracks_dir=rgb_dir)
    with h5py.File(output / "episode_0.hdf5", "r") as result:
        sparse = result["observations/sparse_tracking"]
        assert sparse["rgb_2d_px"][:].tolist() == [[0, 0], [3, 4], [3, 4]]
        assert sparse["rgb_valid"][:].tolist() == [0, 1, 1]
    with pytest.raises(ValueError, match="RGB 2D track unavailable"):
        enrich(episodes, sidecar, tmp_path / "required", require_rgb_2d=True)
