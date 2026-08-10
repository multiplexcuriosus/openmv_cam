import ast
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from openmv_cam.hdf5_replay import (
    HDF5ReplayReader,
    ReplaySchemaError,
    replay_packets,
    scaled_packet_delay_seconds,
)


def write_recording(path, packets=None, monotonic=None):
    packets = packets if packets is not None else [
        [(1, 4, 5, 1_002_003), (0, 8, 9, 1_002_001)],
        [],
        [(1, 12, 13, 1_004_000)],
    ]
    monotonic = monotonic if monotonic is not None else [10, 20, 50]
    types, xs, ys, timestamps, ids = [], [], [], [], []
    starts, ends, counts = [], [], []
    for packet_idx, rows in enumerate(packets):
        starts.append(len(types))
        for event_type, x, y, t_us in rows:
            types.append(event_type)
            xs.append(x)
            ys.append(y)
            timestamps.append(t_us)
            ids.append(packet_idx)
        ends.append(len(types))
        counts.append(len(rows))
    with h5py.File(path, "w") as h5f:
        events = h5f.create_group("events")
        events.create_dataset("type", data=np.asarray(types, dtype=np.uint8))
        events.create_dataset("x", data=np.asarray(xs, dtype=np.uint16))
        events.create_dataset("y", data=np.asarray(ys, dtype=np.uint16))
        events.create_dataset("t_us", data=np.asarray(timestamps, dtype=np.int64))
        events.create_dataset("packet_id", data=np.asarray(ids, dtype=np.int64))
        metadata = h5f.create_group("packets")
        metadata.create_dataset("ros_t_ns", data=np.arange(len(packets), dtype=np.int64) + 100)
        metadata.create_dataset("monotonic_t_ns", data=np.asarray(monotonic, dtype=np.int64))
        metadata.create_dataset("start_event_idx", data=np.asarray(starts, dtype=np.int64))
        metadata.create_dataset("end_event_idx", data=np.asarray(ends, dtype=np.int64))
        metadata.create_dataset("event_count", data=np.asarray(counts, dtype=np.int64))


def test_schema_validation_and_exact_packet_reconstruction(tmp_path):
    path = tmp_path / "events.h5"
    write_recording(path)
    with HDF5ReplayReader(str(path)) as reader:
        packets = [reader.read_packet(i, ros_now_ns=900 + i,
                                      monotonic_now_ns=800 + i)
                   for i in reader.indices()]
    assert [p.event_count for p in packets] == [2, 0, 1]
    assert [p.packet_id for p in packets] == [0, 1, 2]
    assert packets[0].events.tolist() == [
        [1, 1, 2, 3, 4, 5], [0, 1, 2, 1, 8, 9]]
    assert packets[0].timestamps_us.tolist() == [1_002_003, 1_002_001]
    assert packets[0].first_event_timestamp_us == 1_002_001
    assert packets[0].last_event_timestamp_us == 1_002_003
    assert packets[0].source == "hdf5_replay"
    assert packets[0].packet_ros_stamp_ns == 900
    assert packets[0].original_ros_stamp_ns == 100
    assert packets[1].events.shape == (0, 6)


def test_packet_range_is_inclusive_and_preserves_original_indices(tmp_path):
    path = tmp_path / "events.h5"
    write_recording(path)
    with HDF5ReplayReader(str(path), start_packet=1, end_packet=2) as reader:
        assert list(reader.indices()) == [1, 2]
        assert reader.read_packet(2, ros_now_ns=1,
                                  monotonic_now_ns=2).events[:, 4].tolist() == [12]


@pytest.mark.parametrize("start,end", [(-1, 1), (3, -1), (2, 1), (0, 3)])
def test_invalid_packet_ranges(tmp_path, start, end):
    path = tmp_path / "events.h5"
    write_recording(path)
    with pytest.raises(ValueError, match="range|start_packet|outside"):
        HDF5ReplayReader(str(path), start_packet=start, end_packet=end)


def test_missing_file_and_incompatible_schema(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        HDF5ReplayReader(str(tmp_path / "missing.h5"))
    bad = tmp_path / "bad.h5"
    with h5py.File(bad, "w") as h5f:
        h5f.create_group("events")
    with pytest.raises(ReplaySchemaError, match="dataset|group"):
        HDF5ReplayReader(str(bad))


def test_boundary_and_packet_id_corruption_are_rejected(tmp_path):
    path = tmp_path / "events.h5"
    write_recording(path)
    with h5py.File(path, "r+") as h5f:
        h5f["packets/start_event_idx"][1] = 1
    with pytest.raises(ReplaySchemaError, match="boundary"):
        HDF5ReplayReader(str(path))


def test_recorded_and_sensor_schedule_and_rate_scaling():
    assert scaled_packet_delay_seconds(
        1_000_000_000, 1_500_000_000, 1.0,
        units_per_second=1e9) == 0.5
    assert scaled_packet_delay_seconds(
        1_000, 2_000, 2.0, units_per_second=1e6) == 0.0005
    with pytest.raises(ValueError, match="non-monotonic"):
        scaled_packet_delay_seconds(2, 1, 1.0, units_per_second=1e9)
    with pytest.raises(ValueError, match="missing"):
        scaled_packet_delay_seconds(-1, 2, 1.0, units_per_second=1e9)


class FakeStop:
    def __init__(self):
        self.stopped = False
        self.waits = []

    def is_set(self):
        return self.stopped

    def wait(self, delay):
        self.waits.append(delay)
        return self.stopped


def run_replay(path, timing, *, rate=1.0, loop=False, stop=None,
               monotonic_values=None):
    stop = stop or FakeStop()
    output, warnings = [], []
    values = iter(monotonic_values or range(0, 100_000_000_000, 1_000_000))
    with HDF5ReplayReader(str(path)) as reader:
        diagnostics = replay_packets(
            reader, timing=timing, rate=rate, loop=loop, stop_event=stop,
            process_packet=output.append, ros_now_ns=lambda: 999,
            monotonic_now_ns=lambda: next(values), warn=warnings.append)
    return output, stop.waits, warnings, diagnostics


def test_fast_mode_never_waits_and_uses_current_host_stamps(tmp_path):
    path = tmp_path / "events.h5"
    write_recording(path)
    packets, waits, warnings, diagnostics = run_replay(path, "fast")
    assert waits == []
    assert warnings == []
    assert [p.packet_ros_stamp_ns for p in packets] == [999, 999, 999]
    assert diagnostics.replay_packets_processed == 3
    assert diagnostics.replay_events_processed == 3


def test_recorded_timing_waits_are_scaled(tmp_path):
    path = tmp_path / "events.h5"
    write_recording(path, monotonic=[1_000_000_000, 1_100_000_000, 1_300_000_000])
    _, waits, _, _ = run_replay(path, "recorded", rate=2.0)
    assert waits == pytest.approx([0.048, 0.146])


def test_sensor_timing_falls_back_on_empty_packet(tmp_path):
    path = tmp_path / "events.h5"
    write_recording(path)
    packets, waits, warnings, diagnostics = run_replay(path, "sensor")
    assert len(packets) == 3
    assert waits == []
    assert len(warnings) == 1
    assert "falling back to fast" in warnings[0]
    assert diagnostics.replay_timing_fallbacks == 1


def test_recorded_timing_falls_back_on_timestamp_reset(tmp_path):
    path = tmp_path / "events.h5"
    write_recording(path, monotonic=[100, 90, 110])
    packets, waits, warnings, diagnostics = run_replay(path, "recorded")
    assert len(packets) == 3
    assert waits == []
    assert "packet 1" in warnings[0]
    assert diagnostics.replay_timing_fallbacks == 1


def test_loop_restarts_at_selected_start_and_shutdown_is_clean(tmp_path):
    path = tmp_path / "events.h5"
    write_recording(path)
    stop = FakeStop()
    seen = []
    with HDF5ReplayReader(str(path), start_packet=1, end_packet=2) as reader:
        def process(packet):
            seen.append(packet.packet_id)
            if len(seen) == 5:
                stop.stopped = True
        diagnostics = replay_packets(
            reader, timing="fast", rate=1.0, loop=True, stop_event=stop,
            process_packet=process, ros_now_ns=lambda: 1,
            monotonic_now_ns=lambda: 2)
    assert seen == [1, 2, 1, 2, 1]
    assert diagnostics.replay_loops_completed == 2


def test_node_has_exclusive_input_open_and_one_shared_processing_path():
    source_path = Path(__file__).parents[1] / "openmv_cam" / "openmv_cam_node.py"
    source = source_path.read_text()
    tree = ast.parse(source)
    methods = {node.name: node for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    init_source = ast.get_source_segment(source, methods["__init__"])
    hardware_source = ast.get_source_segment(source, methods["_reader_loop"])
    replay_source = ast.get_source_segment(source, methods["_replay_loop"])
    assert 'self.event_input_mode == "hardware"' in init_source
    assert "self._open_serial()" in init_source
    assert "HDF5ReplayReader(" in init_source
    assert "self._process_packet(packet)" in hardware_source
    assert "process_packet=self._process_packet" in replay_source


def test_replay_trace_detail_json_is_finite():
    detail = {
        "source": "hdf5_replay", "packet_index": 2,
        "original_recorded_ros_t_ns": 100,
        "original_recorded_monotonic_t_ns": 200,
    }
    encoded = json.dumps(detail, allow_nan=False)
    assert json.loads(encoded) == detail
