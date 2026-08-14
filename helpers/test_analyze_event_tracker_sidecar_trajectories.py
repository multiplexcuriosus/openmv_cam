#!/usr/bin/env python3
"""Synthetic coverage for native tracker-sidecar trajectory analysis."""

import contextlib
import csv
import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_event_tracker_sidecar_trajectories as analysis  # noqa: E402


FIELDS = ("packet_id", "vx_px_s", "vy_px_s", "speed_px_s", "confidence",
          "velocity_valid", "window_event_count", "candidate_count", "blob_area_px",
          "blob_event_count", "blob_width_px", "blob_height_px", "circularity")


def write_sidecar(path, points, valid=None, timestamps=None, reasons=None,
                  starts=None, ends=None, episode="episode_0"):
    points = np.asarray(points, float); n = len(points)
    valid = np.ones(n, np.uint8) if valid is None else np.asarray(valid, np.uint8)
    timestamps = 1_000_000_000 + np.arange(n) * 20_000_000 if timestamps is None else np.asarray(timestamps, np.int64)
    starts = np.arange(n) * 20_000 if starts is None else np.asarray(starts)
    ends = starts + 10_000 if ends is None else np.asarray(ends)
    with h5py.File(path, "w") as h5:
        h5.create_group("metadata")
        group = h5.require_group("episodes").create_group(episode)
        group.create_dataset("available_ros_t_ns", data=timestamps)
        group.create_dataset("sensor_window_start_us", data=starts)
        group.create_dataset("sensor_window_end_us", data=ends)
        group.create_dataset("x_px", data=points[:, 0]); group.create_dataset("y_px", data=points[:, 1])
        group.create_dataset("valid", data=valid)
        for field in FIELDS:
            data = np.arange(n) + 1 if field in ("packet_id", "window_event_count", "candidate_count") else np.zeros(n)
            group.create_dataset(field, data=data)
        strings = reasons if reasons is not None else ["accepted" if x else "no_blob" for x in valid]
        group.create_dataset("rejection_reason", data=np.asarray(strings, dtype=h5py.string_dtype()))


def load(path, smoothing=1, gap=150):
    with h5py.File(path, "r") as h5:
        return analysis.load_episode(h5["episodes/episode_0"], smoothing, gap)


TURN = np.asarray([[0, 50], [10, 50], [20, 50], [30, 50], [20, 50], [10, 50], [0, 50]])


def test_clear_native_forward_reverse_detects_turn(tmp_path):
    path=tmp_path/"sidecar.h5"; write_sidecar(path,TURN); arrays,tr,_=load(path)
    result=analysis.infer_turning_point(tr,int(arrays["available_ros_t_ns"][0]),2,5,120)
    assert result["selected"]["native_update_index"] == 3
    assert result["selected"]["reversal_angle_deg"] == pytest.approx(180)


def test_monotonic_trajectory_has_no_turn(tmp_path):
    path=tmp_path/"sidecar.h5"; write_sidecar(path,[[x,0] for x in range(0,70,10)]); arrays,tr,_=load(path)
    assert analysis.infer_turning_point(tr,int(arrays["available_ros_t_ns"][0]))["status"] == "no_turn_detected"


def test_invalid_rows_excluded_but_retained_in_diagnostics(tmp_path):
    path=tmp_path/"sidecar.h5"; write_sidecar(path,TURN,valid=[1,1,0,1,1,0,1]); arrays,tr,diag=load(path)
    assert tr.native_update_indices.tolist() == [0,1,3,4,6]
    metric=analysis._metric("episode_0",arrays,tr,diag,{"status":"no_turn_detected","selected":None,"candidates":[],"ambiguous_candidates":[]},4)
    assert metric["invalid_native_row_count"] == 2
    assert metric["longest_valid_detection_gap_ms"] == 40


def test_nonmonotonic_timestamps_stably_sorted(tmp_path):
    path=tmp_path/"sidecar.h5"; times=[100,300,200,200,400,500,600]; write_sidecar(path,TURN,timestamps=times)
    arrays,tr,diag=load(path)
    assert diag["timestamps_stably_sorted"]
    assert arrays["native_update_index"].tolist()[:4] == [0,2,3,1]
    assert np.all(np.diff(tr.timestamps_ns)>=0)


def test_large_gap_breaks_segments_and_turn(tmp_path):
    path=tmp_path/"sidecar.h5"; times=np.asarray([0,20,40,500,520,540,560])*1_000_000; write_sidecar(path,TURN,timestamps=times)
    arrays,tr,_=load(path,gap=150)
    assert len(np.unique(tr.segment_ids)) == 2
    assert analysis.infer_turning_point(tr,int(arrays["available_ros_t_ns"][0]))["status"] == "no_turn_detected"


def test_rejection_reasons_counted(tmp_path):
    path=tmp_path/"sidecar.h5"; out=tmp_path/"out"; write_sidecar(path,TURN,reasons=["ok","no_blob","no_blob","small","ok","ok","small"])
    assert analysis.main([str(path),"--output-dir",str(out),"--smoothing-window","1"]) == 0
    rows={r["rejection_reason"]:int(r["count"]) for r in csv.DictReader((out/"sidecar_rejection_statistics.csv").open())}
    assert rows == {"no_blob":2,"ok":3,"small":2}


def test_sensor_window_durations(tmp_path):
    path=tmp_path/"sidecar.h5"; starts=np.arange(7)*1000; ends=starts+np.asarray([1000,2000,3000,4000,5000,6000,7000]); write_sidecar(path,TURN,starts=starts,ends=ends)
    arrays,tr,diag=load(path); metric=analysis._metric("episode_0",arrays,tr,diag,{"status":"no_turn_detected","selected":None,"candidates":[],"ambiguous_candidates":[]},4)
    assert metric["sensor_window_duration_median_ms"] == 4
    assert metric["sensor_window_duration_max_ms"] == 7


def test_native_and_valid_indices_are_distinct(tmp_path):
    path=tmp_path/"sidecar.h5"; write_sidecar(path,TURN,valid=[0,1,1,1,1,1,1]); arrays,tr,_=load(path)
    result=analysis.infer_turning_point(tr,int(arrays["available_ros_t_ns"][0]),half_window=1)
    assert result["selected"]["native_update_index"] == 3
    assert result["selected"]["valid_detection_index"] == 2


def test_raw_event_input_rejected_usefully(tmp_path):
    path=tmp_path/"recording_raw_events.h5"
    with h5py.File(path,"w") as h5: h5.create_group("packets")
    stderr=io.StringIO()
    with contextlib.redirect_stderr(stderr): result=analysis.main([str(path),"--output-dir",str(tmp_path/"out")])
    assert result == 2
    assert "raw-event packet HDF5" in stderr.getvalue()


def test_optional_episode_pairing(tmp_path):
    sidecar=tmp_path/"sidecar.h5"; episodes=tmp_path/"episodes"; episodes.mkdir(); out=tmp_path/"out"; write_sidecar(sidecar,TURN)
    with h5py.File(episodes/"episode_0.hdf5","w") as h5:
        obs=h5.require_group("observations"); obs.create_dataset("timestamps_ns",data=1_000_000_000+np.arange(7)*20_000_000)
        sparse=obs.create_group("sparse_tracking"); sparse.create_dataset("rgb_2d_px",data=TURN+100); sparse.create_dataset("rgb_valid",data=np.ones(7)); sparse.create_dataset("rgb_source_timestamps",data=1+np.arange(7)*.02)
    assert analysis.main([str(sidecar),"--output-dir",str(out),"--episodes-dir",str(episodes),"--smoothing-window","1"]) == 0
    summary=json.loads((out/"summary.json").read_text())
    assert len(summary["policy_grid_comparisons"]) == 1
    assert "native_event_minus_rgb_turn_time_ms" in summary["policy_grid_comparisons"][0]


def test_all_required_outputs_and_plots(tmp_path):
    pytest.importorskip("matplotlib")
    sidecar=tmp_path/"sidecar.h5"; out=tmp_path/"out"; write_sidecar(sidecar,TURN)
    assert analysis.main([str(sidecar),"--output-dir",str(out),"--plot","--smoothing-window","1"]) == 0
    for name in ("episode_metrics.csv","summary.json","sidecar_rejection_statistics.csv","sidecar_update_intervals.csv","sidecar_sensor_window_statistics.csv"):
        assert (out/name).is_file()
    for name in ("episode_0_event_trajectory.png","episode_0_event_validity_timeline.png","episode_0_event_update_intervals.png","all_event_trajectories.png","all_event_trajectories_turn_aligned.png","turning_point_time_distribution.png","valid_detection_count_distribution.png","native_update_interval_distribution.png","valid_detection_gap_distribution.png","sensor_window_duration_distribution.png","rejection_reason_counts.png"):
        assert (out/"plots"/name).is_file(), name
