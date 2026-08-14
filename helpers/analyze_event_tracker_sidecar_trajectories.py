#!/usr/bin/env python3
"""Analyze native OpenMV tracker-output sidecar trajectories.

The positional input must be an HDF5 tracker sidecar with ``/episodes/episode_N``
groups and native tracker rows.  It is not a ``*_raw_events.h5`` packet file and
it is not a 30/60 Hz enriched policy-grid episode file.  ``available_ros_t_ns``
is authoritative; no policy frame or source-age value is invented.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


REQUIRED = ("available_ros_t_ns", "sensor_window_start_us", "sensor_window_end_us", "x_px", "y_px", "valid")
OPTIONAL_NUMERIC = ("packet_id", "vx_px_s", "vy_px_s", "speed_px_s", "confidence", "velocity_valid", "window_event_count", "candidate_count", "blob_area_px", "blob_event_count", "blob_width_px", "blob_height_px", "circularity")
AMBIGUOUS_SCORE_FRACTION = 0.90


@dataclass
class NativeTrajectory:
    native_update_indices: np.ndarray
    valid_detection_indices: np.ndarray
    timestamps_ns: np.ndarray
    points: np.ndarray
    smoothed_points: np.ndarray
    segment_ids: np.ndarray


def _number(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _number(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_number(v) for v in value]
    return value


def stats(values: Sequence[float]) -> dict[str, Any]:
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if not a.size:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "min": None, "max": None}
    return {"count": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "p95": float(np.percentile(a, 95)),
            "min": float(a.min()), "max": float(a.max())}


def robust_median_smooth(points: np.ndarray, window: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if window <= 1 or not len(points):
        return points.copy()
    half = window // 2
    return np.asarray([np.median(points[max(0, i-half):min(len(points), i+half+1)], axis=0)
                       for i in range(len(points))])


def _decode_reasons(values: np.ndarray) -> list[str]:
    result = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, (bytes, np.bytes_)):
            value = value.decode("utf-8", errors="replace")
        text = str(value).strip()
        result.append(text if text else "<empty>")
    return result


def _episode_sort_key(name: str) -> tuple[int, Any]:
    match = re.fullmatch(r"episode_(\d+)", name)
    return (0, int(match.group(1))) if match else (1, name)


def validate_sidecar(h5: h5py.File, path: Path) -> list[str]:
    if "episodes" not in h5 or not isinstance(h5["episodes"], h5py.Group):
        packet_hints = {"events", "packets", "event_packets", "raw_events"}.intersection(h5.keys())
        hint = " This appears to be a raw-event packet HDF5." if packet_hints or "raw_events" in path.stem and "tracker_outputs" not in path.stem else ""
        raise ValueError("unsupported input: expected tracker-output sidecar layout /episodes/episode_N with x_px/y_px/valid/available_ros_t_ns." + hint)
    names = sorted((name for name, obj in h5["episodes"].items() if isinstance(obj, h5py.Group)), key=_episode_sort_key)
    if not names:
        raise ValueError("unsupported input: /episodes contains no episode groups")
    if not any(all(field in h5["episodes"][name] for field in REQUIRED) for name in names):
        raise ValueError("unsupported input: episode groups lack native tracker fields; raw-event and enriched policy-grid HDF5 files are not accepted")
    return names


def load_episode(group: h5py.Group, smoothing_window: int, max_gap_ms: float) -> tuple[dict[str, np.ndarray], NativeTrajectory, dict[str, Any]]:
    missing = [field for field in REQUIRED if field not in group]
    if missing:
        raise ValueError("missing sidecar fields: " + ", ".join(missing))
    arrays = {name: np.asarray(group[name]).reshape(-1) for name in REQUIRED + OPTIONAL_NUMERIC if name in group}
    arrays["rejection_reason"] = np.asarray(group["rejection_reason"]).reshape(-1) if "rejection_reason" in group else np.asarray([""] * len(arrays["valid"]))
    n = len(arrays["available_ros_t_ns"])
    bad = [name for name, value in arrays.items() if len(value) != n]
    if bad:
        raise ValueError("dataset lengths differ: " + ", ".join(bad))
    original_t = arrays["available_ros_t_ns"].astype(np.int64)
    monotonic = bool(n < 2 or np.all(np.diff(original_t) >= 0))
    order = np.arange(n) if monotonic else np.argsort(original_t, kind="stable")
    arrays = {name: value[order] for name, value in arrays.items()}
    # The index denotes the row in native chronological analysis order. Stable sorting
    # is the sole permitted reordering and duplicate timestamps remain distinct.
    arrays["native_update_index"] = order.astype(np.int64)
    valid_flag = arrays["valid"].astype(bool)
    finite_xy = np.isfinite(arrays["x_px"].astype(float)) & np.isfinite(arrays["y_px"].astype(float))
    usable = valid_flag & finite_xy
    selected = np.flatnonzero(usable)
    ts = arrays["available_ros_t_ns"].astype(np.int64)[selected]
    gaps_ms = np.diff(ts).astype(float) / 1e6
    segments = np.r_[0, np.cumsum(gaps_ms > max_gap_ms)].astype(int) if len(selected) else np.empty(0, dtype=int)
    points = np.column_stack((arrays["x_px"][selected], arrays["y_px"][selected])).astype(float)
    trajectory = NativeTrajectory(arrays["native_update_index"][selected], np.arange(len(selected), dtype=np.int64), ts, points,
                                  robust_median_smooth(points, smoothing_window), segments)
    diagnostics = {"timestamps_were_monotonic": monotonic, "timestamps_stably_sorted": not monotonic,
                   "original_native_update_indices": order.astype(int).tolist(),
                   "valid_flag_count": int(valid_flag.sum()), "finite_valid_detection_count": int(len(selected)),
                   "nonfinite_valid_row_count": int(np.count_nonzero(valid_flag & ~finite_xy))}
    return arrays, trajectory, diagnostics


def infer_turning_point(trajectory: NativeTrajectory, episode_start_ns: int, half_window: int = 2,
                        min_displacement_px: float = 5.0, min_reversal_angle_deg: float = 120.0) -> dict[str, Any]:
    candidates = []
    pts = trajectory.smoothed_points
    for i in range(half_window, len(pts) - half_window):
        left, right = i - half_window, i + half_window
        if trajectory.segment_ids[left] != trajectory.segment_ids[right]:
            continue
        incoming, outgoing = pts[i] - pts[left], pts[right] - pts[i]
        in_mag, out_mag = float(np.linalg.norm(incoming)), float(np.linalg.norm(outgoing))
        if in_mag < min_displacement_px or out_mag < min_displacement_px:
            continue
        cosine = float(np.clip(np.dot(incoming, outgoing) / (in_mag * out_mag), -1, 1))
        angle = float(np.degrees(np.arccos(cosine)))
        if angle < min_reversal_angle_deg:
            continue
        candidates.append({"trajectory_index": i,
                           "native_update_index": int(trajectory.native_update_indices[i]),
                           "valid_detection_index": int(trajectory.valid_detection_indices[i]),
                           "available_ros_t_ns": int(trajectory.timestamps_ns[i]),
                           "elapsed_time_sec": float((trajectory.timestamps_ns[i] - episode_start_ns) / 1e9),
                           "u": float(trajectory.points[i, 0]), "v": float(trajectory.points[i, 1]),
                           "incoming_displacement_px": in_mag, "outgoing_displacement_px": out_mag,
                           "reversal_angle_deg": angle,
                           "score": float((1-cosine) * min(in_mag, out_mag))})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    if not candidates:
        return {"status": "no_turn_detected", "selected": None, "candidates": [], "ambiguous_candidates": []}
    similar = [item for item in candidates if item["score"] >= candidates[0]["score"] * AMBIGUOUS_SCORE_FRACTION]
    return {"status": "ambiguous" if len(similar) > 1 else "detected", "selected": candidates[0],
            "candidates": candidates, "ambiguous_candidates": similar}


def _path_length(trajectory: NativeTrajectory) -> float:
    if len(trajectory.points) < 2:
        return 0.0
    distances = np.linalg.norm(np.diff(trajectory.points, axis=0), axis=1)
    return float(distances[trajectory.segment_ids[1:] == trajectory.segment_ids[:-1]].sum())


def _metric(episode_id: str, arrays: Mapping[str, np.ndarray], trajectory: NativeTrajectory,
            diagnostics: Mapping[str, Any], turn: Mapping[str, Any], min_detections: int) -> dict[str, Any]:
    n = len(arrays["available_ros_t_ns"]); valid_flags = np.asarray(arrays["valid"]).astype(bool)
    update_ms = np.diff(arrays["available_ros_t_ns"].astype(np.int64)).astype(float) / 1e6
    valid_ms = np.diff(trajectory.timestamps_ns).astype(float) / 1e6
    window_ms = (arrays["sensor_window_end_us"].astype(float) - arrays["sensor_window_start_us"].astype(float)) / 1000
    usable = len(trajectory.points) >= min_detections
    selected = turn["selected"] if usable else None
    status = turn["status"] if usable else "not_evaluated"
    row = {"episode_id": episode_id, "total_native_tracker_rows": n,
           "valid_native_row_count": int(valid_flags.sum()), "invalid_native_row_count": int(n-valid_flags.sum()),
           "valid_detection_count": int(len(trajectory.points)), "nonfinite_valid_row_count": diagnostics["nonfinite_valid_row_count"],
           "first_native_availability_t_ns": int(arrays["available_ros_t_ns"][0]) if n else None,
           "last_native_availability_t_ns": int(arrays["available_ros_t_ns"][-1]) if n else None,
           "availability_timestamp_monotonic": diagnostics["timestamps_were_monotonic"],
           "availability_timestamps_stably_sorted": diagnostics["timestamps_stably_sorted"],
           "sensor_window_start_monotonic": bool(n < 2 or np.all(np.diff(arrays["sensor_window_start_us"].astype(float)) >= 0)),
           "sensor_window_end_monotonic": bool(n < 2 or np.all(np.diff(arrays["sensor_window_end_us"].astype(float)) >= 0)),
           "native_update_interval_mean_ms": stats(update_ms)["mean"], "native_update_interval_median_ms": stats(update_ms)["median"],
           "native_update_interval_p95_ms": stats(update_ms)["p95"], "native_update_interval_max_ms": stats(update_ms)["max"],
           "valid_detection_interval_mean_ms": stats(valid_ms)["mean"], "valid_detection_interval_median_ms": stats(valid_ms)["median"],
           "valid_detection_interval_p95_ms": stats(valid_ms)["p95"], "longest_valid_detection_gap_ms": stats(valid_ms)["max"],
           "sensor_window_duration_mean_ms": stats(window_ms)["mean"], "sensor_window_duration_median_ms": stats(window_ms)["median"],
           "sensor_window_duration_p95_ms": stats(window_ms)["p95"], "sensor_window_duration_max_ms": stats(window_ms)["max"],
           "window_event_count_mean": stats(arrays.get("window_event_count", []))["mean"],
           "window_event_count_median": stats(arrays.get("window_event_count", []))["median"],
           "window_event_count_p95": stats(arrays.get("window_event_count", []))["p95"],
           "window_event_count_max": stats(arrays.get("window_event_count", []))["max"],
           "trajectory_path_length_px": _path_length(trajectory), "turning_point_status": status,
           "turning_point_u": selected["u"] if selected else None, "turning_point_v": selected["v"] if selected else None,
           "turning_point_native_update_index": selected["native_update_index"] if selected else None,
           "turning_point_valid_detection_index": selected["valid_detection_index"] if selected else None,
           "turning_point_available_ros_t_ns": selected["available_ros_t_ns"] if selected else None,
           "turning_point_elapsed_time_sec": selected["elapsed_time_sec"] if selected else None,
           "incoming_displacement_px": selected["incoming_displacement_px"] if selected else None,
           "outgoing_displacement_px": selected["outgoing_displacement_px"] if selected else None,
           "reversal_angle_deg": selected["reversal_angle_deg"] if selected else None,
           "candidate_count": len(turn["candidates"]) if usable else 0,
           "similarly_strong_candidate_count": len(turn["ambiguous_candidates"]) if usable else 0,
           "ambiguity_status": "ambiguous" if status == "ambiguous" else "unambiguous" if status == "detected" else "none"}
    row["_native_update_intervals_ms"] = update_ms.tolist(); row["_valid_detection_gaps_ms"] = valid_ms.tolist()
    row["_sensor_window_durations_ms"] = window_ms.tolist()
    return row


def _find_dataset(h5: h5py.File, candidates: Sequence[str]) -> np.ndarray | None:
    for name in candidates:
        if name in h5:
            return np.asarray(h5[name])
    return None


def load_policy_comparison(directory: Path, episode_id: str, native_turn: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    match = re.search(r"(\d+)$", episode_id); suffix = match.group(1) if match else episode_id
    paths = [directory / f"episode_{suffix}.hdf5", directory / f"episode_{suffix}.h5"]
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        return None
    result: dict[str, Any] = {"episode_id": episode_id, "policy_episode_file": str(path),
                              "coordinate_warning": "RGB and event pixels are uncalibrated and are never subtracted."}
    with h5py.File(path, "r") as h5:
        policy_t = _find_dataset(h5, ("observations/timestamps_ns", "observations/timestamps"))
        if policy_t is not None:
            policy_t = np.asarray(policy_t).reshape(-1).astype(float)
            if np.nanmax(np.abs(policy_t)) < 1e12: policy_t *= 1e9
            selected = native_turn.get("selected")
            if selected and len(policy_t):
                nearest = int(np.nanargmin(np.abs(policy_t - selected["available_ros_t_ns"])))
                result["native_turn_nearest_policy_grid_frame_index"] = nearest
                result["native_turn_nearest_policy_grid_delta_ms"] = float((policy_t[nearest] - selected["available_ros_t_ns"]) / 1e6)
        rgb_xy = _find_dataset(h5, ("observations/sparse_tracking/rgb_2d_px",))
        rgb_valid = _find_dataset(h5, ("observations/sparse_tracking/rgb_valid",))
        rgb_source = _find_dataset(h5, ("observations/sparse_tracking/rgb_source_timestamps",))
        if rgb_xy is not None and rgb_valid is not None and rgb_source is not None:
            xy=np.asarray(rgb_xy,float); valid=np.asarray(rgb_valid).reshape(-1).astype(bool) & np.isfinite(xy).all(axis=1)
            inds=np.flatnonzero(valid); source=np.asarray(rgb_source,float).reshape(-1)[inds]
            order=np.argsort(source,kind="stable"); inds=inds[order]; source=source[order]
            if len(inds):
                keep=np.r_[True,np.diff(source)!=0]; inds=inds[keep]; source=source[keep]
            gaps=np.diff(source)*1000; seg=np.r_[0,np.cumsum(gaps>args.max_gap_ms)].astype(int) if len(inds) else np.empty(0,int)
            tr=NativeTrajectory(inds,np.arange(len(inds)),np.asarray(source*1e9,dtype=np.int64),xy[inds],robust_median_smooth(xy[inds],args.smoothing_window),seg)
            rgb_turn=infer_turning_point(tr,int(policy_t[0]) if policy_t is not None and len(policy_t) else int(tr.timestamps_ns[0]) if len(tr.timestamps_ns) else 0,args.turn_half_window,args.min_displacement_px,args.min_reversal_angle_deg) if len(tr.points)>=args.min_detections else {"status":"not_evaluated","selected":None,"candidates":[],"ambiguous_candidates":[]}
            result["rgb_turning_point_status"]=rgb_turn["status"]
            if native_turn.get("selected") and rgb_turn.get("selected"):
                result["native_event_minus_rgb_turn_time_ms"] = float((native_turn["selected"]["available_ros_t_ns"]-rgb_turn["selected"]["available_ros_t_ns"])/1e6)
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fallback: Sequence[str]) -> None:
    fields = [k for k in rows[0] if not k.startswith("_")] if rows else list(fallback)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore"); writer.writeheader()
        for row in rows: writer.writerow({k:_number(row.get(k)) for k in fields})


def _safe(value: str) -> str:
    value = re.sub(r"^episode_", "", str(value))
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _hist(path: Path, values: Sequence[float], xlabel: str, title: str) -> None:
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(); vals=np.asarray(values,float); vals=vals[np.isfinite(vals)]
    if len(vals): ax.hist(vals,bins=min(40,max(5,int(np.sqrt(len(vals))))),color="#3572A5",alpha=.8)
    else: ax.text(.5,.5,"No data",ha="center",va="center",transform=ax.transAxes)
    ax.set(xlabel=xlabel,ylabel="count",title=title); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)


def _set_native_event_axes(ax: Any, title: str) -> None:
    """Use native image coordinates and make the upper-left origin explicit."""
    ax.set(xlim=(0, 320), ylim=(320, 0), xlabel="u / x (native event px)",
           ylabel="v / y (native event px; downward)", title=title)
    ax.set_aspect("equal", adjustable="box")
    ax.scatter([0], [0], marker="o", s=28, color="black", zorder=8,
               clip_on=False)
    ax.annotate("(0, 0)", xy=(0, 0), xytext=(7, -7),
                textcoords="offset points", ha="left", va="bottom",
                annotation_clip=False)


def write_plots(plot_dir: Path, entries: Sequence[Mapping[str, Any]], metrics: Sequence[Mapping[str, Any]], rejection_counts: Counter) -> list[str]:
    import matplotlib.pyplot as plt
    plot_dir.mkdir(parents=True,exist_ok=True); paths=[]
    for entry in entries:
        eid,arrays,tr,metric,turn=entry["episode_id"],entry["arrays"],entry["trajectory"],entry["metric"],entry["turn"]
        fig,ax=plt.subplots(figsize=(8,8));
        for seg in np.unique(tr.segment_ids):
            pts=tr.points[tr.segment_ids==seg]
            if len(pts)>1: ax.plot(pts[:,0],pts[:,1],color=".55",lw=1)
        if len(tr.points):
            elapsed=(tr.timestamps_ns-tr.timestamps_ns[0])/1e9; sc=ax.scatter(tr.points[:,0],tr.points[:,1],c=elapsed,cmap="viridis",s=25,zorder=3); fig.colorbar(sc,ax=ax,label="native elapsed time (s)")
        selected=turn.get("selected",{}).get("trajectory_index") if turn.get("selected") else None
        ambiguous={x["trajectory_index"] for x in turn.get("ambiguous_candidates",[])}
        for candidate in turn.get("candidates",[]):
            idx=candidate["trajectory_index"]
            marker,color,size=("*","red",190) if idx==selected else (("D","purple",65) if idx in ambiguous else ("x","orange",65))
            ax.scatter(candidate["u"],candidate["v"],marker=marker,color=color,s=size,zorder=5)
        _set_native_event_axes(ax, f"{eid} native event trajectory\nvalid={metric['valid_detection_count']}, invalid={metric['invalid_native_row_count']}; update median={metric['native_update_interval_median_ms']} ms; gap max={metric['longest_valid_detection_gap_ms']} ms")
        ax.grid(alpha=.2); target=plot_dir/f"episode_{_safe(eid)}_event_trajectory.png"; fig.tight_layout(); fig.savefig(target,dpi=150); plt.close(fig); paths.append(str(target))
        elapsed=(arrays["available_ros_t_ns"].astype(np.int64)-int(arrays["available_ros_t_ns"][0]))/1e9 if len(arrays["available_ros_t_ns"]) else np.array([])
        fig,ax=plt.subplots(figsize=(10,3)); flag=arrays["valid"].astype(bool); ax.scatter(elapsed[~flag],np.zeros(np.count_nonzero(~flag)),marker="x",color="red",label="invalid native row"); ax.scatter(elapsed[flag],np.ones(np.count_nonzero(flag)),color="green",s=18,label="valid native row"); ax.set(yticks=[0,1],yticklabels=["invalid","valid"],xlabel="elapsed native availability time (s)",title=f"{eid}: recorded tracker updates (not reconstructed 1 ms bins)"); ax.legend(); target=plot_dir/f"episode_{_safe(eid)}_event_validity_timeline.png"; fig.tight_layout(); fig.savefig(target,dpi=150); plt.close(fig); paths.append(str(target))
        fig,ax=plt.subplots(figsize=(9,3)); vals=metric["_native_update_intervals_ms"]; ax.plot(np.arange(1,len(vals)+1),vals,marker="."); ax.axhline(150,color="red",ls="--",alpha=.5,label="150 ms reference"); ax.set(xlabel="native update interval ordinal",ylabel="interval (ms)",title=f"{eid}: native tracker update intervals"); ax.legend(); target=plot_dir/f"episode_{_safe(eid)}_event_update_intervals.png"; fig.tight_layout(); fig.savefig(target,dpi=150); plt.close(fig); paths.append(str(target))
    fig,ax=plt.subplots(figsize=(8,8))
    for e in entries:
        tr=e["trajectory"]
        for seg in np.unique(tr.segment_ids):
            p=tr.points[tr.segment_ids==seg]
            if len(p)>1: ax.plot(p[:,0],p[:,1],alpha=.25)
    _set_native_event_axes(ax, "All native event trajectories (large gaps disconnected)"); target=plot_dir/"all_event_trajectories.png"; fig.tight_layout(); fig.savefig(target,dpi=150); plt.close(fig); paths.append(str(target))
    fig,ax=plt.subplots(figsize=(7,7))
    for e in entries:
        tr,sel=e["trajectory"],e["turn"].get("selected")
        if sel:
            p=tr.points-tr.points[sel["trajectory_index"]]
            for seg in np.unique(tr.segment_ids):
                q=p[tr.segment_ids==seg]
                if len(q)>1: ax.plot(q[:,0],q[:,1],alpha=.3)
    ax.axhline(0,color=".5"); ax.axvline(0,color=".5"); ax.invert_yaxis(); ax.set_aspect("equal"); ax.set(title="Turn-aligned native event trajectories (relative v increases downward)",xlabel="relative u (px)",ylabel="relative v (px; downward)"); target=plot_dir/"all_event_trajectories_turn_aligned.png"; fig.tight_layout(); fig.savefig(target,dpi=150); plt.close(fig); paths.append(str(target))
    hist_specs=(("turning_point_time_distribution.png",[r["turning_point_elapsed_time_sec"] for r in metrics if r["turning_point_elapsed_time_sec"] is not None],"turn elapsed time (s)","Turning-point time distribution"),("valid_detection_count_distribution.png",[r["valid_detection_count"] for r in metrics],"valid detections / episode","Valid detection count distribution"),("native_update_interval_distribution.png",[x for r in metrics for x in r["_native_update_intervals_ms"]],"native update interval (ms)","Native update interval distribution"),("valid_detection_gap_distribution.png",[x for r in metrics for x in r["_valid_detection_gaps_ms"]],"valid detection gap (ms)","Valid detection gap distribution"),("sensor_window_duration_distribution.png",[x for r in metrics for x in r["_sensor_window_durations_ms"]],"sensor-window duration (ms)","Sensor-window duration distribution"))
    for name,vals,xlabel,title in hist_specs: target=plot_dir/name; _hist(target,vals,xlabel,title); paths.append(str(target))
    fig,ax=plt.subplots(figsize=(9,4)); labels=list(rejection_counts); values=[rejection_counts[x] for x in labels]
    if labels: ax.bar(labels,values); ax.tick_params(axis="x",rotation=45)
    else: ax.text(.5,.5,"No rejection reasons",ha="center",va="center",transform=ax.transAxes)
    ax.set(ylabel="native rows",title="Sidecar rejection reason counts"); target=plot_dir/"rejection_reason_counts.png"; fig.tight_layout(); fig.savefig(target,dpi=150); plt.close(fig); paths.append(str(target))
    return paths


def _parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="Analyze a native OpenMV tracker-output sidecar HDF5 (not *_raw_events.h5 packet data and not policy-grid episodes).")
    p.add_argument("sidecar",type=Path,help="Tracker-output sidecar containing /episodes/episode_N native tracker rows")
    p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--plot",action="store_true")
    p.add_argument("--episodes-dir",type=Path,help="Optional enriched episode_N.hdf5 directory for supplementary RGB/policy-grid comparison only")
    p.add_argument("--smoothing-window",type=int,default=3); p.add_argument("--turn-half-window",type=int,default=2)
    p.add_argument("--min-displacement-px",type=float,default=5); p.add_argument("--min-reversal-angle-deg",type=float,default=120)
    p.add_argument("--min-detections",type=int,default=4); p.add_argument("--max-gap-ms",type=float,default=150)
    p.add_argument("--max-episodes",type=int,help="Analyze at most this many sidecar episode groups")
    return p


def _validate(args: argparse.Namespace) -> None:
    if args.smoothing_window<1 or args.smoothing_window%2==0: raise ValueError("smoothing-window must be a positive odd integer")
    if args.turn_half_window<1 or args.min_detections<1: raise ValueError("turn-half-window and min-detections must be positive")
    if args.min_displacement_px<0 or args.max_gap_ms<=0: raise ValueError("min-displacement-px must be nonnegative and max-gap-ms positive")
    if not 0<=args.min_reversal_angle_deg<=180: raise ValueError("min-reversal-angle-deg must be in [0, 180]")
    if args.max_episodes is not None and args.max_episodes<1: raise ValueError("max-episodes must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    args=_parser().parse_args(argv)
    try: _validate(args)
    except ValueError as exc: print(f"error: {exc}",file=sys.stderr); return 2
    metrics=[]; entries=[]; skipped=[]; rejection=Counter(); interval_rows=[]; window_rows=[]; comparisons=[]
    try:
        with h5py.File(args.sidecar,"r") as h5:
            names=validate_sidecar(h5,args.sidecar)
            if args.max_episodes is not None: names=names[:args.max_episodes]
            for name in names:
                try:
                    arrays,tr,diag=load_episode(h5["episodes"][name],args.smoothing_window,args.max_gap_ms)
                    turn=infer_turning_point(tr,int(arrays["available_ros_t_ns"][0]) if len(arrays["available_ros_t_ns"]) else 0,args.turn_half_window,args.min_displacement_px,args.min_reversal_angle_deg) if len(tr.points)>=args.min_detections else {"status":"not_evaluated","selected":None,"candidates":[],"ambiguous_candidates":[]}
                    metric=_metric(name,arrays,tr,diag,turn,args.min_detections); metrics.append(metric); entries.append({"episode_id":name,"arrays":arrays,"trajectory":tr,"metric":metric,"turn":turn})
                    reasons=_decode_reasons(arrays["rejection_reason"])
                    for reason in reasons: rejection[reason]+=1
                    for i,value in enumerate(metric["_native_update_intervals_ms"],1): interval_rows.append({"episode_id":name,"interval_ordinal":i,"previous_native_update_index":int(arrays["native_update_index"][i-1]),"native_update_index":int(arrays["native_update_index"][i]),"interval_ms":value,"is_large_gap":value>args.max_gap_ms})
                    for i,value in enumerate(metric["_sensor_window_durations_ms"]): window_rows.append({"episode_id":name,"native_update_index":int(arrays["native_update_index"][i]),"sensor_window_start_us":arrays["sensor_window_start_us"][i],"sensor_window_end_us":arrays["sensor_window_end_us"][i],"duration_ms":value,"window_event_count":arrays["window_event_count"][i] if "window_event_count" in arrays else None})
                    if args.episodes_dir:
                        paired=load_policy_comparison(args.episodes_dir,name,turn,args)
                        if paired: comparisons.append(paired)
                        else: skipped.append({"episode_id":name,"reason":"no matching episode_N.hdf5 in --episodes-dir (native analysis completed)"})
                except (ValueError,TypeError,KeyError) as exc:
                    skipped.append({"episode_id":name,"reason":str(exc)}); print(f"warning: {name}: {exc}",file=sys.stderr)
    except (OSError,ValueError) as exc:
        print(f"error: {args.sidecar}: {exc}",file=sys.stderr); return 2
    output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=True)
    rejection_rows=[{"rejection_reason":reason,"count":count} for reason,count in sorted(rejection.items())]
    _write_csv(output/"episode_metrics.csv",metrics,("episode_id",)); _write_csv(output/"sidecar_rejection_statistics.csv",rejection_rows,("rejection_reason","count")); _write_csv(output/"sidecar_update_intervals.csv",interval_rows,("episode_id","interval_ordinal","interval_ms")); _write_csv(output/"sidecar_sensor_window_statistics.csv",window_rows,("episode_id","native_update_index","duration_ms"))
    plots=write_plots(output/"plots",entries,metrics,rejection) if args.plot else []
    total_rows=sum(r["total_native_tracker_rows"] for r in metrics); valid=sum(r["valid_native_row_count"] for r in metrics); invalid=sum(r["invalid_native_row_count"] for r in metrics)
    summary={"input_file":str(args.sidecar.resolve()),"input_semantics":"native tracker-output sidecar; available_ros_t_ns and native rows are authoritative","coordinate_system":"320x320 native event pixels, top-left origin, u right, v down; no rotation or homography","episodes_processed":len(metrics),"total_native_tracker_rows":total_rows,"valid_native_rows":valid,"invalid_native_rows":invalid,"turning_points_detected":sum(r["turning_point_status"] in ("detected","ambiguous") for r in metrics),"ambiguous_turning_points":sum(r["turning_point_status"]=="ambiguous" for r in metrics),"native_update_interval_ms":stats([x for r in metrics for x in r["_native_update_intervals_ms"]]),"valid_detection_gap_ms":stats([x for r in metrics for x in r["_valid_detection_gaps_ms"]]),"sensor_window_duration_ms":stats([x for r in metrics for x in r["_sensor_window_durations_ms"]]),"rejection_reason_counts":dict(rejection),"configuration":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},"skipped":skipped,"candidate_details":[{"episode_id":e["episode_id"],"status":e["turn"]["status"],"candidates":_number(e["turn"]["candidates"])} for e in entries],"policy_grid_comparisons":comparisons,"plot_files":plots}
    (output/"summary.json").write_text(json.dumps(_number(summary),indent=2)+"\n",encoding="utf-8")
    print(f"Episodes processed: {len(metrics)}"); print(f"Total native tracker rows: {total_rows}"); print(f"Valid/invalid native rows: {valid}/{invalid}"); print(f"Turning points detected: {summary['turning_points_detected']}"); print(f"Native update interval ms: {summary['native_update_interval_ms']}"); print(f"Valid-detection gap ms: {summary['valid_detection_gap_ms']}"); print(f"Output directory: {output}"); print(f"Skipped episodes/comparisons: {len(skipped)}")
    return 0 if metrics else 1


if __name__ == "__main__":
    raise SystemExit(main())
