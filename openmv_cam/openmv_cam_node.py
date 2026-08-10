#!/usr/bin/env python3

import json
import struct
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import h5py
import numpy as np
import serial

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3Stamped
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from rclpy.qos import QoSProfile, ReliabilityPolicy

from .event_frame_contract import (
    build_event_activity_voxel,
    build_event_frame_3ch,
    build_xyt_signed_voxel,
    has_new_event_data,
    render_event_frame_from_arrays,
    retention_history_ms,
    validate_event_contract_config,
    validate_activity_voxel_config,
    validate_xyt_voxel_config,
)
from .event_diagnostics import EventDiagnosticCounters
from .event_output_mode import resolve_event_outputs
from .image_rotation import SUPPORTED_ROTATIONS_DEG, rotate_event_frame
from .event_ball_tracker import (
    EventBallTracker,
    render_debug_images,
    trace_detail_json,
)
from .evt1_protocol import EventPacket, MAX_EVENT_COUNT, reconstruct_timestamps_us
from .hdf5_replay import HDF5ReplayReader, ReplayDiagnostics, replay_packets


def build_hwc9_image_message(
    image: np.ndarray,
    *,
    encoding: str,
    stamp,
    frame_id: str,
) -> Image:
    """Serialize a contiguous uint8 HWC9 array without cv_bridge assumptions."""
    if image.dtype != np.uint8:
        raise ValueError(f"XYT voxel must have dtype uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 9:
        raise ValueError(f"XYT voxel must have shape (H, W, 9), got {image.shape}")
    if not image.flags.c_contiguous:
        raise ValueError("XYT voxel must be C-contiguous")
    if encoding != "8UC9":
        raise ValueError(f"XYT voxel encoding must be '8UC9', got {encoding!r}")

    height, width, channels = image.shape
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(height)
    msg.width = int(width)
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = int(width * channels)
    msg.data = image.tobytes(order="C")
    return msg


def build_activity_image_message(
    image: np.ndarray,
    *,
    stamp,
    frame_id: str = "openmv_cam",
) -> Image:
    """Serialize an HW/HWC uint8 activity voxel with an explicit ROS layout."""
    if image.dtype != np.uint8:
        raise ValueError(f"activity voxel must have dtype uint8, got {image.dtype}")
    if image.ndim not in (2, 3):
        raise ValueError(f"activity voxel must have shape (H, W) or (H, W, N), got {image.shape}")
    if image.ndim == 3 and image.shape[2] < 2:
        raise ValueError(f"HWC activity voxel must have at least 2 channels, got {image.shape}")
    if not image.flags.c_contiguous:
        raise ValueError("activity voxel must be C-contiguous")

    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(height)
    msg.width = int(width)
    msg.encoding = f"8UC{channels}"
    msg.is_bigendian = 0
    msg.step = int(width * channels)
    msg.data = image.tobytes(order="C")
    return msg


class OpenMVEventCamNode(Node):
    MAGIC = b"EVT1"
    HEADER_FMT = "<LL"   # event_count, payload_len
    HEADER_SIZE = 4 + struct.calcsize(HEADER_FMT)

    print_log = False

    # Cam img dims
    W = 320
    H = 320

    def __init__(self):
        super().__init__("openmv_event_cam")

        # Serial / ROS params
        self.declare_parameter("port", "/dev/openmvcam")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("timeout", 3.0)
        self.declare_parameter("topic", "/openmv_cam/image")
        self.declare_parameter("publish_3_channel_img", True)
        self.declare_parameter("topic_3_channel", "/openmv_cam/event_frame_3ch")
        self.declare_parameter("event_frame_ch0_ms", 50.0)
        self.declare_parameter("event_frame_ch1_ms", 250.0)
        self.declare_parameter("event_frame_ch2_ms", 1000.0)
        self.declare_parameter("event_frame_mode", "cumulative")
        self.declare_parameter("event_scaling", "signed_log1p_fixed_clip")
        self.declare_parameter("event_clip_count", 16.0)
        self.declare_parameter("event_packet_margin_ms", 50.0)
        self.declare_parameter("event_frame_encoding", "8UC3")
        self.declare_parameter("publish_xyt_voxel", False)
        self.declare_parameter("topic_xyt_voxel", "/openmv_cam/event_voxel")
        self.declare_parameter("event_voxel_horizon_ms", 200.0)
        self.declare_parameter("event_voxel_temporal_bins", 9)
        self.declare_parameter("event_voxel_height", 320)
        self.declare_parameter("event_voxel_width", 320)
        self.declare_parameter(
            "event_voxel_scaling", "signed_log1p_fixed_clip"
        )
        self.declare_parameter("event_voxel_clip_count", 16.0)
        self.declare_parameter("event_voxel_encoding", "8UC9")
        self.declare_parameter("publish_event_voxel_1ms", False)
        self.declare_parameter("topic_event_voxel_1ms", "/openmv_cam/event_voxel_1ms")
        self.declare_parameter("event_voxel_bin_ms", 1.0)
        self.declare_parameter("event_voxel_activity_mode", "absolute_activity")
        self.declare_parameter("event_voxel_publish_fps", 30.0)
        self.declare_parameter("event_output_mode", "legacy_flags")
        self.declare_parameter("event_diagnostics_enabled", False)
        self.declare_parameter("event_diagnostics_period_sec", 5.0)
        self.declare_parameter("frame_id", "openmv_cam")
        self.declare_parameter("publish_fps", 30.0)
        self.declare_parameter("event_frame_rotation_degrees", 0)
        self.declare_parameter("event_input_mode", "hardware")
        self.declare_parameter("event_replay_path", "")
        self.declare_parameter("event_replay_timing", "recorded")
        self.declare_parameter("event_replay_rate", 1.0)
        self.declare_parameter("event_replay_start_packet", 0)
        self.declare_parameter("event_replay_end_packet", -1)
        self.declare_parameter("event_replay_loop", False)

        # Raw-packet event ball tracker (native, unrotated OpenMV coordinates).
        self.declare_parameter("event_tracker_enabled", False)
        self.declare_parameter(
            "event_tracker_position_topic",
            "/openmv_cam/event_tracker/ball_2d_px")
        self.declare_parameter(
            "event_tracker_velocity_topic",
            "/openmv_cam/event_tracker/ball_velocity_px_s")
        self.declare_parameter(
            "event_tracker_valid_topic", "/openmv_cam/event_tracker/valid")
        self.declare_parameter("event_tracker_bin_ms", 1.0)
        self.declare_parameter("event_tracker_accumulation_window_ms", 10.0)
        self.declare_parameter("event_tracker_history_limit_ms", 100.0)
        self.declare_parameter("event_tracker_activity_threshold", 1)
        self.declare_parameter("event_tracker_spatial_filter_enabled", False)
        self.declare_parameter(
            "event_tracker_spatial_filter_min_neighbors", 1)
        self.declare_parameter(
            "event_tracker_spatial_filter_min_component_area_px", 1)
        self.declare_parameter("event_tracker_min_event_count", 3)
        self.declare_parameter("event_tracker_min_blob_area_px", 2)
        self.declare_parameter("event_tracker_max_blob_area_px", 2000)
        self.declare_parameter("event_tracker_min_blob_width_px", 1)
        self.declare_parameter("event_tracker_max_blob_width_px", 320)
        self.declare_parameter("event_tracker_min_blob_height_px", 1)
        self.declare_parameter("event_tracker_max_blob_height_px", 320)
        self.declare_parameter("event_tracker_morphology_operation", "close")
        self.declare_parameter("event_tracker_morphology_kernel", 3)
        self.declare_parameter("event_tracker_morphology_iterations", 1)
        self.declare_parameter("event_tracker_use_circularity", False)
        self.declare_parameter("event_tracker_min_circularity", 0.1)
        self.declare_parameter("event_tracker_max_jump_px", 100.0)
        self.declare_parameter("event_tracker_reacquire_after_misses", 3)
        self.declare_parameter("event_tracker_x_crop", [80, 215])
        self.declare_parameter("event_tracker_y_crop", [35, 275, 85, 235])
        self.declare_parameter("event_tracker_velocity_history_size", 5)
        self.declare_parameter("event_tracker_velocity_min_span_ms", 3.0)
        self.declare_parameter("event_tracker_stats_period_sec", 5.0)
        self.declare_parameter("event_tracker_debug_enabled", False)
        self.declare_parameter(
            "event_tracker_debug_topic",
            "/openmv_cam/event_tracker/debug_image")
        self.declare_parameter("event_tracker_debug_fps", 10.0)
        self.declare_parameter("event_tracker_debug_clip_count", 16)
        self.declare_parameter("event_tracker_debug_rotation_degrees", 90)
        self.declare_parameter(
            "event_tracker_debug_activity_topic",
            "/openmv_cam/event_tracker/debug/activity")
        self.declare_parameter(
            "event_tracker_debug_threshold_topic",
            "/openmv_cam/event_tracker/debug/threshold")
        self.declare_parameter(
            "event_tracker_debug_contours_topic",
            "/openmv_cam/event_tracker/debug/contours")
        self.declare_parameter(
            "event_tracker_debug_tracking_topic",
            "/openmv_cam/event_tracker/debug/tracking")
        self.declare_parameter("publish_latency_traces", False)
        self.declare_parameter(
            "latency_trace_topic",
            "/intercept_trace/event_2d_ball_detection")
        self.declare_parameter("latency_trace_run_id", "")

        # Preview/render params
        self.declare_parameter("window_ms", 100.0)
        self.declare_parameter("max_preview_packets", 10)
        self.declare_parameter("max_event_frame_packets", 200)
        self.declare_parameter("contrast", 4.0)
        self.declare_parameter("step", 1.0)
        self.declare_parameter("blur_kernel", 0)
        self.declare_parameter("sort_by_timestamp", False)

        # Raw event recording params
        self.declare_parameter("raw_event_output_path", "")
        self.declare_parameter("flush_every_packets", 50)
        self.declare_parameter("hdf5_chunk_size", 100000)
        self.declare_parameter("hdf5_compression", "gzip")
        self.declare_parameter("hdf5_compression_level", 4)

        self.port_name = self.get_parameter("port").get_parameter_value().string_value
        self.baud = self.get_parameter("baud").get_parameter_value().integer_value
        self.timeout = self.get_parameter("timeout").get_parameter_value().double_value
        self.topic = self.get_parameter("topic").get_parameter_value().string_value.strip()
        self.publish_3_channel_img = self.get_parameter("publish_3_channel_img").get_parameter_value().bool_value
        self.topic_3_channel = self.get_parameter("topic_3_channel").get_parameter_value().string_value.strip()
        self.event_frame_windows_ms = [
            self.get_parameter("event_frame_ch0_ms").get_parameter_value().double_value,
            self.get_parameter("event_frame_ch1_ms").get_parameter_value().double_value,
            self.get_parameter("event_frame_ch2_ms").get_parameter_value().double_value,
        ]
        self.event_frame_mode = self.get_parameter("event_frame_mode").get_parameter_value().string_value.strip().lower()
        self.event_scaling = self.get_parameter("event_scaling").get_parameter_value().string_value.strip().lower()
        self.event_clip_count = self.get_parameter("event_clip_count").get_parameter_value().double_value
        self.event_packet_margin_ms = self.get_parameter("event_packet_margin_ms").get_parameter_value().double_value
        self.event_frame_encoding = self.get_parameter("event_frame_encoding").get_parameter_value().string_value.strip()
        self.publish_xyt_voxel = self.get_parameter("publish_xyt_voxel").value
        self.topic_xyt_voxel = str(self.get_parameter("topic_xyt_voxel").value).strip()
        self.event_voxel_horizon_ms = float(
            self.get_parameter("event_voxel_horizon_ms").value
        )
        self.event_voxel_temporal_bins = int(
            self.get_parameter("event_voxel_temporal_bins").value
        )
        self.legacy_event_voxel_temporal_bins = 9
        self.event_voxel_height = int(self.get_parameter("event_voxel_height").value)
        self.event_voxel_width = int(self.get_parameter("event_voxel_width").value)
        self.event_voxel_scaling = str(
            self.get_parameter("event_voxel_scaling").value
        ).strip().lower()
        self.event_voxel_clip_count = float(
            self.get_parameter("event_voxel_clip_count").value
        )
        self.event_voxel_encoding = str(
            self.get_parameter("event_voxel_encoding").value
        ).strip()
        self.publish_event_voxel_1ms = bool(
            self.get_parameter("publish_event_voxel_1ms").value
        )
        self.topic_event_voxel_1ms = str(
            self.get_parameter("topic_event_voxel_1ms").value
        ).strip()
        self.event_voxel_bin_ms = float(self.get_parameter("event_voxel_bin_ms").value)
        self.event_voxel_activity_mode = str(
            self.get_parameter("event_voxel_activity_mode").value
        ).strip().lower()
        self.event_voxel_publish_fps = float(
            self.get_parameter("event_voxel_publish_fps").value
        )
        self.event_output_mode = str(
            self.get_parameter("event_output_mode").value
        ).strip().lower()
        self.event_diagnostics_enabled = bool(
            self.get_parameter("event_diagnostics_enabled").value
        )
        self.event_diagnostics_period_sec = float(
            self.get_parameter("event_diagnostics_period_sec").value
        )
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.publish_fps = self.get_parameter("publish_fps").get_parameter_value().double_value
        self.event_frame_rotation_degrees = (
            self.get_parameter("event_frame_rotation_degrees")
            .get_parameter_value()
            .integer_value
        )
        self.event_tracker_enabled = bool(
            self.get_parameter("event_tracker_enabled").value)
        self.event_tracker_stats_period_sec = float(
            self.get_parameter("event_tracker_stats_period_sec").value)
        self.event_tracker = None
        self.event_tracker_stats_timer = None
        self.event_tracker_position_pub = None
        self.event_tracker_velocity_pub = None
        self.event_tracker_valid_pub = None
        self.event_tracker_debug_pub = None
        self.event_tracker_debug_stage_pubs = {}
        self.event_tracker_debug_timer = None
        self._last_tracker_debug_bin_start_us = None
        self.latency_trace_pub = None
        self.LatencyTrace = None
        self._tracker_trace_sequence = 0
        if self.event_tracker_enabled:
            self.event_tracker = EventBallTracker(
                width=self.W, height=self.H,
                bin_ms=float(self.get_parameter("event_tracker_bin_ms").value),
                accumulation_window_ms=float(self.get_parameter(
                    "event_tracker_accumulation_window_ms").value),
                history_limit_ms=float(
                    self.get_parameter("event_tracker_history_limit_ms").value),
                activity_threshold=int(
                    self.get_parameter("event_tracker_activity_threshold").value),
                spatial_filter_enabled=bool(self.get_parameter(
                    "event_tracker_spatial_filter_enabled").value),
                spatial_filter_min_neighbors=int(self.get_parameter(
                    "event_tracker_spatial_filter_min_neighbors").value),
                spatial_filter_min_component_area_px=int(self.get_parameter(
                    "event_tracker_spatial_filter_min_component_area_px").value),
                min_event_count=int(
                    self.get_parameter("event_tracker_min_event_count").value),
                min_blob_area_px=int(
                    self.get_parameter("event_tracker_min_blob_area_px").value),
                max_blob_area_px=int(
                    self.get_parameter("event_tracker_max_blob_area_px").value),
                min_blob_width_px=int(
                    self.get_parameter("event_tracker_min_blob_width_px").value),
                max_blob_width_px=int(
                    self.get_parameter("event_tracker_max_blob_width_px").value),
                min_blob_height_px=int(self.get_parameter(
                    "event_tracker_min_blob_height_px").value),
                max_blob_height_px=int(self.get_parameter(
                    "event_tracker_max_blob_height_px").value),
                morphology_operation=str(self.get_parameter(
                    "event_tracker_morphology_operation").value),
                morphology_kernel=int(
                    self.get_parameter("event_tracker_morphology_kernel").value),
                morphology_iterations=int(self.get_parameter(
                    "event_tracker_morphology_iterations").value),
                use_circularity=bool(
                    self.get_parameter("event_tracker_use_circularity").value),
                min_circularity=float(
                    self.get_parameter("event_tracker_min_circularity").value),
                max_jump_px=float(
                    self.get_parameter("event_tracker_max_jump_px").value),
                reacquire_after_misses=int(self.get_parameter(
                    "event_tracker_reacquire_after_misses").value),
                x_crop=list(self.get_parameter("event_tracker_x_crop").value),
                y_crop=list(self.get_parameter("event_tracker_y_crop").value),
                velocity_history_size=int(self.get_parameter(
                    "event_tracker_velocity_history_size").value),
                velocity_min_span_ms=float(self.get_parameter(
                    "event_tracker_velocity_min_span_ms").value),
            )
            self.event_tracker_position_pub = self.create_publisher(
                PointStamped,
                str(self.get_parameter("event_tracker_position_topic").value),
                10)
            self.event_tracker_velocity_pub = self.create_publisher(
                Vector3Stamped,
                str(self.get_parameter("event_tracker_velocity_topic").value),
                10)
            self.event_tracker_valid_pub = self.create_publisher(
                Bool, str(self.get_parameter("event_tracker_valid_topic").value),
                10)
            self.event_tracker_stats_timer = self.create_timer(
                self.event_tracker_stats_period_sec, self._event_tracker_stats_cb)
            if bool(self.get_parameter("event_tracker_debug_enabled").value):
                debug_fps = float(
                    self.get_parameter("event_tracker_debug_fps").value)
                if not np.isfinite(debug_fps) or debug_fps <= 0.0:
                    raise ValueError(
                        "event_tracker_debug_fps must be positive and finite")
                self.event_tracker_debug_pub = self.create_publisher(
                    Image,
                    str(self.get_parameter("event_tracker_debug_topic").value),
                    10)
                stage_topic_parameters = {
                    "activity": "event_tracker_debug_activity_topic",
                    "threshold": "event_tracker_debug_threshold_topic",
                    "contours": "event_tracker_debug_contours_topic",
                    "tracking": "event_tracker_debug_tracking_topic",
                }
                self.event_tracker_debug_stage_pubs = {
                    stage: self.create_publisher(
                        Image, str(self.get_parameter(parameter).value), 10)
                    for stage, parameter in stage_topic_parameters.items()}
                self.event_tracker_debug_timer = self.create_timer(
                    1.0 / debug_fps, self._event_tracker_debug_timer_cb)
            if bool(self.get_parameter("publish_latency_traces").value):
                try:
                    from intercept_latency_monitor.msg import LatencyTrace
                    self.LatencyTrace = LatencyTrace
                    qos = QoSProfile(
                        depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
                    self.latency_trace_pub = self.create_publisher(
                        LatencyTrace,
                        str(self.get_parameter("latency_trace_topic").value), qos)
                except ImportError as error:
                    self.get_logger().error(
                        "Latency tracing requested but intercept_latency_monitor is unavailable; "
                        f"tracing disabled: {error}")

        if self.event_frame_rotation_degrees not in SUPPORTED_ROTATIONS_DEG:
            raise ValueError(
                "event_frame_rotation_degrees must be one of "
                f"{SUPPORTED_ROTATIONS_DEG}, got {self.event_frame_rotation_degrees}"
            )

        self.window_ms = self.get_parameter("window_ms").get_parameter_value().double_value
        self.max_preview_packets = self.get_parameter("max_preview_packets").get_parameter_value().integer_value
        self.max_event_frame_packets = self.get_parameter("max_event_frame_packets").get_parameter_value().integer_value
        self.contrast = self.get_parameter("contrast").get_parameter_value().double_value
        self.step = self.get_parameter("step").get_parameter_value().double_value
        self.blur_kernel = self.get_parameter("blur_kernel").get_parameter_value().integer_value
        self.sort_by_timestamp = self.get_parameter("sort_by_timestamp").get_parameter_value().bool_value
        self.event_input_mode = str(self.get_parameter("event_input_mode").value).strip().lower()
        self.event_replay_path = str(self.get_parameter("event_replay_path").value).strip()
        self.event_replay_timing = str(
            self.get_parameter("event_replay_timing").value).strip().lower()
        self.event_replay_rate = float(self.get_parameter("event_replay_rate").value)
        self.event_replay_start_packet = int(self.get_parameter("event_replay_start_packet").value)
        self.event_replay_end_packet = int(self.get_parameter("event_replay_end_packet").value)
        self.event_replay_loop = bool(self.get_parameter("event_replay_loop").value)
        if self.event_input_mode not in ("hardware", "hdf5_replay"):
            raise ValueError("event_input_mode must be 'hardware' or 'hdf5_replay'")
        if self.event_replay_timing not in ("recorded", "sensor", "fast"):
            raise ValueError("event_replay_timing must be 'recorded', 'sensor', or 'fast'")
        if not np.isfinite(self.event_replay_rate) or self.event_replay_rate <= 0.0:
            raise ValueError("event_replay_rate must be positive and finite")

        configured_topics = [
            self.topic,
            self.topic_3_channel,
            self.topic_xyt_voxel,
            self.topic_event_voxel_1ms,
        ]
        if len(set(configured_topics)) != len(configured_topics):
            error_msg = "Mono, 3-channel, and XYT event image topics must be different."
            self.get_logger().error(error_msg)
            raise ValueError(error_msg)

        contract = validate_event_contract_config(
            windows_ms=self.event_frame_windows_ms,
            mode=self.event_frame_mode,
            event_scaling=self.event_scaling,
            event_clip_count=self.event_clip_count,
            packet_margin_ms=self.event_packet_margin_ms,
        )
        self.event_frame_windows_ms = list(contract.windows_ms)
        self.event_frame_mode = contract.mode
        self.event_scaling = contract.scaling
        self.event_clip_count = contract.clip_count
        self.event_packet_margin_ms = contract.packet_margin_ms

        if self.event_frame_encoding not in ("8UC3", "bgr8", "rgb8"):
            raise ValueError(
                "event_frame_encoding must be one of '8UC3', 'bgr8', or 'rgb8', "
                f"got {self.event_frame_encoding!r}"
            )

        (
            _sensor_width,
            _sensor_height,
            self.event_voxel_width,
            self.event_voxel_height,
            self.event_voxel_horizon_ms,
            self.legacy_event_voxel_temporal_bins,
            self.event_voxel_scaling,
            self.event_voxel_clip_count,
        ) = validate_xyt_voxel_config(
            sensor_width=self.W,
            sensor_height=self.H,
            output_width=self.event_voxel_width,
            output_height=self.event_voxel_height,
            horizon_ms=self.event_voxel_horizon_ms,
            temporal_bins=self.legacy_event_voxel_temporal_bins,
            scaling_mode=self.event_voxel_scaling,
            event_clip_count=self.event_voxel_clip_count,
        )
        if self.event_voxel_encoding != "8UC9":
            raise ValueError(
                "xyt_signed_voxel_v1 requires event_voxel_encoding='8UC9', "
                f"got {self.event_voxel_encoding!r}"
            )

        (
            _activity_width,
            _activity_height,
            _activity_bin_us,
            self.event_voxel_temporal_bins,
            self.event_voxel_activity_mode,
            self.event_voxel_clip_count,
        ) = validate_activity_voxel_config(
            width=self.W,
            height=self.H,
            bin_ms=self.event_voxel_bin_ms,
            temporal_bins=self.event_voxel_temporal_bins,
            activity_mode=self.event_voxel_activity_mode,
            clip_count=self.event_voxel_clip_count,
        )
        if not np.isfinite(self.event_voxel_publish_fps) or self.event_voxel_publish_fps <= 0.0:
            raise ValueError("event_voxel_publish_fps must be positive and finite")
        if (
            not np.isfinite(self.event_diagnostics_period_sec)
            or self.event_diagnostics_period_sec <= 0.0
        ):
            raise ValueError("event_diagnostics_period_sec must be positive and finite")

        outputs = resolve_event_outputs(
            self.event_output_mode,
            publish_3_channel_img=self.publish_3_channel_img,
            publish_xyt_voxel=self.publish_xyt_voxel,
            publish_event_voxel_1ms=self.publish_event_voxel_1ms,
        )
        self.publish_mono_img = outputs.mono
        self.publish_3_channel_img = outputs.event_frame_3ch
        self.publish_xyt_voxel = outputs.legacy_voxel
        self.publish_event_voxel_1ms = outputs.event_voxel_1ms

        self.raw_event_output_path = self.get_parameter("raw_event_output_path").get_parameter_value().string_value
        self.flush_every_packets = self.get_parameter("flush_every_packets").get_parameter_value().integer_value
        self.hdf5_chunk_size = self.get_parameter("hdf5_chunk_size").get_parameter_value().integer_value
        self.hdf5_compression = self.get_parameter("hdf5_compression").get_parameter_value().string_value
        self.hdf5_compression_level = self.get_parameter(
            "hdf5_compression_level"
        ).get_parameter_value().integer_value

        self.bridge = CvBridge()
        self.pub_mono_img = None
        self.pub_3ch = None
        self.pub_xyt_voxel = None
        self.pub_event_voxel_1ms = None
        self.publish_timer = None
        self.event_voxel_timer = None
        self.diagnostics_timer = None
        self._publishing_enabled = False
        self._preview_history_truncated_warned = False
        self._publisher_config_logged = False
        self._last_publish_debug_log_t = 0.0
        self._event_data_generation = 0
        self._last_activity_generation = 0
        self._diagnostics = EventDiagnosticCounters()
        self._last_activity_tick_monotonic = None
        self._last_activity_publish_monotonic = None
        self._last_activity_anchor_t_us = None
        self._diagnostic_last_monotonic = time.monotonic()
        self._diagnostic_last_packets = 0
        self._diagnostic_last_events = 0
        self._diagnostic_last_messages = 0

        self._stop_event = threading.Event()
        self._preview_lock = threading.Lock()
        self._h5_lock = threading.Lock()
        self.serial_port = None
        self._replay_reader = None
        self._replay_diagnostics = ReplayDiagnostics()
        if self.event_input_mode == "hardware":
            self._open_serial()
        else:
            if not self.event_replay_path:
                raise ValueError("event_replay_path is required in hdf5_replay mode")
            self._replay_reader = HDF5ReplayReader(
                self.event_replay_path,
                start_packet=self.event_replay_start_packet,
                end_packet=self.event_replay_end_packet)

        # Stores tuples: (host_arrival_time_monotonic, validated EventPacket).
        self.preview_buffer = deque()

        self._recording_enabled = False
        self._h5_file = None
        self._h5_file_path = ""
        self._h5_events_type = None
        self._h5_events_x = None
        self._h5_events_y = None
        self._h5_events_t_us = None
        self._h5_events_packet_id = None
        self._h5_packets_ros_t_ns = None
        self._h5_packets_monotonic_t_ns = None
        self._h5_packets_start_event_idx = None
        self._h5_packets_end_event_idx = None
        self._h5_packets_event_count = None
        self._h5_packets_first_event_t_us = None
        self._h5_packets_last_event_t_us = None
        self._h5_event_count = 0
        self._h5_packet_count = 0
        self._h5_packets_since_flush = 0

        self.total_packets = 0
        self.total_events = 0
        self.total_payload_bytes = 0
        self.total_protocol_bytes = 0
        self.t0 = time.monotonic()
        self.last_stats_print = self.t0

        worker = (self._reader_loop if self.event_input_mode == "hardware"
                  else self._replay_loop)
        self._reader_thread = threading.Thread(target=worker, daemon=True)
        self._reader_thread.start()

        self._start_pub_srv = self.create_service(
            Trigger,
            "/openmv_cam/start_event_frame_publishing",
            self._handle_start_event_frame_publishing,
        )
        self._stop_pub_srv = self.create_service(
            Trigger,
            "/openmv_cam/stop_event_frame_publishing",
            self._handle_stop_event_frame_publishing,
        )
        self._rotate_plus_90_srv = self.create_service(
            Trigger,
            "/openmv_cam/rotate_event_frame_plus_90",
            self._make_event_frame_rotation_handler(90),
        )
        self._rotate_180_srv = self.create_service(
            Trigger,
            "/openmv_cam/rotate_event_frame_180",
            self._make_event_frame_rotation_handler(180),
        )
        self._rotate_minus_90_srv = self.create_service(
            Trigger,
            "/openmv_cam/rotate_event_frame_minus_90",
            self._make_event_frame_rotation_handler(-90),
        )
        self._start_rec_srv = self.create_service(
            Trigger,
            "/openmv_cam/start_raw_event_recording",
            self._handle_start_raw_event_recording,
        )
        self._stop_rec_srv = self.create_service(
            Trigger,
            "/openmv_cam/stop_raw_event_recording",
            self._handle_stop_raw_event_recording,
        )

        self.get_logger().info(
            "Event input configuration: "
            f"mode={self.event_input_mode}, replay_path={self.event_replay_path!r}, "
            f"replay_timing={self.event_replay_timing}, rate={self.event_replay_rate}, "
            f"packet_range=[{self.event_replay_start_packet},"
            f"{self.event_replay_end_packet}], loop={self.event_replay_loop}")
        if self.event_input_mode == "hardware":
            self.get_logger().info(f"Serial port opened: {self.port_name} @ {self.baud}")
        else:
            self.get_logger().info(
                f"HDF5 replay opened read-only: {self._replay_reader.path}; "
                f"selected_range=[{self._replay_reader.start_packet},"
                f"{self._replay_reader.end_packet}]")
        self.get_logger().info("Event frame publishing initially disabled")
        self.get_logger().info(f"Mono event image topic: {self.topic} encoding=mono8")
        self.get_logger().info(
            f"3-channel event image topic: {self.topic_3_channel} "
            f"encoding={self.event_frame_encoding} enabled={self.publish_3_channel_img}"
        )
        self.get_logger().info(
            f"Native activity voxel topic: {self.topic_event_voxel_1ms} "
            f"enabled={self.publish_event_voxel_1ms} bin_ms={self.event_voxel_bin_ms} "
            f"bins={self.event_voxel_temporal_bins} mode={self.event_voxel_activity_mode} "
            "order=oldest_to_newest frame_id=openmv_cam rotation=0"
        )
        self.get_logger().info(
            f"Event output mode: {self.event_output_mode}; resolved "
            f"mono={self.publish_mono_img}, 3ch={self.publish_3_channel_img}, "
            f"legacy_voxel={self.publish_xyt_voxel}, "
            f"event_voxel_1ms={self.publish_event_voxel_1ms}"
        )
        self.get_logger().info(
            f"XYT event voxel topic: {self.topic_xyt_voxel} "
            f"encoding={self.event_voxel_encoding} enabled={self.publish_xyt_voxel}"
        )
        self.get_logger().info(
            "Live event contract: "
            f"windows_ms={self.event_frame_windows_ms}, "
            f"mode={self.event_frame_mode}, "
            f"scaling={self.event_scaling}, "
            f"event_clip_count={self.event_clip_count}, "
            f"packet_margin_ms={self.event_packet_margin_ms}, "
            f"encoding={self.event_frame_encoding}, "
            "channel_order=recent_to_oldest"
        )
        self.get_logger().info(
            "XYT live contract: "
            f"horizon_ms={self.event_voxel_horizon_ms}, "
            f"temporal_bins={self.legacy_event_voxel_temporal_bins}, "
            f"shape=({self.event_voxel_height},{self.event_voxel_width},9), "
            f"scaling={self.event_voxel_scaling}, "
            f"clip_count={self.event_voxel_clip_count}, "
            "channel_order=oldest_to_newest, causal=true"
        )
        self.get_logger().info("Raw event recording initially disabled")
        self.get_logger().info(
            "Available services: "
            "/openmv_cam/start_event_frame_publishing, "
            "/openmv_cam/stop_event_frame_publishing, "
            "/openmv_cam/rotate_event_frame_plus_90, "
            "/openmv_cam/rotate_event_frame_180, "
            "/openmv_cam/rotate_event_frame_minus_90, "
            "/openmv_cam/start_raw_event_recording, "
            "/openmv_cam/stop_raw_event_recording"
        )

    def _open_serial(self):
        try:
            self.serial_port = serial.Serial(
                self.port_name,
                baudrate=self.baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
                timeout=self.timeout,
            )
            self.serial_port.reset_input_buffer()
        except Exception as e:
            self.get_logger().error(f"Failed to open serial {self.port_name}: {e}")
            raise

    def _normalize_hdf5_compression(self):
        compression = (self.hdf5_compression or "").strip().lower()
        if compression in ("", "none"):
            return None, None
        if compression == "gzip":
            return "gzip", int(self.hdf5_compression_level)
        if compression == "lzf":
            return "lzf", None
        self.get_logger().warn(
            f"Unknown hdf5_compression '{self.hdf5_compression}', using 'gzip'."
        )
        return "gzip", int(self.hdf5_compression_level)

    def _resolve_output_h5_path(self) -> Path:
        configured = (
            self.get_parameter("raw_event_output_path")
            .get_parameter_value()
            .string_value
            .strip()
        )

        if configured:
            return Path(configured).expanduser().resolve()

        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        return (Path.cwd() / f"openmv_raw_events_{ts}.h5").resolve()

    def _create_appendable_1d_dataset(self, group, name, dtype):
        compression, compression_level = self._normalize_hdf5_compression()
        kwargs = {
            "shape": (0,),
            "maxshape": (None,),
            "dtype": dtype,
            "chunks": (max(1, int(self.hdf5_chunk_size)),),
        }
        if compression is not None:
            kwargs["compression"] = compression
        if compression_level is not None:
            kwargs["compression_opts"] = compression_level
        return group.create_dataset(name, **kwargs)

    def _open_h5_for_recording_locked(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        h5f = h5py.File(str(path), mode="x")

        events_group = h5f.create_group("events")
        packets_group = h5f.create_group("packets")

        self._h5_events_type = self._create_appendable_1d_dataset(events_group, "type", np.uint8)
        self._h5_events_x = self._create_appendable_1d_dataset(events_group, "x", np.uint16)
        self._h5_events_y = self._create_appendable_1d_dataset(events_group, "y", np.uint16)
        self._h5_events_t_us = self._create_appendable_1d_dataset(events_group, "t_us", np.int64)
        self._h5_events_packet_id = self._create_appendable_1d_dataset(
            events_group, "packet_id", np.int64
        )

        self._h5_packets_ros_t_ns = self._create_appendable_1d_dataset(
            packets_group, "ros_t_ns", np.int64
        )
        self._h5_packets_monotonic_t_ns = self._create_appendable_1d_dataset(
            packets_group, "monotonic_t_ns", np.int64
        )
        self._h5_packets_start_event_idx = self._create_appendable_1d_dataset(
            packets_group, "start_event_idx", np.int64
        )
        self._h5_packets_end_event_idx = self._create_appendable_1d_dataset(
            packets_group, "end_event_idx", np.int64
        )
        self._h5_packets_event_count = self._create_appendable_1d_dataset(
            packets_group, "event_count", np.int64
        )
        self._h5_packets_first_event_t_us = self._create_appendable_1d_dataset(
            packets_group, "first_event_t_us", np.int64
        )
        self._h5_packets_last_event_t_us = self._create_appendable_1d_dataset(
            packets_group, "last_event_t_us", np.int64
        )

        h5f.attrs["source"] = "openmv_event_cam"
        h5f.attrs["width"] = int(self.W)
        h5f.attrs["height"] = int(self.H)
        h5f.attrs["event_column_order"] = "type,sec,ms,us,x,y"
        h5f.attrs["created_wall_time"] = float(time.time())
        h5f.attrs[
            "timestamp_note"
        ] = (
            "event t_us is reconstructed from OpenMV sec/ms/us fields; "
            "packet ros_t_ns is host ROS receive time for the packet."
        )
        h5f.attrs[
            "packet_first_last_timestamp_note"
        ] = (
            "first_event_t_us and last_event_t_us are reconstructed from the "
            "first/last event in each received packet using OpenMV sec/ms/us fields."
        )

        self._h5_file = h5f
        self._h5_file_path = str(path)
        self._h5_event_count = 0
        self._h5_packet_count = 0
        self._h5_packets_since_flush = 0

    def _close_h5_locked(self):
        closed_path = self._h5_file_path
        try:
            if self._h5_file is not None:
                self._h5_file.flush()
                self._h5_file.close()
        finally:
            self._h5_file = None
            self._h5_file_path = ""
            self._h5_events_type = None
            self._h5_events_x = None
            self._h5_events_y = None
            self._h5_events_t_us = None
            self._h5_events_packet_id = None
            self._h5_packets_ros_t_ns = None
            self._h5_packets_monotonic_t_ns = None
            self._h5_packets_start_event_idx = None
            self._h5_packets_end_event_idx = None
            self._h5_packets_event_count = None
            self._h5_packets_first_event_t_us = None
            self._h5_packets_last_event_t_us = None
            self._h5_event_count = 0
            self._h5_packet_count = 0
            self._h5_packets_since_flush = 0
            self._recording_enabled = False
        return closed_path

    def _append_packet_to_h5(
        self,
        packet: EventPacket,
    ):
        events = packet.events
        event_count = int(events.shape[0])
        if event_count >= 8192:
            print("WARNING: event buffer saturated")
        event_t_us = packet.timestamps_us if event_count > 0 else None

        if event_count > 0:
            packet_first_event_t_us = int(event_t_us[0])
            packet_last_event_t_us = int(event_t_us[-1])
        else:
            packet_first_event_t_us = -1
            packet_last_event_t_us = -1

        if event_count > 0:
            event_type = events[:, 0].astype(np.uint8, copy=False)
            event_x = events[:, 4].astype(np.uint16, copy=False)
            event_y = events[:, 5].astype(np.uint16, copy=False)
        else:
            event_type = None
            event_x = None
            event_y = None
            event_t_us = None

        with self._h5_lock:
            if not self._recording_enabled or self._h5_file is None:
                return

            # Local packet index inside this HDF5 recording.
            packet_idx = self._h5_packet_count

            if event_count > 0:
                event_packet_id = np.full((event_count,), packet_idx, dtype=np.int64)
            else:
                event_packet_id = None

            old_n = self._h5_event_count
            new_n = old_n + event_count


            self._h5_events_type.resize((new_n,))
            self._h5_events_x.resize((new_n,))
            self._h5_events_y.resize((new_n,))
            self._h5_events_t_us.resize((new_n,))
            self._h5_events_packet_id.resize((new_n,))

            if event_count > 0:
                self._h5_events_type[old_n:new_n] = event_type
                self._h5_events_x[old_n:new_n] = event_x
                self._h5_events_y[old_n:new_n] = event_y
                self._h5_events_t_us[old_n:new_n] = event_t_us
                self._h5_events_packet_id[old_n:new_n] = event_packet_id

            packet_idx = self._h5_packet_count
            packet_new_idx = packet_idx + 1
            self._h5_packets_ros_t_ns.resize((packet_new_idx,))
            self._h5_packets_monotonic_t_ns.resize((packet_new_idx,))
            self._h5_packets_start_event_idx.resize((packet_new_idx,))
            self._h5_packets_end_event_idx.resize((packet_new_idx,))
            self._h5_packets_event_count.resize((packet_new_idx,))
            self._h5_packets_first_event_t_us.resize((packet_new_idx,))
            self._h5_packets_last_event_t_us.resize((packet_new_idx,))

            self._h5_packets_ros_t_ns[packet_idx] = np.int64(
                packet.packet_ros_stamp_ns)
            self._h5_packets_monotonic_t_ns[packet_idx] = np.int64(
                packet.packet_monotonic_stamp_ns)
            self._h5_packets_start_event_idx[packet_idx] = np.int64(old_n)
            self._h5_packets_end_event_idx[packet_idx] = np.int64(new_n)
            self._h5_packets_event_count[packet_idx] = np.int64(event_count)
            self._h5_packets_first_event_t_us[packet_idx] = np.int64(packet_first_event_t_us)
            self._h5_packets_last_event_t_us[packet_idx] = np.int64(packet_last_event_t_us)

            self._h5_event_count = new_n
            self._h5_packet_count = packet_new_idx
            self._h5_packets_since_flush += 1

            if self._h5_packets_since_flush >= max(1, int(self.flush_every_packets)):
                self._h5_file.flush()
                self._h5_packets_since_flush = 0

    def _ensure_publisher_and_timer(self):
        if self.publish_mono_img and self.pub_mono_img is None:
            self.pub_mono_img = self.create_publisher(Image, self.topic, 10)
        if self.publish_3_channel_img and self.pub_3ch is None:
            self.pub_3ch = self.create_publisher(Image, self.topic_3_channel, 10)
        if self.publish_xyt_voxel and self.pub_xyt_voxel is None:
            self.pub_xyt_voxel = self.create_publisher(
                Image, self.topic_xyt_voxel, 10
            )
        if self.publish_event_voxel_1ms and self.pub_event_voxel_1ms is None:
            self.pub_event_voxel_1ms = self.create_publisher(
                Image, self.topic_event_voxel_1ms, 10
            )
        if (
            (self.publish_mono_img or self.publish_3_channel_img or self.publish_xyt_voxel)
            and self.publish_timer is None
        ):
            self.publish_timer = self.create_timer(1.0 / self.publish_fps, self._publish_timer_cb)
        if self.publish_event_voxel_1ms and self.event_voxel_timer is None:
            self.event_voxel_timer = self.create_timer(
                1.0 / self.event_voxel_publish_fps,
                self._publish_activity_voxel_timer_cb,
            )
        if self.event_diagnostics_enabled and self.diagnostics_timer is None:
            self.diagnostics_timer = self.create_timer(
                self.event_diagnostics_period_sec,
                self._diagnostics_timer_cb,
            )
        if not self._publisher_config_logged:
            self.get_logger().info(
                "Publishers configured: "
                f"mono_topic={self.topic}, "
                f"topic_3_channel={self.topic_3_channel}, "
                f"publish_3_channel_img={self.publish_3_channel_img}, "
                f"topic_xyt_voxel={self.topic_xyt_voxel}, "
                f"publish_xyt_voxel={self.publish_xyt_voxel}"
                f", topic_event_voxel_1ms={self.topic_event_voxel_1ms}"
                f", publish_event_voxel_1ms={self.publish_event_voxel_1ms}"
            )
            self._publisher_config_logged = True

    def _stop_publishing_internal(self):
        self._publishing_enabled = False
        self._preview_history_truncated_warned = False

        if self.publish_timer is not None:
            self.publish_timer.cancel()
            self.destroy_timer(self.publish_timer)
            self.publish_timer = None
        if self.event_voxel_timer is not None:
            self.event_voxel_timer.cancel()
            self.destroy_timer(self.event_voxel_timer)
            self.event_voxel_timer = None
        if self.diagnostics_timer is not None:
            self.diagnostics_timer.cancel()
            self.destroy_timer(self.diagnostics_timer)
            self.diagnostics_timer = None

        with self._preview_lock:
            self.preview_buffer.clear()

    def _buffer_history_limit_s(self) -> float:
        detector_horizon_ms = (
            self.event_voxel_bin_ms * self.event_voxel_temporal_bins
            if self.publish_event_voxel_1ms
            else 0.0
        )
        voxel_horizon_ms = max(
            self.event_voxel_horizon_ms if self.publish_xyt_voxel else 0.0,
            detector_horizon_ms,
        )
        history_ms = retention_history_ms(
            mono_window_ms=self.window_ms,
            max_event_window_ms=max(self.event_frame_windows_ms),
            event_packet_margin_ms=self.event_packet_margin_ms,
            include_event_channels=self.publish_3_channel_img,
            event_voxel_horizon_ms=voxel_horizon_ms,
            include_event_voxel=self.publish_xyt_voxel or self.publish_event_voxel_1ms,
        )
        return float(history_ms) / 1000.0

    def _handle_start_event_frame_publishing(self, request, response):
        del request
        if self._publishing_enabled:
            response.success = False
            response.message = ""
            return response

        self._ensure_publisher_and_timer()
        self._publishing_enabled = True
        message = f"Event frame publishing started on {self.topic} at {self.publish_fps:.1f} Hz"
        if self.publish_3_channel_img:
            message += f"; 3-channel topic {self.topic_3_channel} active"
        else:
            message += "; 3-channel publishing disabled"
        if self.publish_xyt_voxel:
            message += f"; XYT voxel topic {self.topic_xyt_voxel} active"
        else:
            message += "; XYT voxel publishing disabled"
        if self.publish_event_voxel_1ms:
            message += f"; native activity voxel topic {self.topic_event_voxel_1ms} active"
        self.get_logger().info(message)
        response.success = True
        response.message = ""
        return response

    def _handle_stop_event_frame_publishing(self, request, response):
        del request
        if not self._publishing_enabled:
            response.success = False
            response.message = ""
            return response

        self._stop_publishing_internal()
        self.get_logger().info("Event frame publishing stopped for all configured outputs")
        response.success = True
        response.message = ""
        return response     

    def _make_event_frame_rotation_handler(self, rotation_degrees: int):
        def handle_rotation(request, response):
            del request
            self.event_frame_rotation_degrees = rotation_degrees
            response.success = True
            response.message = (
                f"Event frame output rotation set to {rotation_degrees} degrees "
                "counterclockwise"
            )
            self.get_logger().info(response.message)
            return response

        return handle_rotation

    def _handle_start_raw_event_recording(self, request, response):
        del request
        with self._h5_lock:
            if self._recording_enabled:
                response.success = False
                response.message = ""
                return response

            target_path = self._resolve_output_h5_path()
            if target_path.exists():
                response.success = False
                response.message = ""   

                return response

            try:
                self._open_h5_for_recording_locked(target_path)
                self._recording_enabled = True
            except Exception as e:
                if self._h5_file is not None:
                    try:
                        self._h5_file.close()
                    except Exception:
                        pass
                self._h5_file = None
                self._recording_enabled = False
                response.success = False
                response.message = ""
                return response

        self.get_logger().info(f"Raw event recording started: {target_path}")
        response.success = True
        response.message = ("")
        return response

    def _handle_stop_raw_event_recording(self, request, response):
        del request
        with self._h5_lock:
            if not self._recording_enabled:
                response.success = False
                response.message = ""
                return response

            try:
                closed_path = self._close_h5_locked()
            except Exception as e:
                self.get_logger().warn(f"Failed to flush and close HDF5: {e}")
                response.success = False
                response.message = ""
                return response

        self.get_logger().info(f"Raw event recording stopped: {closed_path}")
        response.success = True
        response.message = ""
        return response

    def _read_exactly(self, n: int) -> bytes:
        data = bytearray()
        while len(data) < n:
            if self._stop_event.is_set():
                raise RuntimeError("Stop requested.")
            chunk = self.serial_port.read(n - len(data))
            if not chunk:
                raise RuntimeError("Serial read timeout.")
            data.extend(chunk)
        return bytes(data)

    def _read_until_magic(self):
        window = bytearray()
        while not self._stop_event.is_set():
            b = self.serial_port.read(1)
            if not b:
                raise RuntimeError("Timeout while waiting for magic.")
            window += b
            if len(window) > len(self.MAGIC):
                window = window[-len(self.MAGIC):]
            if bytes(window) == self.MAGIC:
                return
        raise RuntimeError("Stop requested.")

    @staticmethod
    def event_timestamps_us(events: np.ndarray) -> np.ndarray:
        """
        Reconstruct absolute timestamps in microseconds from columns:
          1: sec
          2: ms
          3: us
        """
        return reconstruct_timestamps_us(events)

    def sort_events_by_timestamp_fn(self, events: np.ndarray) -> np.ndarray:
        if events.size == 0:
            return events
        ts = self.event_timestamps_us(events)
        order = np.argsort(ts, kind="stable")
        return events[order]

    @staticmethod
    def _render_event_frame_from_events(
        events: np.ndarray,
        width: int,
        height: int,
        contrast: float = 4.0,
        step: float = 1.0,
    ) -> np.ndarray:
        frame = np.full((height, width), 128.0, dtype=np.float32)

        if events.size == 0:
            return frame.astype(np.uint8)

        xs = events[:, 4].astype(np.int32)
        ys = events[:, 5].astype(np.int32)
        tp = events[:, 0].astype(np.int32)

        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        xs = xs[valid]
        ys = ys[valid]
        tp = tp[valid]

        if xs.size == 0:
            return frame.astype(np.uint8)

        pos = tp == 1
        neg = ~pos

        acc = np.zeros((height, width), dtype=np.float32)
        np.add.at(acc, (ys[pos], xs[pos]), +step)
        np.add.at(acc, (ys[neg], xs[neg]), -step)

        # m = np.percentile(np.abs(acc), 99.0)
        # m = max(m, 1.0)
        # acc = np.clip(acc / m, -1.0, 1.0)

        m = np.max(np.abs(acc))
        if m > 0:
            acc /= m

        frame = 128.0 + acc * (contrast * 127.0)
        np.clip(frame, 0, 255, out=frame)
        return frame.astype(np.uint8)

    @staticmethod
    def events_to_preview_frame(
        events: np.ndarray,
        width: int,
        height: int,
        contrast: float = 4.0,
        step: float = 1.0,
    ) -> np.ndarray:
        return render_event_frame_from_arrays(
            event_type=events[:, 0],
            event_x=events[:, 4],
            event_y=events[:, 5],
            width=width,
            height=height,
            scaling_mode="legacy_per_frame_max",
            contrast=contrast,
            step=step,
        )

    @staticmethod
    def events_to_shifted_3ch_frame(
        events: np.ndarray,
        event_ts_us: np.ndarray,
        now_event_t_us: int,
        width: int,
        height: int,
        windows_ms,
        mode: str,
        event_scaling: str,
        event_clip_count: float,
        contrast: float,
        step: float,
    ) -> np.ndarray:
        frame_3ch, _counts = build_event_frame_3ch(
            events=events,
            event_ts_us=event_ts_us,
            now_event_t_us=now_event_t_us,
            width=width,
            height=height,
            windows_ms=windows_ms,
            mode=mode,
            scaling_mode=event_scaling,
            event_clip_count=event_clip_count,
            contrast=contrast,
            step=step,
        )
        return frame_3ch

    def _trim_preview_buffer_locked(self, now: float):
        cutoff = now - self._buffer_history_limit_s()

        while self.preview_buffer and self.preview_buffer[0][0] < cutoff:
            self.preview_buffer.popleft()

        safety_cap = (
            max(1, int(self.max_event_frame_packets))
            if (
                self.publish_3_channel_img
                or self.publish_xyt_voxel
                or self.publish_event_voxel_1ms
            )
            else max(1, int(self.max_preview_packets))
        )
        if len(self.preview_buffer) > safety_cap:
            dropped = len(self.preview_buffer) - safety_cap
            for _ in range(dropped):
                dropped_packet = self.preview_buffer.popleft()
                self._diagnostics.packets_dropped_by_cap += 1
                self._diagnostics.events_dropped_by_cap += int(
                    dropped_packet[1].event_count
                )
            if not self._preview_history_truncated_warned:
                self.get_logger().warn(
                    "Preview buffer truncated by safety cap "
                    f"({safety_cap} packets); event-volume history may be incomplete."
                    if (
                        self.publish_3_channel_img
                        or self.publish_xyt_voxel
                        or self.publish_event_voxel_1ms
                    )
                    else f"Preview buffer truncated by safety cap ({safety_cap} packets)."
                )
                self._preview_history_truncated_warned = True

    def _publish_mono_image(self, img: np.ndarray):
        if self.pub_mono_img is None:
            return
        if img.ndim != 2:
            self.get_logger().error(
                f"Refusing to publish mono image on {self.topic}: expected 2D array, got shape={img.shape}"
            )
            return
        msg = self.bridge.cv2_to_imgmsg(img, encoding="mono8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub_mono_img.publish(msg)

    def _publish_3ch_image(self, img_3ch: np.ndarray):
        if self.pub_3ch is None:
            return
        if img_3ch.ndim != 3 or img_3ch.shape[2] != 3:
            self.get_logger().error(
                "Refusing to publish 3-channel image on "
                f"{self.topic_3_channel}: expected shape (H, W, 3), got shape={img_3ch.shape}"
            )
            return
        msg = self.bridge.cv2_to_imgmsg(img_3ch, encoding=self.event_frame_encoding)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub_3ch.publish(msg)

    def _publish_xyt_image(self, voxel: np.ndarray):
        if self.pub_xyt_voxel is None:
            return
        try:
            msg = build_hwc9_image_message(
                voxel,
                encoding=self.event_voxel_encoding,
                stamp=self.get_clock().now().to_msg(),
                frame_id=self.frame_id,
            )
        except ValueError as error:
            self.get_logger().error(
                f"Refusing to publish XYT voxel on {self.topic_xyt_voxel}: {error}"
            )
            return
        self.pub_xyt_voxel.publish(msg)

    def _publish_tracker_trace(self, *, event, sequence, parent_sequence,
                               source_stamp_ns, receipt_ros_stamp_ns,
                               start_ros_stamp_ns, end_ros_stamp_ns,
                               start_steady_ns, end_steady_ns, valid,
                               scalar_value=0.0, detail_json="{}"):
        if self.latency_trace_pub is None:
            return
        msg = self.LatencyTrace()
        msg.run_id = str(self.get_parameter("latency_trace_run_id").value)
        msg.stage = "event_2d_ball_detection"
        msg.event = event
        msg.modality = "event"
        msg.node_name = self.get_name()
        msg.sequence = int(sequence)
        msg.parent_sequence = int(parent_sequence)
        msg.source_stamp_ns = int(source_stamp_ns)
        msg.receipt_ros_stamp_ns = int(receipt_ros_stamp_ns)
        msg.start_ros_stamp_ns = int(start_ros_stamp_ns)
        msg.end_ros_stamp_ns = int(end_ros_stamp_ns)
        msg.start_steady_ns = int(start_steady_ns)
        msg.end_steady_ns = int(end_steady_ns)
        msg.valid = bool(valid)
        msg.scalar_value = float(scalar_value)
        msg.detail_json = detail_json
        self.latency_trace_pub.publish(msg)

    def _handle_tracker_packet(self, packet: EventPacket):
        if self.event_tracker is None:
            return
        start_ros_ns = int(self.get_clock().now().nanoseconds)
        trace_detail = {
            "source": packet.source,
            "packet_index": int(packet.packet_id),
            "sensor_timestamp_domain": "genx320_microseconds",
            "first_event_timestamp_us": int(packet.first_event_timestamp_us),
            "last_event_timestamp_us": int(packet.last_event_timestamp_us),
            "event_count": int(packet.event_count),
        }
        if packet.source == "hdf5_replay":
            trace_detail.update({
                "original_recorded_ros_t_ns": int(packet.original_ros_stamp_ns),
                "original_recorded_monotonic_t_ns": int(
                    packet.original_monotonic_stamp_ns),
            })
        self._publish_tracker_trace(
            event="input", sequence=packet.packet_id, parent_sequence=0,
            source_stamp_ns=packet.packet_ros_stamp_ns,
            receipt_ros_stamp_ns=packet.packet_ros_stamp_ns,
            start_ros_stamp_ns=start_ros_ns, end_ros_stamp_ns=start_ros_ns,
            start_steady_ns=packet.packet_monotonic_stamp_ns,
            end_steady_ns=packet.packet_monotonic_stamp_ns,
            valid=packet.event_count > 0,
            scalar_value=float(packet.event_count),
            detail_json=json.dumps(trace_detail, allow_nan=False, separators=(",", ":")))
        detections = self.event_tracker.update(packet)
        for detection in detections:
            self._tracker_trace_sequence += 1
            if detection.valid:
                stamp = self.get_clock().now().to_msg()
                position = PointStamped()
                position.header.stamp = stamp
                position.header.frame_id = "openmv_cam"
                position.point.x = detection.x_px
                position.point.y = detection.y_px
                velocity = Vector3Stamped()
                velocity.header.stamp = stamp
                velocity.header.frame_id = "openmv_cam"
                velocity.vector.x = detection.vx_px_s
                velocity.vector.y = detection.vy_px_s
                velocity.vector.z = detection.speed_px_s
                self.event_tracker_position_pub.publish(position)
                self.event_tracker_velocity_pub.publish(velocity)
                valid = Bool()
                valid.data = True
                self.event_tracker_valid_pub.publish(valid)
            end_ros_ns = int(self.get_clock().now().nanoseconds)
            self._publish_tracker_trace(
                event="complete", sequence=self._tracker_trace_sequence,
                parent_sequence=detection.parent_packet_id,
                source_stamp_ns=packet.packet_ros_stamp_ns,
                receipt_ros_stamp_ns=packet.packet_ros_stamp_ns,
                start_ros_stamp_ns=start_ros_ns, end_ros_stamp_ns=end_ros_ns,
                start_steady_ns=detection.start_steady_ns,
                end_steady_ns=detection.end_steady_ns, valid=detection.valid,
                scalar_value=detection.confidence,
                detail_json=self._tracker_completion_detail(packet, detection))

    @staticmethod
    def _tracker_completion_detail(packet, detection):
        detail = json.loads(trace_detail_json(detection))
        detail["source"] = packet.source
        detail["packet_index"] = int(packet.packet_id)
        if packet.source == "hdf5_replay":
            detail["original_recorded_ros_t_ns"] = int(packet.original_ros_stamp_ns)
            detail["original_recorded_monotonic_t_ns"] = int(
                packet.original_monotonic_stamp_ns)
        return json.dumps(detail, allow_nan=False, separators=(",", ":"))

    def _event_tracker_stats_cb(self):
        stats = self.event_tracker.statistics()
        timing_names = ("map_build", "blob_detection", "computation",
                        "velocity_fit", "total_tracker_update")
        timings = " ".join(
            f"{name}_p50/p95/max_ms={stats[name][0]:.3f}/{stats[name][1]:.3f}/{stats[name][2]:.3f}"
            for name in timing_names)
        keys = ("packets_received", "packets_with_events",
                "processed_1ms_bins", "window_updates", "empty_bins",
                "late_events_or_bins", "candidate_blob_count",
                "valid_detections", "invalid_detections",
                "velocity_ready_count", "threshold_foreground_pixels",
                "spatial_filter_removed_pixels",
                "spatial_filter_removed_components",
                "spatial_filter_removed_component_pixels")
        counts = " ".join(f"{key}={stats.get(key, 0)}" for key in keys)
        self.get_logger().info(
            "EVENT TRACKER STATS | " + counts +
            f" event_rate_hz={stats['event_rate_hz']:.1f}"
            f" processed_bin_rate_hz={stats['processed_bin_rate_hz']:.2f}"
            f" window_update_rate_hz={stats['window_update_rate_hz']:.2f}"
            f" valid_detection_rate_hz="
            f"{stats['valid_detection_rate_hz']:.2f} " + timings)

    def _event_tracker_debug_timer_cb(self):
        snapshot = self.event_tracker.latest_debug_snapshot()
        if snapshot is None:
            return
        window_end_us = snapshot.detection.window_end_us
        if window_end_us == self._last_tracker_debug_bin_start_us:
            return
        self._last_tracker_debug_bin_start_us = window_end_us
        images = render_debug_images(
            snapshot,
            clip_count=int(
                self.get_parameter("event_tracker_debug_clip_count").value),
            rotation_degrees=int(self.get_parameter(
                "event_tracker_debug_rotation_degrees").value))
        stamp = self.get_clock().now().to_msg()
        for stage, image in images.items():
            message = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            message.header.stamp = stamp
            message.header.frame_id = "openmv_cam"
            self.event_tracker_debug_stage_pubs[stage].publish(message)
        # Preserve the original combined topic as an alias of the tracking stage.
        combined = self.bridge.cv2_to_imgmsg(images["tracking"], encoding="bgr8")
        combined.header.stamp = stamp
        combined.header.frame_id = "openmv_cam"
        self.event_tracker_debug_pub.publish(combined)

    def _reader_loop(self):
        while not self._stop_event.is_set():
            try:
                self._read_until_magic()
                header_rest = self._read_exactly(struct.calcsize(self.HEADER_FMT))
                event_count, payload_len = struct.unpack(self.HEADER_FMT, header_rest)

                if event_count > MAX_EVENT_COUNT:
                    raise RuntimeError(
                        f"Invalid event count: {event_count} exceeds {MAX_EVENT_COUNT}")

                expected_len = event_count * 6 * 2
                if payload_len != expected_len:
                    raise RuntimeError(
                        f"Invalid payload length: got {payload_len}, expected {expected_len}"
                    )

                payload = self._read_exactly(payload_len)
                # Host receipt is sampled only once the complete EVT1 packet is available.
                packet_ros_t_ns = int(self.get_clock().now().nanoseconds)
                packet_mono_t_ns = int(time.monotonic_ns())

                packet_id = int(self.total_packets + 1)
                packet = EventPacket.decode(
                    payload, event_count=event_count, payload_length=payload_len,
                    packet_id=packet_id, packet_ros_stamp_ns=packet_ros_t_ns,
                    packet_monotonic_stamp_ns=packet_mono_t_ns)
                self._process_packet(packet)

            except RuntimeError as e:
                if "Stop requested" in str(e):
                    break
                self.get_logger().warn(f"Event read failed: {e}")
                self._stop_event.wait(0.05)

            except Exception as e:
                self.get_logger().warn(f"Unexpected error in reader loop: {e}")
                self._stop_event.wait(0.05)

    def _process_packet(self, packet: EventPacket):
        """Drive every downstream consumer for hardware and replay packets."""
        if self._stop_event.is_set():
            return
        now = time.monotonic()
        self.total_packets += 1
        self.total_events += packet.event_count
        self.total_payload_bytes += packet.payload_length
        self.total_protocol_bytes += packet.payload_length + self.HEADER_SIZE
        if self.total_packets <= 3 and packet.event_count > 0:
            events = packet.events
            self.get_logger().info(
                f"{packet.source} packet {packet.packet_id}: "
                f"type={np.unique(events[:, 0])[:10]}, "
                f"x=[{int(events[:, 4].min())},{int(events[:, 4].max())}], "
                f"y=[{int(events[:, 5].min())},{int(events[:, 5].max())}]")
        if self._recording_enabled:
            self._append_packet_to_h5(packet)
        self._handle_tracker_packet(packet)
        if self._publishing_enabled and not self._stop_event.is_set():
            with self._preview_lock:
                self.preview_buffer.append((now, packet))
                if packet.event_count > 0:
                    self._event_data_generation += 1
                self._trim_preview_buffer_locked(now)
        if now - self.last_stats_print >= 2.0:
            elapsed = max(now - self.t0, 1e-9)
            if self.print_log:
                self.get_logger().info(
                    "EVENT STREAM STATS | "
                    f"source={packet.source}, packets={self.total_packets}, "
                    f"events={self.total_events}, elapsed={elapsed:.2f}s, "
                    f"packet_rate_hz={self.total_packets / elapsed:.1f}, "
                    f"event_rate_hz={self.total_events / elapsed:.1f}, "
                    f"payload_MBps={self.total_payload_bytes / elapsed / 1e6:.3f}, "
                    f"protocol_MBps={self.total_protocol_bytes / elapsed / 1e6:.3f}")
            self.last_stats_print = now

    def _replay_loop(self):
        replay_started_ns = time.monotonic_ns()
        try:
            replay_packets(
                self._replay_reader, timing=self.event_replay_timing,
                rate=self.event_replay_rate, loop=self.event_replay_loop,
                stop_event=self._stop_event, process_packet=self._process_packet,
                ros_now_ns=lambda: int(self.get_clock().now().nanoseconds),
                monotonic_now_ns=time.monotonic_ns,
                diagnostics=self._replay_diagnostics,
                warn=self.get_logger().warn)
            d = self._replay_diagnostics
            elapsed_s = max((time.monotonic_ns() - replay_started_ns) / 1e9, 1e-9)
            actual_packet_rate = d.replay_packets_processed / elapsed_s
            event_rate = d.replay_events_processed / elapsed_s
            recorded_packet_rate = 0.0
            if d.replay_packets_processed > 1:
                try:
                    first = self._replay_reader.packet_timing_value(
                        self._replay_reader.start_packet, "recorded")
                    last = self._replay_reader.packet_timing_value(
                        self._replay_reader.end_packet, "recorded")
                    span_s = (last - first) / 1e9
                    if first >= 0 and span_s > 0.0:
                        selected_count = (self._replay_reader.end_packet -
                                          self._replay_reader.start_packet + 1)
                        recorded_packet_rate = (selected_count - 1) / span_s
                except (ValueError, KeyError):
                    pass
            self.get_logger().info(
                "HDF5 REPLAY COMPLETE | "
                f"replay_packets_read={d.replay_packets_read}, "
                f"replay_packets_processed={d.replay_packets_processed}, "
                f"replay_events_processed={d.replay_events_processed}, "
                f"replay_packets_skipped={d.replay_packets_skipped}, "
                f"replay_loops_completed={d.replay_loops_completed}, "
                f"replay_timing_fallbacks={d.replay_timing_fallbacks}, "
                f"recorded_packet_rate_hz={recorded_packet_rate:.2f}, "
                f"actual_replay_packet_rate_hz={actual_packet_rate:.2f}, "
                f"event_rate_hz={event_rate:.1f}")
        except Exception as error:
            if not self._stop_event.is_set():
                self.get_logger().error(f"HDF5 replay failed: {error}")
        finally:
            if self._replay_reader is not None:
                self._replay_reader.close()

    def _publish_timer_cb(self):
        if not self._publishing_enabled:
            return

        with self._preview_lock:
            now = time.monotonic()
            self._trim_preview_buffer_locked(now)
            snapshot = list(self.preview_buffer)

        if not snapshot:
            frame = np.full((self.H, self.W), 128, dtype=np.uint8)
            frame_3ch = None
            voxel = (
                np.full(
                    (
                        self.event_voxel_height,
                        self.event_voxel_width,
                        self.legacy_event_voxel_temporal_bins,
                    ),
                    128,
                    dtype=np.uint8,
                )
                if self.publish_xyt_voxel
                else None
            )
        else:
            # preview_buffer keeps enough history for all outputs (mono + 3-channel).
            chunks = [packet.events for _, packet in snapshot]
            timestamp_chunks = [packet.timestamps_us for _, packet in snapshot]
            chunk = np.concatenate(chunks, axis=0)
            event_ts_us = np.concatenate(timestamp_chunks, axis=0)

            if self.sort_by_timestamp:
                order = np.argsort(event_ts_us, kind="stable")
                chunk = chunk[order]
                event_ts_us = event_ts_us[order]

            if event_ts_us.size == 0:
                frame = np.full((self.H, self.W), 128, dtype=np.uint8)
                frame_3ch = (
                    np.full((self.H, self.W, 3), 128, dtype=np.uint8)
                    if self.publish_3_channel_img
                    else None
                )
                voxel = (
                    np.full(
                        (
                            self.event_voxel_height,
                            self.event_voxel_width,
                            self.legacy_event_voxel_temporal_bins,
                        ),
                        128,
                        dtype=np.uint8,
                    )
                    if self.publish_xyt_voxel
                    else None
                )
            else:
                now_event_t_us = int(np.max(event_ts_us))

                # Mono output is explicitly filtered to window_ms.
                mono_window_us = int(self.window_ms * 1000.0)
                mono_mask = event_ts_us >= (now_event_t_us - mono_window_us)
                mono_chunk = chunk[mono_mask]

                if mono_chunk.size == 0:
                    frame = np.full((self.H, self.W), 128, dtype=np.uint8)
                else:
                    frame = self.events_to_preview_frame(
                        mono_chunk,
                        self.W,
                        self.H,
                        contrast=self.contrast,
                        step=self.step,
                    )

                frame_3ch = None
                if self.publish_3_channel_img:
                    # 3-channel output uses full retained chunk with its own configured windows.
                    frame_3ch = self.events_to_shifted_3ch_frame(
                        chunk,
                        event_ts_us,
                        now_event_t_us,
                        self.W,
                        self.H,
                        self.event_frame_windows_ms,
                        self.event_frame_mode,
                        self.event_scaling,
                        float(self.event_clip_count) if self.event_clip_count is not None else 0.0,
                        self.contrast,
                        self.step,
                    )

                voxel = None
                if self.publish_xyt_voxel:
                    voxel, _voxel_counts = build_xyt_signed_voxel(
                        events=chunk,
                        event_ts_us=event_ts_us,
                        anchor_t_us=now_event_t_us,
                        sensor_width=self.W,
                        sensor_height=self.H,
                        output_width=self.event_voxel_width,
                        output_height=self.event_voxel_height,
                        horizon_ms=self.event_voxel_horizon_ms,
                        temporal_bins=self.legacy_event_voxel_temporal_bins,
                        scaling_mode=self.event_voxel_scaling,
                        event_clip_count=self.event_voxel_clip_count,
                    )

                if self.print_log and (now - self._last_publish_debug_log_t) >= 3.0:
                    mono_count = int(np.count_nonzero(mono_mask))
                    full_count = int(chunk.shape[0])
                    self.get_logger().info(
                        "PUBLISH DEBUG | "
                        f"mono_selected_events={mono_count}, "
                        f"full_buffered_events={full_count}, "
                        f"mono_window_ms={self.window_ms:.1f}, "
                        f"event_frame_windows_ms={self.event_frame_windows_ms}, "
                        f"event_scaling={self.event_scaling}, "
                        f"publish_xyt_voxel={self.publish_xyt_voxel}, "
                        f"event_voxel_horizon_ms={self.event_voxel_horizon_ms}"
                    )
                    self._last_publish_debug_log_t = now

            if self.publish_3_channel_img and frame_3ch is not None:
                if self.blur_kernel and self.blur_kernel > 1 and self.event_scaling == "legacy_per_frame_max":
                    for ch_idx in range(frame_3ch.shape[2]):
                        frame_3ch[:, :, ch_idx] = cv2.blur(
                            frame_3ch[:, :, ch_idx],
                            (self.blur_kernel, self.blur_kernel),
                        )

        if self.blur_kernel and self.blur_kernel > 1:
            frame = cv2.blur(frame, (self.blur_kernel, self.blur_kernel))

        frame = rotate_event_frame(frame, self.event_frame_rotation_degrees)
        if frame_3ch is not None:
            frame_3ch = rotate_event_frame(
                frame_3ch, self.event_frame_rotation_degrees
            )
        if voxel is not None:
            voxel = rotate_event_frame(voxel, self.event_frame_rotation_degrees)

        if self.publish_mono_img:
            self._publish_mono_image(frame)
        if self.publish_3_channel_img:
            if frame_3ch is None:
                frame_3ch = np.full((self.H, self.W, 3), 128, dtype=np.uint8)
            self._publish_3ch_image(frame_3ch)
        if self.publish_xyt_voxel:
            if voxel is None:
                voxel = np.full(
                    (
                        self.event_voxel_height,
                        self.event_voxel_width,
                        self.legacy_event_voxel_temporal_bins,
                    ),
                    128,
                    dtype=np.uint8,
                )
            self._publish_xyt_image(voxel)

    def _publish_activity_voxel_timer_cb(self):
        """Publish a rolling event-time voxel only after new non-empty input."""
        tick_started = time.monotonic()
        self._diagnostics.timer_ticks += 1
        if self._last_activity_tick_monotonic is not None:
            self._diagnostics.timer_interval.add_seconds(
                tick_started - self._last_activity_tick_monotonic
            )
        self._last_activity_tick_monotonic = tick_started
        if not self._publishing_enabled or self.pub_event_voxel_1ms is None:
            return

        processing_started = time.monotonic()
        with self._preview_lock:
            now = time.monotonic()
            self._trim_preview_buffer_locked(now)
            generation = self._event_data_generation
            if not has_new_event_data(generation, self._last_activity_generation):
                self._diagnostics.skipped_no_new_events += 1
                return
            snapshot = list(self.preview_buffer)

        nonempty_packets = [item for item in snapshot if item[1].event_count > 0]
        if not nonempty_packets:
            self._diagnostics.skipped_no_new_events += 1
            return

        chunks = [packet.events for _, packet in nonempty_packets]
        events = np.concatenate(chunks, axis=0)
        event_ts_us = np.concatenate(
            [packet.timestamps_us for _, packet in nonempty_packets], axis=0)
        if event_ts_us.size == 0:
            self._diagnostics.skipped_no_new_events += 1
            return
        anchor_t_us = int(np.max(event_ts_us))
        self._diagnostics.buffer_processing.add_seconds(
            time.monotonic() - processing_started
        )
        if (
            self._last_activity_anchor_t_us is None
            or anchor_t_us > self._last_activity_anchor_t_us
        ):
            self._diagnostics.anchors_advanced += 1
        else:
            self._diagnostics.anchors_repeated += 1
        self._last_activity_anchor_t_us = anchor_t_us

        # Use the host ROS receive stamp of the packet containing the anchor.
        # This is the closest available mapping from sensor event time to ROS time.
        anchor_packet_ros_t_ns = nonempty_packets[-1][1].packet_ros_stamp_ns
        for _, packet in nonempty_packets:
            if np.any(packet.timestamps_us == anchor_t_us):
                anchor_packet_ros_t_ns = packet.packet_ros_stamp_ns

        voxel_started = time.monotonic()
        voxel, _bin_counts = build_event_activity_voxel(
            events,
            event_ts_us,
            anchor_t_us,
            width=self.W,
            height=self.H,
            bin_ms=self.event_voxel_bin_ms,
            temporal_bins=self.event_voxel_temporal_bins,
            activity_mode=self.event_voxel_activity_mode,
            clip_count=self.event_voxel_clip_count,
        )
        self._diagnostics.voxel_build.add_seconds(
            time.monotonic() - voxel_started
        )
        stamp = rclpy.time.Time(nanoseconds=int(anchor_packet_ros_t_ns)).to_msg()
        message_started = time.monotonic()
        try:
            msg = build_activity_image_message(
                voxel,
                stamp=stamp,
                frame_id="openmv_cam",
            )
        except ValueError as error:
            self.get_logger().error(
                f"Refusing to publish activity voxel on {self.topic_event_voxel_1ms}: {error}"
            )
            return
        self._diagnostics.message_build.add_seconds(
            time.monotonic() - message_started
        )
        publish_started = time.monotonic()
        self.pub_event_voxel_1ms.publish(msg)
        publish_finished = time.monotonic()
        self._diagnostics.publish_call.add_seconds(
            publish_finished - publish_started
        )
        if self._last_activity_publish_monotonic is not None:
            self._diagnostics.publish_interval.add_seconds(
                publish_finished - self._last_activity_publish_monotonic
            )
        self._last_activity_publish_monotonic = publish_finished
        self._last_activity_generation = generation
        self._diagnostics.messages_published += 1

    def _diagnostics_timer_cb(self):
        """Emit throttled rate and timing diagnostics without per-event logs."""
        now = time.monotonic()
        elapsed = max(now - self._diagnostic_last_monotonic, 1e-9)
        packets = self.total_packets - self._diagnostic_last_packets
        events = self.total_events - self._diagnostic_last_events
        messages = (
            self._diagnostics.messages_published - self._diagnostic_last_messages
        )
        self._diagnostic_last_monotonic = now
        self._diagnostic_last_packets = self.total_packets
        self._diagnostic_last_events = self.total_events
        self._diagnostic_last_messages = self._diagnostics.messages_published

        def timing(name, samples):
            p50, p95, maximum = samples.summary_ms()
            return f"{name}_ms(p50/p95/max)={p50:.3f}/{p95:.3f}/{maximum:.3f}"

        requested_interval_ms = 1_000.0 / self.event_voxel_publish_fps
        self.get_logger().info(
            "EVENT DIAGNOSTICS | "
            f"input_packets_hz={packets / elapsed:.2f}, "
            f"input_events_hz={events / elapsed:.1f}, "
            f"internal_publish_hz={messages / elapsed:.2f}, "
            f"requested_interval_ms={requested_interval_ms:.3f}, "
            f"ticks={self._diagnostics.timer_ticks}, "
            f"published={self._diagnostics.messages_published}, "
            f"skipped_no_new={self._diagnostics.skipped_no_new_events}, "
            f"anchors_advanced={self._diagnostics.anchors_advanced}, "
            f"anchors_repeated={self._diagnostics.anchors_repeated}, "
            f"dropped_packets={self._diagnostics.packets_dropped_by_cap}, "
            f"dropped_events={self._diagnostics.events_dropped_by_cap}, "
            f"{timing('timer_interval', self._diagnostics.timer_interval)}, "
            f"{timing('buffer', self._diagnostics.buffer_processing)}, "
            f"{timing('voxel', self._diagnostics.voxel_build)}, "
            f"{timing('message', self._diagnostics.message_build)}, "
            f"{timing('publish_call', self._diagnostics.publish_call)}, "
            f"{timing('publish_interval', self._diagnostics.publish_interval)}"
        )

    def destroy_node(self):
        self._stop_event.set()

        if self.publish_timer is not None:
            self.publish_timer.cancel()
            self.destroy_timer(self.publish_timer)
            self.publish_timer = None
        if self.event_voxel_timer is not None:
            self.event_voxel_timer.cancel()
            self.destroy_timer(self.event_voxel_timer)
            self.event_voxel_timer = None
        if self.diagnostics_timer is not None:
            self.diagnostics_timer.cancel()
            self.destroy_timer(self.diagnostics_timer)
            self.diagnostics_timer = None
        if self.event_tracker_stats_timer is not None:
            self.event_tracker_stats_timer.cancel()
            self.destroy_timer(self.event_tracker_stats_timer)
            self.event_tracker_stats_timer = None
        if self.event_tracker_debug_timer is not None:
            self.event_tracker_debug_timer.cancel()
            self.destroy_timer(self.event_tracker_debug_timer)
            self.event_tracker_debug_timer = None

        with self._preview_lock:
            self.preview_buffer.clear()

        worker_alive = False
        if hasattr(self, "_reader_thread") and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)
            worker_alive = self._reader_thread.is_alive()
        if self._replay_reader is not None and not worker_alive:
            self._replay_reader.close()
        if worker_alive:
            self.get_logger().warn("Input worker did not stop within 2 seconds")

        with self._h5_lock:
            if self._recording_enabled:
                try:
                    closed_path = self._close_h5_locked()
                    self.get_logger().info(f"Closed raw event recording on shutdown: {closed_path}")
                except Exception as e:
                    self.get_logger().warn(f"Failed to close HDF5 on shutdown: {e}")

        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except Exception:
                pass

        if self.event_input_mode == "hardware":
            self.get_logger().info("Serial port closed")
        else:
            self.get_logger().info("HDF5 replay input closed")

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OpenMVEventCamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
