from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.actions import LogInfo
from launch.substitutions import LaunchConfiguration

from fr3_teleop.config.teleop_config import OPENMV_PARAMS


def generate_launch_description():
    print(
        "[both_cams.launch] OpenMV topics: "
        f"mono={OPENMV_PARAMS['topic']}, "
        f"3ch={OPENMV_PARAMS['topic_3_channel']}, "
        "xyt=/openmv_cam/event_voxel"
    )

    event_frame_mode_arg = DeclareLaunchArgument(
        "event_frame_mode",
        default_value="shifted",
    )
    event_frame_ch0_ms_arg = DeclareLaunchArgument(
        "event_frame_ch0_ms",
        default_value="50.0",
    )
    event_frame_ch1_ms_arg = DeclareLaunchArgument(
        "event_frame_ch1_ms",
        default_value="100.0",
    )
    event_frame_ch2_ms_arg = DeclareLaunchArgument(
        "event_frame_ch2_ms",
        default_value="200.0",
    )
    event_scaling_arg = DeclareLaunchArgument(
        "event_scaling",
        default_value="signed_log1p_fixed_clip",
    )
    event_clip_count_arg = DeclareLaunchArgument(
        "event_clip_count",
        default_value="16.0",
    )
    event_packet_margin_ms_arg = DeclareLaunchArgument(
        "event_packet_margin_ms",
        default_value="50.0",
    )
    publish_xyt_voxel_arg = DeclareLaunchArgument(
        "publish_xyt_voxel",
        default_value="false",
    )
    topic_xyt_voxel_arg = DeclareLaunchArgument(
        "topic_xyt_voxel",
        default_value="/openmv_cam/event_voxel",
    )
    event_voxel_horizon_ms_arg = DeclareLaunchArgument(
        "event_voxel_horizon_ms",
        default_value="200.0",
    )
    event_voxel_temporal_bins_arg = DeclareLaunchArgument(
        "event_voxel_temporal_bins",
        default_value="9",
    )
    event_voxel_height_arg = DeclareLaunchArgument(
        "event_voxel_height",
        default_value="320",
    )
    event_voxel_width_arg = DeclareLaunchArgument(
        "event_voxel_width",
        default_value="320",
    )
    event_voxel_scaling_arg = DeclareLaunchArgument(
        "event_voxel_scaling",
        default_value="signed_log1p_fixed_clip",
    )
    event_voxel_clip_count_arg = DeclareLaunchArgument(
        "event_voxel_clip_count",
        default_value="16.0",
    )
    event_voxel_encoding_arg = DeclareLaunchArgument(
        "event_voxel_encoding",
        default_value="8UC9",
    )
    publish_event_voxel_1ms_arg = DeclareLaunchArgument(
        "publish_event_voxel_1ms", default_value="false"
    )
    topic_event_voxel_1ms_arg = DeclareLaunchArgument(
        "topic_event_voxel_1ms", default_value="/openmv_cam/event_voxel_1ms"
    )
    event_voxel_bin_ms_arg = DeclareLaunchArgument(
        "event_voxel_bin_ms", default_value="1.0"
    )
    event_voxel_activity_mode_arg = DeclareLaunchArgument(
        "event_voxel_activity_mode", default_value="absolute_activity"
    )
    event_voxel_publish_fps_arg = DeclareLaunchArgument(
        "event_voxel_publish_fps", default_value="30.0"
    )
    event_output_mode_arg = DeclareLaunchArgument(
        "event_output_mode",
        default_value="legacy_flags",
        description=(
            "Event image outputs: legacy_flags preserves publish flags; "
            "event_voxel_1ms, all, and none are explicit overrides"
        ),
    )
    event_diagnostics_enabled_arg = DeclareLaunchArgument(
        "event_diagnostics_enabled", default_value="false"
    )
    event_diagnostics_period_sec_arg = DeclareLaunchArgument(
        "event_diagnostics_period_sec", default_value="5.0"
    )
    tracker_defaults = {
        "event_tracker_enabled": "false",
        "event_tracker_position_topic": "/openmv_cam/event_tracker/ball_2d_px",
        "event_tracker_velocity_topic": "/openmv_cam/event_tracker/ball_velocity_px_s",
        "event_tracker_valid_topic": "/openmv_cam/event_tracker/valid",
        "event_tracker_bin_ms": "1.0",
        "event_tracker_history_limit_ms": "100.0",
        "event_tracker_activity_threshold": "1",
        "event_tracker_min_event_count": "3",
        "event_tracker_min_blob_area_px": "2",
        "event_tracker_max_blob_area_px": "500",
        "event_tracker_morphology_kernel": "0",
        "event_tracker_morphology_iterations": "0",
        "event_tracker_use_circularity": "false",
        "event_tracker_min_circularity": "0.1",
        "event_tracker_max_jump_px": "100.0",
        "event_tracker_velocity_history_size": "5",
        "event_tracker_velocity_min_span_ms": "3.0",
        "event_tracker_stats_period_sec": "5.0",
        "publish_latency_traces": "false",
        "latency_trace_topic": "/intercept_trace/event_2d_ball_detection",
        "latency_trace_run_id": "",
    }
    tracker_args = [
        DeclareLaunchArgument(name, default_value=default)
        for name, default in tracker_defaults.items()
    ]

    openmv_params = dict(OPENMV_PARAMS)
    openmv_params.update({
        "event_frame_mode": LaunchConfiguration("event_frame_mode"),
        "event_frame_ch0_ms": LaunchConfiguration("event_frame_ch0_ms"),
        "event_frame_ch1_ms": LaunchConfiguration("event_frame_ch1_ms"),
        "event_frame_ch2_ms": LaunchConfiguration("event_frame_ch2_ms"),
        "event_scaling": LaunchConfiguration("event_scaling"),
        "event_clip_count": LaunchConfiguration("event_clip_count"),
        "event_packet_margin_ms": LaunchConfiguration("event_packet_margin_ms"),
        "publish_xyt_voxel": LaunchConfiguration("publish_xyt_voxel"),
        "topic_xyt_voxel": LaunchConfiguration("topic_xyt_voxel"),
        "event_voxel_horizon_ms": LaunchConfiguration("event_voxel_horizon_ms"),
        "event_voxel_temporal_bins": LaunchConfiguration("event_voxel_temporal_bins"),
        "event_voxel_height": LaunchConfiguration("event_voxel_height"),
        "event_voxel_width": LaunchConfiguration("event_voxel_width"),
        "event_voxel_scaling": LaunchConfiguration("event_voxel_scaling"),
        "event_voxel_clip_count": LaunchConfiguration("event_voxel_clip_count"),
        "event_voxel_encoding": LaunchConfiguration("event_voxel_encoding"),
        "publish_event_voxel_1ms": LaunchConfiguration("publish_event_voxel_1ms"),
        "topic_event_voxel_1ms": LaunchConfiguration("topic_event_voxel_1ms"),
        "event_voxel_bin_ms": LaunchConfiguration("event_voxel_bin_ms"),
        "event_voxel_activity_mode": LaunchConfiguration("event_voxel_activity_mode"),
        "event_voxel_publish_fps": LaunchConfiguration("event_voxel_publish_fps"),
        "event_output_mode": LaunchConfiguration("event_output_mode"),
        "event_diagnostics_enabled": LaunchConfiguration(
            "event_diagnostics_enabled"
        ),
        "event_diagnostics_period_sec": LaunchConfiguration(
            "event_diagnostics_period_sec"
        ),
    })
    openmv_params.update({
        name: LaunchConfiguration(name) for name in tracker_defaults
    })

    xyt_log = LogInfo(msg=[
        "OpenMV XYT voxel: enabled=", LaunchConfiguration("publish_xyt_voxel"),
        ", topic=", LaunchConfiguration("topic_xyt_voxel"),
        ", horizon_ms=", LaunchConfiguration("event_voxel_horizon_ms"),
        ", bins=", LaunchConfiguration("event_voxel_temporal_bins"),
        ", height=", LaunchConfiguration("event_voxel_height"),
        ", width=", LaunchConfiguration("event_voxel_width"),
        ", scaling=", LaunchConfiguration("event_voxel_scaling"),
        ", clip_count=", LaunchConfiguration("event_voxel_clip_count"),
        ", encoding=", LaunchConfiguration("event_voxel_encoding"),
    ])
    activity_log = LogInfo(msg=[
        "OpenMV native activity voxel: enabled=",
        LaunchConfiguration("publish_event_voxel_1ms"),
        ", topic=", LaunchConfiguration("topic_event_voxel_1ms"),
        ", bin_ms=", LaunchConfiguration("event_voxel_bin_ms"),
        ", bins=", LaunchConfiguration("event_voxel_temporal_bins"),
        ", mode=", LaunchConfiguration("event_voxel_activity_mode"),
        ", clip_count=", LaunchConfiguration("event_voxel_clip_count"),
        ", publish_fps=", LaunchConfiguration("event_voxel_publish_fps"),
        ", output_mode=", LaunchConfiguration("event_output_mode"),
    ])

    openmv_node = Node(
        package='openmv_cam',
        executable='openmv_cam_node',
        name='openmv_cam',
        output='screen',
        parameters=[openmv_params],
    )

    return LaunchDescription([
        event_frame_mode_arg,
        event_frame_ch0_ms_arg,
        event_frame_ch1_ms_arg,
        event_frame_ch2_ms_arg,
        event_scaling_arg,
        event_clip_count_arg,
        event_packet_margin_ms_arg,
        publish_xyt_voxel_arg,
        topic_xyt_voxel_arg,
        event_voxel_horizon_ms_arg,
        event_voxel_temporal_bins_arg,
        event_voxel_height_arg,
        event_voxel_width_arg,
        event_voxel_scaling_arg,
        event_voxel_clip_count_arg,
        event_voxel_encoding_arg,
        publish_event_voxel_1ms_arg,
        topic_event_voxel_1ms_arg,
        event_voxel_bin_ms_arg,
        event_voxel_activity_mode_arg,
        event_voxel_publish_fps_arg,
        event_output_mode_arg,
        event_diagnostics_enabled_arg,
        event_diagnostics_period_sec_arg,
        *tracker_args,
        xyt_log,
        activity_log,
        openmv_node,
    ])
