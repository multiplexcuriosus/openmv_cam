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
        xyt_log,
        openmv_node,
    ])
