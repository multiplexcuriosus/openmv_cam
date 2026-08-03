from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from fr3_teleop.config.teleop_config import OPENMV_PARAMS


def generate_launch_description():
    print(
        "[both_cams.launch] OpenMV topics: "
        f"mono={OPENMV_PARAMS['topic']}, "
        f"3ch={OPENMV_PARAMS['topic_3_channel']}"
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

    openmv_params = dict(OPENMV_PARAMS)
    openmv_params.update({
        "event_frame_mode": LaunchConfiguration("event_frame_mode"),
        "event_frame_ch0_ms": LaunchConfiguration("event_frame_ch0_ms"),
        "event_frame_ch1_ms": LaunchConfiguration("event_frame_ch1_ms"),
        "event_frame_ch2_ms": LaunchConfiguration("event_frame_ch2_ms"),
        "event_scaling": LaunchConfiguration("event_scaling"),
        "event_clip_count": LaunchConfiguration("event_clip_count"),
        "event_packet_margin_ms": LaunchConfiguration("event_packet_margin_ms"),
    })

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
        openmv_node,
    ])

