from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

from fr3_teleop.config.teleop_config import OPENMV_PARAMS


def generate_launch_description():
    realsense_launch_path = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch',
        'rs_launch.py'
    )

    print(
        "[both_cams.launch] OpenMV topics: "
        f"mono={OPENMV_PARAMS['topic']}, "
        f"3ch={OPENMV_PARAMS['topic_3_channel']}"
    )

    openmv_node = Node(
        package='openmv_cam',
        executable='openmv_cam_node',
        name='openmv_cam',
        output='screen',
        parameters=[OPENMV_PARAMS]
    )

    realsense_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch_path),
        launch_arguments={
            'enable_depth': 'false',
            'enable_infra1': 'false',
            'enable_infra2': 'false',
            'pointcloud.enable': 'false',
            'align_depth.enable': 'false',
            'rgb_camera.enable_auto_exposure': 'true',
            'rgb_camera.exposure': '3000',
            'rgb_camera.gain': '64',
            'rgb_camera.auto_exposure_priority': 'false',
            'rgb_camera.enable_auto_white_balance': 'false',
            'rgb_camera.white_balance': '4500',
        }.items()
    )

    return LaunchDescription([
        openmv_node,
        realsense_node
    ])
