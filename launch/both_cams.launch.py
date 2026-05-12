from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # === RealSense launch file path ===
    realsense_launch_path = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch',
        'rs_launch.py'
    )

    # === OpenMV node ===
    openmv_node = Node(
        package='openmv_cam',
        executable='openmv_cam_node',
        name='openmv_cam',
        output='screen',
        parameters=[
            {'port': '/dev/openmvcam'},   # adjust if needed
            {'baud': 115200},            # match your setup
            {'fps': 30}                  # optional
        ]
    )

    # === RealSense (depth + IR disabled) ===
    realsense_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch_path),
        launch_arguments={
            'enable_depth': 'false',
            'enable_infra1': 'false',
            'enable_infra2': 'false',
            'pointcloud.enable': 'false',
            'align_depth.enable': 'false',

            # RGB camera settings
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