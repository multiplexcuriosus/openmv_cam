import ast
from pathlib import Path

import pytest

from openmv_cam.event_output_mode import resolve_event_outputs


LEGACY_LAUNCH_DEFAULTS = {
    "event_frame_mode": "shifted",
    "event_frame_ch0_ms": "50.0",
    "event_frame_ch1_ms": "100.0",
    "event_frame_ch2_ms": "200.0",
    "event_scaling": "signed_log1p_fixed_clip",
    "event_clip_count": "16.0",
    "event_packet_margin_ms": "50.0",
    "publish_xyt_voxel": "false",
    "topic_xyt_voxel": "/openmv_cam/event_voxel",
    "event_voxel_horizon_ms": "200.0",
    "event_voxel_temporal_bins": "9",
    "event_voxel_height": "320",
    "event_voxel_width": "320",
    "event_voxel_scaling": "signed_log1p_fixed_clip",
    "event_voxel_clip_count": "16.0",
    "event_voxel_encoding": "8UC9",
}


def _declared_defaults(path):
    tree = ast.parse(Path(path).read_text())
    defaults = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Name) or function.id != "DeclareLaunchArgument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        default = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "default_value"),
            None,
        )
        if isinstance(default, ast.Constant):
            defaults[node.args[0].value] = str(default.value)
    return defaults


def test_all_legacy_openmv_launch_arguments_and_defaults_are_unchanged():
    launch_path = Path(__file__).parents[1] / "launch" / "openmv.launch.py"
    defaults = _declared_defaults(launch_path)
    for name, expected in LEGACY_LAUNCH_DEFAULTS.items():
        assert defaults[name] == expected


def test_default_mode_delegates_to_every_existing_publish_flag():
    outputs = resolve_event_outputs(
        "legacy_flags",
        publish_3_channel_img=False,
        publish_xyt_voxel=True,
        publish_event_voxel_1ms=False,
    )
    assert outputs.mono
    assert not outputs.event_frame_3ch
    assert outputs.legacy_voxel
    assert not outputs.event_voxel_1ms


def test_detector_only_mode_activates_exactly_one_event_representation():
    outputs = resolve_event_outputs(
        "event_voxel_1ms",
        publish_3_channel_img=True,
        publish_xyt_voxel=True,
        publish_event_voxel_1ms=False,
    )
    assert sum(vars(outputs).values()) == 1
    assert outputs.event_voxel_1ms


def test_all_and_none_modes_are_explicit_and_invalid_modes_fail():
    flags = dict(
        publish_3_channel_img=False,
        publish_xyt_voxel=False,
        publish_event_voxel_1ms=False,
    )
    assert all(vars(resolve_event_outputs("all", **flags)).values())
    assert not any(vars(resolve_event_outputs("none", **flags)).values())
    with pytest.raises(ValueError, match="Unsupported event_output_mode"):
        resolve_event_outputs("shifted", **flags)


def test_vision_launch_forwards_detector_mode_when_neighbor_checkout_exists():
    vision_path = Path(__file__).parents[2] / "fr3_teleop" / "launch" / "vision.launch.py"
    if not vision_path.exists():
        pytest.skip("fr3_teleop checkout is not part of this package test environment")
    source = vision_path.read_text()
    assert 'DeclareLaunchArgument(\n        "event_output_mode"' in source
    assert '"event_output_mode": event_output_mode' in source
