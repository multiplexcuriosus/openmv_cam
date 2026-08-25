from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_typed_update_interface_and_topic_are_wired_without_legacy_changes():
    message = (ROOT / "msg" / "EventTrackerUpdate.msg").read_text()
    node = (ROOT / "openmv_cam" / "openmv_cam_node.py").read_text()
    launch = (ROOT / "launch" / "openmv.launch.py").read_text()
    for field in (
        "tracker_update_id", "source_packet_id", "availability_timestamp_ns",
        "sensor_window_start_us", "sensor_window_end_us", "rejection_reason",
            "window_event_count", "velocity_valid"):
        assert field in message
    assert '"/openmv_cam/event_tracker/update"' in node
    assert '"/openmv_cam/event_tracker/update"' in launch
    assert "self.event_tracker_update_pub.publish(update)" in node
    assert "if detection.valid:" in node
    assert "PointStamped()" in node and "Vector3Stamped()" in node
    assert "Bool()" in node


def test_invalid_update_is_published_before_legacy_valid_only_branch():
    node = (ROOT / "openmv_cam" / "openmv_cam_node.py").read_text()
    typed_publish = node.index("self.event_tracker_update_pub.publish(update)")
    legacy_branch = node.index("if detection.valid:", typed_publish)
    assert typed_publish < legacy_branch
