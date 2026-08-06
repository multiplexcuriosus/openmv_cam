# OpenMV event camera ROS node

## Native 1 ms activity voxel

`/openmv_cam/event_voxel_1ms` is a `sensor_msgs/msg/Image` intended as the
frontend for a downstream classical event blob detector. It is disabled by
default. This is a rolling voxel anchored to the newest sensor event available
at a publication tick; it is not a lossless event-time-binned stream. Multiple
sensor-time bins can arrive between publication ticks and are represented only
if they remain inside the configured rolling depth. Raw HDF5 recording remains
the lossless representation.

Every channel is exactly `event_voxel_bin_ms` wide (1.0 ms by default). The HWC
channel order is oldest to newest, with right-closed intervals. For anchor `A`,
depth `N`, and bin width `B`, the represented sensor-time interval is
`(A - N*B, A]`. Events are selected and binned by the reconstructed EVT1 sensor
timestamp, never by serial packet boundaries, and future events are excluded.

With one temporal bin the message is 320x320 `8UC1` (`step=320`). With multiple
bins it is 320x320 `8UCN`, where channel N is temporal, and `step=320*N`.
Absolute activity is the default: both event types increment the same pixel
count, the background is zero, and the deterministic mapping is
`round(255 * min(count, clip_count) / clip_count)`. `signed_activity` is an
optional cancellation/debug mode; its absolute signed accumulation is scaled
with the same fixed rule. There is no per-frame normalization.

The output uses native OpenMV coordinates: origin at the image's top-left,
positive x to the right, positive y downward, 0 <= x,y < 320. It is never
rotated and always has `frame_id="openmv_cam"`; the existing rotation services
continue to affect the legacy preview/3-channel/9-bin outputs only. Consumers
should use these native coordinates, not RGB-calibrated or projected ones.

The header stamp is the host ROS receive timestamp of the serial packet that
contains the anchor event, because EVT1 does not currently provide a reliable
sensor-clock-to-ROS-clock mapping. A timer tick publishes only after a new
non-empty packet has entered the shared buffer. With no new events, no stale
non-empty voxel or falsely fresh timestamp is published. Small timestamp
disorder can amend and republish a rolling voxel with the same anchor/stamp.

Launch example:

```bash
ros2 launch openmv_cam openmv.launch.py \
  publish_event_voxel_1ms:=true \
  event_voxel_bin_ms:=1.0 \
  event_voxel_temporal_bins:=1 \
  event_voxel_publish_fps:=30.0
```

The equivalent detector-only selector is:

```bash
ros2 launch openmv_cam openmv.launch.py \
  event_output_mode:=event_voxel_1ms \
  event_voxel_bin_ms:=1.0 \
  event_voxel_temporal_bins:=1 \
  event_voxel_publish_fps:=30.0 \
  event_diagnostics_enabled:=true
```

`fr3_teleop/launch/vision.launch.py` forwards these detector settings to the
same included OpenMV node. `event_output_mode=legacy_flags` is the default and
preserves historical precedence: the existing publish flags retain their exact
meaning. Explicit `event_voxel_1ms`, `all`, or `none` modes override event-image
flags only. Detector-only mode publishes only the native activity voxel; `all`
enables every event image; `none` disables event images. Services, serial input,
and raw HDF5 recording are unaffected.

Parameters and defaults:

- `publish_event_voxel_1ms=false`
- `topic_event_voxel_1ms=/openmv_cam/event_voxel_1ms`
- `event_voxel_bin_ms=1.0`
- `event_voxel_temporal_bins=9` (set to 1 for a single detector slice; the
  legacy `/openmv_cam/event_voxel` remains fixed at its compatible 9 channels)
- `event_voxel_activity_mode=absolute_activity`
- `event_voxel_clip_count=16.0`
- `event_voxel_publish_fps=30.0`
- `event_output_mode=legacy_flags`
- `event_diagnostics_enabled=false`
- `event_diagnostics_period_sec=5.0`

The output bandwidth is approximately `320*320*N*fps` bytes/s before ROS
overhead (3.1 MB/s for one bin at 30 Hz). The shared packet buffer has a safety
cap (`max_event_frame_packets`), so an undersized cap or host-arrival retention
can make a live rolling voxel incomplete; the node warns when the cap truncates
history. HDF5 recording is unaffected and should be used when lossless capture
is required.

When diagnostics are enabled, one throttled summary reports serial packet/event
rates, requested and actual timer intervals, timer/publish/skip counts,
safety-cap drops, anchor advancement, and rolling p50/p95/max timings for
buffer processing, voxel construction, message construction, and the ROS
publish call.

Headless topic validation:

```bash
python3 scripts/validate_event_voxel.py \
  --topic /openmv_cam/event_voxel_1ms \
  --duration 20 \
  --output-dir /tmp/event_voxel_validation \
  --save-every 10 \
  --save-npz
```

This writes `summary.json`, `frames.csv`, selected PNGs, and optionally
`frames.npz`. It reports arrival/header rates, header-to-arrival delay, layout,
activity and bounding boxes, empty frames, repeated payloads, stale non-empty
frames, and repeated or non-monotonic timestamps.
