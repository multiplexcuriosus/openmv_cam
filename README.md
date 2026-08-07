# OpenMV event camera ROS node

## Raw-packet event ball tracker MVP

The optional event tracker detects a high-activity ball-like blob directly in
validated EVT1 packets. Its path is `EVT1 serial -> EventPacket -> sensor-time
1 ms bins -> activity map -> blob/velocity estimate`. It runs in the serial
reader path: it does not subscribe to `/openmv_cam/event_voxel_1ms` and does not
wait for the 30 Hz image timer. The same timestamp-ordered `EventPacket` is
shared with preview buffering and HDF5 recording, so EVT1 timestamps are
reconstructed once. Wire order is retained for compatibility; packet-local
timestamp disorder is handled explicitly by sensor-time bin assignment.

For each complete half-open sensor-time interval `[start_us, end_us)`, both
polarities increment a dense 320x320 count map. OpenCV contours are filtered by
area and event count, and their center is weighted by per-pixel event count.
The first selection uses highest event count; later selections prefer the
previous/predicted position and enforce a maximum jump. Velocity is a
least-squares line fit over a bounded detection history. Circularity is
available but off by default because fast balls commonly form arcs. There is
no Kalman filter.

Outputs (only for valid detections) are:

- `/openmv_cam/event_tracker/ball_2d_px` (`geometry_msgs/msg/PointStamped`)
- `/openmv_cam/event_tracker/ball_velocity_px_s`
  (`geometry_msgs/msg/Vector3Stamped`; z is scalar speed)
- `/openmv_cam/event_tracker/valid` (`std_msgs/msg/Bool`)

Coordinates are native OpenMV/GENX320 pixels: 320x320, top-left origin, x right,
y down, and `frame_id=openmv_cam`. Tracker coordinates are never rotated.
Message header stamps are the host ROS publication time. Sensor microseconds
are used for binning and velocity only; they are not ROS epoch timestamps.

The tracker defaults to disabled. Important parameters and defaults are:

- `event_tracker_enabled=false`
- `event_tracker_position_topic=/openmv_cam/event_tracker/ball_2d_px`
- `event_tracker_velocity_topic=/openmv_cam/event_tracker/ball_velocity_px_s`
- `event_tracker_valid_topic=/openmv_cam/event_tracker/valid`
- `event_tracker_bin_ms=1.0`, `event_tracker_history_limit_ms=100.0`
- `event_tracker_activity_threshold=1`
- `event_tracker_min_event_count=3`
- `event_tracker_min_blob_area_px=2`, `event_tracker_max_blob_area_px=500`
- `event_tracker_morphology_kernel=0`,
  `event_tracker_morphology_iterations=0`
- `event_tracker_use_circularity=false`, `event_tracker_min_circularity=0.1`
- `event_tracker_max_jump_px=100.0`
- `event_tracker_velocity_history_size=5`
- `event_tracker_velocity_min_span_ms=3.0`
- `event_tracker_stats_period_sec=5.0`

Example:

```bash
ros2 launch openmv_cam openmv.launch.py event_tracker_enabled:=true
```

Optional latency traces use best-effort QoS on
`/intercept_trace/event_2d_ball_detection` when
`publish_latency_traces=true`. `input` traces correspond one-to-one with
validated packets; `complete` traces correspond to every processed bin and
carry packet lineage in `parent_sequence`. Both use stage
`event_2d_ball_detection` and modality `event`. ROS timestamp fields contain
host-clock values. GENX320 bin timestamps and finite detection/blob values are
kept in `detail_json`. Consequently source-to-output latency begins when a
complete EVT1 packet is available on the PC, not at physical sensor exposure.
If `intercept_latency_monitor` is not installed, tracing logs an error and is
disabled without stopping the reader. Configure with:

- `publish_latency_traces=false`
- `latency_trace_topic=/intercept_trace/event_2d_ball_detection`
- `latency_trace_run_id=""`

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
