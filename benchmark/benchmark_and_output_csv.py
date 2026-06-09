# GenX320 raw event benchmark (CSV output version)

import csi
import time
import image

from ulab import numpy as np

# ============================================================
# USER SETTINGS
# ============================================================

BUF_SIZE = 8192
DURATION_SEC = 30
SCENE_LABEL = "hand_motion"   # change manually per run

# ============================================================
# SETUP
# ============================================================

events = np.zeros((BUF_SIZE, 6), dtype=np.uint16)

csi0 = csi.CSI(cid=csi.GENX320)
csi0.reset()
csi0.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, events.shape[0])

time.sleep_ms(200)

# ============================================================
# HELPERS
# ============================================================


def ts_tuple(ev_row):
    return int(ev_row[1]), int(ev_row[2]), int(ev_row[3])


def tuple_lt(a, b):
    if a[0] != b[0]:
        return a[0] < b[0]
    if a[1] != b[1]:
        return a[1] < b[1]
    return a[2] < b[2]

# ============================================================
# STATS
# ============================================================


start_ms = time.ticks_ms()

total_events = 0
total_reads = 0
zero_reads = 0
negative_error_reads = 0
full_buffer_reads = 0

longest_zero_streak = 0
current_zero_streak = 0

timestamp_pairs_checked = 0
non_monotonic_timestamps = 0

window_start_ms = start_ms
window_events = 0
peak_eps_1s = 0.0

last_global_ts = None


# ============================================================
# MAIN LOOP
# ============================================================

while time.ticks_diff(time.ticks_ms(), start_ms) < DURATION_SEC * 1000:
    event_count = csi0.ioctl(csi.IOCTL_GENX320_READ_EVENTS, events)
    total_reads += 1

    if event_count < 0:
        negative_error_reads += 1
        continue

    if event_count == 0:
        zero_reads += 1
        current_zero_streak += 1
        if current_zero_streak > longest_zero_streak:
            longest_zero_streak = current_zero_streak
    else:
        current_zero_streak = 0

    if event_count == BUF_SIZE:
        full_buffer_reads += 1

    total_events += event_count
    window_events += event_count

    # Timestamp checks
    if event_count > 0:
        first_ts = ts_tuple(events[0])

        if last_global_ts is not None:
            timestamp_pairs_checked += 1
            if tuple_lt(first_ts, last_global_ts):
                non_monotonic_timestamps += 1

        prev_ts = first_ts
        for i in range(1, event_count):
            curr_ts = ts_tuple(events[i])
            timestamp_pairs_checked += 1
            if tuple_lt(curr_ts, prev_ts):
                non_monotonic_timestamps += 1
            prev_ts = curr_ts

        last_global_ts = ts_tuple(events[event_count - 1])

    # 1-second window EPS
    now_ms = time.ticks_ms()
    if time.ticks_diff(now_ms, window_start_ms) >= 1000:
        dt_ms = time.ticks_diff(now_ms, window_start_ms)
        if dt_ms > 0:
            eps = (window_events * 1000.0) / dt_ms
            if eps > peak_eps_1s:
                peak_eps_1s = eps

        window_start_ms = now_ms
        window_events = 0

# ============================================================
# FINAL STATS
# ============================================================

end_ms = time.ticks_ms()
elapsed_s = time.ticks_diff(end_ms, start_ms) / 1000.0
avg_eps = total_events / elapsed_s if elapsed_s > 0 else 0.0

frac_full = full_buffer_reads / total_reads if total_reads > 0 else 0.0
frac_zero = zero_reads / total_reads if total_reads > 0 else 0.0

# ============================================================
# CSV OUTPUT
# ============================================================

# Print header (only once ideally)
print("scene,buf_size,duration_s,total_events,avg_eps,peak_eps_1s,total_reads, \
zero_reads,frac_zero,full_buffer_reads,frac_full,longest_zero_streak,nonmono_ts,neg_reads")

# Print result row
print("%s,%d,%d,%d,%.1f,%.1f,%d,%d,%.3f,%d,%.3f,%d,%d,%d" % (
    SCENE_LABEL,
    BUF_SIZE,
    DURATION_SEC,
    total_events,
    avg_eps,
    peak_eps_1s,
    total_reads,
    zero_reads,
    frac_zero,
    full_buffer_reads,
    frac_full,
    longest_zero_streak,
    non_monotonic_timestamps,
    negative_error_reads
))
