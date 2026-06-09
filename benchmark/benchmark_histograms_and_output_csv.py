# GenX320 histogram/event-frame benchmark (CSV output)
# Measures achieved frame rate and short-run stability.

import sensor
import image
import time

# ============================================================
# USER SETTINGS
# ============================================================

TARGET_FPS = 50              # try: 50, 100, 200, 375
DURATION_SEC = 30            # try: 30 or 180
SCENE_LABEL = "fast_motion"  # change manually per run

USE_PALETTE = False          # palette costs CPU, leave False for benchmarking
CONTRAST = 16                # default-ish event visibility scaling
BRIGHTNESS = 128             # neutral center value

# ============================================================
# SENSOR SETUP
# ============================================================

sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.B320X320)
sensor.set_framerate(TARGET_FPS)
sensor.set_contrast(CONTRAST)
sensor.set_brightness(BRIGHTNESS)

if USE_PALETTE:
    sensor.set_color_palette(image.PALETTE_EVT_DARK)

# Give sensor a moment to settle
time.sleep_ms(300)

clock = time.clock()

# ============================================================
# STATS
# ============================================================

start_ms = time.ticks_ms()
window_start_ms = start_ms

total_frames = 0
zero_frames = 0
current_zero_streak = 0
longest_zero_streak = 0

window_frames = 0
peak_fps_1s = 0.0

# Optional crude pixel-activity measure:
# count frames that are completely flat or almost flat.
low_activity_frames = 0

# ============================================================
# MAIN LOOP
# ============================================================

while time.ticks_diff(time.ticks_ms(), start_ms) < DURATION_SEC * 1000:
    clock.tick()
    img = sensor.snapshot()

    total_frames += 1
    window_frames += 1

    # A true "zero frame" is unlikely in histogram mode because the baseline is 128.
    # So instead, define low-activity by checking if image stats are near flat.
    st = img.get_statistics()
    # Very small spread means almost no visible event activity in the histogram
    if (st.max() - st.min()) <= 2:
        zero_frames += 1
        current_zero_streak += 1
        if current_zero_streak > longest_zero_streak:
            longest_zero_streak = current_zero_streak
    else:
        current_zero_streak = 0

    if (st.max() - st.min()) <= 5:
        low_activity_frames += 1

    now_ms = time.ticks_ms()
    if time.ticks_diff(now_ms, window_start_ms) >= 1000:
        dt_ms = time.ticks_diff(now_ms, window_start_ms)
        if dt_ms > 0:
            fps_win = (window_frames * 1000.0) / dt_ms
            if fps_win > peak_fps_1s:
                peak_fps_1s = fps_win
        window_start_ms = now_ms
        window_frames = 0

# ============================================================
# FINAL STATS
# ============================================================

end_ms = time.ticks_ms()
elapsed_s = time.ticks_diff(end_ms, start_ms) / 1000.0
avg_fps = total_frames / elapsed_s if elapsed_s > 0 else 0.0

frac_zero = zero_frames / total_frames if total_frames > 0 else 0.0
frac_low_activity = low_activity_frames / total_frames if total_frames > 0 else 0.0

# ============================================================
# CSV OUTPUT
# ============================================================

print("scene,target_fps,duration_s,total_frames,avg_fps,peak_fps_1s,zero_frames,frac_zero,longest_zero_streak,low_activity_frames,frac_low_activity,contrast,brightness,palette")

print("%s,%d,%d,%d,%.2f,%.2f,%d,%.3f,%d,%d,%.3f,%d,%d,%d" % (
    SCENE_LABEL,
    TARGET_FPS,
    DURATION_SEC,
    total_frames,
    avg_fps,
    peak_fps_1s,
    zero_frames,
    frac_zero,
    longest_zero_streak,
    low_activity_frames,
    frac_low_activity,
    CONTRAST,
    BRIGHTNESS,
    1 if USE_PALETTE else 0
))
