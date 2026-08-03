#!/usr/bin/env bash
set -euo pipefail

PY_SCRIPT="${1:-./genx320_receiver.py}"
OUT_DIR="${2:-./tuning_out}"

PORT="${PORT:-/dev/openmvcam}"
BAUD="${BAUD:-115200}"

mkdir -p "$OUT_DIR"

# Reasonable search space.
DECAYS=(0.1 0.5 0.9)
STEPS=(0.40 0.70 1.00)
CONTRASTS=(32)
WINDOWS_MS=(30 50 100 150)
BLURS=(0 3)
SORTS=(0 1)


FPS=30
RESIZE=2
RUN_SECONDS=3.0
MAX_PACKETS=10

total=0
for decay in "${DECAYS[@]}"; do
  for step in "${STEPS[@]}"; do
    for contrast in "${CONTRASTS[@]}"; do
      for window_ms in "${WINDOWS_MS[@]}"; do
        for blur in "${BLURS[@]}"; do
          for sort in "${SORTS[@]}"; do
            total=$((total + 1))
          done
        done
      done
    done
  done
done

echo "[INFO] total combinations: $total"

i=0
for decay in "${DECAYS[@]}"; do
  for step in "${STEPS[@]}"; do
    for contrast in "${CONTRASTS[@]}"; do
      for window_ms in "${WINDOWS_MS[@]}"; do
        for blur in "${BLURS[@]}"; do
          for sort in "${SORTS[@]}"; do
            i=$((i + 1))
            echo
            echo "[INFO] combo $i / $total"
            echo "       decay=$decay step=$step contrast=$contrast window_ms=$window_ms blur=$blur sort=$sort"

            cmd=(
              python3 "$PY_SCRIPT"
              --port "$PORT"
              --baud "$BAUD"
              --fps "$FPS"
              --window-ms "$window_ms"
              --resize "$RESIZE"
              --max-preview-packets "$MAX_PACKETS"
              --decay "$decay"
              --step "$step"
              --contrast "$contrast"
              --blur "$blur"
              --tune-save-dir "$OUT_DIR"
              --tune-run-seconds "$RUN_SECONDS"
              --tune-prefix "preview"
            )

            if [[ "$sort" == "1" ]]; then
              cmd+=(--sort-ts)
            fi

            "${cmd[@]}"
          done
        done
      done
    done
  done
done

echo
echo "[INFO] finished. images are in $OUT_DIR"