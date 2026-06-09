import cv2
import numpy as np

# ---- config ----
INPUT_PATH = "8bit_afk.png"
OUTPUT_PATH = "quantized_1_to_8_bits.png"

# ---- load image (grayscale, 8-bit) ----
img = cv2.imread(INPUT_PATH, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise RuntimeError(f"Failed to load image from '{INPUT_PATH}'.")

if img.dtype != np.uint8:
    raise RuntimeError(f"Expected uint8 image, got {img.dtype}.")

# ---- helper: quantize to n bits and rescale back to 0..255 for display ----
def quantize_to_n_bits(img_8bit: np.ndarray, bits: int) -> np.ndarray:
    if not (1 <= bits <= 8):
        raise ValueError("bits must be in [1, 8]")

    shift = 8 - bits
    img_nbit = img_8bit >> shift  # values in [0, 2^bits - 1]

    max_val = (1 << bits) - 1
    if max_val == 0:
        raise RuntimeError("Unexpected max_val = 0")

    # rescale to 0..255 for visualization
    img_vis = ((img_nbit.astype(np.float32) / max_val) * 255.0).round().astype(np.uint8)
    return img_vis

# ---- create labeled versions for 1..8 bits ----
labeled_imgs = []

for bits in range(1, 9):
    qimg = quantize_to_n_bits(img, bits)

    # convert to BGR so colored text is possible
    qimg_bgr = cv2.cvtColor(qimg, cv2.COLOR_GRAY2BGR)

    label = f"{bits}-bit"
    cv2.putText(
        qimg_bgr,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    labeled_imgs.append(qimg_bgr)

# ---- arrange into a 2x4 grid ----
top_row = np.hstack(labeled_imgs[:4])     # 1,2,3,4 bit
bottom_row = np.hstack(labeled_imgs[4:])  # 5,6,7,8 bit
grid = np.vstack((top_row, bottom_row))

# ---- show and save ----
cv2.imshow("Quantized images: 1 to 8 bits", grid)
cv2.imwrite(OUTPUT_PATH, grid)

print(f"Saved: {OUTPUT_PATH}")

cv2.waitKey(0)
cv2.destroyAllWindows()