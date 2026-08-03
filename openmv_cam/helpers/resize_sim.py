import cv2
import numpy as np

INPUT_PATH = "8bit_afk.png"

img = cv2.imread(INPUT_PATH, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise RuntimeError("Failed to load image")

h, w = img.shape

# --- Method 1: subsampling (embedded-like) ---
img_subsample = img[::2, ::2]

# --- Method 2: resize (INTER_NEAREST = closest equivalent) ---
img_resize_nn = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_NEAREST)

# --- Method 3: resize (default / bilinear) ---
img_resize_linear = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_LINEAR)

# --- stack for comparison ---
# resize all to same size for display
img_subsample_vis = cv2.resize(img_subsample, (w, h), interpolation=cv2.INTER_NEAREST)
img_resize_nn_vis = cv2.resize(img_resize_nn, (w, h), interpolation=cv2.INTER_NEAREST)
img_resize_linear_vis = cv2.resize(img_resize_linear, (w, h), interpolation=cv2.INTER_NEAREST)

triple = np.hstack((img_subsample_vis, img_resize_nn_vis, img_resize_linear_vis))

cv2.imshow("subsample | resize_nearest | resize_linear", triple)
cv2.waitKey(0)
cv2.destroyAllWindows()