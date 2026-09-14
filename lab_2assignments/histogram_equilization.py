"""
histogram_equalization.py
--------------------------
Contrast enhancement via histogram equalization implemented from first
principles (manual PMF/CDF computation, no library HE call).
Outputs: input/equalized images, their histograms, and a 2x2 comparison grid.

Run:
    python histogram_equalization.py <input_image> <output_image>
"""

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def rgb_to_grey(img_arr: np.ndarray) -> np.ndarray:
    r, g, b = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)


def histogram_equalize(grey: np.ndarray) -> np.ndarray:
    """Classic global histogram equalization built from scratch."""
    L = 256
    hist = np.zeros(L, dtype=np.int64)
    for v in grey.ravel():
        hist[v] += 1

    total_pixels = grey.size
    pmf = hist / total_pixels

    cdf = np.cumsum(pmf)
    # Standard HE transform function: round((L-1) * cdf(v))
    transform = np.round((L - 1) * cdf).astype(np.uint8)

    equalized = transform[grey]
    return equalized, hist


def compute_hist(arr: np.ndarray):
    hist = np.zeros(256, dtype=np.int64)
    for v in arr.ravel():
        hist[v] += 1
    return hist


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "Input_images/sample.png"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "Output/histogram_equalization_grid.png"

    img = Image.open(in_path).convert("RGB")
    grey = rgb_to_grey(np.array(img))
    equalized, in_hist = histogram_equalize(grey)
    out_hist = compute_hist(equalized)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    axes[0, 0].imshow(grey, cmap="gray")
    axes[0, 0].set_title("Input (Greyscale)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(equalized, cmap="gray")
    axes[0, 1].set_title("Histogram Equalized")
    axes[0, 1].axis("off")

    axes[1, 0].bar(range(256), in_hist, color="steelblue", width=1)
    axes[1, 0].set_title("Input Histogram")
    axes[1, 0].set_xlabel("Intensity")
    axes[1, 0].set_ylabel("Frequency")

    axes[1, 1].bar(range(256), out_hist, color="darkorange", width=1)
    axes[1, 1].set_title("Equalized Histogram")
    axes[1, 1].set_xlabel("Intensity")
    axes[1, 1].set_ylabel("Frequency")

    plt.suptitle("Histogram Equalization (from first principles)", fontsize=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved grid to {out_path}")


if __name__ == "__main__":
    main()