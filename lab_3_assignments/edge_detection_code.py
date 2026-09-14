"""
edge_detection.py
-------------------
First-order (Forward, Backward, Central Difference) and Second-order
(LoG, DoG) edge detection implemented from scratch (manual convolution),
including zero-crossing detection for the second-order operators.

Run:
    python edge_detection.py <input_image> <output_image>
"""

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def rgb_to_grey(img_arr: np.ndarray) -> np.ndarray:
    r, g, b = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float64)


def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Manual 2D convolution (correlation, standard in image processing) with zero padding."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(img, dtype=np.float64)

    # Vectorized sliding-window sum via stride tricks
    h, w = img.shape
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    out = np.tensordot(windows, kernel, axes=([2, 3], [0, 1]))
    return out


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def log_kernel(size: int, sigma: float) -> np.ndarray:
    """Laplacian of Gaussian kernel."""
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    r2 = xx**2 + yy**2
    factor = -1 / (np.pi * sigma**4)
    kernel = factor * (1 - r2 / (2 * sigma**2)) * np.exp(-r2 / (2 * sigma**2))
    kernel -= kernel.mean()  # zero-sum so flat regions -> 0
    return kernel


def zero_crossings(response: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Detect zero crossings: sign changes between a pixel and its neighbours."""
    h, w = response.shape
    zc = np.zeros((h, w), dtype=np.uint8)
    padded = np.pad(response, 1, mode="edge")

    center = padded[1:-1, 1:-1]
    neighbours = [
        padded[0:-2, 1:-1], padded[2:, 1:-1],   # up, down
        padded[1:-1, 0:-2], padded[1:-1, 2:],   # left, right
        padded[0:-2, 0:-2], padded[0:-2, 2:],   # diagonals
        padded[2:, 0:-2], padded[2:, 2:],
    ]
    for nb in neighbours:
        sign_change = (center * nb) < 0
        magnitude_ok = np.abs(center - nb) > threshold
        zc |= (sign_change & magnitude_ok).astype(np.uint8)
    return zc * 255


def normalize(arr: np.ndarray) -> np.ndarray:
    a = arr - arr.min()
    if a.max() > 0:
        a = a / a.max()
    return (a * 255).astype(np.uint8)


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "Input_images/sample.png"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "Output/edge_detection_grid.png"

    img = Image.open(in_path).convert("RGB")
    grey = rgb_to_grey(np.array(img))

    # ---- First order ----
    fwd_x = np.zeros_like(grey)
    fwd_x[:, :-1] = grey[:, 1:] - grey[:, :-1]
    fwd_y = np.zeros_like(grey)
    fwd_y[:-1, :] = grey[1:, :] - grey[:-1, :]
    forward_diff = np.hypot(fwd_x, fwd_y)

    bwd_x = np.zeros_like(grey)
    bwd_x[:, 1:] = grey[:, 1:] - grey[:, :-1]
    bwd_y = np.zeros_like(grey)
    bwd_y[1:, :] = grey[1:, :] - grey[:-1, :]
    backward_diff = np.hypot(bwd_x, bwd_y)

    central_x = np.zeros_like(grey)
    central_x[:, 1:-1] = (grey[:, 2:] - grey[:, :-2]) / 2
    central_y = np.zeros_like(grey)
    central_y[1:-1, :] = (grey[2:, :] - grey[:-2, :]) / 2
    central_diff = np.hypot(central_x, central_y)

    # ---- Second order: LoG ----
    log_k = log_kernel(13, sigma=2.0)
    log_response = convolve2d(grey, log_k)
    log_zc = zero_crossings(log_response, threshold=3.0)

    # ---- Second order: DoG (difference of Gaussians, approximates LoG) ----
    g1 = convolve2d(grey, gaussian_kernel(13, sigma=1.6))
    g2 = convolve2d(grey, gaussian_kernel(13, sigma=2.6))
    dog_response = g1 - g2
    dog_zc = zero_crossings(dog_response, threshold=5.0)

    results = [
        ("Original", grey, "gray"),
        ("Forward Difference", forward_diff, "gray"),
        ("Backward Difference", backward_diff, "gray"),
        ("Central Difference", central_diff, "gray"),
        ("LoG Response", log_response, "gray"),
        ("LoG Zero-Crossings", log_zc, "gray"),
        ("DoG Response", dog_response, "gray"),
        ("DoG Zero-Crossings", dog_zc, "gray"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.ravel()
    for ax, (name, im, cmap) in zip(axes, results):
        disp = normalize(im) if name not in ("LoG Zero-Crossings", "DoG Zero-Crossings") else im
        ax.imshow(disp, cmap=cmap)
        ax.set_title(name)
        ax.axis("off")

    plt.suptitle("Edge Detection: First-Order vs Second-Order (from scratch)", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Saved grid to {out_path}")


if __name__ == "__main__":
    main()