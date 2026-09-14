"""
discrete_wavelet_transform.py
------------------------------
2D Discrete Wavelet Transform (Haar DWT / IDWT) implemented from first
principles. Shows 1-level and 2-level multiresolution subbands
(LL, LH, HL, HH) and verifies exact reconstruction via IDWT.

Run:
    python discrete_wavelet_transform.py <input_image> <output_image>
"""

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def rgb_to_grey(img_arr: np.ndarray) -> np.ndarray:
    r, g, b = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float64)


def haar_1d(signal: np.ndarray) -> np.ndarray:
    """1D Haar transform on the last axis: returns concat([approx, detail])."""
    evens = signal[..., 0::2]
    odds = signal[..., 1::2]
    approx = (evens + odds) / np.sqrt(2)
    detail = (evens - odds) / np.sqrt(2)
    return np.concatenate([approx, detail], axis=-1)


def inverse_haar_1d(coeffs: np.ndarray) -> np.ndarray:
    n = coeffs.shape[-1] // 2
    approx = coeffs[..., :n]
    detail = coeffs[..., n:]
    evens = (approx + detail) / np.sqrt(2)
    odds = (approx - detail) / np.sqrt(2)
    out = np.empty(coeffs.shape[:-1] + (n * 2,), dtype=coeffs.dtype)
    out[..., 0::2] = evens
    out[..., 1::2] = odds
    return out


def dwt2(img: np.ndarray):
    """One level 2D Haar DWT -> LL, LH, HL, HH subbands."""
    h, w = img.shape
    if h % 2:
        img = img[:-1, :]
    if w % 2:
        img = img[:, :-1]

    # Transform along width (axis=1)
    step1 = haar_1d(img)
    # Transform along height (axis=0) -> transpose trick
    step2 = haar_1d(step1.T).T

    half_h, half_w = step2.shape[0] // 2, step2.shape[1] // 2
    LL = step2[:half_h, :half_w]
    HL = step2[half_h:, :half_w]
    LH = step2[:half_h, half_w:]
    HH = step2[half_h:, half_w:]
    return LL, LH, HL, HH


def idwt2(LL, LH, HL, HH):
    top = np.concatenate([LL, LH], axis=1)
    bottom = np.concatenate([HL, HH], axis=1)
    combined = np.concatenate([top, bottom], axis=0)

    # Inverse along height then width (reverse order of forward transform)
    step1 = inverse_haar_1d(combined.T).T
    reconstructed = inverse_haar_1d(step1)
    return reconstructed


def normalize_for_display(band: np.ndarray) -> np.ndarray:
    b = band - band.min()
    if b.max() > 0:
        b = b / b.max()
    return (b * 255).astype(np.uint8)


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "Input_images/sample.png"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "Output/dwt_grid.png"

    img = Image.open(in_path).convert("RGB")
    grey = rgb_to_grey(np.array(img))

    # ---- Level 1 ----
    LL1, LH1, HL1, HH1 = dwt2(grey)

    # ---- Level 2 (applied on LL1) ----
    LL2, LH2, HL2, HH2 = dwt2(LL1)

    # ---- Reconstruction check ----
    recon1 = idwt2(LL1, LH1, HL1, HH1)
    h, w = grey.shape[0] - grey.shape[0] % 2, grey.shape[1] - grey.shape[1] % 2
    max_err = np.max(np.abs(recon1 - grey[:h, :w]))

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    axes[0, 0].imshow(grey, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")
    for ax in axes[0, 1:]:
        ax.axis("off")

    level1_bands = [("LL1", LL1), ("LH1", LH1), ("HL1", HL1), ("HH1", HH1)]
    for ax, (name, band) in zip(axes[1], level1_bands):
        ax.imshow(normalize_for_display(band), cmap="gray")
        ax.set_title(f"{name} (Level 1)")
        ax.axis("off")

    level2_bands = [("LL2", LL2), ("LH2", LH2), ("HL2", HL2), ("HH2", HH2)]
    for ax, (name, band) in zip(axes[2], level2_bands):
        ax.imshow(normalize_for_display(band), cmap="gray")
        ax.set_title(f"{name} (Level 2, from LL1)")
        ax.axis("off")

    plt.suptitle(
        f"2D Haar DWT - 1 & 2 Level Decomposition\n"
        f"Exact reconstruction max abs error = {max_err:.2e}",
        fontsize=15,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Saved grid to {out_path}")
    print(f"Max reconstruction error (Level 1 IDWT vs original): {max_err:.2e}")


if __name__ == "__main__":
    main()