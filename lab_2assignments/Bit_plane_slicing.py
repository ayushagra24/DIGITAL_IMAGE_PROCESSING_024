"""
bit_plane_slicing.py
---------------------
Converts an image to greyscale, extracts all 8 bit planes (Bit 0 to Bit 7),
and outputs a 3x3 comparison grid (original greyscale + 8 bit planes).

Run:
    python bit_plane_slicing.py <input_image> <output_image>
"""

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def rgb_to_grey(img_arr: np.ndarray) -> np.ndarray:
    """Convert an RGB array to greyscale using the ITU-R BT.601 luma formula."""
    r, g, b = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2]
    grey = 0.299 * r + 0.587 * g + 0.114 * b
    return grey.astype(np.uint8)


def extract_bit_planes(grey: np.ndarray):
    """Return a list of 8 bit-plane images (Bit 0 = LSB ... Bit 7 = MSB)."""
    planes = []
    for bit in range(8):
        plane = (grey >> bit) & 1
        # Scale 0/1 -> 0/255 for visibility
        planes.append((plane * 255).astype(np.uint8))
    return planes


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "Input_images/sample.png"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "Output/bit_plane_slicing_grid.png"

    img = Image.open(in_path).convert("RGB")
    img_arr = np.array(img)
    grey = rgb_to_grey(img_arr)
    planes = extract_bit_planes(grey)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()

    axes[0].imshow(grey, cmap="gray")
    axes[0].set_title("Greyscale (original)")
    axes[0].axis("off")

    for bit in range(8):
        ax = axes[bit + 1]
        ax.imshow(planes[bit], cmap="gray")
        ax.set_title(f"Bit Plane {bit} ({'LSB' if bit == 0 else 'MSB' if bit == 7 else ''})")
        ax.axis("off")

    plt.suptitle("Bit Plane Slicing (Bit 0 -> Bit 7)", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved grid to {out_path}")


if __name__ == "__main__":
    main()