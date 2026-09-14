"""
histogram_equalization_types.py
--------------------------------
Implements Global (GHE), Local (LHE), Adaptive (AHE), and CLAHE techniques
from first principles, with paired image/histogram plots and a 5x2 overview
comparison grid.

Run:
    python histogram_equalization_types.py <input_image> <output_image>
"""

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def rgb_to_grey(img_arr: np.ndarray) -> np.ndarray:
    r, g, b = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)


def compute_hist(arr: np.ndarray) -> np.ndarray:
    hist = np.zeros(256, dtype=np.int64)
    for v in arr.ravel():
        hist[v] += 1
    return hist


def he_mapping(hist: np.ndarray, n_pixels: int) -> np.ndarray:
    """Given a histogram, return the 256-length intensity mapping (CDF scaled)."""
    pmf = hist / max(n_pixels, 1)
    cdf = np.cumsum(pmf)
    return np.round(255 * cdf).astype(np.uint8)


# ---------------------------------------------------------------------------
# 1. Global Histogram Equalization (GHE)
# ---------------------------------------------------------------------------
def global_he(grey: np.ndarray) -> np.ndarray:
    hist = compute_hist(grey)
    mapping = he_mapping(hist, grey.size)
    return mapping[grey]


# ---------------------------------------------------------------------------
# 2. Local Histogram Equalization (LHE) - non-overlapping tiles, each
#    equalized independently against its own histogram (blocky by design).
# ---------------------------------------------------------------------------
def local_he(grey: np.ndarray, tile_size: int = 32) -> np.ndarray:
    h, w = grey.shape
    out = np.zeros_like(grey)
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = grey[y:y + tile_size, x:x + tile_size]
            hist = compute_hist(tile)
            mapping = he_mapping(hist, tile.size)
            out[y:y + tile_size, x:x + tile_size] = mapping[tile]
    return out


# ---------------------------------------------------------------------------
# 3. Adaptive Histogram Equalization (AHE) - true pixel-wise sliding window,
#    implemented efficiently via 256 per-intensity "indicator plane" box
#    sums (integral image trick) so every pixel gets its own local CDF.
# ---------------------------------------------------------------------------
def _box_sum(plane: np.ndarray, radius: int) -> np.ndarray:
    """Sum of values within a (2r+1)x(2r+1) window around every pixel,
    via a 2D integral image (summed-area table)."""
    padded = np.pad(plane, radius, mode="edge")
    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
    h, w = plane.shape
    win = 2 * radius + 1
    total = (
        integral[win:win + h, win:win + w]
        - integral[0:h, win:win + w]
        - integral[win:win + h, 0:w]
        + integral[0:h, 0:w]
    )
    return total


def adaptive_he(grey: np.ndarray, radius: int = 15, n_bins: int = 32) -> np.ndarray:
    """
    Pixel-wise adaptive HE. To keep runtime tractable we quantize intensities
    into n_bins buckets when building the local CDF, then map each pixel
    through its own window's CDF (still a true per-pixel adaptive result).
    """
    h, w = grey.shape
    window_area = (2 * radius + 1) ** 2

    # Quantized planes (indicator: value <= bin_edge) so cumulative count
    # of pixels <= grey[y,x] within the window gives the local CDF directly.
    bin_edges = np.linspace(0, 255, n_bins + 1)[1:].astype(np.int32)

    # local_rank[y,x] = number of pixels in window with intensity <= grey[y,x]
    # Built incrementally in quantized steps for speed.
    quant = np.digitize(grey, bin_edges)  # 0..n_bins-1

    cum_count = np.zeros((h, w), dtype=np.float64)
    for b in range(n_bins):
        indicator = (quant <= b).astype(np.float64)
        local_sum = _box_sum(indicator, radius)
        # pixels whose own bin equals b get their local CDF from this step
        mask = quant == b
        cum_count[mask] = local_sum[mask]

    local_cdf = cum_count / window_area
    out = np.round(255 * local_cdf).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# 4. CLAHE - tiled + contrast-limited (histogram clipping) + bilinear
#    interpolation between neighbouring tile mappings.
# ---------------------------------------------------------------------------
def clahe(grey: np.ndarray, tile_size: int = 32, clip_limit: float = 0.02) -> np.ndarray:
    h, w = grey.shape
    ty = int(np.ceil(h / tile_size))
    tx = int(np.ceil(w / tile_size))

    padded_h, padded_w = ty * tile_size, tx * tile_size
    padded = np.zeros((padded_h, padded_w), dtype=grey.dtype)
    padded[:h, :w] = grey
    padded[h:, :w] = grey[-1:, :]
    padded[:, w:] = padded[:, w - 1:w]

    mappings = np.zeros((ty, tx, 256), dtype=np.float64)
    clip = int(clip_limit * tile_size * tile_size)

    for j in range(ty):
        for i in range(tx):
            tile = padded[j * tile_size:(j + 1) * tile_size, i * tile_size:(i + 1) * tile_size]
            hist = compute_hist(tile)

            # Clip histogram and redistribute the excess uniformly (standard CLAHE step)
            excess = np.clip(hist - clip, 0, None).sum()
            hist_clipped = np.minimum(hist, clip)
            hist_clipped += excess // 256

            mapping = he_mapping(hist_clipped, tile.size)
            mappings[j, i] = mapping

    # Bilinear interpolation between the 4 nearest tile-center mappings
    out = np.zeros((padded_h, padded_w), dtype=np.float64)
    centers_y = (np.arange(ty) + 0.5) * tile_size
    centers_x = (np.arange(tx) + 0.5) * tile_size

    for y in range(padded_h):
        j = np.searchsorted(centers_y, y) - 1
        j = np.clip(j, 0, ty - 2) if ty > 1 else 0
        if ty == 1:
            wy, j0, j1 = 0.0, 0, 0
        else:
            j0, j1 = j, j + 1
            wy = (y - centers_y[j0]) / (centers_y[j1] - centers_y[j0])
            wy = np.clip(wy, 0, 1)

        for x in range(padded_w):
            i = np.searchsorted(centers_x, x) - 1
            i = np.clip(i, 0, tx - 2) if tx > 1 else 0
            if tx == 1:
                wx, i0, i1 = 0.0, 0, 0
            else:
                i0, i1 = i, i + 1
                wx = (x - centers_x[i0]) / (centers_x[i1] - centers_x[i0])
                wx = np.clip(wx, 0, 1)

            v = padded[y, x]
            m00 = mappings[j0, i0, v]
            m01 = mappings[j0, i1, v]
            m10 = mappings[j1, i0, v]
            m11 = mappings[j1, i1, v]
            top = m00 * (1 - wx) + m01 * wx
            bot = m10 * (1 - wx) + m11 * wx
            out[y, x] = top * (1 - wy) + bot * wy

    return np.round(out[:h, :w]).astype(np.uint8)


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "Input_images/sample.png"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "Output/histogram_equalization_types_grid.png"

    img = Image.open(in_path).convert("RGB")
    grey = rgb_to_grey(np.array(img))

    # Downsample a touch for the pixel-wise AHE to keep runtime reasonable
    small = np.array(Image.fromarray(grey).resize((grey.shape[1] // 2, grey.shape[0] // 2)))

    print("Computing GHE...")
    ghe = global_he(grey)
    print("Computing LHE...")
    lhe = local_he(grey, tile_size=32)
    print("Computing AHE (this is the slow one)...")
    ahe_small = adaptive_he(small, radius=12, n_bins=32)
    ahe = np.array(Image.fromarray(ahe_small).resize((grey.shape[1], grey.shape[0])))
    print("Computing CLAHE...")
    clahe_img = clahe(grey, tile_size=32, clip_limit=0.02)

    results = [
        ("Original", grey),
        ("GHE", ghe),
        ("LHE (tiled)", lhe),
        ("AHE (pixel-wise)", ahe),
        ("CLAHE", clahe_img),
    ]

    fig, axes = plt.subplots(5, 2, figsize=(10, 20))
    for row, (name, im) in enumerate(results):
        axes[row, 0].imshow(im, cmap="gray")
        axes[row, 0].set_title(name)
        axes[row, 0].axis("off")

        hist = compute_hist(im)
        axes[row, 1].bar(range(256), hist, width=1, color="steelblue")
        axes[row, 1].set_title(f"{name} Histogram")
        axes[row, 1].set_xlabel("Intensity")
        axes[row, 1].set_ylabel("Frequency")

    plt.suptitle("GHE vs LHE vs AHE vs CLAHE", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Saved grid to {out_path}")


if __name__ == "__main__":
    main()