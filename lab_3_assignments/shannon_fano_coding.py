"""
shannon_fano_coding.py
------------------------
Shannon-Fano source coding implemented from first principles, with
prefix-tree decoding, entropy, coding-efficiency metrics, and a codeword
length heatmap (mapped onto the image itself).

Run:
    python shannon_fano_coding.py <input_image> <output_image>
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


class Node:
    def __init__(self, symbol=None):
        self.symbol = symbol
        self.left = None
        self.right = None


def shannon_fano_recursive(symbols_probs, codes):
    """symbols_probs: list of (symbol, prob), sorted descending by prob."""
    if len(symbols_probs) == 1:
        sym, _ = symbols_probs[0]
        codes[sym] = codes.get(sym, "") or "0"
        return

    total = sum(p for _, p in symbols_probs)
    running = 0.0
    split = 0
    best_diff = float("inf")
    for i in range(len(symbols_probs) - 1):
        running += symbols_probs[i][1]
        diff = abs((total - running) - running)
        if diff < best_diff:
            best_diff = diff
            split = i + 1

    left_group = symbols_probs[:split]
    right_group = symbols_probs[split:]

    for sym, _ in left_group:
        codes[sym] = codes.get(sym, "") + "0"
    for sym, _ in right_group:
        codes[sym] = codes.get(sym, "") + "1"

    if len(left_group) > 1:
        shannon_fano_recursive(left_group, codes)
    if len(right_group) > 1:
        shannon_fano_recursive(right_group, codes)


def build_shannon_fano_codes(hist: np.ndarray):
    symbols_probs = [(s, c) for s, c in enumerate(hist) if c > 0]
    symbols_probs.sort(key=lambda x: x[1], reverse=True)
    codes = {}
    shannon_fano_recursive(symbols_probs, codes)
    return codes


class PrefixTreeNode:
    __slots__ = ("symbol", "left", "right")

    def __init__(self):
        self.symbol = None
        self.left = None
        self.right = None


def build_prefix_tree(codes: dict) -> PrefixTreeNode:
    root = PrefixTreeNode()
    for sym, code in codes.items():
        node = root
        for bit in code:
            if bit == "0":
                if node.left is None:
                    node.left = PrefixTreeNode()
                node = node.left
            else:
                if node.right is None:
                    node.right = PrefixTreeNode()
                node = node.right
        node.symbol = sym
    return root


def decode(bitstring: str, root: PrefixTreeNode, n_symbols: int):
    decoded = []
    node = root
    for bit in bitstring:
        node = node.left if bit == "0" else node.right
        if node.symbol is not None:
            decoded.append(node.symbol)
            node = root
            if len(decoded) == n_symbols:
                break
    return decoded


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "Input_images/sample.png"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "Output/shannon_fano_grid.png"

    img = Image.open(in_path).convert("RGB")
    grey = rgb_to_grey(np.array(img))
    hist = compute_hist(grey)
    n_pixels = grey.size
    pmf = hist / n_pixels

    codes = build_shannon_fano_codes(hist)

    entropy = -sum(p * np.log2(p) for p in pmf if p > 0)
    avg_len = sum(pmf[s] * len(c) for s, c in codes.items())
    efficiency = entropy / avg_len * 100

    # Encode a sample chunk and verify decoding on it (full image would be slow in pure Python)
    flat = grey.ravel()
    sample = flat[:2000]
    encoded_bits = "".join(codes[v] for v in sample)
    tree_root = build_prefix_tree(codes)
    decoded = decode(encoded_bits, tree_root, len(sample))
    lossless = np.array_equal(np.array(decoded, dtype=np.uint8), sample)

    # Codeword-length heatmap over the whole image
    length_map = np.zeros(256, dtype=np.int32)
    for s, c in codes.items():
        length_map[s] = len(c)
    heatmap = length_map[grey]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(grey, cmap="gray")
    axes[0, 0].set_title("Original (Greyscale)")
    axes[0, 0].axis("off")

    im = axes[0, 1].imshow(heatmap, cmap="viridis")
    axes[0, 1].set_title("Shannon-Fano Codeword Length Heatmap")
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, label="bits")

    axes[1, 0].bar(range(256), hist, width=1, color="teal")
    axes[1, 0].set_title("Symbol Histogram (Intensity)")
    axes[1, 0].set_xlabel("Intensity")
    axes[1, 0].set_ylabel("Frequency")

    axes[1, 1].axis("off")
    stats_text = (
        f"Number of unique symbols: {len(codes)}\n\n"
        f"Entropy H(X): {entropy:.4f} bits/symbol\n"
        f"Average codeword length: {avg_len:.4f} bits/symbol\n"
        f"Coding efficiency: {efficiency:.2f}%\n\n"
        f"Fixed-length baseline: 8 bits/symbol\n"
        f"Compression ratio vs fixed-length: {8/avg_len:.3f}x\n\n"
        f"Lossless round-trip check (first {len(sample)} px): "
        f"{'PASS' if lossless else 'FAIL'}"
    )
    axes[1, 1].text(0.02, 0.5, stats_text, fontsize=12, va="center", family="monospace")

    plt.suptitle("Shannon-Fano Coding (from first principles)", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Saved grid to {out_path}")
    print(f"Entropy={entropy:.4f}, AvgLen={avg_len:.4f}, Efficiency={efficiency:.2f}%, Lossless={lossless}")


if __name__ == "__main__":
    main()