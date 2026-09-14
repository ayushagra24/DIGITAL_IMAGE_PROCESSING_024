"""
huffman_coding.py
-------------------
Optimal Huffman coding implemented from first principles: binary tree
construction via a min-heap, prefix decoding, entropy, coding-efficiency
metrics, and a bit-length heatmap.

Run:
    python huffman_coding.py <input_image> <output_image>
"""

import sys
import heapq
import itertools
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


class HuffmanNode:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(hist: np.ndarray) -> HuffmanNode:
    heap = []
    counter = itertools.count()  # tie-breaker so heap never compares nodes directly
    for symbol, freq in enumerate(hist):
        if freq > 0:
            heapq.heappush(heap, (freq, next(counter), HuffmanNode(freq, symbol)))

    if len(heap) == 1:
        # Degenerate case: only one distinct symbol
        _, _, only = heap[0]
        return HuffmanNode(only.freq, left=only)

    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        merged = HuffmanNode(f1 + f2, left=n1, right=n2)
        heapq.heappush(heap, (merged.freq, next(counter), merged))

    return heap[0][2]


def generate_codes(node: HuffmanNode, prefix="", codes=None):
    if codes is None:
        codes = {}
    if node is None:
        return codes
    if node.symbol is not None:
        codes[node.symbol] = prefix or "0"
        return codes
    generate_codes(node.left, prefix + "0", codes)
    generate_codes(node.right, prefix + "1", codes)
    return codes


def decode(bitstring: str, root: HuffmanNode, n_symbols: int):
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
    out_path = sys.argv[2] if len(sys.argv) > 2 else "Output/huffman_grid.png"

    img = Image.open(in_path).convert("RGB")
    grey = rgb_to_grey(np.array(img))
    hist = compute_hist(grey)
    n_pixels = grey.size
    pmf = hist / n_pixels

    tree_root = build_huffman_tree(hist)
    codes = generate_codes(tree_root)

    entropy = -sum(p * np.log2(p) for p in pmf if p > 0)
    avg_len = sum(pmf[s] * len(c) for s, c in codes.items())
    efficiency = entropy / avg_len * 100

    flat = grey.ravel()
    sample = flat[:2000]
    encoded_bits = "".join(codes[v] for v in sample)
    decoded = decode(encoded_bits, tree_root, len(sample))
    lossless = np.array_equal(np.array(decoded, dtype=np.uint8), sample)

    length_map = np.zeros(256, dtype=np.int32)
    for s, c in codes.items():
        length_map[s] = len(c)
    heatmap = length_map[grey]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(grey, cmap="gray")
    axes[0, 0].set_title("Original (Greyscale)")
    axes[0, 0].axis("off")

    im = axes[0, 1].imshow(heatmap, cmap="magma")
    axes[0, 1].set_title("Huffman Codeword Length Heatmap")
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, label="bits")

    axes[1, 0].bar(range(256), hist, width=1, color="firebrick")
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

    plt.suptitle("Huffman Coding (from first principles)", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Saved grid to {out_path}")
    print(f"Entropy={entropy:.4f}, AvgLen={avg_len:.4f}, Efficiency={efficiency:.2f}%, Lossless={lossless}")


if __name__ == "__main__":
    main()