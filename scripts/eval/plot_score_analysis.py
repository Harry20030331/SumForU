import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def extract_suitability(pred: str) -> float | None:
    """
    Extract suitability score from a model output string.
    Expect patterns like: 'Suitability: 8/10'.
    Return the raw 0-10 integer as float, or None if not found.
    """
    pred = pred.replace("<|im_end|>", "")
    m = re.search(r"Suitability\s*:\s*([0-9]+)\s*/\s*10", pred)
    if not m:
        return None
    try:
        val = float(m.group(1))
        return val
    except ValueError:
        return None


def load_suitability_scores(path: Path) -> List[float]:
    """
    Load model outputs JSON (list[str]) and extract suitability scores.
    Convert each to 0-5 scale by dividing by 2.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    scores: List[float] = []
    for pred in data:
        raw = extract_suitability(pred)
        if raw is None:
            scores.append(float("nan"))
        else:
            scores.append(raw / 2.0)
    return scores


def extract_ratings_from_reviews(reviews_text: str) -> List[float]:
    """
    Extract all 'rating x.y' patterns from the concatenated reviews text.
    Return list of floats (e.g., [5.0, 3.0, 4.0]).
    """
    matches = re.findall(r"rating\s+([0-9]+(?:\.[0-9]+)?)", reviews_text)
    ratings: List[float] = []
    for m in matches:
        try:
            ratings.append(float(m))
        except ValueError:
            continue
    return ratings


def load_avg_ratings(gt_path: Path) -> List[float]:
    """
    For each sample in ground truth JSON, compute the average rating
    over all 'rating x.y' occurrences in its 'reviews' field.
    If no rating is found, use NaN.
    """
    with gt_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    avg_ratings: List[float] = []
    for item in data:
        reviews_text = item.get("reviews", "")
        ratings = extract_ratings_from_reviews(reviews_text)
        if not ratings:
            avg_ratings.append(float("nan"))
        else:
            avg_ratings.append(float(sum(ratings) / len(ratings)))
    return avg_ratings


def align_and_filter(pred_scores: List[float], gt_scores: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align prediction and ground truth scores by index, and drop any pair
    where either side is NaN.
    """
    n = min(len(pred_scores), len(gt_scores))
    pred_arr = np.array(pred_scores[:n], dtype=float)
    gt_arr = np.array(gt_scores[:n], dtype=float)

    mask = ~np.isnan(pred_arr) & ~np.isnan(gt_arr)
    return pred_arr[mask], gt_arr[mask]


def scores_to_labels(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map continuous scores (0-5) to integer labels 1-5 for confusion matrix:
      gt_label   = round(gt) clipped to [1,5]
      pred_label = round(pred) clipped to [1,5]
    """
    gt_label = np.rint(gt)
    pred_label = np.rint(pred)

    gt_label = np.clip(gt_label, 1, 5).astype(int)
    pred_label = np.clip(pred_label, 1, 5).astype(int)
    return pred_label, gt_label


def compute_confusion_matrix(pred_labels: np.ndarray, gt_labels: np.ndarray) -> np.ndarray:
    """
    Compute a 5x5 confusion matrix for labels in {1,2,3,4,5}:
    rows = true labels, cols = predicted labels.
    """
    num_classes = 5
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(gt_labels, pred_labels):
        if 1 <= true <= 5 and 1 <= pred <= 5:
            cm[true - 1, pred - 1] += 1
    return cm


def compute_bin_bias(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each true rating bin k in {1,...,5}, compute average bias:
      bias_k = mean(pred - gt) over samples with rounded gt == k.
    Return:
      bins: np.array([1,2,3,4,5])
      biases: np.array of length 5 with NaN for empty bins.
    """
    gt_round = np.rint(gt)
    gt_round = np.clip(gt_round, 1, 5).astype(int)

    bins = np.arange(1, 6)
    biases = np.full_like(bins, np.nan, dtype=float)

    for k in bins:
        mask = gt_round == k
        if np.any(mask):
            diffs = pred[mask] - gt[mask]
            biases[k - 1] = float(np.mean(diffs))
    return bins, biases


def main():
    parser = argparse.ArgumentParser(
        description="Plot stacked confusion matrices and bias per rating bin for multiple models."
    )
    parser.add_argument(
        "--gt-path",
        type=Path,
        required=True,
        help="Ground truth JSON path, e.g. dataset/data/raw/v1_test_preprocessed.json",
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        required=True,
        help="Baseline outputs JSON",
    )
    parser.add_argument(
        "--pe-path",
        type=Path,
        required=True,
        help="PE model outputs JSON",
    )
    parser.add_argument(
        "--sft-path",
        type=Path,
        required=True,
        help="SFT model outputs JSON",
    )
    parser.add_argument(
        "--rl-path",
        type=Path,
        required=True,
        help="RL model outputs JSON",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("confusion_and_bias_all.png"),
        help="Path to save the combined figure.",
    )
    args = parser.parse_args()

    gt_avg_ratings = load_avg_ratings(args.gt_path)

    methods = [
        ("baseline", args.baseline_path),
        ("pe", args.pe_path),
        ("sft", args.sft_path),
        ("rl", args.rl_path),
    ]

    # Prepare data for each method
    results = []
    for name, path in methods:
        pred_scores_raw = load_suitability_scores(path)
        pred_arr, gt_arr = align_and_filter(pred_scores_raw, gt_avg_ratings)

        if pred_arr.size == 0:
            print(f"{name}: no valid suitability scores, skipping.")
            results.append((name, None, None, None))
            continue

        pred_labels, gt_labels = scores_to_labels(pred_arr, gt_arr)
        cm = compute_confusion_matrix(pred_labels, gt_labels)
        bins, biases = compute_bin_bias(pred_arr, gt_arr)
        results.append((name, cm, bins, biases))

    # Create a stacked 4x2 figure
    fig, axes = plt.subplots(len(methods), 2, figsize=(10, 3 * len(methods)))

    for row_idx, (name, cm, bins, biases) in enumerate(results):
        ax_cm = axes[row_idx, 0]
        ax_bias = axes[row_idx, 1]

        if cm is None or biases is None:
            ax_cm.axis("off")
            ax_bias.axis("off")
            continue

        # Confusion matrix heatmap
        im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
        ax_cm.set_title(f"{name} - Confusion Matrix", fontsize=9)
        ax_cm.set_xlabel("Predicted rating", fontsize=8)
        ax_cm.set_ylabel("True rating", fontsize=8)
        ax_cm.set_xticks(np.arange(5))
        ax_cm.set_yticks(np.arange(5))
        ax_cm.set_xticklabels([str(i) for i in range(1, 6)], fontsize=7)
        ax_cm.set_yticklabels([str(i) for i in range(1, 6)], fontsize=7)

        for i in range(5):
            for j in range(5):
                val = cm[i, j]
                if val > 0:
                    ax_cm.text(
                        j, i, str(val),
                        ha="center", va="center",
                        fontsize=6, color="black",
                    )

        fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

        # Bias bar plot
        ax_bias.set_title(f"{name} - Avg Bias per True Rating", fontsize=9)
        ax_bias.set_xlabel("True rating", fontsize=8)
        ax_bias.set_ylabel("Average (pred - true)", fontsize=8)

        has_data = ~np.isnan(biases)
        plot_biases = np.where(has_data, biases, 0.0)

        bars = ax_bias.bar(bins, plot_biases, color="skyblue", edgecolor="black")
        for idx, bar in enumerate(bars):
            if not has_data[idx]:
                bar.set_hatch("//")
                bar.set_alpha(0.3)

        ax_bias.axhline(0.0, color="gray", linewidth=0.8)
        ax_bias.set_xticks(bins)
        ax_bias.set_xticklabels([str(int(b)) for b in bins], fontsize=7)
        ax_bias.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    fig.savefig(args.output_path, dpi=200)
    plt.close(fig)
    print(f"Saved combined figure to {args.output_path}")


if __name__ == "__main__":
    main()
