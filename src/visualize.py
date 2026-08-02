"""
visualize.py — Summary charts and plots for the README / final deliverable

Reads the already-computed results from classifier.py (classification_summary.json)
and detector.py (change_detection_summary.csv) and produces polished charts:

1. Per-class F1 score bar chart (Stage 1 classification)
2. Change detection F1 score per image pair, sorted (Stage 2)
3. NDVI change per image pair, highlighting vegetation loss vs gain (Stage 2)
4. A combined 2x2 summary dashboard figure

Does not retrain or reprocess anything — purely visualizes existing results.

Run directly:
    python src/visualize.py
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
VIZ_DIR = os.path.join(OUTPUT_DIR, "visualizations")

CLASSIFICATION_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "classification_summary.json")
CHANGE_DETECTION_CSV_PATH = os.path.join(OUTPUT_DIR, "change_detection_summary.csv")


def load_classification_summary():
    with open(CLASSIFICATION_SUMMARY_PATH, "r") as f:
        return json.load(f)


def load_change_detection_df():
    return pd.read_csv(CHANGE_DETECTION_CSV_PATH)


def plot_class_f1_scores(classification_summary, ax=None):
    """Bar chart of per-class F1 scores, sorted descending."""
    per_class = classification_summary["per_class_metrics"]
    class_names = list(per_class.keys())
    f1_scores = [per_class[c]["f1_score"] for c in class_names]

    order = np.argsort(f1_scores)[::-1]
    sorted_names = [class_names[i] for i in order]
    sorted_scores = [f1_scores[i] for i in order]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(sorted_names, sorted_scores, color="#2c7fb8")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Land Use Classification: Per-Class F1 Score\n"
                 f"(Best model: {classification_summary['best_model']}, "
                 f"Overall accuracy: {classification_summary['test_accuracy']:.1%})")
    ax.tick_params(axis="x", rotation=45)
    for bar, score in zip(bars, sorted_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.01, f"{score:.2f}",
                ha="center", va="bottom", fontsize=8)

    if standalone:
        plt.tight_layout()
        path = os.path.join(VIZ_DIR, "classification_f1_by_class.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")


def plot_change_detection_f1(change_df, ax=None):
    """F1 score per image pair, sorted, colored by train/test split."""
    df_sorted = change_df.sort_values("f1", ascending=False).reset_index(drop=True)
    colors = ["#d95f02" if s == "test" else "#1b9e77" for s in df_sorted["split"]]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6))

    labels = [f"{row['split']}#{row['pair_id']}" for _, row in df_sorted.iterrows()]
    ax.bar(labels, df_sorted["f1"], color=colors)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Change Detection: F1 Score per Image Pair\n"
                 f"(Mean F1 = {change_df['f1'].mean():.3f} across {len(change_df)} pairs)")
    ax.tick_params(axis="x", rotation=90)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1b9e77", label="Train"),
        Patch(facecolor="#d95f02", label="Test"),
    ]
    ax.legend(handles=legend_elements)

    if standalone:
        plt.tight_layout()
        path = os.path.join(VIZ_DIR, "change_detection_f1_by_pair.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")


def plot_ndvi_change(change_df, ax=None):
    """NDVI change per pair, colored red (loss) vs green (gain)."""
    df_sorted = change_df.sort_values("mean_ndvi_change").reset_index(drop=True)
    colors = ["#e41a1c" if v < 0 else "#4daf4a" for v in df_sorted["mean_ndvi_change"]]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6))

    labels = [f"{row['split']}#{row['pair_id']}" for _, row in df_sorted.iterrows()]
    ax.bar(labels, df_sorted["mean_ndvi_change"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean NDVI Change (after - before)")
    ax.set_title("Vegetation Change per Image Pair\n(red = net loss, green = net gain)")
    ax.tick_params(axis="x", rotation=90)

    if standalone:
        plt.tight_layout()
        path = os.path.join(VIZ_DIR, "ndvi_change_by_pair.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")


def plot_summary_dashboard(classification_summary, change_df):
    """Combined 2x2 dashboard: class F1, change F1, NDVI change, and a metrics text panel."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    plot_class_f1_scores(classification_summary, ax=axes[0, 0])
    plot_change_detection_f1(change_df, ax=axes[0, 1])
    plot_ndvi_change(change_df, ax=axes[1, 0])

    axes[1, 1].axis("off")
    summary_text = (
        "PIPELINE SUMMARY\n\n"
        f"Stage 1 — Land Use Classification\n"
        f"  Model: {classification_summary['best_model']}\n"
        f"  Test accuracy: {classification_summary['test_accuracy']:.1%}\n"
        f"  Macro F1: {classification_summary['macro_avg_f1']:.3f}\n"
        f"  Test samples: {classification_summary['num_test_samples']}\n\n"
        f"Stage 2 — Change Detection\n"
        f"  Pairs analyzed: {len(change_df)}\n"
        f"  Mean precision: {change_df['precision'].mean():.3f}\n"
        f"  Mean recall: {change_df['recall'].mean():.3f}\n"
        f"  Mean F1: {change_df['f1'].mean():.3f}\n"
        f"  Pairs with vegetation loss: {(change_df['mean_ndvi_change'] < 0).sum()}/{len(change_df)}"
    )
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                     fontsize=13, verticalalignment="top", family="monospace")

    plt.suptitle("Satellite Climate Monitoring System — Results Dashboard", fontsize=16)
    plt.tight_layout()

    path = os.path.join(VIZ_DIR, "summary_dashboard.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    os.makedirs(VIZ_DIR, exist_ok=True)

    print("Loading Stage 1 and Stage 2 results...")
    classification_summary = load_classification_summary()
    change_df = load_change_detection_df()

    print("\nGenerating charts...")
    plot_class_f1_scores(classification_summary)
    plot_change_detection_f1(change_df)
    plot_ndvi_change(change_df)
    plot_summary_dashboard(classification_summary, change_df)

    print(f"\nAll visualizations saved to: {VIZ_DIR}")
    print("\nStep 6 (visualize.py) complete.")


if __name__ == "__main__":
    main()