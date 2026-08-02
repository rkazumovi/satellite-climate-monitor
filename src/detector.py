"""
detector.py — Change detection on real Sentinel-2 before/after image pairs

Uses the OSCD dataset (blanchon/OSCD_MSI): 24 real registered Sentinel-2
image pairs (14 train + 10 test) with real pixel-level ground truth change
masks (urban changes: new buildings, new roads).

NOTE on this dataset's schema: it declares image1/image2 as a fixed shape
Array3D(13, 10000, 10000), but that's inaccurate — actual per-pair image
size varies (confirmed via direct Arrow access, e.g. one pair is 13x522x582).
Using the library's default decoder crashes trying to force that fixed
reshape, so this script reads image1/image2 via with_format("arrow") to
get each pair's true raw shape instead.

For each pair, this script computes two statistical change indicators
purely with NumPy/Pandas (no trained model):

1. Spectral change magnitude: mean absolute difference across all 13 bands,
   per pixel. Thresholded (mean + 2*std) into a predicted binary change
   mask, then compared against OSCD's real ground truth (precision/recall/
   F1/accuracy) since the ground truth specifically covers this kind of
   urban change.

2. NDVI (vegetation index) change: computed from Red (B04) and NIR (B08)
   bands, before vs after. This flags vegetation loss/gain (deforestation-
   style signal). OSCD's ground truth only covers urban change, not
   vegetation, so this indicator is reported descriptively without a
   ground-truth validation step.

Run directly:
    python src/detector.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

HF_DATASET_ID = "blanchon/OSCD_MSI"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
CHANGE_DIR = os.path.join(OUTPUT_DIR, "change_detection")

# Sentinel-2 13-band order: B01,B02,B03,B04,B05,B06,B07,B08,B08A,B09,B10,B11,B12
BAND_BLUE = 1
BAND_GREEN = 2
BAND_RED = 3
BAND_NIR = 7

NUM_SAMPLE_VISUALIZATIONS = 4  # how many pairs to save full before/after/mask plots for


def load_raw_image_pair(arrow_view, index):
    """
    Fetch image1/image2 for one row via raw Arrow access (bypasses the
    dataset's buggy fixed-shape Array3D decode) and return as numpy
    arrays of shape (13, H, W).
    """
    row_table = arrow_view[index:index + 1]

    image1_nested = row_table.column("image1")[0].as_py()
    image2_nested = row_table.column("image2")[0].as_py()

    image1 = np.array(image1_nested, dtype=np.float32)  # shape (13, H, W)
    image2 = np.array(image2_nested, dtype=np.float32)

    return image1, image2


def load_mask(split, index):
    """Fetch the ground truth mask for one row, isolating the mask column
    so we don't trigger the image1/image2 decode crash."""
    mask_row = split.select_columns(["mask"])[index]
    mask_img = mask_row["mask"]
    mask_array = np.array(mask_img)
    return mask_array  # shape (H, W), values 0 or 1


def compute_ndvi(image, epsilon=1e-6):
    """NDVI = (NIR - Red) / (NIR + Red). Image shape: (13, H, W)."""
    red = image[BAND_RED]
    nir = image[BAND_NIR]
    ndvi = (nir - red) / (nir + red + epsilon)
    return ndvi


def compute_spectral_change_magnitude(image1, image2):
    """
    Mean absolute difference across all 13 bands, per pixel, normalized by
    each band's own standard deviation so no single band dominates purely
    due to having a larger numeric range.
    """
    diff = np.abs(image2 - image1)  # shape (13, H, W)

    band_stds = diff.reshape(diff.shape[0], -1).std(axis=1)
    band_stds = np.where(band_stds == 0, 1.0, band_stds)  # avoid divide-by-zero

    normalized_diff = diff / band_stds[:, None, None]
    change_magnitude = normalized_diff.mean(axis=0)  # shape (H, W)

    return change_magnitude


def threshold_change_mask(change_magnitude, num_std=2.0):
    """Flag pixels whose change magnitude exceeds mean + num_std*std as 'changed'."""
    threshold = change_magnitude.mean() + num_std * change_magnitude.std()
    predicted_mask = (change_magnitude > threshold).astype(np.uint8)
    return predicted_mask, threshold


def compute_classification_metrics(true_mask, predicted_mask):
    """Precision/recall/F1/accuracy computed manually with NumPy — binary change vs no-change."""
    true_flat = true_mask.flatten().astype(bool)
    pred_flat = predicted_mask.flatten().astype(bool)

    true_positive = np.sum(true_flat & pred_flat)
    false_positive = np.sum(~true_flat & pred_flat)
    false_negative = np.sum(true_flat & ~pred_flat)
    true_negative = np.sum(~true_flat & ~pred_flat)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (true_positive + true_negative) / true_flat.size

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "pct_pixels_changed_true": true_flat.mean() * 100,
        "pct_pixels_changed_predicted": pred_flat.mean() * 100,
    }


def normalize_for_display(band_2d, low_pct=2, high_pct=98):
    """Percentile-stretch a single band to 0-1 range for matplotlib display."""
    low = np.percentile(band_2d, low_pct)
    high = np.percentile(band_2d, high_pct)
    stretched = np.clip((band_2d - low) / (high - low + 1e-6), 0, 1)
    return stretched


def make_rgb_thumbnail(image):
    """Build a true-color RGB thumbnail (Red=B04, Green=B03, Blue=B02) from a (13,H,W) image."""
    red = normalize_for_display(image[BAND_RED])
    green = normalize_for_display(image[BAND_GREEN])
    blue = normalize_for_display(image[BAND_BLUE])
    rgb = np.stack([red, green, blue], axis=-1)
    return rgb


def save_visualization(image1, image2, true_mask, predicted_mask, ndvi_diff, pair_id, split_name):
    os.makedirs(CHANGE_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    axes[0].imshow(make_rgb_thumbnail(image1))
    axes[0].set_title("Before")
    axes[0].axis("off")

    axes[1].imshow(make_rgb_thumbnail(image2))
    axes[1].set_title("After")
    axes[1].axis("off")

    axes[2].imshow(true_mask, cmap="Reds", vmin=0, vmax=1)
    axes[2].set_title("Ground Truth Change")
    axes[2].axis("off")

    axes[3].imshow(predicted_mask, cmap="Reds", vmin=0, vmax=1)
    axes[3].set_title("Predicted Change")
    axes[3].axis("off")

    ndvi_plot = axes[4].imshow(ndvi_diff, cmap="RdYlGn", vmin=-0.5, vmax=0.5)
    axes[4].set_title("NDVI Change\n(green=growth, red=loss)")
    axes[4].axis("off")
    fig.colorbar(ndvi_plot, ax=axes[4], fraction=0.046, pad=0.04)

    plt.suptitle(f"{split_name} pair #{pair_id}")
    plt.tight_layout()

    path = os.path.join(CHANGE_DIR, f"{split_name}_pair_{pair_id}.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)


def process_split(dataset, split_name):
    split = dataset[split_name]
    arrow_view = split.with_format("arrow")

    records = []

    for i in range(len(split)):
        print(f"\n[{split_name}] Processing pair {i + 1}/{len(split)}...")

        image1, image2 = load_raw_image_pair(arrow_view, i)
        true_mask = load_mask(split, i)

        # Images and mask may differ by a pixel or two due to registration;
        # crop to the smallest common shape before comparing.
        h = min(image1.shape[1], image2.shape[1], true_mask.shape[0])
        w = min(image1.shape[2], image2.shape[2], true_mask.shape[1])
        image1, image2 = image1[:, :h, :w], image2[:, :h, :w]
        true_mask = true_mask[:h, :w]

        change_magnitude = compute_spectral_change_magnitude(image1, image2)
        predicted_mask, threshold = threshold_change_mask(change_magnitude)

        metrics = compute_classification_metrics(true_mask, predicted_mask)

        ndvi_before = compute_ndvi(image1)
        ndvi_after = compute_ndvi(image2)
        ndvi_diff = ndvi_after - ndvi_before

        record = {
            "split": split_name,
            "pair_id": i,
            "height": h,
            "width": w,
            "change_threshold": threshold,
            **metrics,
            "mean_ndvi_before": float(np.mean(ndvi_before)),
            "mean_ndvi_after": float(np.mean(ndvi_after)),
            "mean_ndvi_change": float(np.mean(ndvi_diff)),
            "pct_pixels_vegetation_loss": float(np.mean(ndvi_diff < -0.2) * 100),
        }
        records.append(record)

        print(
            f"  precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
            f"f1={metrics['f1']:.3f} | true_change={metrics['pct_pixels_changed_true']:.2f}% "
            f"predicted_change={metrics['pct_pixels_changed_predicted']:.2f}%"
        )

        if i < NUM_SAMPLE_VISUALIZATIONS:
            save_visualization(image1, image2, true_mask, predicted_mask, ndvi_diff, i, split_name)

    return records


def main():
    print(f"Loading dataset '{HF_DATASET_ID}' (uses local cache if already downloaded)...")
    dataset = load_dataset(HF_DATASET_ID)

    all_records = []
    for split_name in ["train", "test"]:
        all_records.extend(process_split(dataset, split_name))

    results_df = pd.DataFrame(all_records)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "change_detection_summary.csv")
    results_df.to_csv(csv_path, index=False)

    print("\n--- Aggregate results across all 24 pairs ---")
    print(f"Mean precision: {results_df['precision'].mean():.3f}")
    print(f"Mean recall:    {results_df['recall'].mean():.3f}")
    print(f"Mean F1:        {results_df['f1'].mean():.3f}")
    print(f"Mean accuracy:  {results_df['accuracy'].mean():.3f}")
    print(f"\nPairs with net vegetation loss (mean NDVI change < 0): "
          f"{(results_df['mean_ndvi_change'] < 0).sum()} / {len(results_df)}")

    print(f"\nSaved per-pair results table: {csv_path}")
    print(f"Saved sample visualizations to: {CHANGE_DIR}")
    print("\nStep 4 (detector.py) complete.")


if __name__ == "__main__":
    main()