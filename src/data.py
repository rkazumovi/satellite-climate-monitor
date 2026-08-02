"""
data.py — EuroSAT download + preprocessing

Downloads the EuroSAT RGB satellite image dataset from the Hugging Face
Hub (blanchon/EuroSAT_RGB: 27,000 images, 10 land-use classes, 64x64 RGB).
The repo ships its own train/test/validation splits, which this script
uses directly, then saves sample images per class for visual inspection
and prints dataset statistics.

Run directly:
    python src/data.py
"""

import os
from collections import Counter

from datasets import load_dataset
from PIL import Image

HF_DATASET_ID = "blanchon/EuroSAT_RGB"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "sample_images")


def load_eurosat():
    """Download (or load from local HF cache) the EuroSAT RGB dataset."""
    print(f"Loading dataset '{HF_DATASET_ID}' from Hugging Face Hub...")
    dataset = load_dataset(HF_DATASET_ID)
    print("Dataset loaded. Available splits:", list(dataset.keys()))
    return dataset


def inspect_dataset(dataset):
    """Print basic structure info: splits, features, label names, sizes."""
    print("\n--- Dataset structure ---")
    for split_name, split in dataset.items():
        print(f"  {split_name:<12} {len(split)} examples")

    train_split = dataset["train"]
    label_feature = train_split.features["label"]
    class_names = label_feature.names
    print(f"\nNumber of classes: {len(class_names)}")
    print(f"Class names: {class_names}")

    sample = train_split[0]
    img = sample["image"]
    print(f"Sample image mode: {img.mode}, size: {img.size}")

    return class_names


def get_splits(dataset):
    """
    Use the dataset's own train/test/validation splits rather than
    re-splitting, since they were built by the dataset maintainer.
    Total: 27,000 images (16,200 train / 5,400 test / 5,400 validation).
    """
    train_split = dataset["train"]
    test_split = dataset["test"]
    validation_split = dataset["validation"]

    print("\n--- Using dataset's native splits ---")
    print(f"Train examples:      {len(train_split)}")
    print(f"Test examples:       {len(test_split)}")
    print(f"Validation examples: {len(validation_split)}")

    return train_split, test_split, validation_split


def print_class_distribution(split, class_names, name):
    labels = split["label"]
    counts = Counter(labels)
    print(f"\nClass distribution ({name}):")
    for class_idx in sorted(counts.keys()):
        print(f"  {class_names[class_idx]:<20} {counts[class_idx]}")


def save_sample_images(split, class_names, samples_per_class=2):
    """Save a few sample images per class to outputs/sample_images for a quick visual check."""
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    seen_per_class = Counter()
    saved = 0

    for idx in range(len(split)):
        example = split[idx]
        label = example["label"]
        class_name = class_names[label]

        if seen_per_class[label] >= samples_per_class:
            continue

        img: Image.Image = example["image"]
        filename = f"{class_name}_{seen_per_class[label]}.png"
        img.save(os.path.join(SAMPLES_DIR, filename))

        seen_per_class[label] += 1
        saved += 1

        if all(seen_per_class[i] >= samples_per_class for i in range(len(class_names))):
            break

    print(f"\nSaved {saved} sample images to: {SAMPLES_DIR}")


def main():
    dataset = load_eurosat()
    class_names = inspect_dataset(dataset)
    train_split, test_split, validation_split = get_splits(dataset)

    print_class_distribution(train_split, class_names, "train")
    print_class_distribution(test_split, class_names, "test")
    print_class_distribution(validation_split, class_names, "validation")

    save_sample_images(train_split, class_names)

    print("\nStep 1 (data.py) complete. Dataset is downloaded, split, and verified.")


if __name__ == "__main__":
    main()