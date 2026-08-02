"""
inspect_oscd.py — one-off inspection of the OSCD_MSI dataset structure

Before writing the full change-detection pipeline (detector.py), we need to
know the real schema: field names, image dimensions, band count/order, and
how the ground-truth change mask is represented. This script downloads the
dataset once and prints everything needed to design detector.py correctly.

Run directly:
    python src/inspect_oscd.py
"""

from datasets import load_dataset

HF_DATASET_ID = "blanchon/OSCD_MSI"


def main():
    print(f"Loading dataset '{HF_DATASET_ID}' from Hugging Face Hub...")
    dataset = load_dataset(HF_DATASET_ID)

    print("\n--- Splits ---")
    for split_name, split in dataset.items():
        print(f"  {split_name}: {len(split)} examples")

    train_split = dataset["train"]

    print("\n--- Features (schema) ---")
    print(train_split.features)

    print("\n--- First example: top-level keys ---")
    example = train_split[0]
    for key in example.keys():
        value = example[key]
        value_type = type(value)
        print(f"  key='{key}'  type={value_type}")

        # If it looks like a PIL image, print size/mode
        if hasattr(value, "size") and hasattr(value, "mode"):
            print(f"      -> PIL Image: size={value.size}, mode={value.mode}")
        # If it's a list (could be a list of band images, or nested structure)
        elif isinstance(value, list):
            print(f"      -> list of length {len(value)}")
            if len(value) > 0:
                first_item = value[0]
                print(f"      -> first item type: {type(first_item)}")
                if hasattr(first_item, "size") and hasattr(first_item, "mode"):
                    print(f"      -> first item is PIL Image: size={first_item.size}, mode={first_item.mode}")
        # If it's a string, print it directly (likely metadata like location name)
        elif isinstance(value, str):
            print(f"      -> value: {value}")
        else:
            print(f"      -> value preview: {str(value)[:200]}")

    print("\nInspection complete. Share this output before we write detector.py.")


if __name__ == "__main__":
    main()