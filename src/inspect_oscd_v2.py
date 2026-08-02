"""
inspect_oscd_v2.py — bypass the buggy Array3D decoder in blanchon/OSCD_MSI

The dataset's declared schema says image1/image2 are Array3D(shape=(13,10000,10000)),
but that can't be literally true (the math doesn't fit the file size), so the
per-row images actually vary in size and the library's default decoder breaks
trying to force a fixed reshape. This script reads the raw Arrow data directly
(format="arrow"), which skips that buggy fixed-shape reshape and preserves each
row's true, possibly-ragged structure.

Run directly (uses the already-downloaded local cache, no re-download):
    python src/inspect_oscd_v2.py
"""

from datasets import load_dataset

HF_DATASET_ID = "blanchon/OSCD_MSI"


def describe_nested_list(value, name, max_depth=4):
    """Recursively print length at each nesting level of a raw nested Python list."""
    print(f"\n{name}:")
    current = value
    depth = 0
    while isinstance(current, list) and depth < max_depth:
        print(f"  depth {depth}: length {len(current)}")
        if len(current) == 0:
            break
        current = current[0]
        depth += 1
    print(f"  final element type at depth {depth}: {type(current)}")


def main():
    print(f"Loading dataset '{HF_DATASET_ID}' (should use local cache, no re-download)...")
    dataset = load_dataset(HF_DATASET_ID)
    train_split = dataset["train"]

    print("\n--- Trying raw Arrow-format access (bypasses Array3D fixed-shape decode) ---")
    arrow_view = train_split.with_format("arrow")

    # Fetch row 0 as a raw pyarrow Table (1 row, all columns)
    row0_table = arrow_view[0:1]
    print(f"Columns available: {row0_table.column_names}")

    for col_name in ["image1", "image2"]:
        col = row0_table.column(col_name)
        raw_value = col[0].as_py()  # true nested Python list, no forced reshape
        describe_nested_list(raw_value, col_name)

    # mask uses the standard Image feature, should decode fine on its own
    print("\n--- Mask column (standard HF Image feature) ---")
    mask_only = train_split.select_columns(["mask"])[0]
    mask_img = mask_only["mask"]
    print(f"mask type: {type(mask_img)}")
    if hasattr(mask_img, "size") and hasattr(mask_img, "mode"):
        print(f"mask PIL Image: size={mask_img.size}, mode={mask_img.mode}")
        # Print unique pixel values to confirm binary change mask (0/255 or 0/1)
        import numpy as np
        mask_array = np.array(mask_img)
        print(f"mask array shape: {mask_array.shape}, dtype: {mask_array.dtype}")
        print(f"unique values in mask: {np.unique(mask_array)}")

    print("\nInspection complete.")


if __name__ == "__main__":
    main()