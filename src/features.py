"""
features.py — Feature extraction from satellite images using a pre-trained CNN

Uses a pre-trained ResNet18 (ImageNet weights) as a fixed feature extractor:
strips the final classification layer and takes the 512-dim penultimate
(avgpool) representation for every image. These embeddings feed Stage 1's
scikit-learn classifier (classifier.py) — we don't fine-tune the CNN, we
just use it to turn each 64x64 satellite image into a compact feature
vector that captures visual structure (edges, textures, shapes).

Run directly:
    python src/features.py
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from datasets import load_dataset

HF_DATASET_ID = "blanchon/EuroSAT_RGB"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization stats, required since ResNet18 was pre-trained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ResNet expects at least 224x224 input; EuroSAT images are 64x64, so we upscale
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class EuroSATTorchDataset(Dataset):
    """Wraps a Hugging Face split so it can be used with a PyTorch DataLoader."""

    def __init__(self, hf_split, transform):
        self.hf_split = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        example = self.hf_split[idx]
        image = example["image"].convert("RGB")
        label = example["label"]
        return self.transform(image), label


def build_feature_extractor():
    """
    Load pre-trained ResNet18 and strip the final fully-connected layer,
    leaving the 512-dim avgpool output as our feature vector.
    """
    print(f"Building ResNet18 feature extractor on device: {DEVICE}")
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()  # replace classification head with a passthrough
    resnet.eval()
    resnet.to(DEVICE)
    return resnet


def extract_features(model, hf_split, split_name):
    """Run every image in a split through the CNN and collect feature vectors + labels."""
    dataset = EuroSATTorchDataset(hf_split, preprocess)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_features = []
    all_labels = []

    total_batches = len(loader)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(DEVICE)
            features = model(images)  # shape: (batch_size, 512)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == total_batches:
                print(f"  [{split_name}] batch {batch_idx + 1}/{total_batches}")

    features_array = np.concatenate(all_features, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)

    print(f"[{split_name}] features shape: {features_array.shape}, labels shape: {labels_array.shape}")
    return features_array, labels_array


def save_features(features, labels, split_name):
    os.makedirs(FEATURES_DIR, exist_ok=True)
    features_path = os.path.join(FEATURES_DIR, f"{split_name}_features.npy")
    labels_path = os.path.join(FEATURES_DIR, f"{split_name}_labels.npy")
    np.save(features_path, features)
    np.save(labels_path, labels)
    print(f"Saved: {features_path}")
    print(f"Saved: {labels_path}")


def main():
    print(f"Loading dataset '{HF_DATASET_ID}' from Hugging Face Hub (cached if already downloaded)...")
    dataset = load_dataset(HF_DATASET_ID)

    model = build_feature_extractor()

    for split_name in ["train", "test", "validation"]:
        print(f"\n--- Extracting features: {split_name} ---")
        features, labels = extract_features(model, dataset[split_name], split_name)
        save_features(features, labels, split_name)

    print("\nStep 2 (features.py) complete. CNN feature vectors saved for all splits.")


if __name__ == "__main__":
    main()