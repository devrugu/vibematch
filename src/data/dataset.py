"""
VibeMatch — Dataset

PyTorch Dataset that loads game screenshots and tag strings.
Applies image augmentations for training and clean transforms for validation.

How it works:
    1. Reads the master CSV to get app_ids, tag strings, and splits
    2. For each game, randomly picks 1 of 3 screenshots (training augmentation)
    3. Applies image transforms (resize, crop, flip, color jitter, normalize)
    4. Returns (image_tensor, tag_string) pairs

Data augmentation strategy:
    - Random screenshot selection: each epoch sees different screenshots
    - RandomResizedCrop: random region of the image, scaled to 300x300
    - RandomHorizontalFlip: games look the same mirrored
    - ColorJitter: slight color/brightness variation builds robustness
    - These augmentations prevent overfitting and teach the model to focus
      on game aesthetics rather than specific screenshot details

ImageNet normalization:
    - We use ImageNet mean/std because EfficientNet-B3 was trained with them
    - The model expects inputs normalized this way
    - mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
"""
import os
import random
import pandas as pd
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms():
    """
    Training transforms with data augmentation.

    RandomResizedCrop(300): picks a random region (80-100% of the image area),
    resizes it to 300x300. This is the most important augmentation — it forces
    the model to recognize game aesthetics from partial views.

    RandomHorizontalFlip: 50% chance of mirroring. Game visuals are usually
    symmetric in style (a pixel art game looks like pixel art whether flipped
    or not).

    ColorJitter: slight random changes to brightness, contrast, saturation.
    Teaches the model that a slightly darker screenshot is still the same game.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(300, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms():
    """
    Validation/test transforms — no augmentation, deterministic.

    Resize to 320 (slightly larger), then center crop to 300x300.
    This ensures consistent evaluation across epochs.
    """
    return transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class GameDataset(Dataset):
    """
    PyTorch Dataset for VibeMatch training.

    Each item is a (image_tensor, tag_string) pair representing one game.
    During training, a random screenshot is selected from the 3 available.
    """

    def __init__(self, csv_path, image_dir, split='train', transform=None):
        """
        Args:
            csv_path: Path to games_master.csv
            image_dir: Path to data/images/ directory
            split: 'train', 'val', or 'test'
            transform: Image transform pipeline (uses defaults if None)
        """
        self.image_dir = Path(image_dir)
        self.split = split

        # Load and filter to requested split
        df = pd.read_csv(csv_path)
        self.df = df[df['split'] == split].reset_index(drop=True)

        # Filter out games with missing screenshots
        valid_mask = []
        for _, row in self.df.iterrows():
            game_dir = self.image_dir / str(row['app_id'])
            has_any = any(
                (game_dir / f"screenshot_{i}.jpg").exists()
                for i in range(3)
            )
            valid_mask.append(has_any)

        self.df = self.df[valid_mask].reset_index(drop=True)

        # Set transforms
        if transform is not None:
            self.transform = transform
        elif split == 'train':
            self.transform = get_train_transforms()
        else:
            self.transform = get_val_transforms()

        print(f"  {split} dataset: {len(self.df)} games loaded")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            image: Transformed image tensor [3, 300, 300]
            tag_string: Comma-separated tag string
        """
        row = self.df.iloc[idx]
        app_id = row['app_id']
        tag_string = row['tags_string']

        # Randomly select one of 3 screenshots (training augmentation)
        # For validation, we always use screenshot_0 for consistency
        game_dir = self.image_dir / str(app_id)

        if self.split == 'train':
            # Find which screenshots exist for this game
            available = [
                i for i in range(3)
                if (game_dir / f"screenshot_{i}.jpg").exists()
            ]
            img_idx = random.choice(available)
        else:
            # Validation/test: always use first available
            for i in range(3):
                if (game_dir / f"screenshot_{i}.jpg").exists():
                    img_idx = i
                    break

        img_path = game_dir / f"screenshot_{img_idx}.jpg"

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # If image is corrupted, return a black image
            # (this should be very rare)
            print(f"  Warning: failed to load {img_path}: {e}")
            image = Image.new('RGB', (300, 300), (0, 0, 0))

        # Apply transforms
        image = self.transform(image)

        return image, tag_string


def collate_fn(batch):
    """
    Custom collate function for DataLoader.

    Default collate can't handle mixed (tensor, string) batches.
    We stack the image tensors and keep tag strings as a list.

    Args:
        batch: List of (image_tensor, tag_string) tuples

    Returns:
        images: Stacked tensor [B, 3, 300, 300]
        tag_strings: List of strings, length B
    """
    images, tag_strings = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(tag_strings)


# Quick test when running this file directly
if __name__ == "__main__":
    print("Testing GameDataset...\n")

    csv_path = "data/processed/games_master.csv"
    image_dir = "data/images"

    # Test train dataset
    print("Loading train split:")
    train_dataset = GameDataset(csv_path, image_dir, split='train')
    print(f"  Length: {len(train_dataset)}")

    # Get a sample
    image, tags = train_dataset[0]
    print(f"  Sample image shape: {image.shape}")
    print(f"  Sample image dtype: {image.dtype}")
    print(f"  Sample image range: [{image.min():.2f}, {image.max():.2f}]")
    print(f"  Sample tags: {tags[:80]}...")

    # Test DataLoader
    print("\nTesting DataLoader:")
    loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,  # 0 for testing, increase for training
        collate_fn=collate_fn,
    )

    batch_images, batch_tags = next(iter(loader))
    print(f"  Batch images shape: {batch_images.shape}")
    print(f"  Batch tags count: {len(batch_tags)}")
    print(f"  First tag: {batch_tags[0][:60]}...")

    # Test val dataset
    print("\nLoading val split:")
    val_dataset = GameDataset(csv_path, image_dir, split='val')
    print(f"  Length: {len(val_dataset)}")

    print("\n  PASSED!")