"""
VibeMatch — Image Encoder

Takes a batch of game screenshot images and produces 256-dimensional
L2-normalized embedding vectors.

Architecture:
    Input image [B, 3, 300, 300]
        ↓
    EfficientNet-B3 backbone (pretrained on ImageNet)
        ↓
    Pooled features [B, 1536]
        ↓
    Projection head: Linear(1536, 256)
        ↓
    L2 normalization
        ↓
    Output embedding [B, 256]

Why EfficientNet-B3?
    - Strong accuracy/compute trade-off (12.2M params vs ResNet-50's 25.6M)
    - Native input size is 300x300, matching our target
    - Pretrained ImageNet weights give us a head start on visual features
    - The compound scaling (depth + width + resolution) captures both
      fine details (textures, UI elements) and global structure (scene layout)

Why 256 dimensions?
    - Large enough to capture game aesthetics (genre, art style, mood)
    - Small enough for fast FAISS nearest-neighbor search at inference
    - Standard choice in contrastive learning literature (CLIP uses 512,
      but we have a simpler domain)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256, pretrained=True):
        """
        Args:
            embed_dim: Output embedding dimension (default 256)
            pretrained: Whether to load ImageNet weights (always True for us)
        """
        super().__init__()

        # Load EfficientNet-B3 with pretrained ImageNet weights
        # This gives us a powerful visual feature extractor out of the box
        if pretrained:
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
        else:
            weights = None

        efficientnet = models.efficientnet_b3(weights=weights)

        # EfficientNet structure:
        #   .features    -> convolutional backbone (extracts visual features)
        #   .avgpool     -> adaptive average pooling (any input size -> 1x1)
        #   .classifier  -> final classification layer (1000 ImageNet classes)
        #
        # We keep features + avgpool, and REPLACE the classifier with our
        # own projection head that maps to embed_dim instead of 1000 classes.

        self.backbone = efficientnet.features      # Conv layers
        self.pool = efficientnet.avgpool            # Adaptive avg pool -> [B, 1536, 1, 1]

        # The backbone outputs 1536 channels for EfficientNet-B3
        # We project this down to our embedding dimension
        self.projection = nn.Sequential(
            nn.Dropout(p=0.3),      # Regularization (same rate as original EfficientNet-B3)
            nn.Linear(1536, embed_dim),
        )

    def forward(self, x):
        """
        Args:
            x: Batch of images [B, 3, 300, 300] (normalized with ImageNet stats)

        Returns:
            embeddings: L2-normalized vectors [B, embed_dim]
        """
        # Extract visual features
        # Input: [B, 3, 300, 300] -> Output: [B, 1536, 10, 10]
        features = self.backbone(x)

        # Global average pooling
        # [B, 1536, 10, 10] -> [B, 1536, 1, 1] -> [B, 1536]
        pooled = self.pool(features).flatten(1)

        # Project to embedding space
        # [B, 1536] -> [B, 256]
        projected = self.projection(pooled)

        # L2 normalize so all vectors lie on the unit hypersphere
        # This is critical for contrastive learning:
        # - Cosine similarity becomes a simple dot product
        # - Prevents the model from cheating by just scaling vectors up
        # - Makes the temperature parameter in InfoNCE meaningful
        embeddings = F.normalize(projected, p=2, dim=-1)

        return embeddings


# Quick test when running this file directly
if __name__ == "__main__":
    print("Testing ImageEncoder...")
    model = ImageEncoder(embed_dim=256, pretrained=True)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    projection_params = sum(p.numel() for p in model.projection.parameters())

    print(f"  Total parameters: {total_params / 1e6:.1f}M")
    print(f"  Backbone parameters: {backbone_params / 1e6:.1f}M")
    print(f"  Projection parameters: {projection_params / 1e6:.1f}M")

    # Test forward pass with dummy data
    dummy_input = torch.randn(4, 3, 300, 300)  # Batch of 4 images
    output = model(dummy_input)

    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output norm (should be ~1.0): {output.norm(dim=-1).mean().item():.4f}")
    print("  PASSED!")