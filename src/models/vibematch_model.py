"""
VibeMatch — Combined Model

Wraps both encoders and the loss function into a single module.
This is the top-level model used during training.

Architecture overview:
    ┌─────────────────┐     ┌─────────────────┐
    │  Screenshot      │     │  Tag string      │
    │  [B, 3, 300, 300]│     │  List[str], len B│
    └────────┬────────┘     └────────┬────────┘
             │                       │
             ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐
    │  Image Encoder   │     │  Text Encoder    │
    │  EfficientNet-B3 │     │  Sentence-BERT   │
    │  + projection    │     │  + projection    │
    └────────┬────────┘     └────────┬────────┘
             │                       │
             ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐
    │  Image embedding │     │  Text embedding  │
    │  [B, 256], L2    │     │  [B, 256], L2    │
    └────────┬────────┘     └────────┬────────┘
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Contrastive Loss   │
              │  (InfoNCE / Triplet)│
              └─────────────────────┘
"""
import torch
import torch.nn as nn

# Import from sibling modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.image_encoder import ImageEncoder
from src.models.text_encoder import TextEncoder
from src.models.losses import InfoNCELoss, TripletLoss


class VibeMatchModel(nn.Module):
    """
    The complete VibeMatch model.

    Holds both encoders and the loss function. During training, call forward()
    with a batch of (images, tag_strings) to get the loss. During inference,
    use encode_images() or encode_tags() separately.
    """

    def __init__(
        self,
        embed_dim=256,
        loss_type='infonce',
        temperature=0.07,
        triplet_margin=0.3,
        freeze_text_backbone=True,
        pretrained_image=True,
    ):
        """
        Args:
            embed_dim: Dimension of the shared embedding space
            loss_type: 'infonce' or 'triplet'
            temperature: Initial temperature for InfoNCE
            triplet_margin: Margin for triplet loss
            freeze_text_backbone: Whether to freeze SBERT weights
            pretrained_image: Whether to use ImageNet pretrained weights
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.loss_type = loss_type

        # Create encoders
        self.image_encoder = ImageEncoder(
            embed_dim=embed_dim,
            pretrained=pretrained_image,
        )

        self.text_encoder = TextEncoder(
            embed_dim=embed_dim,
            freeze_backbone=freeze_text_backbone,
        )

        # Create loss function
        if loss_type == 'infonce':
            self.loss_fn = InfoNCELoss(initial_temperature=temperature)
        elif loss_type == 'triplet':
            self.loss_fn = TripletLoss(margin=triplet_margin)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, images, tag_strings):
        """
        Full forward pass for training.

        Args:
            images: Batch of screenshots [B, 3, 300, 300]
            tag_strings: List of tag strings, length B

        Returns:
            loss: Scalar loss value
            stats: Dict with training metrics
            image_embs: Image embeddings [B, embed_dim] (for logging)
            text_embs: Text embeddings [B, embed_dim] (for logging)
        """
        # Encode both modalities
        image_embs = self.image_encoder(images)
        text_embs = self.text_encoder(tag_strings)

        # Compute loss
        loss, stats = self.loss_fn(image_embs, text_embs)

        return loss, stats, image_embs, text_embs

    def encode_images(self, images):
        """
        Encode images only (used at inference time).

        Args:
            images: [B, 3, 300, 300]
        Returns:
            embeddings: [B, embed_dim]
        """
        return self.image_encoder(images)

    def encode_tags(self, tag_strings):
        """
        Encode tag strings only (used for building the index).

        Args:
            tag_strings: List[str]
        Returns:
            embeddings: [B, embed_dim]
        """
        return self.text_encoder(tag_strings)

    def get_parameter_groups(self, backbone_lr=1e-5, head_lr=1e-3):
        """
        Create parameter groups with different learning rates.

        The backbone (EfficientNet-B3) uses a small LR because it's already
        pretrained — we want to fine-tune gently, not destroy the learned features.

        The projection heads and temperature use a larger LR because they're
        initialized randomly and need to learn from scratch.

        Args:
            backbone_lr: Learning rate for EfficientNet-B3 backbone
            head_lr: Learning rate for projection heads + loss params

        Returns:
            List of parameter groups for the optimizer
        """
        # Group 1: Image encoder backbone (gentle fine-tuning)
        backbone_params = list(self.image_encoder.backbone.parameters())

        # Group 2: All projection heads + loss parameters (faster learning)
        head_params = (
            list(self.image_encoder.projection.parameters())
            + list(self.text_encoder.projection.parameters())
            + list(self.loss_fn.parameters())
        )

        # Group 3: Text backbone (only if unfrozen)
        text_backbone_params = [
            p for p in self.text_encoder.sbert.parameters()
            if p.requires_grad
        ]

        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr, 'name': 'image_backbone'},
            {'params': head_params, 'lr': head_lr, 'name': 'heads'},
        ]

        if text_backbone_params:
            param_groups.append({
                'params': text_backbone_params,
                'lr': backbone_lr,
                'name': 'text_backbone',
            })

        return param_groups

    def count_parameters(self):
        """Print a summary of parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        img_backbone = sum(p.numel() for p in self.image_encoder.backbone.parameters())
        img_proj = sum(p.numel() for p in self.image_encoder.projection.parameters())
        txt_backbone = sum(p.numel() for p in self.text_encoder.sbert.parameters())
        txt_proj = sum(p.numel() for p in self.text_encoder.projection.parameters())
        loss_params = sum(p.numel() for p in self.loss_fn.parameters())

        print(f"{'=' * 50}")
        print(f"VibeMatch Model Summary")
        print(f"{'=' * 50}")
        print(f"  Image backbone (EfficientNet-B3): {img_backbone / 1e6:.1f}M")
        print(f"  Image projection head:            {img_proj / 1e6:.3f}M")
        print(f"  Text backbone (SBERT):            {txt_backbone / 1e6:.1f}M")
        print(f"  Text projection head:             {txt_proj / 1e6:.3f}M")
        print(f"  Loss parameters:                  {loss_params}")
        print(f"{'=' * 50}")
        print(f"  Total parameters:     {total / 1e6:.1f}M")
        print(f"  Trainable parameters: {trainable / 1e6:.1f}M")
        print(f"  Frozen parameters:    {(total - trainable) / 1e6:.1f}M")
        print(f"{'=' * 50}")


# Quick test when running this file directly
if __name__ == "__main__":
    print("Testing VibeMatchModel...\n")

    # Create model
    model = VibeMatchModel(
        embed_dim=256,
        loss_type='infonce',
        temperature=0.07,
        freeze_text_backbone=True,
    )

    # Show parameter summary
    model.count_parameters()

    # Test forward pass
    print("\nForward pass test:")
    dummy_images = torch.randn(4, 3, 300, 300)
    dummy_tags = [
        "action, rpg, pixel graphics",
        "casual, puzzle, 2d, colorful",
        "horror, atmospheric, first-person",
        "simulation, strategy, city builder",
    ]

    loss, stats, img_embs, txt_embs = model(dummy_images, dummy_tags)

    print(f"  Loss: {stats['loss']:.4f}")
    print(f"  Image embeddings: {img_embs.shape}")
    print(f"  Text embeddings: {txt_embs.shape}")

    # Test parameter groups
    print("\nParameter groups:")
    param_groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)
    for group in param_groups:
        n_params = sum(p.numel() for p in group['params'])
        print(f"  {group['name']}: {n_params/1e6:.3f}M params, lr={group['lr']}")

    # Test inference methods
    print("\nInference test:")
    with torch.no_grad():
        img_only = model.encode_images(dummy_images)
        txt_only = model.encode_tags(dummy_tags)
    print(f"  encode_images: {img_only.shape}")
    print(f"  encode_tags: {txt_only.shape}")

    print("\n  PASSED!")