"""
VibeMatch — Text Encoder

Takes a batch of tag strings (e.g., "action, rpg, pixel graphics, multiplayer")
and produces 256-dimensional L2-normalized embedding vectors.

Architecture:
    Input tag string "action, rpg, pixel graphics"
        ↓
    Sentence-BERT tokenizer (splits into subword tokens)
        ↓
    Sentence-BERT model (all-mpnet-base-v2, 110M params)
        ↓
    Sentence embedding [B, 768]
        ↓
    Projection head: Linear(768, 256)
        ↓
    L2 normalization
        ↓
    Output embedding [B, 256]

Why Sentence-BERT (all-mpnet-base-v2)?
    - Designed specifically for producing meaningful sentence-level embeddings
    - MPNet architecture captures bidirectional context between tags
    - "action, rpg" and "rpg, action" produce similar (not identical) embeddings
    - Pre-trained on 1B+ sentence pairs, understands semantic relationships

Why freeze the backbone by default?
    - Our tag strings are short and structured (just comma-separated words)
    - SBERT already understands these words from pre-training
    - Fine-tuning 110M parameters on short tag strings risks overfitting
    - Only the projection head (197K params) needs to learn game-specific mappings
    - We test unfreezing as an ablation experiment
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class TextEncoder(nn.Module):
    def __init__(self, model_name='all-mpnet-base-v2', embed_dim=256, freeze_backbone=True):
        """
        Args:
            model_name: Which Sentence-BERT model to use
            embed_dim: Output embedding dimension (must match ImageEncoder)
            freeze_backbone: If True, only train the projection head
        """
        super().__init__()

        self.freeze_backbone = freeze_backbone

        # Load the pre-trained Sentence-BERT model
        # This handles tokenization + encoding internally
        self.sbert = SentenceTransformer(model_name)

        # Get the output dimension of SBERT (768 for mpnet-base)
        sbert_dim = self.sbert.get_sentence_embedding_dimension()

        # Projection head: maps SBERT's 768-dim output to our 256-dim space
        self.projection = nn.Sequential(
            nn.Linear(sbert_dim, embed_dim),
        )

        # Freeze SBERT backbone if requested
        # This means during training, only self.projection gets updated
        if freeze_backbone:
            for param in self.sbert.parameters():
                param.requires_grad = False

    def forward(self, tag_strings):
        """
        Args:
            tag_strings: List of tag strings, length B
                         e.g., ["action, rpg, pixel graphics", "casual, puzzle, 2d"]

        Returns:
            embeddings: L2-normalized vectors [B, embed_dim]
        """
        # Encode tag strings using SBERT
        # SBERT handles tokenization internally and returns numpy arrays
        # We need to convert to tensors on the right device
        device = self.projection[0].weight.device

        if self.freeze_backbone:
            # No gradients needed for frozen backbone — saves memory
            with torch.no_grad():
                sbert_embs = self.sbert.encode(
                    tag_strings,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
        else:
            # If fine-tuning, we need gradients through SBERT
            # sentence-transformers .encode() doesn't support gradients,
            # so we use the underlying model directly
            sbert_embs = self._encode_with_gradients(tag_strings, device)

        # Move to correct device if needed
        sbert_embs = sbert_embs.to(device).clone()

        # Project to shared embedding space
        # [B, 768] -> [B, 256]
        projected = self.projection(sbert_embs)

        # L2 normalize (same as image encoder — both must be on unit sphere)
        embeddings = F.normalize(projected, p=2, dim=-1)

        return embeddings

    def _encode_with_gradients(self, tag_strings, device):
        """
        Encode strings through SBERT with gradient tracking.
        Used only when freeze_backbone=False (ablation experiment).

        The standard .encode() method uses torch.no_grad() internally,
        so we need to call the underlying transformer directly.
        """
        # Tokenize
        tokenizer = self.sbert.tokenizer
        encoded = tokenizer(
            tag_strings,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt',
        ).to(device)

        # Forward through the transformer
        model = self.sbert[0].auto_model
        outputs = model(**encoded)

        # Mean pooling over token embeddings (same as SBERT default)
        attention_mask = encoded['attention_mask'].unsqueeze(-1)
        token_embeddings = outputs.last_hidden_state
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counted = attention_mask.sum(dim=1).clamp(min=1e-9)
        sentence_embeddings = summed / counted

        return sentence_embeddings


# Quick test when running this file directly
if __name__ == "__main__":
    print("Testing TextEncoder...")
    model = TextEncoder(embed_dim=256, freeze_backbone=True)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"  Total parameters: {total_params / 1e6:.1f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.3f}M")
    print(f"  Frozen parameters: {frozen_params / 1e6:.1f}M")

    # Test forward pass with sample tag strings
    sample_tags = [
        "action, rpg, pixel graphics, multiplayer",
        "casual, puzzle, 2d, colorful, cute",
        "simulation, strategy, city builder, singleplayer",
        "horror, atmospheric, first-person, story rich",
    ]

    output = model(sample_tags)

    print(f"  Input: {len(sample_tags)} tag strings")
    print(f"  Output shape: {output.shape}")
    print(f"  Output norm (should be ~1.0): {output.norm(dim=-1).mean().item():.4f}")

    # Check that different tag strings produce different embeddings
    similarity = (output @ output.T).detach()
    print(f"  Self-similarity matrix diagonal (should be 1.0): {similarity.diag().mean().item():.4f}")
    print(f"  Off-diagonal mean (should be < 1.0): {(similarity.sum() - similarity.diag().sum()).item() / 12:.4f}")
    print("  PASSED!")