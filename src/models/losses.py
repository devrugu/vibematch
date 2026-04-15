"""
VibeMatch — Loss Functions

Implements the contrastive loss functions that train the image and text
encoders to produce aligned embeddings.

The core idea: if image_i and tags_i belong to the same game, their
embeddings should be close (high cosine similarity). If they belong to
different games, they should be far apart (low cosine similarity).

Two loss functions are implemented:
    1. InfoNCE (primary) — the same loss used in CLIP
    2. TripletLoss (alternative) — classic metric learning loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE loss (also called NT-Xent or CLIP loss).

    Given a batch of N (image, tag) pairs:
    1. Compute the NxN cosine similarity matrix between all image and tag embeddings
    2. Scale by a learned temperature parameter
    3. The diagonal entries are the "positive" pairs (correct matches)
    4. All off-diagonal entries are "negative" pairs (wrong matches)
    5. Apply cross-entropy loss in both directions:
       - Image→Text: for each image, which text is the correct match?
       - Text→Image: for each text, which image is the correct match?
    6. Average the two losses

    Why symmetric? Because we want BOTH the image encoder to find the right
    text AND the text encoder to find the right image. One-directional training
    leads to weaker embeddings.

    Temperature parameter:
    - Controls how "sharp" the similarity distribution is
    - Low temperature (0.01) = very peaked, only the closest match matters
    - High temperature (1.0) = flat, all similarities weighted equally
    - We make it learnable so the model finds the optimal sharpness
    - Initialized at 0.07 (CLIP's default)
    """

    def __init__(self, initial_temperature=0.07):
        super().__init__()

        # Store log(1/temperature) as a learnable parameter
        # We use log scale for numerical stability
        # log(1/0.07) ≈ 2.659
        self.log_temperature = nn.Parameter(
            torch.tensor(1.0 / initial_temperature).log()
        )

    def forward(self, image_embs, text_embs):
        """
        Args:
            image_embs: L2-normalized image embeddings [N, D]
            text_embs: L2-normalized text embeddings [N, D]
            (image_embs[i] and text_embs[i] are a matched pair)

        Returns:
            loss: Scalar loss value
            stats: Dict with useful metrics for logging
        """
        N = image_embs.shape[0]
        device = image_embs.device

        # Compute temperature (clamped for stability)
        # exp(log_temp) gives us 1/temperature, then we take reciprocal
        temperature = (1.0 / self.log_temperature.exp()).clamp(min=0.01, max=1.0)

        # Compute cosine similarity matrix
        # Since embeddings are L2-normalized, dot product = cosine similarity
        # Result: [N, N] matrix where entry [i,j] = similarity(image_i, text_j)
        logits = (image_embs @ text_embs.T) / temperature

        # Labels: the diagonal entries are the correct matches
        # image_0 should match text_0, image_1 should match text_1, etc.
        labels = torch.arange(N, device=device)

        # Cross-entropy loss in both directions
        # Image→Text: given image_i, find the correct text (row-wise softmax)
        loss_i2t = F.cross_entropy(logits, labels)

        # Text→Image: given text_j, find the correct image (column-wise softmax)
        loss_t2i = F.cross_entropy(logits.T, labels)

        # Average both directions
        loss = (loss_i2t + loss_t2i) / 2

        # Compute useful stats for monitoring training health
        with torch.no_grad():
            # Accuracy: how often is the correct match ranked #1?
            i2t_acc = (logits.argmax(dim=1) == labels).float().mean()
            t2i_acc = (logits.T.argmax(dim=1) == labels).float().mean()

            # Average similarity of positive pairs (diagonal)
            pos_sim = logits.diag().mean() * temperature  # Undo temp scaling

            # Average similarity of negative pairs (off-diagonal)
            mask = ~torch.eye(N, dtype=torch.bool, device=device)
            neg_sim = (logits[mask].mean()) * temperature

        stats = {
            'loss': loss.item(),
            'temperature': temperature.item(),
            'i2t_accuracy': i2t_acc.item(),
            't2i_accuracy': t2i_acc.item(),
            'pos_similarity': pos_sim.item(),
            'neg_similarity': neg_sim.item(),
        }

        return loss, stats


class TripletLoss(nn.Module):
    """
    Triplet loss for metric learning (alternative to InfoNCE).

    For each game in the batch:
    - Anchor: the image embedding
    - Positive: the matching tag embedding (same game)
    - Negative: tag embeddings from other games in the batch

    The loss pushes: similarity(anchor, positive) > similarity(anchor, negative) + margin

    We use the "batch hard" mining strategy: for each anchor, we pick the
    hardest negative (most similar wrong match) in the batch. This focuses
    training on the most confusing cases.

    When to prefer Triplet over InfoNCE:
    - Smaller batch sizes (InfoNCE needs many negatives to work well)
    - When you want explicit control over the margin
    - Historical baseline comparison

    When to prefer InfoNCE:
    - Large batches (more negatives = better signal)
    - When you want symmetric training (both directions)
    - Generally achieves better results in practice (CLIP, etc.)
    """

    def __init__(self, margin=0.3):
        """
        Args:
            margin: Minimum gap required between positive and negative similarity.
                    0.3 means: sim(anchor, positive) must exceed
                    sim(anchor, hardest_negative) by at least 0.3
        """
        super().__init__()
        self.margin = margin

    def forward(self, image_embs, text_embs):
        """
        Args:
            image_embs: L2-normalized image embeddings [N, D]
            text_embs: L2-normalized text embeddings [N, D]

        Returns:
            loss: Scalar loss value
            stats: Dict with useful metrics
        """
        N = image_embs.shape[0]

        # Compute all pairwise similarities
        # [N, N] where [i,j] = cosine_similarity(image_i, text_j)
        sim_matrix = image_embs @ text_embs.T

        # Positive similarities: diagonal entries (correct matches)
        pos_sim = sim_matrix.diag()  # [N]

        # For each anchor, find the hardest negative
        # (highest similarity among wrong matches)
        # Mask out the diagonal (positive pairs)
        mask = ~torch.eye(N, dtype=torch.bool, device=image_embs.device)
        neg_sim = sim_matrix.masked_fill(~mask, -float('inf'))
        hardest_neg_sim = neg_sim.max(dim=1).values  # [N]

        # Triplet loss: max(0, neg_sim - pos_sim + margin)
        losses = F.relu(hardest_neg_sim - pos_sim + self.margin)
        loss = losses.mean()

        # Stats for monitoring
        with torch.no_grad():
            # Fraction of triplets where the margin is already satisfied
            satisfied = (losses == 0).float().mean()

        stats = {
            'loss': loss.item(),
            'pos_similarity': pos_sim.mean().item(),
            'neg_similarity': hardest_neg_sim.mean().item(),
            'margin_satisfied': satisfied.item(),
        }

        return loss, stats


# Quick test when running this file directly
if __name__ == "__main__":
    print("Testing loss functions...\n")

    # Create fake embeddings (4 pairs)
    torch.manual_seed(42)
    N, D = 4, 256
    img_embs = F.normalize(torch.randn(N, D), dim=-1)
    txt_embs = F.normalize(torch.randn(N, D), dim=-1)

    # Test InfoNCE
    print("InfoNCE Loss:")
    infonce = InfoNCELoss(initial_temperature=0.07)
    loss, stats = infonce(img_embs, txt_embs)
    print(f"  Loss: {stats['loss']:.4f}")
    print(f"  Temperature: {stats['temperature']:.4f}")
    print(f"  I→T accuracy: {stats['i2t_accuracy']:.2%}")
    print(f"  T→I accuracy: {stats['t2i_accuracy']:.2%}")
    print(f"  Positive similarity: {stats['pos_similarity']:.4f}")
    print(f"  Negative similarity: {stats['neg_similarity']:.4f}")

    # Test Triplet
    print("\nTriplet Loss:")
    triplet = TripletLoss(margin=0.3)
    loss, stats = triplet(img_embs, txt_embs)
    print(f"  Loss: {stats['loss']:.4f}")
    print(f"  Positive similarity: {stats['pos_similarity']:.4f}")
    print(f"  Negative similarity: {stats['neg_similarity']:.4f}")
    print(f"  Margin satisfied: {stats['margin_satisfied']:.2%}")

    print("\n  PASSED!")