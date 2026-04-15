"""
VibeMatch — Training Script

Trains the VibeMatch model using contrastive learning.
Handles the complete training pipeline: data loading, optimization,
validation, logging, and checkpointing.

Run from project root:
    python src/training/train.py

Options:
    --epochs N          Number of training epochs (default: 40)
    --batch_size N      Batch size (default: 128)
    --backbone_lr F     Learning rate for EfficientNet backbone (default: 1e-5)
    --head_lr F         Learning rate for projection heads (default: 1e-3)
    --embed_dim N       Embedding dimension (default: 256)
    --loss_type STR     'infonce' or 'triplet' (default: infonce)
    --num_workers N     DataLoader workers (default: 4)
    --resume PATH       Resume from checkpoint
    --experiment STR    Experiment name for logging (default: 'default')

Training strategy:
    - AdamW optimizer with differential learning rates
    - Linear warmup (500 steps) + cosine decay schedule
    - Gradient clipping (max norm 1.0)
    - Validation every epoch with Recall@5
    - Save best model by validation loss
    - Early stopping with patience of 10 epochs
"""
import os
import sys
import time
import math
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.vibematch_model import VibeMatchModel
from src.data.dataset import GameDataset, collate_fn


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Learning rate schedule: linear warmup then cosine decay.

    Why warmup?
        The model starts with random projection heads. If we use a large
        learning rate immediately, the gradients from random outputs create
        wild updates that can destabilize the pretrained backbone.
        Warmup gradually increases the LR, giving the heads time to produce
        reasonable embeddings before we start aggressively updating.

    Why cosine decay?
        After warmup, the LR follows a smooth cosine curve from peak to ~0.
        This is gentler than step decay and generally gives better final
        performance in contrastive learning.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup: 0 -> 1 over warmup steps
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay: 1 -> 0 over remaining steps
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_recall_at_k(image_embs, text_embs, k=5):
    """
    Compute Recall@K on a set of image-text pairs.

    For each image, we check if its matching text is in the top-K
    most similar texts (and vice versa).

    This is our primary evaluation metric during training.

    Args:
        image_embs: [N, D] normalized image embeddings
        text_embs: [N, D] normalized text embeddings
        k: Number of top results to check

    Returns:
        i2t_recall: Image-to-text Recall@K
        t2i_recall: Text-to-image Recall@K
    """
    # Similarity matrix
    sim = image_embs @ text_embs.T  # [N, N]

    # Image→Text: for each image, is the correct text in top-K?
    topk_i2t = sim.topk(k, dim=1).indices  # [N, K]
    labels = torch.arange(sim.shape[0], device=sim.device).unsqueeze(1)  # [N, 1]
    i2t_recall = (topk_i2t == labels).any(dim=1).float().mean().item()

    # Text→Image: for each text, is the correct image in top-K?
    topk_t2i = sim.T.topk(k, dim=1).indices
    t2i_recall = (topk_t2i == labels).any(dim=1).float().mean().item()

    return i2t_recall, t2i_recall


@torch.no_grad()
def validate(model, val_loader, device):
    """
    Run validation: compute loss and Recall@5 on the validation set.
    """
    model.eval()

    total_loss = 0
    all_image_embs = []
    all_text_embs = []
    num_batches = 0

    for images, tag_strings in val_loader:
        images = images.to(device)

        loss, stats, image_embs, text_embs = model(images, tag_strings)

        total_loss += loss.item()
        all_image_embs.append(image_embs.cpu())
        all_text_embs.append(text_embs.cpu())
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)

    # Compute Recall@5 on collected embeddings
    # Limit to first 2000 pairs (Recall computation is O(N^2))
    all_image_embs = torch.cat(all_image_embs, dim=0)[:2000]
    all_text_embs = torch.cat(all_text_embs, dim=0)[:2000]
    i2t_r5, t2i_r5 = compute_recall_at_k(all_image_embs, all_text_embs, k=5)

    model.train()

    return {
        'val_loss': avg_loss,
        'val_i2t_recall@5': i2t_r5,
        'val_t2i_recall@5': t2i_r5,
    }


def train(args):
    """Main training function."""

    print("=" * 60)
    print(f"VibeMatch Training — {args.experiment}")
    print("=" * 60)

    # -------------------------------------------------------
    # Setup
    # -------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create output directories
    checkpoint_dir = Path("checkpoints") / args.experiment
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("runs") / args.experiment
    writer = SummaryWriter(log_dir=str(log_dir))

    # -------------------------------------------------------
    # Data
    # -------------------------------------------------------
    print("\nLoading datasets...")
    train_dataset = GameDataset(
        csv_path="data/processed/games_master.csv",
        image_dir="data/images",
        split='train',
    )
    val_dataset = GameDataset(
        csv_path="data/processed/games_master.csv",
        image_dir="data/images",
        split='val',
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,       # Faster CPU→GPU transfer
        drop_last=True,        # Drop incomplete last batch (contrastive learning
                               # needs consistent batch sizes for fair negatives)
        persistent_workers=True if args.num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    print(f"  Train batches per epoch: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # -------------------------------------------------------
    # Model
    # -------------------------------------------------------
    print("\nCreating model...")
    model = VibeMatchModel(
        embed_dim=args.embed_dim,
        loss_type=args.loss_type,
        temperature=0.07,
        freeze_text_backbone=True,
    )
    model.count_parameters()
    model = model.to(device)

    # -------------------------------------------------------
    # Optimizer and Scheduler
    # -------------------------------------------------------
    param_groups = model.get_parameter_groups(
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=1e-4,
    )

    # Learning rate schedule
    total_steps = len(train_loader) * args.epochs
    warmup_steps = 500
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\nOptimizer: AdamW (weight_decay=1e-4)")
    print(f"Schedule: {warmup_steps} warmup steps, {total_steps} total steps")
    print(f"Backbone LR: {args.backbone_lr}, Head LR: {args.head_lr}")

    # -------------------------------------------------------
    # Resume from checkpoint if requested
    # -------------------------------------------------------
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0

    if args.resume:
        print(f"\nResuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"  Resumed at epoch {start_epoch}, best val loss: {best_val_loss:.4f}")

    # -------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Starting training for {args.epochs} epochs...")
    print(f"{'=' * 60}\n")

    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        epoch_loss = 0
        epoch_steps = 0

        accumulation_steps = 2  # Simulate batch_size * 2 = effective 128
        optimizer.zero_grad()

        for batch_idx, (images, tag_strings) in enumerate(train_loader):
            images = images.to(device)

            # Forward pass
            loss, stats, _, _ = model(images, tag_strings)

            # Scale loss by accumulation steps so gradients average correctly
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()

            # Only update weights every accumulation_steps batches
            if (batch_idx + 1) % accumulation_steps == 0:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            else:
                grad_norm = torch.tensor(0.0)

            # Track progress
            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            # Log to TensorBoard every 50 steps
            if global_step % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/grad_norm', grad_norm.item(), global_step)
                writer.add_scalar('train/lr', current_lr, global_step)

                if 'temperature' in stats:
                    writer.add_scalar('train/temperature', stats['temperature'], global_step)
                if 'i2t_accuracy' in stats:
                    writer.add_scalar('train/i2t_accuracy', stats['i2t_accuracy'], global_step)
                    writer.add_scalar('train/t2i_accuracy', stats['t2i_accuracy'], global_step)

                writer.add_scalar('train/pos_similarity', stats['pos_similarity'], global_step)
                writer.add_scalar('train/neg_similarity', stats['neg_similarity'], global_step)

            # Print progress every 100 steps
            if (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - epoch_start
                eta = elapsed / (batch_idx + 1) * (len(train_loader) - batch_idx - 1)
                print(
                    f"  Epoch {epoch+1}/{args.epochs} | "
                    f"Step {batch_idx+1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"ETA: {eta:.0f}s"
                )

        # -------------------------------------------------------
        # End of epoch
        # -------------------------------------------------------
        avg_train_loss = epoch_loss / max(1, epoch_steps)
        epoch_time = time.time() - epoch_start

        # Validation
        val_metrics = validate(model, val_loader, device)

        # Log epoch metrics
        writer.add_scalar('epoch/train_loss', avg_train_loss, epoch)
        writer.add_scalar('epoch/val_loss', val_metrics['val_loss'], epoch)
        writer.add_scalar('epoch/val_i2t_recall@5', val_metrics['val_i2t_recall@5'], epoch)
        writer.add_scalar('epoch/val_t2i_recall@5', val_metrics['val_t2i_recall@5'], epoch)

        # Print epoch summary
        print(
            f"\n  Epoch {epoch+1}/{args.epochs} complete ({epoch_time:.0f}s)\n"
            f"    Train loss: {avg_train_loss:.4f}\n"
            f"    Val loss:   {val_metrics['val_loss']:.4f}\n"
            f"    Val I→T Recall@5: {val_metrics['val_i2t_recall@5']:.4f}\n"
            f"    Val T→I Recall@5: {val_metrics['val_t2i_recall@5']:.4f}\n"
        )

        # -------------------------------------------------------
        # Checkpointing
        # -------------------------------------------------------
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_metrics['val_loss'],
            'best_val_loss': best_val_loss,
            'args': vars(args),
        }

        # Save every 5 epochs
        if (epoch + 1) % 5 == 0:
            path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(checkpoint_data, path)
            print(f"    Saved checkpoint: {path}")

        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
            path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint_data, path)
            print(f"    New best model! Val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"    No improvement ({patience_counter}/{args.patience})")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n  Early stopping after {patience_counter} epochs without improvement.")
            break

    # -------------------------------------------------------
    # Done
    # -------------------------------------------------------
    writer.close()

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best model saved to: {checkpoint_dir / 'best_model.pt'}")
    print(f"  TensorBoard logs: {log_dir}")
    print(f"\n  View logs with: tensorboard --logdir runs/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VibeMatch")
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--backbone_lr', type=float, default=1e-5)
    parser.add_argument('--head_lr', type=float, default=1e-3)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--loss_type', type=str, default='infonce', choices=['infonce', 'triplet'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--experiment', type=str, default='default')
    args = parser.parse_args()

    train(args)