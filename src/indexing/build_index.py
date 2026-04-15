"""
VibeMatch — Phase 4: Build FAISS Index

Runs all game screenshots through the trained image encoder to produce
per-game embeddings, then stores them in a FAISS index for fast retrieval.

Steps:
    1. Load the trained model (best_model.pt)
    2. For each game in the catalogue:
       - Load all 3 screenshots
       - Pass through frozen image encoder → 3 vectors of [256]
       - Average the 3 vectors
       - L2-normalize the result
    3. Store all game embeddings in a FAISS IndexFlatIP (inner product = cosine)
    4. Save the index and metadata to disk

Run from project root: python src/indexing/build_index.py

Output:
    data/processed/game_embeddings.npy   — [N, 256] embedding matrix
    data/processed/game_metadata.json    — ordered list of {app_id, name, tags}
    data/processed/vibematch.index       — FAISS index file
"""
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import faiss
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import get_val_transforms
from src.models.vibematch_model import VibeMatchModel


def main():
    print("=" * 60)
    print("VibeMatch — Build FAISS Index")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # -------------------------------------------------------
    # 1. Load trained model
    # -------------------------------------------------------
    print("\n[1/4] Loading trained model...")
    checkpoint_path = "checkpoints/infonce_v1/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = VibeMatchModel(
        embed_dim=checkpoint['args']['embed_dim'],
        loss_type=checkpoint['args']['loss_type'],
        freeze_text_backbone=True,
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    print(f"  Loaded from epoch {checkpoint['epoch']+1}")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")

    # -------------------------------------------------------
    # 2. Load dataset info
    # -------------------------------------------------------
    print("\n[2/4] Loading dataset...")
    df = pd.read_csv("data/processed/games_master.csv")
    image_dir = Path("data/images")
    transform = get_val_transforms()

    # Filter to games that have all 3 screenshots
    valid_games = []
    for _, row in df.iterrows():
        game_dir = image_dir / str(row['app_id'])
        has_all = all(
            (game_dir / f"screenshot_{i}.jpg").exists()
            for i in range(3)
        )
        if has_all:
            valid_games.append(row)

    valid_df = pd.DataFrame(valid_games)
    print(f"  Total games in CSV: {len(df)}")
    print(f"  Games with all 3 screenshots: {len(valid_df)}")

    # -------------------------------------------------------
    # 3. Generate embeddings
    # -------------------------------------------------------
    print("\n[3/4] Generating embeddings...")
    print("  This takes ~15-20 minutes on GPU...")

    all_embeddings = []
    metadata = []
    batch_size = 32  # Process 32 games at a time (96 images)
    start_time = time.time()

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(valid_df), batch_size), desc="  Encoding"):
            batch_end = min(batch_start + batch_size, len(valid_df))
            batch_df = valid_df.iloc[batch_start:batch_end]

            for _, row in batch_df.iterrows():
                app_id = row['app_id']
                game_dir = image_dir / str(app_id)

                # Load and transform all 3 screenshots
                images = []
                for i in range(3):
                    img_path = game_dir / f"screenshot_{i}.jpg"
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = transform(img)
                        images.append(img)
                    except Exception as e:
                        print(f"\n  Warning: failed to load {img_path}: {e}")
                        continue

                if len(images) == 0:
                    continue

                # Stack and encode
                images_tensor = torch.stack(images).to(device)  # [3, 3, 300, 300]
                embeddings = model.encode_images(images_tensor)  # [3, 256]

                # Average and normalize
                avg_embedding = embeddings.mean(dim=0)  # [256]
                avg_embedding = torch.nn.functional.normalize(avg_embedding, p=2, dim=0)

                all_embeddings.append(avg_embedding.cpu().numpy())
                metadata.append({
                    'app_id': int(app_id),
                    'name': str(row['name']),
                    'tags_string': str(row['tags_string']),
                    'split': str(row['split']),
                })

    elapsed = time.time() - start_time
    print(f"\n  Encoded {len(all_embeddings)} games in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Stack into numpy array
    embeddings_matrix = np.stack(all_embeddings).astype('float32')  # [N, 256]
    print(f"  Embeddings shape: {embeddings_matrix.shape}")
    print(f"  Embedding norm (should be ~1.0): {np.linalg.norm(embeddings_matrix, axis=1).mean():.4f}")

    # -------------------------------------------------------
    # 4. Build FAISS index and save everything
    # -------------------------------------------------------
    print("\n[4/4] Building FAISS index...")

    # Since vectors are L2-normalized, inner product = cosine similarity
    index = faiss.IndexFlatIP(256)
    index.add(embeddings_matrix)

    print(f"  Index size: {index.ntotal} vectors")

    # Quick sanity check: query the first game, it should return itself as #1
    D, I = index.search(embeddings_matrix[:1], 5)
    first_game = metadata[0]['name']
    top5_names = [metadata[i]['name'] for i in I[0]]
    print(f"\n  Sanity check: query '{first_game}'")
    print(f"  Top-5 results:")
    for rank, (name, score) in enumerate(zip(top5_names, D[0]), 1):
        print(f"    {rank}. {name} (score: {score:.4f})")

    # Save everything
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "game_embeddings.npy", embeddings_matrix)
    print(f"\n  Saved embeddings to {output_dir / 'game_embeddings.npy'}")

    with open(output_dir / "game_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  Saved metadata to {output_dir / 'game_metadata.json'}")

    faiss.write_index(index, str(output_dir / "vibematch.index"))
    print(f"  Saved FAISS index to {output_dir / 'vibematch.index'}")

    # Summary stats
    print(f"\n{'=' * 60}")
    print("INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"  Total games indexed: {index.ntotal}")
    print(f"  Embedding dimension: 256")
    print(f"  Index type: Flat Inner Product (exact search)")
    print(f"  Index file size: {(output_dir / 'vibematch.index').stat().st_size / 1e6:.1f} MB")

    # Show split distribution
    splits = pd.DataFrame(metadata)['split'].value_counts()
    print(f"\n  Split distribution in index:")
    for split, count in splits.items():
        print(f"    {split}: {count}")


if __name__ == "__main__":
    main()