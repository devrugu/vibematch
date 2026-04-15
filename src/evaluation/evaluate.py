"""
VibeMatch — Phase 5: Evaluation

Computes all evaluation metrics on the test set:
    1. Recall@1 and Recall@5 — self-retrieval (can the model find the same game?)
    2. MRR — Mean Reciprocal Rank
    3. Tag Jaccard@5 — do retrieved games share similar tags?
    4. Qualitative examples — show best and worst retrievals

Run from project root: python src/evaluation/evaluate.py

Requires: build_index.py must have been run first.
"""
import sys
import json
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def compute_tag_jaccard(tags_a, tags_b):
    """
    Compute Jaccard similarity between two tag sets.
    Jaccard = |intersection| / |union|
    Returns 0-1 where 1 means identical tags.
    """
    set_a = set(tags_a.lower().split(', '))
    set_b = set(tags_b.lower().split(', '))
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def main():
    print("=" * 60)
    print("VibeMatch — Evaluation")
    print("=" * 60)

    # -------------------------------------------------------
    # Load index and metadata
    # -------------------------------------------------------
    print("\n[1/5] Loading index and metadata...")
    embeddings = np.load("data/processed/game_embeddings.npy")
    index = faiss.read_index("data/processed/vibematch.index")

    with open("data/processed/game_metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"  Index size: {index.ntotal} games")
    print(f"  Embeddings shape: {embeddings.shape}")

    # Separate by split
    split_indices = {'train': [], 'val': [], 'test': []}
    for i, m in enumerate(metadata):
        split_indices[m['split']].append(i)

    for split, indices in split_indices.items():
        print(f"  {split}: {len(indices)} games")

    # -------------------------------------------------------
    # Self-retrieval evaluation on test set
    # -------------------------------------------------------
    print("\n[2/5] Self-retrieval evaluation (test set)...")
    print("  For each test game, query with its embedding and check if")
    print("  the game itself appears in the top-K results.\n")

    test_indices = split_indices['test']
    test_embeddings = embeddings[test_indices]

    # Search against the FULL index (all games, not just test)
    K = 20  # Retrieve top-20 for MRR calculation
    D, I = index.search(test_embeddings, K)

    # Compute metrics
    recall_at_1 = 0
    recall_at_5 = 0
    reciprocal_ranks = []

    for q_idx, (distances, retrieved_indices) in enumerate(zip(D, I)):
        # The query game's index in the full index
        true_idx = test_indices[q_idx]

        # Check if the true game appears in results
        found_at = None
        for rank, ret_idx in enumerate(retrieved_indices):
            if ret_idx == true_idx:
                found_at = rank
                break

        if found_at is not None:
            if found_at < 1:
                recall_at_1 += 1
            if found_at < 5:
                recall_at_5 += 1
            reciprocal_ranks.append(1.0 / (found_at + 1))
        else:
            reciprocal_ranks.append(0.0)

    n_test = len(test_indices)
    r1 = recall_at_1 / n_test
    r5 = recall_at_5 / n_test
    mrr = np.mean(reciprocal_ranks)

    print(f"  Recall@1:  {r1:.4f} ({recall_at_1}/{n_test})")
    print(f"  Recall@5:  {r5:.4f} ({recall_at_5}/{n_test})")
    print(f"  MRR:       {mrr:.4f}")

    # -------------------------------------------------------
    # Tag Jaccard evaluation
    # -------------------------------------------------------
    print("\n[3/5] Tag Jaccard evaluation (test set)...")
    print("  For each test game, compute avg tag overlap with top-5 results.\n")

    jaccard_scores = []
    jaccard_at_1 = []

    for q_idx in range(len(test_indices)):
        query_meta = metadata[test_indices[q_idx]]
        query_tags = query_meta['tags_string']

        # Top-5 retrieved (skip self if it appears at rank 0)
        retrieved = []
        for ret_idx in I[q_idx]:
            if ret_idx != test_indices[q_idx]:  # Skip self
                retrieved.append(ret_idx)
            if len(retrieved) >= 5:
                break

        # Compute Jaccard with each retrieved game
        jaccards = []
        for ret_idx in retrieved:
            ret_tags = metadata[ret_idx]['tags_string']
            j = compute_tag_jaccard(query_tags, ret_tags)
            jaccards.append(j)

        if jaccards:
            jaccard_scores.append(np.mean(jaccards))
            jaccard_at_1.append(jaccards[0])

    avg_jaccard_5 = np.mean(jaccard_scores)
    avg_jaccard_1 = np.mean(jaccard_at_1)

    print(f"  Avg Tag Jaccard@1:  {avg_jaccard_1:.4f}")
    print(f"  Avg Tag Jaccard@5:  {avg_jaccard_5:.4f}")

    # -------------------------------------------------------
    # Per-genre analysis
    # -------------------------------------------------------
    print("\n[4/5] Per-genre Tag Jaccard analysis...")
    print("  Breaking down performance by primary tag.\n")

    # Find each test game's most distinctive tag (least common)
    genre_jaccards = {}
    common_tags = {'singleplayer', 'indie', 'action', 'adventure', 'casual',
                   '2d', '3d', 'simulation', 'strategy', 'rpg', 'atmospheric',
                   'puzzle', 'colorful', 'exploration', 'story rich'}

    for q_idx in range(len(test_indices)):
        query_meta = metadata[test_indices[q_idx]]
        tags = query_meta['tags_string'].split(', ')

        # Use the first non-generic tag as the "genre"
        genre = tags[0]
        for t in tags:
            if t in common_tags:
                genre = t
                break

        if genre not in genre_jaccards:
            genre_jaccards[genre] = []
        genre_jaccards[genre].append(jaccard_scores[q_idx])

    # Show top genres by count
    genre_stats = []
    for genre, scores in genre_jaccards.items():
        if len(scores) >= 20:  # Only genres with enough samples
            genre_stats.append({
                'genre': genre,
                'count': len(scores),
                'avg_jaccard': np.mean(scores),
            })

    genre_stats.sort(key=lambda x: x['avg_jaccard'], reverse=True)

    print(f"  {'Genre':<20} {'Count':>6} {'Avg Jaccard@5':>14}")
    print(f"  {'-'*20} {'-'*6} {'-'*14}")
    for gs in genre_stats[:15]:
        print(f"  {gs['genre']:<20} {gs['count']:>6} {gs['avg_jaccard']:>14.4f}")

    # -------------------------------------------------------
    # Qualitative examples
    # -------------------------------------------------------
    print(f"\n[5/5] Qualitative examples...")

    # Best retrievals (highest avg Jaccard)
    sorted_by_jaccard = sorted(
        range(len(test_indices)),
        key=lambda i: jaccard_scores[i],
        reverse=True
    )

    print(f"\n  TOP 5 BEST RETRIEVALS (highest tag overlap):")
    print(f"  {'='*55}")
    for rank in range(5):
        q_idx = sorted_by_jaccard[rank]
        query_meta = metadata[test_indices[q_idx]]

        print(f"\n  Query: {query_meta['name']}")
        print(f"  Tags: {query_meta['tags_string'][:70]}...")
        print(f"  Jaccard@5: {jaccard_scores[q_idx]:.4f}")
        print(f"  Retrieved:")

        retrieved = []
        for ret_idx in I[q_idx]:
            if ret_idx != test_indices[q_idx]:
                retrieved.append(ret_idx)
            if len(retrieved) >= 3:
                break

        for i, ret_idx in enumerate(retrieved, 1):
            ret_meta = metadata[ret_idx]
            j = compute_tag_jaccard(query_meta['tags_string'], ret_meta['tags_string'])
            print(f"    {i}. {ret_meta['name']} (Jaccard: {j:.3f})")

    # Worst retrievals
    print(f"\n  TOP 5 WORST RETRIEVALS (lowest tag overlap):")
    print(f"  {'='*55}")
    for rank in range(5):
        q_idx = sorted_by_jaccard[-(rank+1)]
        query_meta = metadata[test_indices[q_idx]]

        print(f"\n  Query: {query_meta['name']}")
        print(f"  Tags: {query_meta['tags_string'][:70]}...")
        print(f"  Jaccard@5: {jaccard_scores[q_idx]:.4f}")
        print(f"  Retrieved:")

        retrieved = []
        for ret_idx in I[q_idx]:
            if ret_idx != test_indices[q_idx]:
                retrieved.append(ret_idx)
            if len(retrieved) >= 3:
                break

        for i, ret_idx in enumerate(retrieved, 1):
            ret_meta = metadata[ret_idx]
            j = compute_tag_jaccard(query_meta['tags_string'], ret_meta['tags_string'])
            print(f"    {i}. {ret_meta['name']} (Jaccard: {j:.3f})")

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Dataset: {index.ntotal} total games, {n_test} test games")
    print(f"  Embedding dim: 256")
    print(f"")
    print(f"  Self-Retrieval Metrics:")
    print(f"    Recall@1:        {r1:.4f}")
    print(f"    Recall@5:        {r5:.4f}")
    print(f"    MRR:             {mrr:.4f}")
    print(f"")
    print(f"  Similarity Metrics:")
    print(f"    Tag Jaccard@1:   {avg_jaccard_1:.4f}")
    print(f"    Tag Jaccard@5:   {avg_jaccard_5:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()