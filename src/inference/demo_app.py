"""
VibeMatch — Demo App

Interactive web interface where users upload up to 3 game screenshots
and receive the top-5 most visually similar games from the database.

Run from project root: python src/inference/demo_app.py

Then open http://localhost:7860 in your browser.

How it works:
    1. User uploads 1-3 screenshots
    2. Each screenshot goes through the trained image encoder → 256-dim vector
    3. Vectors are averaged and L2-normalized → single query vector
    4. FAISS searches the index of 114K games for nearest neighbors
    5. Top-5 games are returned with names, tags, similarity scores, and thumbnails
"""
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import faiss
import gradio as gr
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import get_val_transforms
from src.models.vibematch_model import VibeMatchModel


# -------------------------------------------------------
# Global variables (loaded once at startup)
# -------------------------------------------------------
MODEL = None
INDEX = None
METADATA = None
TRANSFORM = None
DEVICE = None


def load_model():
    """Load trained model, FAISS index, and metadata."""
    global MODEL, INDEX, METADATA, TRANSFORM, DEVICE

    print("Loading VibeMatch model and index...")

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {DEVICE}")

    # Load model
    checkpoint_path = "checkpoints/infonce_v1/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    MODEL = VibeMatchModel(
        embed_dim=checkpoint['args']['embed_dim'],
        loss_type=checkpoint['args']['loss_type'],
        freeze_text_backbone=True,
    )
    MODEL.load_state_dict(checkpoint['model_state_dict'], strict=False)
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()
    print(f"  Model loaded (epoch {checkpoint['epoch']+1}, val_loss={checkpoint['val_loss']:.4f})")

    # Load FAISS index
    INDEX = faiss.read_index("data/processed/vibematch.index")
    print(f"  FAISS index loaded ({INDEX.ntotal} games)")

    # Load metadata
    with open("data/processed/game_metadata.json", 'r', encoding='utf-8') as f:
        METADATA = json.load(f)
    print(f"  Metadata loaded ({len(METADATA)} games)")

    # Image transform
    TRANSFORM = get_val_transforms()

    print("  Ready!")


def get_thumbnail_path(app_id):
    """Get the first screenshot path for a game (used as thumbnail)."""
    game_dir = Path("data/images") / str(app_id)
    for i in range(3):
        path = game_dir / f"screenshot_{i}.jpg"
        if path.exists():
            return str(path)
    return None


def retrieve_similar_games(img1, img2, img3):
    """
    Main retrieval function called by Gradio.

    Args:
        img1, img2, img3: PIL Images or None (user may upload 1-3)

    Returns:
        results: List of (image, caption) tuples for Gradio Gallery
    """
    # Collect uploaded images
    images = [img for img in [img1, img2, img3] if img is not None]

    if len(images) == 0:
        return [], "Please upload at least one screenshot."

    # Preprocess and encode
    tensors = []
    for img in images:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert('RGB')
        elif isinstance(img, str):
            img = Image.open(img).convert('RGB')
        else:
            img = img.convert('RGB')
        tensor = TRANSFORM(img)
        tensors.append(tensor)

    batch = torch.stack(tensors).to(DEVICE)  # [N, 3, 300, 300]

    with torch.no_grad():
        embeddings = MODEL.encode_images(batch)  # [N, 256]

    # Average and normalize
    query = embeddings.mean(dim=0, keepdim=True)  # [1, 256]
    query = F.normalize(query, p=2, dim=-1)
    query_np = query.cpu().numpy().astype('float32')

    # Search FAISS
    k = 10  # Retrieve 10, show top 5 (some might not have thumbnails)
    scores, indices = INDEX.search(query_np, k)

    # Build results
    gallery_items = []
    result_text = ""

    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if len(gallery_items) >= 5:
            break

        meta = METADATA[idx]
        thumbnail_path = get_thumbnail_path(meta['app_id'])

        if thumbnail_path is None:
            continue

        # Load thumbnail
        try:
            thumb = Image.open(thumbnail_path).convert('RGB')
        except Exception:
            continue

        # Create caption
        tags_preview = meta['tags_string'][:60]
        caption = f"#{len(gallery_items)+1} | {meta['name']}\nScore: {score:.4f}\nTags: {tags_preview}..."

        gallery_items.append((thumb, caption))

        # Build text summary
        result_text += (
            f"**#{len(gallery_items)}. {meta['name']}**\n"
            f"- Similarity: {score:.4f}\n"
            f"- Tags: {meta['tags_string'][:80]}...\n\n"
        )

    status = f"Found {len(gallery_items)} similar games from {len(images)} uploaded screenshot(s)."

    return gallery_items, result_text


def create_demo():
    """Create the Gradio interface."""

    with gr.Blocks(
        title="VibeMatch — Find Games by Vibe",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            """
            # VibeMatch: Cross-Modal Game Retrieval
            ### Upload 1-3 game screenshots to find visually similar games

            The model analyzes visual aesthetics — art style, color palette, UI design,
            scene composition — and finds games with similar "vibes" from a database
            of 114,000+ Steam games.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Upload Screenshots")
                img1 = gr.Image(label="Screenshot 1", type="pil")
                img2 = gr.Image(label="Screenshot 2 (optional)", type="pil")
                img3 = gr.Image(label="Screenshot 3 (optional)", type="pil")
                search_btn = gr.Button("Find Similar Games", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### Results")
                gallery = gr.Gallery(
                    label="Top 5 Similar Games",
                    columns=5,
                    rows=1,
                    height=250,
                    object_fit="cover",
                )
                results_text = gr.Markdown(label="Details")

        # Connect button to function
        search_btn.click(
            fn=retrieve_similar_games,
            inputs=[img1, img2, img3],
            outputs=[gallery, results_text],
        )

        gr.Markdown(
            """
            ---
            **How it works:** Screenshots are encoded using EfficientNet-B3 trained with
            contrastive learning (CLIP-style) to align visual features with game tags.
            The model was trained on 114K Steam games with 342K screenshots.

            **CS566 Spring 2026** | Beste Nur Pacci & Uğurcan Yılmaz
            """
        )

    return demo


if __name__ == "__main__":
    load_model()
    demo = create_demo()
    demo.launch(
        share=False,       # Set to True if you want a public link
        server_name="0.0.0.0",
        server_port=7860,
    )