"""
VibeMatch setup verification script.
Run: python verify_setup.py
"""
import sys

def check(name, test_fn):
    try:
        result = test_fn()
        print(f"  [OK] {name}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False

print("=" * 60)
print("VibeMatch Environment Verification")
print("=" * 60)

all_ok = True

print("\n1. Python")
all_ok &= check("Version", lambda: sys.version.split()[0])

print("\n2. PyTorch + GPU")
import torch
all_ok &= check("PyTorch version", lambda: torch.__version__)
all_ok &= check("CUDA available", lambda: torch.cuda.is_available())
if torch.cuda.is_available():
    all_ok &= check("GPU name", lambda: torch.cuda.get_device_name(0))
    all_ok &= check("VRAM (GB)", lambda: f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}")
    # Quick GPU test
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.mm(x, x)
    all_ok &= check("GPU compute test", lambda: f"Passed ({y.shape})")

print("\n3. EfficientNet-B3 (image encoder)")
from torchvision import models
model = models.efficientnet_b3(weights='IMAGENET1K_V1')
param_count = sum(p.numel() for p in model.parameters())
all_ok &= check("Load pretrained", lambda: f"{param_count/1e6:.1f}M parameters")

print("\n4. Sentence-BERT (text encoder)")
from sentence_transformers import SentenceTransformer
sbert = SentenceTransformer('all-mpnet-base-v2')
test_emb = sbert.encode(["action, rpg, pixel graphics"])
all_ok &= check("Load + encode", lambda: f"Output dim = {test_emb.shape[1]}")

print("\n5. FAISS")
import faiss
index = faiss.IndexFlatIP(256)
import numpy as np
test_vecs = np.random.randn(100, 256).astype('float32')
faiss.normalize_L2(test_vecs)
index.add(test_vecs)
D, I = index.search(test_vecs[:1], 5)
all_ok &= check("Index + search", lambda: f"Top-5 retrieved, best score = {D[0][0]:.3f}")

print("\n6. Other libraries")
import pandas; all_ok &= check("pandas", lambda: pandas.__version__)
import PIL; all_ok &= check("Pillow", lambda: PIL.__version__)
import aiohttp; all_ok &= check("aiohttp", lambda: aiohttp.__version__)
import sklearn; all_ok &= check("scikit-learn", lambda: sklearn.__version__)
import tqdm; all_ok &= check("tqdm", lambda: tqdm.__version__)

print("\n" + "=" * 60)
if all_ok:
    print("ALL CHECKS PASSED — You're ready to build VibeMatch!")
else:
    print("SOME CHECKS FAILED — Fix the issues above before continuing.")
print("=" * 60)