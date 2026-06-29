from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    pass

HAS_SENTENCE_TRANSFORMERS = False
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    pass


def info_nce_loss(
    embeddings: np.ndarray,
    temperature: float = 0.05,
) -> float:
    if embeddings.shape[0] < 2:
        return 0.0
    if HAS_TORCH:
        return _info_nce_torch(embeddings, temperature)
    return _info_nce_numpy(embeddings, temperature)


def _info_nce_torch(
    embeddings: np.ndarray,
    temperature: float = 0.05,
) -> float:
    if not HAS_TORCH:
        return _info_nce_numpy(embeddings, temperature)
    emb = torch.from_numpy(embeddings).float()
    emb = F.normalize(emb, p=2, dim=1)
    sim = emb @ emb.T / temperature
    batch_size = emb.shape[0]
    labels = torch.arange(batch_size, device=emb.device)
    loss = F.cross_entropy(sim, labels)
    return float(loss.item())


def _info_nce_numpy(
    embeddings: np.ndarray,
    temperature: float = 0.05,
) -> float:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    emb_norm = embeddings / norms
    sim = (emb_norm @ emb_norm.T) / temperature
    exp_sim = np.exp(sim - np.max(sim, axis=1, keepdims=True))
    pos = np.diag(exp_sim)
    neg = np.sum(exp_sim, axis=1) - pos
    neg = np.maximum(neg, 1e-10)
    loss = -np.mean(np.log(pos / (pos + neg)))
    return float(loss)


def compute_simcse_embeddings(
    sentences: list[str],
    dropout_mask: float = 0.1,
) -> np.ndarray | None:
    if not HAS_SENTENCE_TRANSFORMERS or len(sentences) < 2:
        return None
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        base = model.encode(sentences, show_progress_bar=False)
        if dropout_mask > 0 and HAS_TORCH:
            import torch
            with torch.no_grad():
                noisy = base + np.random.normal(
                    0, dropout_mask, size=base.shape
                ).astype(np.float32)
            combined = np.concatenate([base, noisy], axis=0)
        else:
            combined = base
        return combined.astype(np.float32)
    except Exception as exc:
        logger.debug("SimCSE embedding failed: %s", exc)
        return None


def estimate_anisotropy(embeddings: np.ndarray) -> dict:
    if embeddings.shape[0] < 3:
        return {"anisotropy_score": 1.0, "condition_number": 1.0, "uniformity": 1.0}
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    emb_norm = embeddings / norms
    mean_vec = np.mean(emb_norm, axis=0)
    anisotropy = float(np.linalg.norm(mean_vec))

    try:
        _, s, _ = np.linalg.svd(emb_norm - mean_vec, full_matrices=False)
        condition = float(s[0] / max(s[-1], 1e-10))
    except Exception:
        condition = 1.0

    sims = emb_norm @ emb_norm.T
    triu = np.triu(sims, k=1)
    non_zero = triu[triu != 0]
    uniformity = float(np.mean(non_zero)) if len(non_zero) > 0 else 0.0

    return {
        "anisotropy_score": round(anisotropy, 4),
        "condition_number": round(condition, 4),
        "uniformity": round(uniformity, 4),
    }
