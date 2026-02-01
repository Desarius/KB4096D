"""Utility functions: cosine similarity, formatting, vector ops."""

from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two vectors. Returns scalar float."""
    a = a.float().flatten()
    b = b.float().flatten()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def cosine_sim_batch(query: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between a query vector and a batch of vectors.

    Args:
        query: shape (D,) or (1, D)
        vectors: shape (N, D)

    Returns:
        Tensor of shape (N,) with similarity scores.
    """
    query = query.float().flatten().unsqueeze(0)  # (1, D)
    vectors = vectors.float()  # (N, D)
    return F.cosine_similarity(query, vectors, dim=1)


def top_k_similar(
    query: torch.Tensor,
    names: List[str],
    vectors: torch.Tensor,
    k: int = 5,
    threshold: float = 0.0,
) -> List[Tuple[str, float]]:
    """Find top-k most similar vectors by cosine similarity.

    Args:
        query: shape (D,)
        names: list of N names
        vectors: shape (N, D)
        k: number of results
        threshold: minimum similarity

    Returns:
        List of (name, similarity) tuples, sorted descending.
    """
    if vectors.size(0) == 0:
        return []

    sims = cosine_sim_batch(query, vectors)
    k = min(k, sims.size(0))
    topk_vals, topk_ids = torch.topk(sims, k)

    results = []
    for val, idx in zip(topk_vals.tolist(), topk_ids.tolist()):
        if val >= threshold:
            results.append((names[idx], val))
    return results


def normalize_vector(v: torch.Tensor) -> torch.Tensor:
    """L2-normalize a vector."""
    v = v.float()
    norm = v.norm(p=2)
    if norm > 0:
        return v / norm
    return v


def interpolate_vectors(
    a: torch.Tensor, b: torch.Tensor, alpha: float = 0.5
) -> torch.Tensor:
    """Linear interpolation between two vectors, then normalize.

    alpha=0.0 -> a, alpha=1.0 -> b
    """
    result = (1 - alpha) * a.float() + alpha * b.float()
    return normalize_vector(result)


def analogy_vector(
    base: torch.Tensor, src: torch.Tensor, dst: torch.Tensor
) -> torch.Tensor:
    """Compute analogy: base + (dst - src), then normalize.

    "base is to ? as src is to dst"
    e.g., France:Paris :: Japan:? -> Japan + (Paris - France) ≈ Tokyo
    """
    result = base.float() + (dst.float() - src.float())
    return normalize_vector(result)


def format_results(results: List[Tuple[str, float]], prefix: str = "") -> str:
    """Format search results for display."""
    if not results:
        return f"{prefix}No results found."
    lines = []
    for i, (name, score) in enumerate(results, 1):
        lines.append(f"{prefix}{i}. {name} (similarity: {score:.4f})")
    return "\n".join(lines)
