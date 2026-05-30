"""Fréchet distance for two sets of feature vectors."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import linalg


def fid_from_features(
    feats_a: np.ndarray, feats_b: np.ndarray, eps: float = 1e-6
) -> float:
    """Fréchet distance between two feature sets (each ``(N, D)``)."""
    mu_a = feats_a.mean(axis=0)
    mu_b = feats_b.mean(axis=0)
    sigma_a = np.cov(feats_a, rowvar=False)
    sigma_b = np.cov(feats_b, rowvar=False)

    diff = mu_a - mu_b
    covmean, _ = linalg.sqrtm(sigma_a.dot(sigma_b), disp=False)
    if not np.isfinite(covmean).all():
        warnings.warn(f"fid sqrtm produced singular product; adding {eps} to diagonals")
        offset = np.eye(sigma_a.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_a + offset).dot(sigma_b + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"imaginary FID component: {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return float(
        diff.dot(diff) + np.trace(sigma_a) + np.trace(sigma_b) - 2 * np.trace(covmean)
    )
