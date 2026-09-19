"""Feature engineering utilities (standardization + PCA).

To make them trainable and reasonably well-conditioned, this module standardize
each feature (zero mean, unit variance) and apply PCA to keep
only a percentage of the original 3072 dimensions. All transformations
are fit on the training split only and applied to the test split to avoid
data leakage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureSet:
    # Short handle for the feature set (e.g., "original", "pca_10").
    name: str
    # Training features (already standardized and optionally PCA-transformed).
    X_train: np.ndarray
    # Test features (same transformation as training).
    X_test: np.ndarray
    # Dimensionality of the transformed feature space.
    n_features: int
    # Sum of explained variance ratios (1.0 for full-dimensional copies).
    explained_variance: float
    # Human-readable description for reports.
    description: str


def create_feature_sets(
    X_train: np.ndarray,
    X_test: np.ndarray,
    pca_targets: Sequence[int],
    random_state: int = 42,
) -> List[FeatureSet]:
    """Build standardized feature sets plus PCA variants for downstream models.

    Steps:
      1) Fit StandardScaler on training data, then transform both train/test.
      2) Always include a full-dimensional standardized copy ("original").
      3) For each percentage in pca_targets, compute how many components that
         percentage corresponds to, fit PCA on the standardized train data,
         and transform both splits.
      4) Package each variant in a FeatureSet with metadata for downstream use.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit only on train to avoid leakage
    X_test_scaled = scaler.transform(X_test)        # apply same scaling to test

    feature_sets: List[FeatureSet] = [
        FeatureSet(
            name="original",
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            n_features=X_train_scaled.shape[1],
            explained_variance=1.0,
            description="Standardized flattened RGB pixels.",
        )
    ]

    original_dim = X_train_scaled.shape[1]
    for pct in sorted(set(pca_targets)):
        label = f"pca_{pct}"
        if pct >= 100:
            # Store a copy to keep metadata consistent with other PCA variants.
            feature_sets.append(
                FeatureSet(
                    name=label,
                    X_train=X_train_scaled,
                    X_test=X_test_scaled,
                    n_features=original_dim,
                    explained_variance=1.0,
                    description="Reference copy of the full feature space.",
                )
            )
            continue
        n_components = max(1, int(round(original_dim * (pct / 100.0))))
        n_components = min(n_components, original_dim)  # never exceed original dimension
        pca = PCA(
            n_components=n_components,
            svd_solver="full",
            random_state=random_state,
            whiten=False,
        )
        X_train_pca = pca.fit_transform(X_train_scaled)  # fit on train
        X_test_pca = pca.transform(X_test_scaled)        # transform test
        explained = float(np.sum(pca.explained_variance_ratio_))  # cumulative variance
        feature_sets.append(
            FeatureSet(
                name=label,
                X_train=X_train_pca,
                X_test=X_test_pca,
                n_features=X_train_pca.shape[1],
                explained_variance=explained,
                description=(
                    f"PCA keeping ~{pct}% of original dimensions "
                    f"({n_components} comps, {explained:.2%} variance)."
                ),
            )
        )
    return feature_sets


def dump_feature_metadata(feature_sets: Iterable[FeatureSet], output_path: Path) -> None:
    """Write a JSON summary describing each feature set that was generated.
    """
    serializable = []
    for fs in feature_sets:
        serializable.append(
            {
                "name": fs.name,
                "n_features": fs.n_features,
                "explained_variance": fs.explained_variance,
                "description": fs.description,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serializable, indent=2))
