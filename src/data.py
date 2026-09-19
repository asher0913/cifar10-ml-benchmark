"""Data loading helpers for the ML experiment suite experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
@dataclass
class DatasetBundle:
    # Flattened training features (N x D).
    X_train: np.ndarray
    # Integer labels for training set.
    y_train: np.ndarray
    # Flattened test features (N x D).
    X_test: np.ndarray
    # Integer labels for test set.
    y_test: np.ndarray
    # Human-readable class names (CIFAR-10).
    class_names: Sequence[str]
    # Raw training images (H x W x C) kept for reference/plots.
    train_images: np.ndarray
    # Raw test images.
    test_images: np.ndarray


def _flatten_and_normalize(images: np.ndarray) -> np.ndarray:
    """Flatten HWC images to vectors and scale pixels into [0, 1]."""
    flat = images.reshape(images.shape[0], -1).astype(np.float32)
    return flat / 255.0  # keep float in [0,1] to stabilize downstream models


def _take_subset(X: np.ndarray, y: np.ndarray, subset: int | None, seed: int):
    """Optionally subsample a fixed-size subset for quicker experiments."""
    if subset is None or subset >= len(X):
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=subset, replace=False)
    return X[idx], y[idx]


def load_cifar10(
    data_root: str,
    train_subset: int | None = None,
    test_subset: int | None = None,
    random_state: int = 42,
) -> DatasetBundle:
    """Download CIFAR-10 (if missing), optionally subsample, and return flattened arrays."""
    from torchvision import datasets, transforms

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    # torchvision handles caching; repeated calls reuse the same downloaded files.
    train_dataset = datasets.CIFAR10(
        root=str(root), train=True, download=True, transform=transforms.ToTensor()
    )
    test_dataset = datasets.CIFAR10(
        root=str(root), train=False, download=True, transform=transforms.ToTensor()
    )

    train_images = train_dataset.data  # uint8 images (N x H x W x C)
    y_train = np.array(train_dataset.targets)  # list -> np.ndarray
    test_images = test_dataset.data
    y_test = np.array(test_dataset.targets)

    train_images, y_train = _take_subset(train_images, y_train, train_subset, random_state)
    test_images, y_test = _take_subset(test_images, y_test, test_subset, random_state + 1)

    X_train = _flatten_and_normalize(train_images)  # shape: (N_train, 3072)
    X_test = _flatten_and_normalize(test_images)    # shape: (N_test, 3072)

    return DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_names=train_dataset.classes,
        train_images=train_images,
        test_images=test_images,
    )
