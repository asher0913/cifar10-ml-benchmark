"""Configuration dataclasses used across the ML experiment suite experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class DataConfig:
    """Settings that control how CIFAR-10 is loaded and optionally subsampled."""

    # Location to cache CIFAR-10 data.
    data_root: str = "data"
    # Number of training samples to keep (None or 0 means full 50k).
    train_subset: int | None = 10000
    # Number of test samples to keep (None or 0 means full 10k).
    test_subset: int | None = 2000
    # Seed that drives data shuffling and subset selection.
    random_state: int = 42


@dataclass
class FeatureConfig:
    """Targets for PCA dimensionality reduction (as percentages of original dims)."""

    # Percentages of original dimension to retain (e.g., 10 -> 10% of 3072 dims).
    pca_targets: Tuple[int, ...] = (10, 30, 50, 70, 100)


@dataclass
class MLPConfig:
    """Default hyper-parameters for the Torch MLP used in MLP experiments."""

    # Width of hidden layers.
    hidden_layer_sizes: Tuple[int, ...] = (512, 256)
    # Placeholder (for parity with sklearn-style configs).
    activation: str = "relu"
    # Placeholder solver name; actual optimizer is set in mlp_experiments.py.
    solver: str = "adam"
    # Weight decay (L2 penalty) coefficient.
    alpha: float = 1e-4
    # Batch size for DataLoader.
    batch_size: int = 256
    # Initial learning rate.
    learning_rate_init: float = 1e-3
    # Maximum training epochs.
    max_iter: int = 80
    # Whether to use early stopping.
    early_stopping: bool = True
    # Patience for early stopping.
    n_iter_no_change: int = 10
    # Optimization tolerance.
    tol: float = 1e-4


@dataclass
class RandomForestConfig:
    """Default hyper-parameters for the RandomForest baseline in Random Forest experiments."""

    # Number of trees to grow in the forest.
    n_estimators: int = 400
    # Maximum depth of each tree (None = unlimited).
    max_depth: int | None = None
    # Minimum samples required to split an internal node.
    min_samples_split: int = 2
    # CPU parallelism.
    n_jobs: int = -1
    # RNG seed for reproducibility.
    random_state: int = 42

@dataclass
class ExperimentConfig:
    """Bundle of configs plus helper factories for reproducible sweeps."""
    # Data-related settings.
    data: DataConfig = field(default_factory=DataConfig)
    # PCA targets and feature construction.
    features: FeatureConfig = field(default_factory=FeatureConfig)
    # Base MLP hyper-parameters.
    mlp: MLPConfig = field(default_factory=MLPConfig)
    # Base RandomForest hyper-parameters.
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    # Number of CV folds used across tasks.
    cv_splits: int = 5

    def mlp_hparam_grid(self) -> List[Dict]:
        return [
            {
                "name": "compact",
                "hidden_layer_sizes": (256,),
                "learning_rate": 2e-3,
                "weight_decay": 5e-5,
                "dropout": 0.05,
                "epochs": 60,
            },
            {
                "name": "baseline",
                "hidden_layer_sizes": self.mlp.hidden_layer_sizes,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "dropout": 0.1,
                "epochs": 80,
            },
            {
                "name": "deep",
                "hidden_layer_sizes": (1024, 512, 256),
                "learning_rate": 5e-4,
                "weight_decay": 2e-4,
                "dropout": 0.15,
                "epochs": 110,
            },
        ]

    def rf_hparam_grid(self) -> List[Dict]:
        return [
            {
                "name": "shallow",
                "n_estimators": 200,
                "max_depth": 40,
                "min_samples_split": 2,
            },
            {
                "name": "baseline",
                "n_estimators": self.random_forest.n_estimators,
                "max_depth": self.random_forest.max_depth,
                "min_samples_split": self.random_forest.min_samples_split,
            },
            {
                "name": "regularized",
                "n_estimators": 600,
                "max_depth": 60,
                "min_samples_split": 4,
            },
        ]
