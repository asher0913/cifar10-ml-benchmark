"""CLI entry that orchestrates the experiment suite.

The pipeline is: parse arguments → load CIFAR-10 →
standardize and build PCA feature sets → run MLP experiments (MLP sweeps) →
run Random Forest experiments (Random Forest sweeps). All outputs are written under the outputs
folder in a deterministic subfolder structure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .data import load_cifar10
from .features import FeatureSet, create_feature_sets, dump_feature_metadata
from .mlp_experiments import (
    run_feature_dimension_experiment as mlp_feature_sweep,
)
from .mlp_experiments import run_hparam_experiment as mlp_hparam_sweep
from .random_forest_experiments import (
    run_feature_dimension_experiment as rf_feature_sweep,
)
from .random_forest_experiments import run_hparam_experiment as rf_hparam_sweep
from .utils import ensure_dir, set_global_seed


def _parse_args() -> argparse.Namespace:
    """Define every CLI flag so users can tweak the full experiment from the shell."""
    parser = argparse.ArgumentParser(
        description="CIFAR-10 benchmark runner (PCA, MLP, and Random Forest)"
    )
    # Where to download/cache CIFAR-10.
    parser.add_argument("--data-root", type=str, default="data")
    # Where all CSV/JSON/PNG outputs will be written.
    parser.add_argument("--output-root", type=str, default="outputs")
    # Optional subsampling of training/testing for quicker runs; 0/None means full data.
    parser.add_argument("--train-subset", type=int, default=10000)
    parser.add_argument("--test-subset", type=int, default=2000)
    # PCA targets as percentages of the original 3072-dimensional vector.
    parser.add_argument("--pca-targets", nargs="+", type=int, default=[10, 30, 50, 70, 100])
    # Global random seed so folds and subsets are repeatable.
    parser.add_argument("--random-seed", type=int, default=42)
    # How many cross-validation folds to use in both tasks.
    parser.add_argument("--cv-splits", type=int, default=5)
    # Skip flags allow running only a subset of tasks.
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--skip-rf", action="store_true")

    # Base MLP hyper-parameters (MLP experiments).
    parser.add_argument("--mlp-hidden", nargs="+", type=int, default=[1024, 512])
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-dropout", type=float, default=0.3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-batch-size", type=int, default=256)
    parser.add_argument("--mlp-epochs", type=int, default=80)
    parser.add_argument("--mlp-patience", type=int, default=10)
    parser.add_argument("--mlp-no-bn", action="store_true", help="disable BatchNorm")
    parser.add_argument("--mlp-max-grad-norm", type=float, default=1.0)

    # Base Random Forest hyper-parameters (Random Forest experiments).
    parser.add_argument("--rf-estimators", type=int, default=400)
    parser.add_argument("--rf-max-depth", type=int, default=0, help="0 represents None")
    parser.add_argument("--rf-min-split", type=int, default=2)
    parser.add_argument("--rf-min-leaf", type=int, default=1)
    parser.add_argument("--rf-max-features", type=str, default="auto", help="auto/sqrt/log2 or int/float")
    parser.add_argument("--rf-max-samples", type=float, default=0.0, help="0 represents using all samples")
    parser.add_argument("--rf-n-jobs", type=int, default=-1)
    return parser.parse_args()


def _select_feature_set(feature_sets: Sequence[FeatureSet], name: str) -> FeatureSet:
    """Pick a FeatureSet by name or raise a clear error."""
    for fs in feature_sets:
        if fs.name == name:
            return fs
    raise ValueError(f"Feature set '{name}' not found. Available: {[fs.name for fs in feature_sets]}")


def main() -> None:
    """Orchestrate data prep, PCA, MLP sweeps, and RF sweeps in order."""
    args = _parse_args()
    set_global_seed(args.random_seed)
    output_root = Path(args.output_root)  # base output directory
    ensure_dir(output_root)               # create outputs/ if missing

    print("=== Loading CIFAR-10 dataset ===")
    dataset = load_cifar10(
        data_root=args.data_root,
        train_subset=args.train_subset if args.train_subset > 0 else None,
        test_subset=args.test_subset if args.test_subset > 0 else None,
        random_state=args.random_seed,
    )
    print(f"Train samples: {len(dataset.X_train)} | Test samples: {len(dataset.X_test)}")

    print("=== Building PCA feature sets ===")
    feature_sets = create_feature_sets(
        dataset.X_train,
        dataset.X_test,
        args.pca_targets,
        random_state=args.random_seed,
    )
    # Save lightweight JSON describing each feature set (name, dims, variance).
    dump_feature_metadata(feature_sets, output_root / "features" / "feature_metadata.json")

    # Collect base hyper-parameters to reuse across sweeps.
    mlp_params = {
        "hidden_layer_sizes": tuple(args.mlp_hidden),
        "learning_rate": args.mlp_lr,
        "weight_decay": args.mlp_weight_decay,
        "dropout": args.mlp_dropout,
        "batch_size": args.mlp_batch_size,
        "epochs": args.mlp_epochs,
        "patience": args.mlp_patience,
        "use_batchnorm": not args.mlp_no_bn,
        "max_grad_norm": args.mlp_max_grad_norm,
        "use_sgd": False,
    }
    # RF parameters mirror scikit-learn's API; 0/None conventions are normalized.
    rf_params = {
        "n_estimators": args.rf_estimators,
        "max_depth": None if args.rf_max_depth <= 0 else args.rf_max_depth,
        "min_samples_split": args.rf_min_split,
        "min_samples_leaf": args.rf_min_leaf,
        "max_features": None if args.rf_max_features == "auto" else args.rf_max_features,
        "max_samples": None if args.rf_max_samples <= 0 else args.rf_max_samples,
        "n_jobs": args.rf_n_jobs,
    }

    original_features = _select_feature_set(feature_sets, "original")  # full-dim standardized set

    if not args.skip_mlp:
        print("=== Running MLP experiments: MLP feature sweep ===")
        # For each PCA setting, run K-fold CV, then train/test once on the full split.
        mlp_feature_sweep(
            feature_sets=feature_sets,
            y_train=dataset.y_train,
            y_test=dataset.y_test,
            class_names=dataset.class_names,
            base_params=mlp_params,
            output_dir=output_root,
            cv_splits=args.cv_splits,
            random_state=args.random_seed,
        )
        print("=== Running MLP experiments: MLP hyper-parameter sweep ===")
        # Learning-rate-focused sweep; structure kept fixed for fairness.
        mlp_hparam_grid = [
            {
                "name": "lr_7e4",
                "hidden_layer_sizes": tuple(args.mlp_hidden),
                "learning_rate": 7e-4,
                "weight_decay": args.mlp_weight_decay,
                "dropout": args.mlp_dropout,
                "epochs": args.mlp_epochs,
                "use_batchnorm": not args.mlp_no_bn,
                "max_grad_norm": args.mlp_max_grad_norm,
                "use_sgd": False,
            },
            {
                "name": "lr_1e3",
                "hidden_layer_sizes": tuple(args.mlp_hidden),
                "learning_rate": 1e-3,
                "weight_decay": args.mlp_weight_decay,
                "dropout": args.mlp_dropout,
                "epochs": args.mlp_epochs,
                "use_batchnorm": not args.mlp_no_bn,
                "max_grad_norm": args.mlp_max_grad_norm,
                "use_sgd": False,
            },
            {
                "name": "lr_2e3",
                "hidden_layer_sizes": tuple(args.mlp_hidden),
                "learning_rate": 2e-3,
                "weight_decay": args.mlp_weight_decay,
                "dropout": args.mlp_dropout,
                "epochs": args.mlp_epochs,
                "use_batchnorm": not args.mlp_no_bn,
                "max_grad_norm": args.mlp_max_grad_norm,
                "use_sgd": False,
            },
        ]
        mlp_hparam_sweep(
            feature_set=original_features,
            y_train=dataset.y_train,
            y_test=dataset.y_test,
            class_names=dataset.class_names,
            base_params=mlp_params,
            hparam_grid=mlp_hparam_grid,
            output_dir=output_root,
            cv_splits=args.cv_splits,
            random_state=args.random_seed,
        )
    else:
        print("Skipping MLP experiments as requested.")

    if not args.skip_rf:
        print("=== Running Random Forest experiments: Random Forest feature sweep ===")
        rf_feature_sweep(
            feature_sets=feature_sets,
            y_train=dataset.y_train,
            y_test=dataset.y_test,
            class_names=dataset.class_names,
            base_params=rf_params,
            output_dir=output_root,
            cv_splits=args.cv_splits,
            random_state=args.random_seed + 101,
        )
        print("=== Running Random Forest experiments: Random Forest hyper-parameter sweep ===")
        rf_hparam_grid = [
            {
                "name": "shallow",
                "n_estimators": max(200, args.rf_estimators // 2),
                "max_depth": 40 if args.rf_max_depth == 0 else args.rf_max_depth,
                "min_samples_split": max(2, args.rf_min_split),
            },
            {
                "name": "baseline",
                "n_estimators": args.rf_estimators,
                "max_depth": None if args.rf_max_depth == 0 else args.rf_max_depth,
                "min_samples_split": args.rf_min_split,
            },
            {
                "name": "regularized",
                "n_estimators": args.rf_estimators + 200,
                "max_depth": None if args.rf_max_depth == 0 else args.rf_max_depth + 20,
                "min_samples_split": args.rf_min_split + 2,
            },
        ]
        rf_hparam_sweep(
            feature_set=original_features,
            y_train=dataset.y_train,
            y_test=dataset.y_test,
            class_names=dataset.class_names,
            base_params=rf_params,
            hparam_grid=rf_hparam_grid,
            output_dir=output_root,
            cv_splits=args.cv_splits,
            random_state=args.random_seed + 211,
        )
    else:
        print("Skipping Random Forest experiments as requested.")

    print("All requested experiments have completed.")


if __name__ == "__main__":
    main()
