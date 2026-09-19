# CIFAR-10 Classical Machine Learning Benchmark

A reproducible benchmark for image classification without convolutional
networks. The pipeline flattens and standardizes CIFAR-10 images, builds PCA
representations, and compares a PyTorch multilayer perceptron with a
scikit-learn Random Forest under the same cross-validation and held-out test
protocol.

The workflow has three stages:

1) **PCA feature extraction** – Standardize flattened CIFAR-10 images and build PCA feature sets (10/30/50/70/100%).
2) **MLP experiments** – Train and evaluate a GPU/MPS-based MLP on all PCA feature sets (5-fold CV + test) and an MLP hyper-parameter sweep.
3) **Random Forest experiments** – Train and evaluate a Random Forest on all PCA feature sets (5-fold CV + test) and an RF hyper-parameter sweep.

All outputs are written under `outputs/` as CSV summaries, per-class JSON
reports, and plots. The default dataset cache is `data/`; torchvision downloads
the dataset automatically when it is not present.

## Results

Experiments used a 10,000-image training subset, five-fold stratified
cross-validation, and an independent 2,000-image test subset.

| Model | Best configuration | Test accuracy | Test macro F1 |
| --- | --- | ---: | ---: |
| MLP | learning rate 0.002, two hidden layers | 48.1% | 48.4% |
| Random Forest | 600 trees, `min_samples_split=4` | 42.5% | 42.0% |

The PCA sweep also showed that the MLP retained 47.3% test accuracy with only
307 principal components, close to the 47.6% result from all 3,072 flattened
pixel features.

## Prerequisites
- Python 3.10+
- PyTorch with CUDA or Apple MPS for the MLP; the Random Forest runs on CPU

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## How to run
Run the full benchmark with the default 10,000/2,000 subsets:

```bash
python main.py
```

Recommended for better MLP performance (full dataset + moderate network):

```bash
python main.py \
  --train-subset 0 --test-subset 0 \
  --mlp-hidden 1024 512 \
  --mlp-dropout 0.1 \
  --mlp-lr 0.001 \
  --mlp-epochs 220 --mlp-patience 35
```

Key CLI options (see `python main.py --help` for full list):
- `--data-root`, `--output-root`: dataset cache and output directories (defaults: `data`, `outputs`).
- `--train-subset`, `--test-subset`: set to `0` to use full CIFAR-10; otherwise number of samples.
- `--pca-targets`: PCA variance percentages (default `10 30 50 70 100`).
- MLP knobs: `--mlp-hidden`, `--mlp-lr`, `--mlp-dropout`, `--mlp-weight-decay`, `--mlp-batch-size`, `--mlp-epochs`, `--mlp-patience`, `--mlp-no-bn`, `--mlp-max-grad-norm`.
- RF knobs: `--rf-estimators`, `--rf-max-depth`, `--rf-min-split`, `--rf-min-leaf`, `--rf-max-features`, `--rf-max-samples`, `--rf-n-jobs`.
- `--skip-mlp`, `--skip-rf`: skip MLP or RF if needed.

## Outputs (under `outputs/`)
- `features/feature_metadata.json`: PCA feature dimensions & explained variance.
- `mlp/mlp_feature_sweep/` and `mlp/mlp_hparam_sweep/`: CV/test summaries (`summary.csv`), fold metrics, per-class JSON reports, plots.
- `random_forest/rf_feature_sweep/` and `random_forest/rf_hparam_sweep/`: same structure for Random Forest.

## Hardware note

The MLP runner currently requires a CUDA or Apple MPS device. Use
`--skip-mlp` to run the CPU-only Random Forest experiments.
