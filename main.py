"""Entry point for Machine Learning experiment suite pipeline (PCA, MLP, and Random Forest: PCA + MLP + RF)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # repository root
if str(ROOT) not in sys.path:
    # Ensure local src/ package is importable when running as a script.
    sys.path.insert(0, str(ROOT))

from src import run_pipeline


def main() -> None:
    """Delegate all orchestration to src.run_pipeline."""
    run_pipeline.main()


if __name__ == "__main__":
    main()
