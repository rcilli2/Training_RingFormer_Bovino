#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Uncertainty-aware training script for MiniTransformer sequence labelling.

What this script does:
1. Trains the existing MiniTransformer with standard cross-entropy.
2. Estimates uncertainty with either MC dropout or deep ensembles.
3. Decomposes predictive uncertainty into:
   - total uncertainty: predictive entropy
   - aleatoric uncertainty: expected entropy
   - epistemic uncertainty: mutual information
4. Applies post-hoc temperature scaling on a dedicated split.
5. Builds split-conformal prediction sets on a second dedicated split.
6. Reports whether the uncertainty scores are actually informative for errors.

Important note on guarantees:
- The conformal guarantee implemented here is token-level marginal coverage.
- It is valid when the conformal calibration tokens and the evaluation tokens are
  exchangeable with respect to the chosen nonconformity score.
- This is not a formal sequence-level guarantee, and dependencies inside each
  borehole can make the guarantee approximate in practice.

Example:
    python train_sequence_labelling_uq.py ^
        --data-glob "data/*.csv" ^
        --output-dir "results_uq" ^
        --uq-method mc_dropout ^
        --mc-samples 30 ^
        --alpha 0.1 ^
        --fold-index 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

from models.MiniTransformer import MiniTransformerPerElement


UNIT_TO_CLASS = {
    "riporto": 0,
    "t": 0,
    "4": 1,
    "3": 2,
    "3a": 2,
    "2b": 3,
    "2a": 4,
    "2c": 3,
    "1": 5,
}

CLASS_NAMES = {
    0: "Topsoil",
    1: "Elu (4)",
    2: "Debris (3)",
    3: "Transitional Flysch (2b)",
    4: "Clay Flysch (2a)",
    5: "Fyr Flysch (1)",
    #6: "1",
}


def filesystem_path(path: Path) -> Path:
    """Use Windows extended paths so deeply nested fold artifacts remain writable."""
    resolved = path.resolve()
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path(f"\\\\?\\{resolved}")
    return resolved


DEFAULT_CAMPAIGN_PREFIX_MAP = (
    "FI=GeoConsul,FP=GeoConsul,"
    "SC=PST,SG=GeoIng,SP=Geo-Invest,"
    "C=Coppolella,T=Botticelli,N=Troncone,L=Troncone,P=Troncone,"
    "S=AdBacino,A=GeoDaunia,V=Giordano,U=Rampino"
)


@dataclass
class RawBoreholeDataset:
    names: list[str]
    embeddings: np.ndarray
    labels: np.ndarray
    valid_mask: np.ndarray
    depths: np.ndarray
    x_values: np.ndarray
    y_values: np.ndarray
    z_values: np.ndarray


@dataclass
class ProcessedBoreholeDataset:
    names: list[str]
    features: np.ndarray
    labels: np.ndarray
    valid_mask: np.ndarray
    depths: np.ndarray
    x_values: np.ndarray
    y_values: np.ndarray


@dataclass
class FoldSplit:
    test_index: int
    train_indices: list[int]
    model_selection_indices: list[int]
    temperature_indices: list[int]
    conformal_indices: list[int]
    eligible_indices: list[int]
    buffer_excluded_indices: list[int]
    test_indices: list[int] | None = None
    fold_name: str | None = None
    split_label: str = "borehole"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_output_dir = script_dir / "results_uq"
    default_cache = script_dir / "embedding_cache_all_MiniLM_L6_v2.npz"

    parser = argparse.ArgumentParser(
        description="Train MiniTransformer for sequence labelling with uncertainty estimation."
    )
    parser.add_argument(
        "--data-glob",
        type=str,
        default=str(script_dir / "data" / "*.csv"),
        help="Glob pattern for borehole CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_output_dir),
        help="Directory where metrics, predictions, and optional checkpoints are saved.",
    )
    parser.add_argument(
        "--embedding-cache",
        type=str,
        default=str(default_cache),
        help="Compressed NPZ cache for raw text embeddings.",
    )
    parser.add_argument(
        "--force-recompute-cache",
        action="store_true",
        help="Recompute text embeddings even if the cache already exists.",
    )
    parser.add_argument(
        "--description-col",
        type=str,
        default="Description",
        help="Name of the text column used to compute embeddings.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name used for text embeddings.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=128,
        help="Batch size for SentenceTransformer encoding.",
    )
    parser.add_argument("--epochs", type=int, default=1500, help="Fixed number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=95, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay.")
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw"],
        default="adamw",
        help="Optimizer used for model training.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Deprecated compatibility option; early stopping is disabled.",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=20,
        help="Deprecated compatibility option; early stopping is disabled.",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Deprecated compatibility option; fixed-epoch training is always used.",
    )
    parser.add_argument("--d-model", type=int, default=128, help="Transformer hidden dimension.")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of encoder layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument(
        "--embedding-pca-dim",
        type=int,
        default=None,
        help=(
            "If set, reduce SentenceBERT embeddings to this many PCA components. "
            "The PCA is fitted separately inside each fold using only training tokens."
        ),
    )
    parser.add_argument(
        "--fold-pca-cache-dir",
        type=Path,
        default=None,
        help="Optional shared directory for fold-wise PCA transforms reused across methods.",
    )
    parser.add_argument(
        "--train-objective",
        type=str,
        choices=["cross_entropy", "beta_nll", "evidential"],
        default="cross_entropy",
        help="Training objective. beta_nll enables a sigma head; evidential interprets the classifier head as Dirichlet evidence.",
    )
    parser.add_argument(
        "--beta-nll-beta",
        type=float,
        default=0.5,
        help="Beta parameter used by the beta-NLL attenuation term.",
    )
    parser.add_argument(
        "--beta-nll-eps",
        type=float,
        default=1e-8,
        help="Numerical epsilon used in the beta-NLL objective.",
    )
    parser.add_argument(
        "--beta-nll-log-sigma-min",
        type=float,
        default=-6.0,
        help="Lower clamp applied to log-sigma before exponentiation in beta-NLL.",
    )
    parser.add_argument(
        "--beta-nll-log-sigma-max",
        type=float,
        default=3.0,
        help="Upper clamp applied to log-sigma before exponentiation in beta-NLL.",
    )
    parser.add_argument(
        "--evidential-kl-weight",
        type=float,
        default=0.01,
        help="KL regularization weight for evidential Dirichlet training.",
    )
    parser.add_argument(
        "--evidential-eps",
        type=float,
        default=1e-8,
        help="Numerical epsilon used by evidential probability computations.",
    )
    parser.add_argument(
        "--pooling-power",
        type=int,
        default=5,
        help="Pool by a factor 2**N along depth. Use 0 to disable pooling.",
    )
    parser.add_argument(
        "--valid-pooling-threshold",
        type=float,
        default=0.5,
        help="A pooled token is marked valid if mean(valid) >= threshold.",
    )
    parser.add_argument(
        "--uq-method",
        type=str,
        choices=["mc_dropout", "deep_ensemble", "subsample_ensemble"],
        default="mc_dropout",
        help="Uncertainty estimation method.",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=20,
        help="Number of stochastic forward passes for MC dropout.",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=5,
        help="Number of independently trained members for deep or subsampling ensembles.",
    )
    parser.add_argument(
        "--subsample-fraction",
        type=float,
        default=0.8,
        help="Fraction of training boreholes used by each member when --uq-method subsample_ensemble.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Miscoverage level for split-conformal prediction sets.",
    )
    parser.add_argument(
        "--conformal-method",
        type=str,
        choices=["aps", "lac"],
        default="aps",
        help="Conformal score: APS or LAC.",
    )
    parser.add_argument(
        "--model-selection-fraction",
        type=float,
        default=0.1,
        help="Fraction of non-test boreholes reserved for early stopping.",
    )
    parser.add_argument(
        "--temperature-fraction",
        type=float,
        default=0.1,
        help="Fraction of non-test boreholes reserved for temperature scaling.",
    )
    parser.add_argument(
        "--conformal-fraction",
        type=float,
        default=0.1,
        help="Fraction of non-test boreholes reserved for conformal calibration.",
    )
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=15,
        help="Number of bins for expected calibration error.",
    )
    parser.add_argument(
        "--temperature-steps",
        type=int,
        default=200,
        help="Gradient steps used to fit the temperature scalar.",
    )
    parser.add_argument(
        "--temperature-learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for temperature scaling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for training and inference.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["lobo", "campaign"],
        default="lobo",
        help="Cross-validation unit: leave-one-borehole-out or leave-one-campaign-out.",
    )
    parser.add_argument(
        "--campaign-prefix-map",
        type=str,
        default=DEFAULT_CAMPAIGN_PREFIX_MAP,
        help=(
            "Comma-separated prefix=campaign rules used by --split-mode campaign. "
            "Rules are matched by longest prefix first."
        ),
    )
    parser.add_argument(
        "--test-campaign",
        type=str,
        default=None,
        help="Run only one campaign fold when --split-mode campaign is active.",
    )
    parser.add_argument(
        "--max-campaigns",
        type=int,
        default=None,
        help="Run only the first N campaign folds after sorting by campaign name.",
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        default=None,
        help="Run only one specific leave-one-borehole-out fold by index.",
    )
    parser.add_argument(
        "--test-borehole",
        type=str,
        default=None,
        help="Run only one fold specified by borehole filename stem, e.g. A1.",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Run only the first N folds after sorting by borehole name.",
    )
    parser.add_argument(
        "--save-member-models",
        action="store_true",
        help="Save trained state_dict files for each fold/member.",
    )
    parser.add_argument(
        "--save-representation-npz",
        action="store_true",
        help="Save per-fold compressed NPZ files with train/test token representations before and after the Transformer encoder.",
    )
    parser.add_argument(
        "--save-test-predictions-by-epoch",
        action="store_true",
        help=(
            "At the end of each epoch, save deterministic predictions on the held-out test borehole. "
            "This is useful for LOBO learning curves against non-contextual baselines."
        ),
    )
    parser.add_argument(
        "--epoch-snapshot-interval",
        type=int,
        default=0,
        help=(
            "Save model state_dict snapshots every N epochs for post-hoc evolution analyses. "
            "Use 0 to disable snapshots. The final epoch is always included when enabled."
        ),
    )
    parser.add_argument(
        "--exclusion-buffer-m",
        type=float,
        default=0.0,
        help="Exclude non-test boreholes whose planar XY distance from the test borehole is <= this radius.",
    )
    parser.add_argument(
        "--skip-impossible-folds",
        action="store_true",
        help="Skip folds that become impossible after buffer exclusion instead of stopping the full run.",
    )
    parser.add_argument(
        "--max-train-boreholes",
        type=int,
        default=None,
        help="If set, randomly subsample at most this many training boreholes per fold after all exclusions/hold-outs.",
    )

    parser.set_defaults(
        train_on_valid_only=True,
        eval_on_valid_only=True,
        use_temperature_scaling=True,
        use_embeddings_as_feature=True,
        use_x_as_feature=True,
        use_y_as_feature=True,
        use_depth_as_feature=True,
        use_z_as_feature=True,
        use_valid_as_feature=True,
    )
    parser.add_argument(
        "--use-embeddings-as-feature",
        dest="use_embeddings_as_feature",
        action="store_true",
        help="Use the pooled sentence embeddings as model inputs (default).",
    )
    parser.add_argument(
        "--ignore-embeddings-as-feature",
        dest="use_embeddings_as_feature",
        action="store_false",
        help="Exclude sentence embeddings and train on the selected auxiliary features only.",
    )
    parser.add_argument(
        "--train-on-valid-only",
        dest="train_on_valid_only",
        action="store_true",
        help="Train only on tokens whose Valid flag is 1 after pooling.",
    )
    parser.add_argument(
        "--train-on-all-tokens",
        dest="train_on_valid_only",
        action="store_false",
        help="Train on every pooled token, even when Valid is 0.",
    )
    parser.add_argument(
        "--eval-on-valid-only",
        dest="eval_on_valid_only",
        action="store_true",
        help="Evaluate metrics only on tokens whose Valid flag is 1 after pooling.",
    )
    parser.add_argument(
        "--eval-on-all-tokens",
        dest="eval_on_valid_only",
        action="store_false",
        help="Evaluate metrics on every pooled token.",
    )
    parser.add_argument(
        "--use-temperature-scaling",
        dest="use_temperature_scaling",
        action="store_true",
        help="Fit a temperature scalar on a dedicated split before conformal calibration.",
    )
    parser.add_argument(
        "--no-temperature-scaling",
        dest="use_temperature_scaling",
        action="store_false",
        help="Skip temperature scaling and keep temperature=1.",
    )
    parser.add_argument(
        "--use-valid-as-feature",
        dest="use_valid_as_feature",
        action="store_true",
        help="Append pooled Valid ratio as an input feature.",
    )
    parser.add_argument(
        "--ignore-valid-as-feature",
        dest="use_valid_as_feature",
        action="store_false",
        help="Do not expose Valid as an input feature. This is the safer default if Valid is an annotation artifact.",
    )
    parser.add_argument(
        "--use-x-as-feature",
        dest="use_x_as_feature",
        action="store_true",
        help="Append pooled x coordinate as an input feature.",
    )
    parser.add_argument(
        "--ignore-x-as-feature",
        dest="use_x_as_feature",
        action="store_false",
        help="Do not append x as an input feature.",
    )
    parser.add_argument(
        "--use-y-as-feature",
        dest="use_y_as_feature",
        action="store_true",
        help="Append pooled y coordinate as an input feature.",
    )
    parser.add_argument(
        "--ignore-y-as-feature",
        dest="use_y_as_feature",
        action="store_false",
        help="Do not append y as an input feature.",
    )
    parser.add_argument(
        "--use-depth-as-feature",
        dest="use_depth_as_feature",
        action="store_true",
        help="Append pooled depth as an input feature.",
    )
    parser.add_argument(
        "--ignore-depth-as-feature",
        dest="use_depth_as_feature",
        action="store_false",
        help="Do not append depth as an input feature.",
    )
    parser.add_argument(
        "--use-z-as-feature",
        dest="use_z_as_feature",
        action="store_true",
        help="Append pooled elevation z as an input feature.",
    )
    parser.add_argument(
        "--ignore-z-as-feature",
        dest="use_z_as_feature",
        action="store_false",
        help="Do not append z as an input feature.",
    )

    args = parser.parse_args()

    if args.fold_index is not None and args.test_borehole is not None:
        parser.error("Use only one between --fold-index and --test-borehole.")
    if args.split_mode == "campaign" and (args.fold_index is not None or args.test_borehole is not None):
        parser.error("--fold-index and --test-borehole are only valid with --split-mode lobo.")
    if args.split_mode == "lobo" and (args.test_campaign is not None or args.max_campaigns is not None):
        parser.error("--test-campaign and --max-campaigns are only valid with --split-mode campaign.")
    # Preserve legacy CLI arguments without allowing them to change the
    # fixed-epoch protocol recorded in new run configurations.
    args.patience = 0
    args.no_early_stopping = True
    args.checkpoint_selection = "final_epoch"
    if args.alpha <= 0.0 or args.alpha >= 1.0:
        parser.error("--alpha must be in (0, 1).")
    if args.pooling_power < 0:
        parser.error("--pooling-power must be >= 0.")
    if args.exclusion_buffer_m < 0.0:
        parser.error("--exclusion-buffer-m must be >= 0.")
    if args.max_train_boreholes is not None and args.max_train_boreholes <= 0:
        parser.error("--max-train-boreholes must be > 0.")
    if args.max_campaigns is not None and args.max_campaigns <= 0:
        parser.error("--max-campaigns must be > 0.")
    if args.embedding_pca_dim is not None and args.embedding_pca_dim <= 0:
        parser.error("--embedding-pca-dim must be > 0.")
    if args.epoch_snapshot_interval < 0:
        parser.error("--epoch-snapshot-interval must be >= 0.")
    if args.train_objective == "evidential" and args.use_temperature_scaling:
        parser.error("Use --no-temperature-scaling with --train-objective evidential.")
    if args.evidential_kl_weight < 0.0:
        parser.error("--evidential-kl-weight must be >= 0.")
    if args.ensemble_size <= 0:
        parser.error("--ensemble-size must be > 0.")
    if args.subsample_fraction <= 0.0 or args.subsample_fraction > 1.0:
        parser.error("--subsample-fraction must be in (0, 1].")
    if not 0.0 <= args.valid_pooling_threshold <= 1.0:
        parser.error("--valid-pooling-threshold must be in [0, 1].")
    for field_name in (
        "model_selection_fraction",
        "temperature_fraction",
        "conformal_fraction",
    ):
        value = getattr(args, field_name)
        if value < 0.0 or value >= 1.0:
            parser.error(f"--{field_name.replace('_', '-')} must be in [0, 1).")

    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_unit(value: Any) -> str:
    text = str(value).strip().lower()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def unit_to_class(value: Any) -> int:
    unit = normalize_unit(value)
    if unit not in UNIT_TO_CLASS:
        raise ValueError(f"Unsupported Unit value {value!r}. Extend UNIT_TO_CLASS before running.")
    return UNIT_TO_CLASS[unit]


def json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def higher_quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot compute a quantile from an empty array.")
    q = min(max(float(q), 0.0), 1.0)
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="higher"))


def softmax_mean_probs(logits_samples: torch.Tensor, temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    scaled = logits_samples / float(max(temperature, 1e-6))
    probs_samples = torch.softmax(scaled, dim=-1)
    mean_probs = probs_samples.mean(dim=0)
    return probs_samples.cpu().numpy(), mean_probs.cpu().numpy()


def evidential_alpha(outputs: torch.Tensor) -> torch.Tensor:
    return F.softplus(outputs) + 1.0


def output_probs_tensor(outputs: torch.Tensor, args: argparse.Namespace, temperature: float = 1.0) -> torch.Tensor:
    if args.train_objective == "evidential":
        alpha = evidential_alpha(outputs)
        return alpha / torch.clamp(alpha.sum(dim=-1, keepdim=True), min=float(args.evidential_eps))
    scaled = outputs / float(max(temperature, 1e-6))
    return torch.softmax(scaled, dim=-1)


def output_samples_to_probs(
    output_samples: torch.Tensor,
    args: argparse.Namespace,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    probs_samples = output_probs_tensor(output_samples, args=args, temperature=temperature)
    mean_probs = probs_samples.mean(dim=0)
    return probs_samples.cpu().numpy(), mean_probs.cpu().numpy()


def evidential_uncertainty_from_outputs(
    output_samples: torch.Tensor,
    args: argparse.Namespace,
) -> np.ndarray | None:
    if args.train_objective != "evidential":
        return None
    alpha = evidential_alpha(output_samples)
    strength = alpha.sum(dim=-1).mean(dim=0)
    uncertainty = len(CLASS_NAMES) / torch.clamp(strength, min=float(args.evidential_eps))
    return uncertainty.cpu().numpy()


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(probs * np.log(probs), axis=-1)


def mutual_information_from_samples(probs_samples: np.ndarray, mean_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_uncertainty = predictive_entropy(mean_probs)
    sample_entropies = predictive_entropy(np.clip(probs_samples, 1e-12, 1.0))
    aleatoric = sample_entropies.mean(axis=0)
    epistemic = np.clip(total_uncertainty - aleatoric, 0.0, None)
    return total_uncertainty, aleatoric, epistemic


def multiclass_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))


def series_to_bool(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y"}).to_numpy(dtype=bool)


def expected_calibration_error(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int,
) -> float:
    if confidences.size == 0:
        return float("nan")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
            in_bin = (confidences >= left) & (confidences <= right)
        else:
            in_bin = (confidences >= left) & (confidences < right)
        if not np.any(in_bin):
            continue
        bin_conf = float(confidences[in_bin].mean())
        bin_acc = float((predictions[in_bin] == labels[in_bin]).mean())
        ece += float(np.mean(in_bin)) * abs(bin_acc - bin_conf)
    return float(ece)


def safe_binary_metric(metric_name: str, y_true: np.ndarray, score: np.ndarray) -> float | None:
    unique = np.unique(y_true)
    if unique.size < 2:
        return None
    if metric_name == "auroc":
        return float(roc_auc_score(y_true, score))
    if metric_name == "auprc":
        return float(average_precision_score(y_true, score))
    raise ValueError(f"Unknown metric_name {metric_name!r}.")


def selective_metrics(correct: np.ndarray, uncertainty: np.ndarray) -> dict[str, float]:
    if correct.size == 0:
        return {}

    order = np.argsort(uncertainty)
    sorted_correct = correct[order].astype(np.float64)
    coverages = [1.0, 0.95, 0.9, 0.8, 0.7, 0.5]

    metrics: dict[str, float] = {}
    for coverage in coverages:
        keep = max(1, int(np.ceil(coverage * sorted_correct.size)))
        accuracy = float(sorted_correct[:keep].mean())
        label = int(round(coverage * 100))
        metrics[f"accuracy_at_{label}pct_coverage"] = accuracy
        metrics[f"risk_at_{label}pct_coverage"] = float(1.0 - accuracy)

    cumulative_risk = 1.0 - (np.cumsum(sorted_correct) / np.arange(1, sorted_correct.size + 1))
    coverage_axis = np.arange(1, sorted_correct.size + 1) / sorted_correct.size
    metrics["aurc"] = float(np.trapezoid(cumulative_risk, coverage_axis))
    return metrics


def load_or_build_raw_dataset(
    csv_paths: list[Path],
    args: argparse.Namespace,
) -> RawBoreholeDataset:
    cache_path = Path(args.embedding_cache)
    if cache_path.exists() and not args.force_recompute_cache:
        cached = np.load(cache_path, allow_pickle=True)
        cache_keys = set(cached.files)
        required_keys = {"names", "embeddings", "labels", "valid_mask", "depths", "x_values", "y_values", "z_values"}
        if required_keys.issubset(cache_keys):
            return RawBoreholeDataset(
                names=[str(name) for name in cached["names"].tolist()],
                embeddings=cached["embeddings"].astype(np.float32),
                labels=cached["labels"].astype(np.int64),
                valid_mask=cached["valid_mask"].astype(bool),
                depths=cached["depths"].astype(np.float32),
                x_values=cached["x_values"].astype(np.float32),
                y_values=cached["y_values"].astype(np.float32),
                z_values=cached["z_values"].astype(np.float32),
            )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required to build the embedding cache. "
            "Install it or provide an existing cache via --embedding-cache."
        ) from exc

    text_model = SentenceTransformer(args.embedding_model)

    names: list[str] = []
    embedding_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    valid_list: list[np.ndarray] = []
    depth_list: list[np.ndarray] = []
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    z_list: list[np.ndarray] = []

    expected_length: int | None = None

    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        texts = frame[args.description_col].fillna("").astype(str).tolist()
        embeddings = text_model.encode(
            texts,
            batch_size=args.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype(np.float32)

        labels = np.array([unit_to_class(value) for value in frame["Unit"]], dtype=np.int64)
        valid_mask = frame["Valid"].astype(int).to_numpy(dtype=np.int64) > 0
        depths = frame["Depth"].to_numpy(dtype=np.float32)
        x_values = frame["x"].to_numpy(dtype=np.float32)
        y_values = frame["y"].to_numpy(dtype=np.float32)
        z_values = frame["z"].to_numpy(dtype=np.float32)

        if expected_length is None:
            expected_length = len(frame)
        if len(frame) != expected_length:
            raise ValueError(
                f"Inconsistent sequence length: {csv_path.name} has {len(frame)} rows, "
                f"expected {expected_length}."
            )

        names.append(csv_path.stem)
        embedding_list.append(embeddings)
        label_list.append(labels)
        valid_list.append(valid_mask.astype(bool))
        depth_list.append(depths)
        x_list.append(x_values)
        y_list.append(y_values)
        z_list.append(z_values)

    raw_dataset = RawBoreholeDataset(
        names=names,
        embeddings=np.stack(embedding_list, axis=0),
        labels=np.stack(label_list, axis=0),
        valid_mask=np.stack(valid_list, axis=0),
        depths=np.stack(depth_list, axis=0),
        x_values=np.stack(x_list, axis=0),
        y_values=np.stack(y_list, axis=0),
        z_values=np.stack(z_list, axis=0),
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        names=np.array(raw_dataset.names, dtype=object),
        embeddings=raw_dataset.embeddings.astype(np.float32),
        labels=raw_dataset.labels.astype(np.int64),
        valid_mask=raw_dataset.valid_mask.astype(np.uint8),
        depths=raw_dataset.depths.astype(np.float32),
        x_values=raw_dataset.x_values.astype(np.float32),
        y_values=raw_dataset.y_values.astype(np.float32),
        z_values=raw_dataset.z_values.astype(np.float32),
    )
    return raw_dataset


def mode_pool_labels(labels: np.ndarray, step: int, num_classes: int) -> np.ndarray:
    num_boreholes, seq_len = labels.shape
    trimmed_len = seq_len - (seq_len % step)
    windows = labels[:, :trimmed_len].reshape(num_boreholes, -1, step)
    counts = np.stack([(windows == cls).sum(axis=-1) for cls in range(num_classes)], axis=-1)
    return counts.argmax(axis=-1).astype(np.int64)


def average_pool_3d(values: np.ndarray, step: int) -> np.ndarray:
    num_boreholes, seq_len, channels = values.shape
    trimmed_len = seq_len - (seq_len % step)
    return values[:, :trimmed_len, :].reshape(num_boreholes, -1, step, channels).mean(axis=2)


def average_pool_2d(values: np.ndarray, step: int) -> np.ndarray:
    num_boreholes, seq_len = values.shape
    trimmed_len = seq_len - (seq_len % step)
    return values[:, :trimmed_len].reshape(num_boreholes, -1, step).mean(axis=2)


def preprocess_raw_dataset(raw_dataset: RawBoreholeDataset, args: argparse.Namespace) -> ProcessedBoreholeDataset:
    step = 2 ** args.pooling_power
    num_classes = len(CLASS_NAMES)

    if step == 1:
        pooled_embeddings = raw_dataset.embeddings.astype(np.float32)
        pooled_labels = raw_dataset.labels.astype(np.int64)
        valid_ratio = raw_dataset.valid_mask.astype(np.float32)
        pooled_valid_mask = raw_dataset.valid_mask.astype(bool)
        pooled_depths = raw_dataset.depths.astype(np.float32)
        pooled_x = raw_dataset.x_values.astype(np.float32)
        pooled_y = raw_dataset.y_values.astype(np.float32)
        pooled_z = raw_dataset.z_values.astype(np.float32)
    else:
        pooled_embeddings = average_pool_3d(raw_dataset.embeddings.astype(np.float32), step)
        pooled_labels = mode_pool_labels(raw_dataset.labels.astype(np.int64), step, num_classes)
        valid_ratio = average_pool_2d(raw_dataset.valid_mask.astype(np.float32), step).astype(np.float32)
        pooled_valid_mask = valid_ratio >= float(args.valid_pooling_threshold)
        pooled_depths = average_pool_2d(raw_dataset.depths.astype(np.float32), step).astype(np.float32)
        pooled_x = average_pool_2d(raw_dataset.x_values.astype(np.float32), step).astype(np.float32)
        pooled_y = average_pool_2d(raw_dataset.y_values.astype(np.float32), step).astype(np.float32)
        pooled_z = average_pool_2d(raw_dataset.z_values.astype(np.float32), step).astype(np.float32)

    aux_features: list[np.ndarray] = []
    if args.use_valid_as_feature:
        aux_features.append(valid_ratio[..., None])
    if args.use_x_as_feature:
        aux_features.append(pooled_x[..., None])
    if args.use_y_as_feature:
        aux_features.append(pooled_y[..., None])
    if args.use_depth_as_feature:
        aux_features.append(pooled_depths[..., None])
    if args.use_z_as_feature:
        aux_features.append(pooled_z[..., None])

    if aux_features:
        aux_matrix = np.concatenate(aux_features, axis=-1).astype(np.float32)
    else:
        aux_matrix = np.zeros((pooled_embeddings.shape[0], pooled_embeddings.shape[1], 0), dtype=np.float32)

    features = np.concatenate([pooled_embeddings, aux_matrix], axis=-1).astype(np.float32)

    return ProcessedBoreholeDataset(
        names=raw_dataset.names,
        features=features,
        labels=pooled_labels,
        valid_mask=pooled_valid_mask.astype(bool),
        depths=pooled_depths.astype(np.float32),
        x_values=pooled_x.astype(np.float32),
        y_values=pooled_y.astype(np.float32),
    )


def normalize_features(
    processed: ProcessedBoreholeDataset,
    train_indices: list[int],
    embedding_dim: int,
) -> ProcessedBoreholeDataset:
    features = processed.features.copy()
    if features.shape[-1] <= embedding_dim:
        return ProcessedBoreholeDataset(
            names=processed.names,
            features=features.astype(np.float32),
            labels=processed.labels,
            valid_mask=processed.valid_mask,
            depths=processed.depths,
            x_values=processed.x_values,
            y_values=processed.y_values,
        )

    aux = features[..., embedding_dim:]
    train_aux = aux[train_indices]
    mean = train_aux.mean(axis=(0, 1), keepdims=True)
    std = train_aux.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    features[..., embedding_dim:] = (aux - mean) / std

    return ProcessedBoreholeDataset(
        names=processed.names,
        features=features.astype(np.float32),
        labels=processed.labels,
        valid_mask=processed.valid_mask,
        depths=processed.depths,
        x_values=processed.x_values,
        y_values=processed.y_values,
    )


def select_model_input_features(
    processed: ProcessedBoreholeDataset,
    raw_embedding_dim: int,
    use_embeddings_as_feature: bool,
) -> tuple[ProcessedBoreholeDataset, int]:
    """Apply the global embedding ablation before fold-specific preprocessing."""
    if use_embeddings_as_feature:
        return processed, raw_embedding_dim

    return (
        ProcessedBoreholeDataset(
            names=processed.names,
            features=processed.features[..., raw_embedding_dim:].astype(np.float32),
            labels=processed.labels,
            valid_mask=processed.valid_mask,
            depths=processed.depths,
            x_values=processed.x_values,
            y_values=processed.y_values,
        ),
        0,
    )


def reduce_embedding_dim_with_fold_pca(
    processed: ProcessedBoreholeDataset,
    train_indices: list[int],
    embedding_dim: int,
    target_dim: int | None,
    seed: int,
    cache_path: Path | None = None,
) -> tuple[ProcessedBoreholeDataset, int]:
    if target_dim is None or target_dim >= embedding_dim:
        return processed, embedding_dim
    if target_dim <= 0:
        raise ValueError("target_dim must be positive.")

    features = processed.features
    embeddings = features[..., :embedding_dim]
    aux = features[..., embedding_dim:]

    train_embeddings = embeddings[train_indices]
    train_valid = processed.valid_mask[train_indices]
    fit_matrix = train_embeddings[train_valid]
    if fit_matrix.shape[0] == 0:
        fit_matrix = train_embeddings.reshape(-1, embedding_dim)

    n_components = min(int(target_dim), int(fit_matrix.shape[0]), int(embedding_dim))
    if n_components < int(target_dim):
        raise ValueError(
            f"Cannot fit {target_dim} PCA components with only {fit_matrix.shape[0]} training tokens."
        )

    # Keeping the fold PCA in float32 avoids large transient allocations for
    # high-dimensional models such as E5-Mistral (4096 embedding features).
    flat_embeddings = embeddings.reshape(-1, embedding_dim).astype(np.float32, copy=False)
    components: np.ndarray | None = None
    mean: np.ndarray | None = None
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            if (
                int(cached["embedding_dim"]) == embedding_dim
                and int(cached["n_components"]) == n_components
                and np.array_equal(
                    cached["train_indices"].astype(np.int64),
                    np.asarray(train_indices, dtype=np.int64),
                )
            ):
                components = cached["components"].astype(np.float32)
                mean = cached["mean"].astype(np.float32)

    if components is None or mean is None:
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
        pca.fit(fit_matrix.astype(np.float32, copy=False))
        components = pca.components_.astype(np.float32)
        mean = pca.mean_.astype(np.float32)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                components=components,
                mean=mean,
                train_indices=np.asarray(train_indices, dtype=np.int64),
                embedding_dim=np.asarray(embedding_dim, dtype=np.int64),
                n_components=np.asarray(n_components, dtype=np.int64),
            )

    reduced = ((flat_embeddings - mean) @ components.T).reshape(
        embeddings.shape[0],
        embeddings.shape[1],
        n_components,
    )
    reduced_features = np.concatenate([reduced.astype(np.float32), aux.astype(np.float32)], axis=-1)

    return (
        ProcessedBoreholeDataset(
            names=processed.names,
            features=reduced_features.astype(np.float32),
            labels=processed.labels,
            valid_mask=processed.valid_mask,
            depths=processed.depths,
            x_values=processed.x_values,
            y_values=processed.y_values,
        ),
        n_components,
    )


def prepare_fold_features(
    processed: ProcessedBoreholeDataset,
    train_indices: list[int],
    embedding_dim: int,
    embedding_pca_dim: int | None,
    seed: int,
    pca_cache_path: Path | None = None,
) -> tuple[ProcessedBoreholeDataset, int]:
    """Reproduce the exact fold-specific PCA and normalization used in training."""
    processed_for_fold, effective_embedding_dim = reduce_embedding_dim_with_fold_pca(
        processed=processed,
        train_indices=train_indices,
        embedding_dim=embedding_dim,
        target_dim=embedding_pca_dim,
        seed=seed,
        cache_path=pca_cache_path,
    )
    normalized = normalize_features(
        processed_for_fold,
        train_indices=train_indices,
        embedding_dim=effective_embedding_dim,
    )
    return normalized, effective_embedding_dim


def subset_processed_dataset(processed: ProcessedBoreholeDataset, indices: list[int]) -> ProcessedBoreholeDataset:
    return ProcessedBoreholeDataset(
        names=[processed.names[index] for index in indices],
        features=processed.features[indices].astype(np.float32),
        labels=processed.labels[indices].astype(np.int64),
        valid_mask=processed.valid_mask[indices].astype(bool),
        depths=processed.depths[indices].astype(np.float32),
        x_values=processed.x_values[indices].astype(np.float32),
        y_values=processed.y_values[indices].astype(np.float32),
    )


class BoreholeSequenceDataset(Dataset):
    def __init__(self, processed: ProcessedBoreholeDataset):
        self.processed = processed

    def __len__(self) -> int:
        return self.processed.features.shape[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "name": self.processed.names[index],
            "inputs": torch.from_numpy(self.processed.features[index]).float(),
            "labels": torch.from_numpy(self.processed.labels[index]).long(),
            "valid_mask": torch.from_numpy(self.processed.valid_mask[index]).bool(),
            "depths": torch.from_numpy(self.processed.depths[index]).float(),
            "x_values": torch.from_numpy(self.processed.x_values[index]).float(),
            "y_values": torch.from_numpy(self.processed.y_values[index]).float(),
        }


def collate_borehole_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": [item["name"] for item in batch],
        "inputs": torch.stack([item["inputs"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
        "valid_mask": torch.stack([item["valid_mask"] for item in batch], dim=0),
        "depths": torch.stack([item["depths"] for item in batch], dim=0),
        "x_values": torch.stack([item["x_values"] for item in batch], dim=0),
        "y_values": torch.stack([item["y_values"] for item in batch], dim=0),
    }


def build_dataloader(
    processed: ProcessedBoreholeDataset,
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    dataset = BoreholeSequenceDataset(processed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collate_borehole_batch,
    )


def build_token_mask(batch: dict[str, Any], train_mode: bool, args: argparse.Namespace) -> torch.Tensor:
    valid_mask = batch["valid_mask"]
    if train_mode and args.train_on_valid_only:
        return valid_mask
    if (not train_mode) and args.eval_on_valid_only:
        return valid_mask
    return torch.ones_like(valid_mask, dtype=torch.bool)


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    selected_logits = logits[token_mask]
    selected_labels = labels[token_mask]
    if selected_labels.numel() == 0:
        raise ValueError("The selected supervision mask contains zero tokens.")
    return F.cross_entropy(selected_logits, selected_labels)


def split_model_outputs(model_outputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(model_outputs, tuple):
        if len(model_outputs) != 2:
            raise ValueError(f"Unsupported model output tuple length: {len(model_outputs)}")
        logits, log_sigma = model_outputs
        return logits, log_sigma
    return model_outputs, None


def masked_beta_nll_loss(
    logits: torch.Tensor,
    log_sigma: torch.Tensor,
    labels: torch.Tensor,
    token_mask: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    selected_logits = logits[token_mask]
    selected_labels = labels[token_mask]
    selected_log_sigma = log_sigma[token_mask]
    if selected_labels.numel() == 0:
        raise ValueError("The selected supervision mask contains zero tokens.")

    clamped_log_sigma = torch.clamp(
        selected_log_sigma.squeeze(-1),
        min=float(args.beta_nll_log_sigma_min),
        max=float(args.beta_nll_log_sigma_max),
    )
    sigma = torch.exp(clamped_log_sigma)

    log_probs = F.log_softmax(selected_logits, dim=-1)
    log_prob_correct = log_probs.gather(dim=-1, index=selected_labels.unsqueeze(-1)).squeeze(-1)
    nll = -log_prob_correct

    base_loss = 0.5 * (nll / (sigma ** 2) + torch.log(sigma + float(args.beta_nll_eps)))
    beta = float(args.beta_nll_beta)
    if beta > 0.0:
        base_loss = base_loss * (sigma.detach() ** (2.0 * beta))
    return base_loss.mean()


def dirichlet_kl_to_uniform(alpha: torch.Tensor, eps: float) -> torch.Tensor:
    num_classes = alpha.shape[-1]
    alpha = torch.clamp(alpha, min=eps)
    strength = torch.clamp(alpha.sum(dim=-1, keepdim=True), min=eps)
    log_b_alpha = torch.lgamma(alpha).sum(dim=-1) - torch.lgamma(strength.squeeze(-1))
    log_b_uniform = -torch.lgamma(torch.tensor(float(num_classes), device=alpha.device, dtype=alpha.dtype))
    digamma_term = torch.sum((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(strength)), dim=-1)
    return -log_b_alpha + log_b_uniform + digamma_term


def masked_evidential_loss(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    token_mask: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    selected_outputs = outputs[token_mask]
    selected_labels = labels[token_mask]
    if selected_labels.numel() == 0:
        raise ValueError("The selected supervision mask contains zero tokens.")

    eps = float(args.evidential_eps)
    alpha = evidential_alpha(selected_outputs)
    strength = torch.clamp(alpha.sum(dim=-1, keepdim=True), min=eps)
    one_hot = F.one_hot(selected_labels, num_classes=len(CLASS_NAMES)).to(dtype=alpha.dtype)

    # Expected cross-entropy under a Dirichlet predictive distribution.
    ce = torch.sum(one_hot * (torch.digamma(strength) - torch.digamma(alpha)), dim=-1)

    # Penalize evidence assigned to non-target classes toward a uniform Dirichlet prior.
    adjusted_alpha = one_hot + (1.0 - one_hot) * alpha
    kl = dirichlet_kl_to_uniform(adjusted_alpha, eps=eps)
    return (ce + float(args.evidential_kl_weight) * kl).mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    model.train()
    running_loss = 0.0
    total_batches = 0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        labels = batch["labels"].to(device)
        token_mask = build_token_mask(batch, train_mode=True, args=args).to(device)

        if not torch.any(token_mask):
            continue

        optimizer.zero_grad(set_to_none=True)
        logits, log_sigma = split_model_outputs(model(inputs))
        if args.train_objective == "beta_nll":
            if log_sigma is None:
                raise ValueError("beta_nll training requires the model to return log_sigma.")
            loss = masked_beta_nll_loss(logits, log_sigma, labels, token_mask, args)
        elif args.train_objective == "evidential":
            loss = masked_evidential_loss(logits, labels, token_mask, args)
        else:
            loss = masked_cross_entropy(logits, labels, token_mask)
        loss.backward()
        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        running_loss += float(loss.item())
        total_batches += 1

    if total_batches == 0:
        raise ValueError("No supervised batch was available during training. Check the masking settings.")
    return running_loss / total_batches


def evaluate_deterministic(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].to(device)
            token_mask = build_token_mask(batch, train_mode=False, args=args).to(device)
            if not torch.any(token_mask):
                continue

            logits, _ = split_model_outputs(model(inputs))
            selected_logits = logits[token_mask]
            selected_labels = labels[token_mask]
            if args.train_objective == "evidential":
                loss = masked_evidential_loss(logits, labels, token_mask, args) * selected_labels.numel()
                selected_probs = output_probs_tensor(selected_logits, args=args)
                predictions = selected_probs.argmax(dim=-1)
            else:
                loss = F.cross_entropy(selected_logits, selected_labels, reduction="sum")
                predictions = selected_logits.argmax(dim=-1)
            total_loss += float(loss.item())
            total_tokens += int(selected_labels.numel())
            total_correct += int((predictions == selected_labels).sum().item())

    if total_tokens == 0:
        return {"loss": float("inf"), "accuracy": float("nan")}
    return {
        "loss": total_loss / total_tokens,
        "accuracy": total_correct / total_tokens,
    }


def collect_deterministic_prediction_frame(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    member_index: int,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].cpu().numpy()
            valid_mask = batch["valid_mask"].cpu().numpy().astype(bool)
            eval_mask = build_token_mask(batch, train_mode=False, args=args).cpu().numpy().astype(bool)
            depths = batch["depths"].cpu().numpy()
            x_values = batch["x_values"].cpu().numpy()
            y_values = batch["y_values"].cpu().numpy()
            names = batch["name"]

            logits, _ = split_model_outputs(model(inputs))
            probs = output_probs_tensor(logits, args=args).cpu().numpy()
            predictions = probs.argmax(axis=-1)
            confidence = probs.max(axis=-1)

            batch_size, seq_len = labels.shape
            for batch_index in range(batch_size):
                borehole_name = str(names[batch_index])
                for token_index in range(seq_len):
                    true_id = int(labels[batch_index, token_index])
                    pred_id = int(predictions[batch_index, token_index])
                    rows.append(
                        {
                            "fold_name": borehole_name,
                            "member_index": int(member_index),
                            "epoch": int(epoch),
                            "borehole": borehole_name,
                            "token_index": int(token_index),
                            "depth": float(depths[batch_index, token_index]),
                            "x": float(x_values[batch_index, token_index]),
                            "y": float(y_values[batch_index, token_index]),
                            "valid_mask": bool(valid_mask[batch_index, token_index]),
                            "eval_mask": bool(eval_mask[batch_index, token_index]),
                            "true_class_id": true_id,
                            "true_class_name": CLASS_NAMES[true_id],
                            "pred_class_id": pred_id,
                            "pred_class_name": CLASS_NAMES[pred_id],
                            "correct": int(pred_id == true_id),
                            "confidence": float(confidence[batch_index, token_index]),
                        }
                    )

    return pd.DataFrame(rows)


def maybe_enable_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def collect_logits_from_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_samples: int,
    stochastic: bool,
) -> dict[str, Any]:
    labels_ref: torch.Tensor | None = None
    valid_mask_ref: torch.Tensor | None = None
    depths_ref: torch.Tensor | None = None
    x_values_ref: torch.Tensor | None = None
    y_values_ref: torch.Tensor | None = None
    names_ref: list[str] | None = None

    member_logits: list[torch.Tensor] = []
    member_log_sigma: list[torch.Tensor] = []
    has_sigma_head = False

    for _ in range(num_samples):
        if stochastic:
            model.eval()
            maybe_enable_dropout(model)
        else:
            model.eval()

        batch_logits: list[torch.Tensor] = []
        batch_labels: list[torch.Tensor] = []
        batch_masks: list[torch.Tensor] = []
        batch_depths: list[torch.Tensor] = []
        batch_x_values: list[torch.Tensor] = []
        batch_y_values: list[torch.Tensor] = []
        batch_names: list[str] = []
        batch_log_sigma: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                inputs = batch["inputs"].to(device)
                logits, log_sigma = split_model_outputs(model(inputs))
                logits = logits.cpu()
                batch_logits.append(logits)
                if log_sigma is not None:
                    has_sigma_head = True
                    batch_log_sigma.append(log_sigma.cpu())

                if labels_ref is None:
                    batch_labels.append(batch["labels"].cpu())
                    batch_masks.append(batch["valid_mask"].cpu())
                    batch_depths.append(batch["depths"].cpu())
                    batch_x_values.append(batch["x_values"].cpu())
                    batch_y_values.append(batch["y_values"].cpu())
                    batch_names.extend(batch["name"])

        member_logits.append(torch.cat(batch_logits, dim=0))
        if has_sigma_head:
            member_log_sigma.append(torch.cat(batch_log_sigma, dim=0))

        if labels_ref is None:
            labels_ref = torch.cat(batch_labels, dim=0)
            valid_mask_ref = torch.cat(batch_masks, dim=0)
            depths_ref = torch.cat(batch_depths, dim=0)
            x_values_ref = torch.cat(batch_x_values, dim=0)
            y_values_ref = torch.cat(batch_y_values, dim=0)
            names_ref = batch_names

    if labels_ref is None or valid_mask_ref is None or depths_ref is None or x_values_ref is None or y_values_ref is None or names_ref is None:
        raise ValueError("Cannot collect logits from an empty dataloader.")

    outputs = {
        "logits_samples": torch.stack(member_logits, dim=0),
        "labels": labels_ref,
        "valid_mask": valid_mask_ref,
        "depths": depths_ref,
        "x_values": x_values_ref,
        "y_values": y_values_ref,
        "names": names_ref,
    }
    if has_sigma_head and member_log_sigma:
        outputs["log_sigma_samples"] = torch.stack(member_log_sigma, dim=0)
    return outputs


def collect_logits_from_ensemble(
    models: list[nn.Module],
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    metadata: dict[str, Any] | None = None
    ensemble_logits: list[torch.Tensor] = []
    ensemble_log_sigma: list[torch.Tensor] = []

    for model in models:
        outputs = collect_logits_from_model(
            model=model,
            loader=loader,
            device=device,
            num_samples=1,
            stochastic=False,
        )
        ensemble_logits.append(outputs["logits_samples"][0])
        if "log_sigma_samples" in outputs:
            ensemble_log_sigma.append(outputs["log_sigma_samples"][0])
        if metadata is None:
            metadata = {
                "labels": outputs["labels"],
                "valid_mask": outputs["valid_mask"],
                "depths": outputs["depths"],
                "x_values": outputs["x_values"],
                "y_values": outputs["y_values"],
                "names": outputs["names"],
            }

    if metadata is None:
        raise ValueError("The ensemble is empty.")

    metadata["logits_samples"] = torch.stack(ensemble_logits, dim=0)
    if ensemble_log_sigma:
        metadata["log_sigma_samples"] = torch.stack(ensemble_log_sigma, dim=0)
    return metadata


def collect_token_representations(
    model: nn.Module,
    processed: ProcessedBoreholeDataset,
    args: argparse.Namespace,
    device: torch.device,
    raw_embedding_dim: int,
) -> dict[str, Any]:
    loader = build_dataloader(processed, args, shuffle=False)
    model.eval()

    model_inputs_batches: list[torch.Tensor] = []
    sentencebert_batches: list[torch.Tensor] = []
    aux_batches: list[torch.Tensor] = []
    encoder_input_batches: list[torch.Tensor] = []
    hidden_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    valid_batches: list[torch.Tensor] = []
    depth_batches: list[torch.Tensor] = []
    x_batches: list[torch.Tensor] = []
    y_batches: list[torch.Tensor] = []
    names: list[str] = []

    with torch.no_grad():
        for batch in loader:
            inputs_cpu = batch["inputs"].float()
            outputs = model.forward_with_representations(inputs_cpu.to(device))

            model_inputs_batches.append(inputs_cpu.cpu())
            sentencebert_batches.append(inputs_cpu[..., :raw_embedding_dim].cpu())
            aux_batches.append(inputs_cpu[..., raw_embedding_dim:].cpu())
            encoder_input_batches.append(outputs["encoder_input"].cpu())
            hidden_batches.append(outputs["hidden"].cpu())
            label_batches.append(batch["labels"].cpu())
            valid_batches.append(batch["valid_mask"].cpu())
            depth_batches.append(batch["depths"].cpu())
            x_batches.append(batch["x_values"].cpu())
            y_batches.append(batch["y_values"].cpu())
            names.extend(batch["name"])

    if not names:
        raise ValueError("Cannot collect token representations from an empty dataset.")

    labels = torch.cat(label_batches, dim=0).numpy().astype(np.int64)
    valid_mask = torch.cat(valid_batches, dim=0).numpy().astype(bool)
    eval_mask = valid_mask.copy() if args.eval_on_valid_only else np.ones_like(valid_mask, dtype=bool)
    seq_len = labels.shape[1]
    token_index = np.broadcast_to(np.arange(seq_len, dtype=np.int64), labels.shape)

    return {
        "names": np.asarray(names),
        "model_inputs": torch.cat(model_inputs_batches, dim=0).numpy().astype(np.float32),
        "sentencebert_embeddings": torch.cat(sentencebert_batches, dim=0).numpy().astype(np.float32),
        "aux_features": torch.cat(aux_batches, dim=0).numpy().astype(np.float32),
        "encoder_inputs": torch.cat(encoder_input_batches, dim=0).numpy().astype(np.float32),
        "hidden_representations": torch.cat(hidden_batches, dim=0).numpy().astype(np.float32),
        "labels": labels,
        "valid_mask": valid_mask,
        "eval_mask": eval_mask,
        "depths": torch.cat(depth_batches, dim=0).numpy().astype(np.float32),
        "x_values": torch.cat(x_batches, dim=0).numpy().astype(np.float32),
        "y_values": torch.cat(y_batches, dim=0).numpy().astype(np.float32),
        "token_index": token_index,
    }


def save_fold_representation_npz(
    model: nn.Module,
    train_data: ProcessedBoreholeDataset,
    test_data: ProcessedBoreholeDataset,
    fold_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    raw_embedding_dim: int,
    member_index: int = 0,
) -> Path:
    payload: dict[str, Any] = {
        "class_ids": np.asarray(sorted(CLASS_NAMES.keys()), dtype=np.int64),
        "class_names": np.asarray([CLASS_NAMES[class_id] for class_id in sorted(CLASS_NAMES.keys())]),
        "raw_embedding_dim": np.asarray([raw_embedding_dim], dtype=np.int64),
        "hidden_dim": np.asarray([args.d_model], dtype=np.int64),
        "representation_member_index": np.asarray([member_index], dtype=np.int64),
    }

    split_map = {
        "train": train_data,
        "test": test_data,
    }
    for split_name, split_data in split_map.items():
        split_repr = collect_token_representations(
            model=model,
            processed=split_data,
            args=args,
            device=device,
            raw_embedding_dim=raw_embedding_dim,
        )
        for key, value in split_repr.items():
            payload[f"{split_name}_{key}"] = value

    output_path = fold_dir / "representations.npz"
    np.savez_compressed(output_path, **payload)
    return output_path


def train_single_model(
    train_data: ProcessedBoreholeDataset,
    model_selection_data: ProcessedBoreholeDataset,
    epoch_eval_data: ProcessedBoreholeDataset | None,
    args: argparse.Namespace,
    device: torch.device,
    input_dim: int,
    seq_len: int,
    member_seed: int,
    member_index: int,
    epoch_snapshot_callback: Callable[[nn.Module, int], None] | None = None,
) -> tuple[nn.Module, dict[str, Any], pd.DataFrame | None]:
    set_seed(member_seed)

    model = MiniTransformerPerElement(
        input_dim=input_dim,
        num_classes=len(CLASS_NAMES),
        seq_len=seq_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout_prob=args.dropout,
        predict_log_sigma=(args.train_objective == "beta_nll"),
    ).to(device)

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    train_loader = build_dataloader(train_data, args, shuffle=True)
    model_sel_loader = build_dataloader(model_selection_data, args, shuffle=False)
    epoch_eval_loader = (
        build_dataloader(epoch_eval_data, args, shuffle=False)
        if args.save_test_predictions_by_epoch and epoch_eval_data is not None and len(epoch_eval_data.names) > 0
        else None
    )

    history = {
        "train_loss": [],
        "model_selection_loss": [],
        "model_selection_accuracy": [],
        "test_accuracy": [],
        "checkpoint_selection": "final_epoch",
        "best_epoch": None,
        "best_model_selection_loss": None,
        "best_model_selection_accuracy": None,
    }
    test_prediction_frames: list[pd.DataFrame] = []

    has_model_selection = len(model_selection_data.names) > 0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args)
        history["train_loss"].append(float(train_loss))

        if has_model_selection:
            metrics = evaluate_deterministic(model, model_sel_loader, device, args)
            model_sel_loss = float(metrics["loss"])
            model_sel_accuracy = float(metrics["accuracy"])
        else:
            model_sel_loss = float("nan")
            model_sel_accuracy = float("nan")

        history["model_selection_loss"].append(model_sel_loss)
        history["model_selection_accuracy"].append(model_sel_accuracy)

        if epoch_eval_loader is not None:
            epoch_predictions = collect_deterministic_prediction_frame(
                model=model,
                loader=epoch_eval_loader,
                device=device,
                args=args,
                epoch=epoch + 1,
                member_index=member_index,
            )
            eval_rows = epoch_predictions[epoch_predictions["eval_mask"]]
            test_accuracy = float(eval_rows["correct"].mean()) if not eval_rows.empty else float("nan")
            history["test_accuracy"].append(test_accuracy)
            test_prediction_frames.append(epoch_predictions)
        else:
            history["test_accuracy"].append(float("nan"))

        if epoch_snapshot_callback is not None:
            epoch_snapshot_callback(model, epoch + 1)

    # Use the requested fixed training budget for every fold. Model-selection
    # metrics, when available, are diagnostic only and never alter the model.
    final_epoch = len(history["train_loss"])
    history["best_epoch"] = final_epoch
    history["best_model_selection_loss"] = float("nan")
    history["best_model_selection_accuracy"] = float("nan")
    test_prediction_frame = pd.concat(test_prediction_frames, axis=0, ignore_index=True) if test_prediction_frames else None
    return model, history, test_prediction_frame


def histories_to_frame(fold_name: str, member_histories: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for member_index, history in enumerate(member_histories):
        train_losses = history.get("train_loss", [])
        model_selection_losses = history.get("model_selection_loss", [])
        model_selection_accuracies = history.get("model_selection_accuracy", [])
        test_accuracies = history.get("test_accuracy", [])
        best_epoch = int(history.get("best_epoch", 0) or 0)

        num_epochs = max(len(train_losses), len(model_selection_losses), len(model_selection_accuracies), len(test_accuracies))
        for epoch_index in range(num_epochs):
            rows.append(
                {
                    "fold_name": fold_name,
                    "member_index": member_index,
                    "epoch": epoch_index + 1,
                    "train_loss": float(train_losses[epoch_index]) if epoch_index < len(train_losses) else float("nan"),
                    "model_selection_loss": float(model_selection_losses[epoch_index])
                    if epoch_index < len(model_selection_losses)
                    else float("nan"),
                    "model_selection_accuracy": float(model_selection_accuracies[epoch_index])
                    if epoch_index < len(model_selection_accuracies)
                    else float("nan"),
                    "test_accuracy": float(test_accuracies[epoch_index])
                    if epoch_index < len(test_accuracies)
                    else float("nan"),
                    "is_best_epoch": int((epoch_index + 1) == best_epoch),
                }
            )
    return pd.DataFrame(rows)


def fit_temperature(
    logits_samples: torch.Tensor,
    labels: torch.Tensor,
    token_mask: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    selected_count = int(token_mask.sum().item())
    if selected_count == 0:
        raise ValueError("Temperature scaling received zero calibration tokens.")

    logits_samples = logits_samples.to(device)
    labels = labels.to(device)
    token_mask = token_mask.to(device)

    log_temperature = nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.Adam([log_temperature], lr=args.temperature_learning_rate)

    for _ in range(args.temperature_steps):
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temperature)
        probs = torch.softmax(logits_samples / temperature, dim=-1).mean(dim=0)
        selected_probs = probs[token_mask].clamp_min(1e-12)
        selected_labels = labels[token_mask]
        loss = F.nll_loss(selected_probs.log(), selected_labels)
        loss.backward()
        optimizer.step()

    return float(torch.exp(log_temperature.detach()).cpu().item())


def compute_conformal_scores(
    probs: np.ndarray,
    labels: np.ndarray,
    method: str,
) -> np.ndarray:
    scores = np.zeros(labels.shape[0], dtype=np.float64)

    if method == "lac":
        return 1.0 - probs[np.arange(labels.shape[0]), labels]

    if method != "aps":
        raise ValueError(f"Unsupported conformal method {method!r}.")

    for index, (prob_vector, true_label) in enumerate(zip(probs, labels)):
        order = np.argsort(-prob_vector)
        cumulative = np.cumsum(prob_vector[order])
        true_rank = int(np.where(order == true_label)[0][0])
        scores[index] = float(cumulative[true_rank])
    return scores


def fit_conformal_predictor(
    logits_samples: torch.Tensor,
    labels: torch.Tensor,
    token_mask: torch.Tensor,
    temperature: float,
    alpha: float,
    method: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if int(token_mask.sum().item()) == 0:
        raise ValueError("Conformal calibration received zero calibration tokens.")

    _, mean_probs = output_samples_to_probs(logits_samples, args=args, temperature=temperature)
    flat_probs = mean_probs[token_mask.cpu().numpy()]
    flat_labels = labels.cpu().numpy()[token_mask.cpu().numpy()]

    scores = compute_conformal_scores(flat_probs, flat_labels, method=method)
    n = scores.shape[0]
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    qhat = higher_quantile(scores, level)

    return {
        "alpha": float(alpha),
        "method": method,
        "qhat": float(qhat),
        "num_tokens": int(n),
    }


def build_prediction_set(prob_vector: np.ndarray, qhat: float, method: str) -> list[int]:
    if method == "lac":
        threshold = 1.0 - qhat
        chosen = np.flatnonzero(prob_vector >= threshold).tolist()
        if not chosen:
            chosen = [int(np.argmax(prob_vector))]
        return chosen

    if method != "aps":
        raise ValueError(f"Unsupported conformal method {method!r}.")

    order = np.argsort(-prob_vector)
    cumulative = np.cumsum(prob_vector[order])
    cutoff = int(np.searchsorted(cumulative, qhat, side="left"))
    return [int(label) for label in order[: cutoff + 1]]


def summarize_fold_predictions(
    logits_samples: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    depths: torch.Tensor,
    x_values: torch.Tensor,
    y_values: torch.Tensor,
    names: list[str],
    temperature: float,
    args: argparse.Namespace,
    conformal: dict[str, Any] | None,
    log_sigma_samples: torch.Tensor | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    probs_samples, mean_probs = output_samples_to_probs(logits_samples, args=args, temperature=temperature)
    total_uncertainty, aleatoric, epistemic = mutual_information_from_samples(probs_samples, mean_probs)
    evidential_uncertainty = evidential_uncertainty_from_outputs(logits_samples, args=args)
    confidence = mean_probs.max(axis=-1)
    predictions = mean_probs.argmax(axis=-1)
    variation_ratio = 1.0 - confidence
    difficulty_sigma: np.ndarray | None = None
    if log_sigma_samples is not None:
        sigma_samples = torch.exp(
            torch.clamp(
                log_sigma_samples,
                min=float(args.beta_nll_log_sigma_min),
                max=float(args.beta_nll_log_sigma_max),
            )
        ).cpu().numpy()
        difficulty_sigma = sigma_samples.mean(axis=0).squeeze(-1)

    eval_mask = valid_mask.cpu().numpy() if args.eval_on_valid_only else np.ones_like(valid_mask.cpu().numpy(), dtype=bool)
    all_labels = labels.cpu().numpy()
    all_depths = depths.cpu().numpy()
    all_x_values = x_values.cpu().numpy()
    all_y_values = y_values.cpu().numpy()

    flat_eval = eval_mask.reshape(-1)
    flat_labels = all_labels.reshape(-1)[flat_eval]
    flat_predictions = predictions.reshape(-1)[flat_eval]
    flat_probs = mean_probs.reshape(-1, mean_probs.shape[-1])[flat_eval]
    flat_confidence = confidence.reshape(-1)[flat_eval]
    flat_total_uncertainty = total_uncertainty.reshape(-1)[flat_eval]
    flat_aleatoric = aleatoric.reshape(-1)[flat_eval]
    flat_epistemic = epistemic.reshape(-1)[flat_eval]
    flat_evidential_uncertainty = (
        evidential_uncertainty.reshape(-1)[flat_eval] if evidential_uncertainty is not None else None
    )
    flat_correct = (flat_predictions == flat_labels).astype(np.int64)
    flat_error = 1 - flat_correct
    flat_difficulty_sigma = difficulty_sigma.reshape(-1)[flat_eval] if difficulty_sigma is not None else None

    metrics: dict[str, Any] = {}
    metrics["num_eval_tokens"] = int(flat_labels.shape[0])
    metrics["accuracy"] = float(flat_correct.mean()) if flat_correct.size else float("nan")
    metrics["kappa"] = float(cohen_kappa_score(flat_labels, flat_predictions)) if flat_labels.size else float("nan")
    if flat_labels.size:
        metrics["nll"] = float(
            -np.log(np.clip(flat_probs[np.arange(flat_labels.shape[0]), flat_labels], 1e-12, 1.0)).mean()
        )
    else:
        metrics["nll"] = float("nan")
    metrics["brier"] = multiclass_brier_score(flat_probs, flat_labels)
    metrics["ece"] = expected_calibration_error(flat_confidence, flat_predictions, flat_labels, args.ece_bins)
    metrics["mean_confidence"] = float(flat_confidence.mean()) if flat_confidence.size else float("nan")
    metrics["mean_total_uncertainty"] = float(flat_total_uncertainty.mean()) if flat_total_uncertainty.size else float("nan")
    metrics["mean_aleatoric_uncertainty"] = float(flat_aleatoric.mean()) if flat_aleatoric.size else float("nan")
    metrics["mean_epistemic_uncertainty"] = float(flat_epistemic.mean()) if flat_epistemic.size else float("nan")
    if flat_evidential_uncertainty is not None and flat_evidential_uncertainty.size:
        metrics["mean_evidential_uncertainty"] = float(flat_evidential_uncertainty.mean())
        metrics["error_auroc_evidential_uncertainty"] = safe_binary_metric("auroc", flat_error, flat_evidential_uncertainty)
        metrics["error_auprc_evidential_uncertainty"] = safe_binary_metric("auprc", flat_error, flat_evidential_uncertainty)
    if flat_difficulty_sigma is not None and flat_difficulty_sigma.size:
        metrics["mean_difficulty_sigma"] = float(flat_difficulty_sigma.mean())
        metrics["error_auroc_difficulty_sigma"] = safe_binary_metric("auroc", flat_error, flat_difficulty_sigma)
        metrics["error_auprc_difficulty_sigma"] = safe_binary_metric("auprc", flat_error, flat_difficulty_sigma)
    metrics["error_auroc_total_uncertainty"] = safe_binary_metric("auroc", flat_error, flat_total_uncertainty)
    metrics["error_auprc_total_uncertainty"] = safe_binary_metric("auprc", flat_error, flat_total_uncertainty)
    metrics["error_auroc_epistemic"] = safe_binary_metric("auroc", flat_error, flat_epistemic)
    metrics["error_auprc_epistemic"] = safe_binary_metric("auprc", flat_error, flat_epistemic)
    metrics["confusion_matrix"] = confusion_matrix(
        flat_labels,
        flat_predictions,
        labels=list(range(len(CLASS_NAMES))),
    ).tolist()
    metrics.update(selective_metrics(flat_correct.astype(bool), flat_total_uncertainty))

    conformal_rows: list[list[int]] | None = None
    if conformal is not None:
        qhat = float(conformal["qhat"])
        method = str(conformal["method"])
        conformal_rows = []
        for prob_vector in mean_probs.reshape(-1, mean_probs.shape[-1]):
            conformal_rows.append(build_prediction_set(prob_vector, qhat, method))

        covered = []
        set_sizes = []
        flat_sets = [conformal_rows[index] for index, keep in enumerate(flat_eval) if keep]
        for true_label, pred_set in zip(flat_labels, flat_sets):
            covered.append(int(int(true_label) in pred_set))
            set_sizes.append(len(pred_set))

        metrics["conformal_coverage"] = float(np.mean(covered)) if covered else float("nan")
        metrics["conformal_mean_set_size"] = float(np.mean(set_sizes)) if set_sizes else float("nan")
        metrics["conformal_singleton_rate"] = float(np.mean(np.array(set_sizes) == 1)) if set_sizes else float("nan")
        metrics["conformal_qhat"] = qhat
        metrics["conformal_method"] = method

    rows: list[dict[str, Any]] = []
    flattened_names = []
    for borehole_name in names:
        flattened_names.extend([borehole_name] * all_labels.shape[1])

    flat_all_labels = all_labels.reshape(-1)
    flat_all_predictions = predictions.reshape(-1)
    flat_all_valid = valid_mask.cpu().numpy().reshape(-1)
    flat_all_eval = eval_mask.reshape(-1)
    flat_all_depths = all_depths.reshape(-1)
    flat_all_x_values = all_x_values.reshape(-1)
    flat_all_y_values = all_y_values.reshape(-1)
    flat_all_confidence = confidence.reshape(-1)
    flat_all_total_uncertainty = total_uncertainty.reshape(-1)
    flat_all_aleatoric = aleatoric.reshape(-1)
    flat_all_epistemic = epistemic.reshape(-1)
    flat_all_evidential_uncertainty = evidential_uncertainty.reshape(-1) if evidential_uncertainty is not None else None
    flat_all_variation_ratio = variation_ratio.reshape(-1)
    flat_all_probs = mean_probs.reshape(-1, mean_probs.shape[-1])
    flat_all_difficulty_sigma = difficulty_sigma.reshape(-1) if difficulty_sigma is not None else None

    for index, borehole_name in enumerate(flattened_names):
        row = {
            "borehole": borehole_name,
            "depth": float(flat_all_depths[index]),
            "x": float(flat_all_x_values[index]),
            "y": float(flat_all_y_values[index]),
            "valid_mask": bool(flat_all_valid[index]),
            "eval_mask": bool(flat_all_eval[index]),
            "true_class_id": int(flat_all_labels[index]),
            "true_class_name": CLASS_NAMES[int(flat_all_labels[index])],
            "pred_class_id": int(flat_all_predictions[index]),
            "pred_class_name": CLASS_NAMES[int(flat_all_predictions[index])],
            "correct": int(flat_all_predictions[index] == flat_all_labels[index]),
            "confidence": float(flat_all_confidence[index]),
            "variation_ratio": float(flat_all_variation_ratio[index]),
            "total_uncertainty": float(flat_all_total_uncertainty[index]),
            "aleatoric_uncertainty": float(flat_all_aleatoric[index]),
            "epistemic_uncertainty": float(flat_all_epistemic[index]),
        }
        if flat_all_evidential_uncertainty is not None:
            row["evidential_uncertainty"] = float(flat_all_evidential_uncertainty[index])
        if flat_all_difficulty_sigma is not None:
            row["difficulty_sigma"] = float(flat_all_difficulty_sigma[index])
        for class_id in range(len(CLASS_NAMES)):
            row[f"prob_class_{class_id}"] = float(flat_all_probs[index, class_id])

        if conformal_rows is not None:
            pred_set = conformal_rows[index]
            row["conformal_set_size"] = int(len(pred_set))
            row["conformal_covered"] = int(int(flat_all_labels[index]) in pred_set)
            row["conformal_set_class_ids"] = ";".join(str(class_id) for class_id in pred_set)
            row["conformal_set_class_names"] = ";".join(CLASS_NAMES[class_id] for class_id in pred_set)

        rows.append(row)

    return metrics, pd.DataFrame(rows)


def compute_global_metrics_from_prediction_frame(prediction_frame: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    if prediction_frame.empty:
        raise ValueError("The combined prediction frame is empty.")

    eval_mask = series_to_bool(prediction_frame["eval_mask"])
    valid_mask = series_to_bool(prediction_frame["valid_mask"])
    eval_frame = prediction_frame[eval_mask].copy()
    if eval_frame.empty:
        raise ValueError("No evaluation token is available after applying eval_mask.")

    labels = eval_frame["true_class_id"].to_numpy(dtype=np.int64)
    predictions = eval_frame["pred_class_id"].to_numpy(dtype=np.int64)
    confidences = eval_frame["confidence"].to_numpy(dtype=np.float64)
    total_uncertainty = eval_frame["total_uncertainty"].to_numpy(dtype=np.float64)
    aleatoric_uncertainty = eval_frame["aleatoric_uncertainty"].to_numpy(dtype=np.float64)
    epistemic_uncertainty = eval_frame["epistemic_uncertainty"].to_numpy(dtype=np.float64)
    evidential_uncertainty = (
        eval_frame["evidential_uncertainty"].to_numpy(dtype=np.float64)
        if "evidential_uncertainty" in eval_frame.columns
        else None
    )
    difficulty_sigma = eval_frame["difficulty_sigma"].to_numpy(dtype=np.float64) if "difficulty_sigma" in eval_frame.columns else None
    correct = eval_frame["correct"].to_numpy(dtype=np.int64)
    error = 1 - correct

    prob_columns = [f"prob_class_{class_id}" for class_id in range(len(CLASS_NAMES))]
    probs = eval_frame[prob_columns].to_numpy(dtype=np.float64)

    metrics: dict[str, Any] = {
        "num_boreholes": int(prediction_frame["borehole"].nunique()),
        "num_total_tokens": int(prediction_frame.shape[0]),
        "num_valid_tokens": int(valid_mask.sum()),
        "num_eval_tokens": int(labels.shape[0]),
        "accuracy": float(correct.mean()),
        "kappa": float(cohen_kappa_score(labels, predictions)),
        "nll": float(-np.log(np.clip(probs[np.arange(labels.shape[0]), labels], 1e-12, 1.0)).mean()),
        "brier": multiclass_brier_score(probs, labels),
        "ece": expected_calibration_error(confidences, predictions, labels, args.ece_bins),
        "mean_confidence": float(confidences.mean()),
        "mean_total_uncertainty": float(total_uncertainty.mean()),
        "mean_aleatoric_uncertainty": float(aleatoric_uncertainty.mean()),
        "mean_epistemic_uncertainty": float(epistemic_uncertainty.mean()),
        "mean_total_uncertainty_correct": float(total_uncertainty[correct == 1].mean()) if np.any(correct == 1) else float("nan"),
        "mean_total_uncertainty_error": float(total_uncertainty[error == 1].mean()) if np.any(error == 1) else float("nan"),
        "mean_epistemic_uncertainty_correct": float(epistemic_uncertainty[correct == 1].mean()) if np.any(correct == 1) else float("nan"),
        "mean_epistemic_uncertainty_error": float(epistemic_uncertainty[error == 1].mean()) if np.any(error == 1) else float("nan"),
        "error_auroc_total_uncertainty": safe_binary_metric("auroc", error, total_uncertainty),
        "error_auprc_total_uncertainty": safe_binary_metric("auprc", error, total_uncertainty),
        "error_auroc_epistemic": safe_binary_metric("auroc", error, epistemic_uncertainty),
        "error_auprc_epistemic": safe_binary_metric("auprc", error, epistemic_uncertainty),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=list(range(len(CLASS_NAMES))),
        ).tolist(),
    }
    if evidential_uncertainty is not None:
        metrics["mean_evidential_uncertainty"] = float(evidential_uncertainty.mean())
        metrics["mean_evidential_uncertainty_correct"] = (
            float(evidential_uncertainty[correct == 1].mean()) if np.any(correct == 1) else float("nan")
        )
        metrics["mean_evidential_uncertainty_error"] = (
            float(evidential_uncertainty[error == 1].mean()) if np.any(error == 1) else float("nan")
        )
        metrics["error_auroc_evidential_uncertainty"] = safe_binary_metric("auroc", error, evidential_uncertainty)
        metrics["error_auprc_evidential_uncertainty"] = safe_binary_metric("auprc", error, evidential_uncertainty)
    if difficulty_sigma is not None:
        metrics["mean_difficulty_sigma"] = float(difficulty_sigma.mean())
        metrics["mean_difficulty_sigma_correct"] = float(difficulty_sigma[correct == 1].mean()) if np.any(correct == 1) else float("nan")
        metrics["mean_difficulty_sigma_error"] = float(difficulty_sigma[error == 1].mean()) if np.any(error == 1) else float("nan")
        metrics["error_auroc_difficulty_sigma"] = safe_binary_metric("auroc", error, difficulty_sigma)
        metrics["error_auprc_difficulty_sigma"] = safe_binary_metric("auprc", error, difficulty_sigma)
    metrics.update(selective_metrics(correct.astype(bool), total_uncertainty))

    classwise_metrics: dict[str, dict[str, float]] = {}
    for class_id, class_name in CLASS_NAMES.items():
        mask = labels == class_id
        if not np.any(mask):
            continue
        classwise_metrics[class_name] = {
            "support": int(mask.sum()),
            "accuracy": float((predictions[mask] == labels[mask]).mean()),
            "mean_confidence": float(confidences[mask].mean()),
            "mean_total_uncertainty": float(total_uncertainty[mask].mean()),
            "mean_epistemic_uncertainty": float(epistemic_uncertainty[mask].mean()),
        }
        if difficulty_sigma is not None:
            classwise_metrics[class_name]["mean_difficulty_sigma"] = float(difficulty_sigma[mask].mean())
        if evidential_uncertainty is not None:
            classwise_metrics[class_name]["mean_evidential_uncertainty"] = float(evidential_uncertainty[mask].mean())
    metrics["classwise"] = classwise_metrics

    if "conformal_covered" in eval_frame.columns:
        set_sizes = eval_frame["conformal_set_size"].to_numpy(dtype=np.int64)
        covered = eval_frame["conformal_covered"].to_numpy(dtype=np.int64)
        metrics["conformal_coverage"] = float(covered.mean())
        metrics["conformal_mean_set_size"] = float(set_sizes.mean())
        metrics["conformal_singleton_rate"] = float(np.mean(set_sizes == 1))

    return metrics


def allocate_split_count(total: int, fraction: float) -> int:
    if fraction <= 0.0:
        return 0
    return max(1, int(round(total * fraction)))


def compute_borehole_xy_centers(processed: ProcessedBoreholeDataset) -> np.ndarray:
    x_centers = processed.x_values.mean(axis=1, dtype=np.float64)
    y_centers = processed.y_values.mean(axis=1, dtype=np.float64)
    return np.column_stack((x_centers, y_centers)).astype(np.float64)


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())


def parse_campaign_prefix_map(mapping_text: str) -> list[tuple[str, str]]:
    rules: list[tuple[str, str]] = []
    for raw_item in mapping_text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid campaign mapping item {item!r}. Use comma-separated prefix=campaign rules."
            )
        prefix, campaign = item.split("=", 1)
        prefix = prefix.strip()
        campaign = campaign.strip()
        if not prefix or not campaign:
            raise ValueError(f"Invalid campaign mapping item {item!r}.")
        rules.append((prefix, campaign))
    if not rules:
        raise ValueError("The campaign prefix map is empty.")
    return sorted(rules, key=lambda item: len(item[0]), reverse=True)


def infer_campaign_from_name(name: str, rules: list[tuple[str, str]]) -> str:
    for prefix, campaign in rules:
        if name.startswith(prefix):
            return campaign
    raise ValueError(
        f"Cannot assign borehole {name!r} to a campaign. Extend --campaign-prefix-map."
    )


def resolve_campaign_names(campaigns: list[str], args: argparse.Namespace) -> list[str]:
    unique_campaigns = sorted(set(campaigns))
    if args.test_campaign is not None:
        if args.test_campaign not in unique_campaigns:
            raise ValueError(
                f"Campaign {args.test_campaign!r} not found. Available campaigns: {unique_campaigns}."
            )
        return [args.test_campaign]
    if args.max_campaigns is not None:
        return unique_campaigns[: args.max_campaigns]
    return unique_campaigns


def split_reserved_indices(
    remaining: list[int],
    args: argparse.Namespace,
    rng_seed: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    shuffled = list(remaining)
    rng = np.random.default_rng(rng_seed)
    rng.shuffle(shuffled)

    model_sel_count = allocate_split_count(len(shuffled), args.model_selection_fraction)
    temperature_count = allocate_split_count(len(shuffled), args.temperature_fraction) if args.use_temperature_scaling else 0
    conformal_count = allocate_split_count(len(shuffled), args.conformal_fraction)
    reserved = model_sel_count + temperature_count + conformal_count

    if reserved >= len(shuffled):
        raise ValueError(
            "Too many boreholes are reserved for model selection/calibration. "
            "Reduce the hold-out fractions or use more data."
        )

    cursor = 0
    model_selection_indices = shuffled[cursor : cursor + model_sel_count]
    cursor += model_sel_count
    temperature_indices = shuffled[cursor : cursor + temperature_count]
    cursor += temperature_count
    conformal_indices = shuffled[cursor : cursor + conformal_count]
    cursor += conformal_count
    train_indices = shuffled[cursor:]

    if args.max_train_boreholes is not None and len(train_indices) > args.max_train_boreholes:
        train_indices = train_indices[: args.max_train_boreholes]

    if len(train_indices) == 0:
        raise ValueError("The training split is empty.")

    return train_indices, model_selection_indices, temperature_indices, conformal_indices


def build_fold_split(
    num_boreholes: int,
    test_index: int,
    borehole_xy: np.ndarray,
    args: argparse.Namespace,
    names: list[str] | None = None,
) -> FoldSplit:
    all_non_test = [index for index in range(num_boreholes) if index != test_index]
    buffer_excluded_indices: list[int] = []
    if args.exclusion_buffer_m > 0.0:
        deltas = borehole_xy - borehole_xy[test_index]
        distances = np.sqrt(np.sum(deltas * deltas, axis=1))
        buffer_excluded_indices = [
            index
            for index in all_non_test
            if distances[index] <= args.exclusion_buffer_m
        ]

    remaining = [index for index in all_non_test if index not in set(buffer_excluded_indices)]
    if len(remaining) == 0:
        fold_name = names[test_index] if names is not None else str(test_index)
        raise ValueError(
            f"Exclusion buffer {args.exclusion_buffer_m:.1f} m around fold {fold_name} leaves no eligible non-test boreholes."
        )

    try:
        train_indices, model_selection_indices, temperature_indices, conformal_indices = split_reserved_indices(
            remaining=remaining,
            args=args,
            rng_seed=args.seed + test_index,
        )
    except ValueError as exc:
        fold_name = names[test_index] if names is not None else str(test_index)
        raise ValueError(
            f"{exc} Fold={fold_name}, exclusion_buffer_m={args.exclusion_buffer_m:.1f}."
        ) from exc

    return FoldSplit(
        test_index=test_index,
        train_indices=train_indices,
        model_selection_indices=model_selection_indices,
        temperature_indices=temperature_indices,
        conformal_indices=conformal_indices,
        eligible_indices=remaining,
        buffer_excluded_indices=buffer_excluded_indices,
    )


def build_campaign_split(
    campaign_name: str,
    campaigns: list[str],
    borehole_xy: np.ndarray,
    args: argparse.Namespace,
    names: list[str],
) -> FoldSplit:
    test_indices = [index for index, campaign in enumerate(campaigns) if campaign == campaign_name]
    if not test_indices:
        raise ValueError(f"Campaign {campaign_name!r} has no boreholes.")

    test_set = set(test_indices)
    all_non_test = [index for index in range(len(names)) if index not in test_set]
    buffer_excluded_indices: list[int] = []
    if args.exclusion_buffer_m > 0.0:
        test_xy = borehole_xy[test_indices]
        for index in all_non_test:
            deltas = test_xy - borehole_xy[index]
            distances = np.sqrt(np.sum(deltas * deltas, axis=1))
            if float(distances.min()) <= args.exclusion_buffer_m:
                buffer_excluded_indices.append(index)

    remaining = [index for index in all_non_test if index not in set(buffer_excluded_indices)]
    if len(remaining) == 0:
        raise ValueError(
            f"Campaign fold {campaign_name!r} leaves no eligible training boreholes."
        )

    campaign_order = sorted(set(campaigns)).index(campaign_name)
    try:
        train_indices, model_selection_indices, temperature_indices, conformal_indices = split_reserved_indices(
            remaining=remaining,
            args=args,
            rng_seed=args.seed + 100_000 + campaign_order,
        )
    except ValueError as exc:
        raise ValueError(
            f"{exc} Campaign={campaign_name}, exclusion_buffer_m={args.exclusion_buffer_m:.1f}."
        ) from exc

    return FoldSplit(
        test_index=test_indices[0],
        test_indices=test_indices,
        fold_name=f"campaign_{safe_name(campaign_name)}",
        split_label="campaign",
        train_indices=train_indices,
        model_selection_indices=model_selection_indices,
        temperature_indices=temperature_indices,
        conformal_indices=conformal_indices,
        eligible_indices=remaining,
        buffer_excluded_indices=buffer_excluded_indices,
    )


def resolve_test_indices(names: list[str], args: argparse.Namespace) -> list[int]:
    if args.test_borehole is not None:
        lookup = {name: index for index, name in enumerate(names)}
        if args.test_borehole not in lookup:
            raise ValueError(
                f"Borehole {args.test_borehole!r} not found. Available names include {names[:10]}."
            )
        return [lookup[args.test_borehole]]

    if args.fold_index is not None:
        if args.fold_index < 0 or args.fold_index >= len(names):
            raise ValueError(f"--fold-index must be in [0, {len(names) - 1}].")
        return [args.fold_index]

    indices = list(range(len(names)))
    if args.max_folds is not None:
        indices = indices[: args.max_folds]
    return indices


def run_fold(
    processed: ProcessedBoreholeDataset,
    raw_embedding_dim: int,
    fold_split: FoldSplit,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    device = torch.device(args.device)
    test_indices = fold_split.test_indices if fold_split.test_indices is not None else [fold_split.test_index]
    fold_name = fold_split.fold_name or processed.names[fold_split.test_index]
    pca_cache_path = None
    if args.fold_pca_cache_dir is not None and args.embedding_pca_dim is not None:
        pca_cache_path = Path(args.fold_pca_cache_dir) / f"{fold_name}.npz"
    normalized, effective_embedding_dim = prepare_fold_features(
        processed=processed,
        train_indices=fold_split.train_indices,
        embedding_dim=raw_embedding_dim,
        embedding_pca_dim=args.embedding_pca_dim,
        seed=args.seed + fold_split.test_index,
        pca_cache_path=pca_cache_path,
    )

    train_data = subset_processed_dataset(normalized, fold_split.train_indices)
    model_selection_data = subset_processed_dataset(normalized, fold_split.model_selection_indices)
    temperature_data = subset_processed_dataset(normalized, fold_split.temperature_indices)
    conformal_data = subset_processed_dataset(normalized, fold_split.conformal_indices)
    test_data = subset_processed_dataset(normalized, test_indices)

    input_dim = train_data.features.shape[-1]
    seq_len = train_data.features.shape[1]

    models: list[nn.Module] = []
    member_histories: list[dict[str, Any]] = []
    epoch_test_prediction_frames: list[pd.DataFrame] = []

    num_members = args.ensemble_size if args.uq_method in {"deep_ensemble", "subsample_ensemble"} else 1
    member_train_names: list[list[str]] = []

    for member_index in range(num_members):
        member_seed = args.seed + (fold_split.test_index * 1000) + member_index
        member_train_data = train_data
        member_train_indices = list(fold_split.train_indices)
        if args.uq_method == "subsample_ensemble":
            rng = np.random.default_rng(member_seed)
            subsample_count = max(1, int(round(len(fold_split.train_indices) * float(args.subsample_fraction))))
            subsample_count = min(subsample_count, len(fold_split.train_indices))
            member_train_indices = sorted(
                rng.choice(fold_split.train_indices, size=subsample_count, replace=False).tolist()
            )
            member_train_data = subset_processed_dataset(normalized, member_train_indices)
        member_train_names.append([normalized.names[index] for index in member_train_indices])

        def save_epoch_snapshot(model_to_save: nn.Module, epoch_number: int) -> None:
            interval = int(args.epoch_snapshot_interval)
            if interval <= 0:
                return
            if epoch_number % interval != 0 and epoch_number != int(args.epochs):
                return
            snapshot_model_dir = (
                output_dir
                / "epoch_snapshots"
                / f"epoch_{epoch_number:04d}"
                / "models"
            )
            snapshot_model_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = snapshot_model_dir / f"fold_{fold_name}_member_{member_index}.pt"
            torch.save(model_to_save.state_dict(), snapshot_path)

        model, history, epoch_test_predictions = train_single_model(
            train_data=member_train_data,
            model_selection_data=model_selection_data,
            epoch_eval_data=test_data,
            args=args,
            device=device,
            input_dim=input_dim,
            seq_len=seq_len,
            member_seed=member_seed,
            member_index=member_index,
            epoch_snapshot_callback=save_epoch_snapshot,
        )
        models.append(model)
        member_histories.append(history)
        if epoch_test_predictions is not None:
            epoch_test_prediction_frames.append(epoch_test_predictions)

        if args.save_member_models:
            model_dir = output_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"fold_{fold_name}_member_{member_index}.pt"
            torch.save(model.state_dict(), model_path)

    temperature = 1.0
    conformal_predictor: dict[str, Any] | None = None

    if args.uq_method == "mc_dropout":
        reference_model = models[0]
        test_loader = build_dataloader(test_data, args, shuffle=False)
        test_outputs = collect_logits_from_model(
            model=reference_model,
            loader=test_loader,
            device=device,
            num_samples=args.mc_samples,
            stochastic=True,
        )

        if len(temperature_data.names) > 0:
            temperature_loader = build_dataloader(temperature_data, args, shuffle=False)
            temperature_outputs = collect_logits_from_model(
                model=reference_model,
                loader=temperature_loader,
                device=device,
                num_samples=args.mc_samples,
                stochastic=True,
            )
            if args.use_temperature_scaling:
                temperature = fit_temperature(
                    logits_samples=temperature_outputs["logits_samples"],
                    labels=temperature_outputs["labels"],
                    token_mask=temperature_outputs["valid_mask"] if args.eval_on_valid_only else torch.ones_like(temperature_outputs["valid_mask"], dtype=torch.bool),
                    args=args,
                    device=device,
                )

        if len(conformal_data.names) > 0:
            conformal_loader = build_dataloader(conformal_data, args, shuffle=False)
            conformal_outputs = collect_logits_from_model(
                model=reference_model,
                loader=conformal_loader,
                device=device,
                num_samples=args.mc_samples,
                stochastic=True,
            )
            conformal_mask = conformal_outputs["valid_mask"] if args.eval_on_valid_only else torch.ones_like(conformal_outputs["valid_mask"], dtype=torch.bool)
            conformal_predictor = fit_conformal_predictor(
                logits_samples=conformal_outputs["logits_samples"],
                labels=conformal_outputs["labels"],
                token_mask=conformal_mask,
                temperature=temperature,
                alpha=args.alpha,
                method=args.conformal_method,
                args=args,
            )
    else:
        test_loader = build_dataloader(test_data, args, shuffle=False)
        test_outputs = collect_logits_from_ensemble(models=models, loader=test_loader, device=device)

        if len(temperature_data.names) > 0:
            temperature_loader = build_dataloader(temperature_data, args, shuffle=False)
            temperature_outputs = collect_logits_from_ensemble(models=models, loader=temperature_loader, device=device)
            if args.use_temperature_scaling:
                temperature = fit_temperature(
                    logits_samples=temperature_outputs["logits_samples"],
                    labels=temperature_outputs["labels"],
                    token_mask=temperature_outputs["valid_mask"] if args.eval_on_valid_only else torch.ones_like(temperature_outputs["valid_mask"], dtype=torch.bool),
                    args=args,
                    device=device,
                )

        if len(conformal_data.names) > 0:
            conformal_loader = build_dataloader(conformal_data, args, shuffle=False)
            conformal_outputs = collect_logits_from_ensemble(models=models, loader=conformal_loader, device=device)
            conformal_mask = conformal_outputs["valid_mask"] if args.eval_on_valid_only else torch.ones_like(conformal_outputs["valid_mask"], dtype=torch.bool)
            conformal_predictor = fit_conformal_predictor(
                logits_samples=conformal_outputs["logits_samples"],
                labels=conformal_outputs["labels"],
                token_mask=conformal_mask,
                temperature=temperature,
                alpha=args.alpha,
                method=args.conformal_method,
                args=args,
            )

    fold_metrics, prediction_frame = summarize_fold_predictions(
        logits_samples=test_outputs["logits_samples"],
        labels=test_outputs["labels"],
        valid_mask=test_outputs["valid_mask"],
        depths=test_outputs["depths"],
        x_values=test_outputs["x_values"],
        y_values=test_outputs["y_values"],
        names=test_outputs["names"],
        temperature=temperature,
        args=args,
        conformal=conformal_predictor,
        log_sigma_samples=test_outputs.get("log_sigma_samples"),
    )

    fold_dir = output_dir / "folds" / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)

    representation_path: Path | None = None
    if args.save_representation_npz:
        representation_path = save_fold_representation_npz(
            model=models[0],
            train_data=train_data,
            test_data=test_data,
            fold_dir=fold_dir,
            args=args,
            device=device,
            raw_embedding_dim=effective_embedding_dim,
            member_index=0,
        )

    prediction_frame.insert(0, "fold_name", fold_name)
    prediction_frame.insert(1, "split_label", fold_split.split_label)
    prediction_frame.to_csv(fold_dir / "predictions.csv", index=False)
    learning_curve_frame = histories_to_frame(fold_name, member_histories)
    learning_curve_frame.to_csv(fold_dir / "learning_curves.csv", index=False)
    test_predictions_by_epoch_path: Path | None = None
    if epoch_test_prediction_frames:
        test_predictions_by_epoch_path = fold_dir / "test_predictions_by_epoch.csv"
        pd.concat(epoch_test_prediction_frames, axis=0, ignore_index=True).to_csv(
            test_predictions_by_epoch_path,
            index=False,
        )

    fold_artifacts = {
        "fold_name": fold_name,
        "test_index": fold_split.test_index,
        "buffer": {
            "radius_m": float(args.exclusion_buffer_m),
            "num_eligible_boreholes": len(fold_split.eligible_indices),
            "num_excluded_boreholes": len(fold_split.buffer_excluded_indices),
            "excluded_boreholes": [normalized.names[index] for index in fold_split.buffer_excluded_indices],
        },
        "split": {
            "train": [normalized.names[index] for index in fold_split.train_indices],
            "model_selection": [normalized.names[index] for index in fold_split.model_selection_indices],
            "temperature": [normalized.names[index] for index in fold_split.temperature_indices],
            "conformal": [normalized.names[index] for index in fold_split.conformal_indices],
            "test": [normalized.names[index] for index in test_indices],
        },
        "temperature": float(temperature),
        "optimizer": args.optimizer,
        "conformal": conformal_predictor,
        "member_histories": member_histories,
        "member_train_boreholes": member_train_names,
        "metrics": fold_metrics,
        "representation_path": str(representation_path) if representation_path is not None else None,
        "test_predictions_by_epoch_path": str(test_predictions_by_epoch_path) if test_predictions_by_epoch_path is not None else None,
    }

    with (fold_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(fold_artifacts, handle, indent=2, default=json_default)

    return {
        "fold_name": fold_name,
        "split_label": fold_split.split_label,
        "metrics": fold_metrics,
        "temperature": float(temperature),
        "conformal": conformal_predictor,
        "buffer_radius_m": float(args.exclusion_buffer_m),
        "num_eligible_boreholes": len(fold_split.eligible_indices),
        "num_buffer_excluded_boreholes": len(fold_split.buffer_excluded_indices),
        "num_train_boreholes": len(fold_split.train_indices),
        "member_train_boreholes": member_train_names,
        "num_test_boreholes": len(test_indices),
        "test_boreholes": [normalized.names[index] for index in test_indices],
        "prediction_path": str(fold_dir / "predictions.csv"),
        "learning_curve_path": str(fold_dir / "learning_curves.csv"),
        "representation_path": str(representation_path) if representation_path is not None else None,
        "test_predictions_by_epoch_path": str(test_predictions_by_epoch_path) if test_predictions_by_epoch_path is not None else None,
    }


def aggregate_fold_metrics(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"num_folds": len(fold_results)}
    numeric_keys: set[str] = set()

    for result in fold_results:
        for key, value in result["metrics"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_keys.add(key)

    for key in sorted(numeric_keys):
        values = [
            float(result["metrics"][key])
            for result in fold_results
            if result["metrics"].get(key) is not None and not np.isnan(result["metrics"][key])
        ]
        if not values:
            continue
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_std"] = float(np.std(values))

    aggregate["fold_names"] = [result["fold_name"] for result in fold_results]
    return aggregate


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    output_dir = filesystem_path(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, default=json_default)

    csv_paths = [Path(path) for path in sorted(glob(args.data_glob))]
    if not csv_paths:
        raise FileNotFoundError(f"No CSV file matched --data-glob {args.data_glob!r}.")

    raw_dataset = load_or_build_raw_dataset(csv_paths, args)
    processed = preprocess_raw_dataset(raw_dataset, args)
    processed, raw_embedding_dim = select_model_input_features(
        processed=processed,
        raw_embedding_dim=raw_dataset.embeddings.shape[-1],
        use_embeddings_as_feature=args.use_embeddings_as_feature,
    )
    borehole_xy = compute_borehole_xy_centers(processed)

    fold_results: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    if args.split_mode == "campaign":
        campaign_rules = parse_campaign_prefix_map(args.campaign_prefix_map)
        campaigns = [infer_campaign_from_name(name, campaign_rules) for name in processed.names]
        campaign_assignments = pd.DataFrame(
            {
                "borehole": processed.names,
                "campaign": campaigns,
            }
        )
        campaign_assignments.to_csv(output_dir / "campaign_assignments.csv", index=False)
        campaign_names = resolve_campaign_names(campaigns, args)

        for campaign_name in campaign_names:
            try:
                split = build_campaign_split(
                    campaign_name=campaign_name,
                    campaigns=campaigns,
                    borehole_xy=borehole_xy,
                    args=args,
                    names=processed.names,
                )
                result = run_fold(
                    processed=processed,
                    raw_embedding_dim=raw_embedding_dim,
                    fold_split=split,
                    args=args,
                    output_dir=output_dir,
                )
                result["campaign"] = campaign_name
                fold_results.append(result)
            except ValueError as exc:
                if not args.skip_impossible_folds:
                    raise
                skipped_rows.append(
                    {
                        "fold_name": f"campaign_{safe_name(campaign_name)}",
                        "campaign": campaign_name,
                        "split_label": "campaign",
                        "buffer_radius_m": float(args.exclusion_buffer_m),
                        "reason": str(exc),
                    }
                )
    else:
        test_indices = resolve_test_indices(processed.names, args)
        for test_index in test_indices:
            try:
                split = build_fold_split(
                    num_boreholes=len(processed.names),
                    test_index=test_index,
                    borehole_xy=borehole_xy,
                    args=args,
                    names=processed.names,
                )
                result = run_fold(
                    processed=processed,
                    raw_embedding_dim=raw_embedding_dim,
                    fold_split=split,
                    args=args,
                    output_dir=output_dir,
                )
                fold_results.append(result)
            except ValueError as exc:
                if not args.skip_impossible_folds:
                    raise
                skipped_rows.append(
                    {
                        "fold_name": processed.names[test_index],
                        "test_index": test_index,
                        "split_label": "borehole",
                        "buffer_radius_m": float(args.exclusion_buffer_m),
                        "reason": str(exc),
                    }
                )

    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(output_dir / "skipped_folds.csv", index=False)

    if not fold_results:
        raise RuntimeError(
            f"No fold completed successfully for exclusion buffer {args.exclusion_buffer_m:.1f} m."
        )

    fold_rows = []
    for result in fold_results:
        row = {
            "fold_name": result["fold_name"],
            "split_label": result.get("split_label"),
            "campaign": result.get("campaign"),
            "temperature": result["temperature"],
            "buffer_radius_m": result["buffer_radius_m"],
            "num_eligible_boreholes": result["num_eligible_boreholes"],
            "num_buffer_excluded_boreholes": result["num_buffer_excluded_boreholes"],
            "num_train_boreholes": result["num_train_boreholes"],
            "num_test_boreholes": result.get("num_test_boreholes"),
            "test_boreholes": ";".join(result.get("test_boreholes", [])),
            "representation_path": result.get("representation_path"),
            "test_predictions_by_epoch_path": result.get("test_predictions_by_epoch_path"),
        }
        for key, value in result["metrics"].items():
            if isinstance(value, list):
                continue
            row[key] = value
        if result["conformal"] is not None:
            row["conformal_qhat"] = result["conformal"]["qhat"]
            row["conformal_num_tokens"] = result["conformal"]["num_tokens"]
        fold_rows.append(row)

    fold_metrics_frame = pd.DataFrame(fold_rows)
    fold_metrics_frame.to_csv(output_dir / "fold_metrics.csv", index=False)

    aggregate_metrics = aggregate_fold_metrics(fold_results)
    with (output_dir / "aggregate_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate_metrics, handle, indent=2, default=json_default)

    prediction_frames = [pd.read_csv(result["prediction_path"]) for result in fold_results]
    combined_predictions = pd.concat(prediction_frames, axis=0, ignore_index=True)
    combined_predictions.to_csv(output_dir / "all_folds_predictions.csv", index=False)

    learning_curve_frames = [pd.read_csv(result["learning_curve_path"]) for result in fold_results]
    combined_learning_curves = pd.concat(learning_curve_frames, axis=0, ignore_index=True)
    combined_learning_curves.to_csv(output_dir / "all_folds_learning_curves.csv", index=False)

    epoch_summary = (
        combined_learning_curves.groupby("epoch", as_index=False)
        .agg(
            train_loss_mean=("train_loss", "mean"),
            train_loss_std=("train_loss", "std"),
            model_selection_loss_mean=("model_selection_loss", "mean"),
            model_selection_loss_std=("model_selection_loss", "std"),
            model_selection_accuracy_mean=("model_selection_accuracy", "mean"),
            model_selection_accuracy_std=("model_selection_accuracy", "std"),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            folds_observed=("fold_name", "nunique"),
        )
    )
    epoch_summary.to_csv(output_dir / "learning_curve_epoch_summary.csv", index=False)

    epoch_prediction_paths = [
        result.get("test_predictions_by_epoch_path")
        for result in fold_results
        if result.get("test_predictions_by_epoch_path")
    ]
    if epoch_prediction_paths:
        epoch_prediction_frames = [pd.read_csv(path) for path in epoch_prediction_paths]
        combined_epoch_predictions = pd.concat(epoch_prediction_frames, axis=0, ignore_index=True)
        combined_epoch_predictions.to_csv(output_dir / "all_folds_test_predictions_by_epoch.csv", index=False)

        epoch_eval = combined_epoch_predictions[series_to_bool(combined_epoch_predictions["eval_mask"])].copy()
        fold_epoch_accuracy = (
            epoch_eval.groupby(["epoch", "fold_name"], as_index=False)
            .agg(
                accuracy=("correct", "mean"),
                mean_confidence=("confidence", "mean"),
                num_eval_tokens=("correct", "size"),
            )
        )
        fold_epoch_accuracy.to_csv(output_dir / "test_prediction_fold_epoch_metrics.csv", index=False)

        test_prediction_epoch_summary = (
            fold_epoch_accuracy.groupby("epoch", as_index=False)
            .agg(
                test_accuracy_mean=("accuracy", "mean"),
                test_accuracy_std=("accuracy", "std"),
                test_accuracy_q025=("accuracy", lambda values: float(np.nanquantile(values, 0.025))),
                test_accuracy_q975=("accuracy", lambda values: float(np.nanquantile(values, 0.975))),
                mean_confidence_mean=("mean_confidence", "mean"),
                folds_observed=("fold_name", "nunique"),
                num_eval_tokens=("num_eval_tokens", "sum"),
            )
        )
        test_prediction_epoch_summary.to_csv(output_dir / "test_prediction_epoch_summary.csv", index=False)

    global_metrics = compute_global_metrics_from_prediction_frame(combined_predictions, args)
    with (output_dir / "global_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(global_metrics, handle, indent=2, default=json_default)


if __name__ == "__main__":
    main()
