from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
COLORS = {
    "MC dropout": "#24476b",
    "Deep ensemble": "#e68632",
    "Subsample ensemble": "#3f9560",
    "LLLA": "#c94b45",
    "TabICLv2 (hidden)": "#76559b",
}
CLASS_COLORS = {
    0: "#2b8c3b",
    1: "#f4e7a1",
    2: "#f4a623",
    3: "#ef9474",
    4: "#8f79c6",
    5: "#4f86c6",
}
CLASS_NAMES = {
    0: "Topsoil",
    1: "Elu (4)",
    2: "Debris (3)",
    3: "Transitional Flysch (2b)",
    4: "Clay Flysch (2a)",
    5: "Fyr Flysch (1)",
}


def load_predictions(name: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA / name)
    if "eval_mask" in frame:
        mask = frame["eval_mask"].astype(str).str.lower().isin({"true", "1", "yes"})
        frame = frame.loc[mask].copy()
    return frame


def global_accuracy(frame: pd.DataFrame) -> float:
    if "correct" in frame:
        return float(pd.to_numeric(frame["correct"], errors="coerce").mean())
    return float((frame["true_class_id"] == frame["pred_class_id"]).mean())


def plot_protocol_confusions() -> tuple[plt.Figure, np.ndarray]:
    frames = [load_predictions("lobo_predictions.csv"), load_predictions("loco_mc_dropout_predictions.csv")]
    labels = sorted(set().union(*[set(f.true_class_id).union(f.pred_class_id) for f in frames]))
    names = {}
    for frame in frames:
        names.update(dict(frame[["true_class_id", "true_class_name"]].drop_duplicates().values))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for axis, frame, title in zip(axes, frames, ["Leave-one-borehole-out", "Leave-one-campaign-out"]):
        counts = confusion_matrix(frame.true_class_id, frame.pred_class_id, labels=labels)
        normalized = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for row in range(len(labels)):
            for col in range(len(labels)):
                value = normalized[row, col]
                axis.text(col, row, f"{value:.0%}\n({counts[row, col]})", ha="center", va="center", fontsize=8,
                          color="white" if value >= 0.52 else "#17212b")
        axis.set_title(f"{title}\nOA = {global_accuracy(frame):.3f}", fontweight="bold")
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Observed class")
        class_names = [names.get(label, str(label)) for label in labels]
        axis.set_xticks(range(len(labels)), class_names, rotation=35, ha="right")
        axis.set_yticks(range(len(labels)), class_names)
    fig.colorbar(image, ax=axes, label="Row-normalized fraction", shrink=0.84)
    return fig, axes


def _reliability_points(frame: pd.DataFrame, bins: int = 10):
    confidence = frame.confidence.clip(0, 1).to_numpy()
    correct = frame.correct.astype(float).to_numpy()
    index = np.clip(np.digitize(confidence, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    points = [(confidence[index == i].mean(), correct[index == i].mean(), (index == i).sum())
              for i in range(bins) if np.any(index == i)]
    x, y, n = map(np.asarray, zip(*points))
    return x, y, float(np.sum(n * np.abs(x - y)) / np.sum(n))


def plot_loco_reliability_and_roc() -> tuple[plt.Figure, np.ndarray]:
    files = {
        "MC dropout": "loco_mc_dropout_predictions.csv",
        "Deep ensemble": "loco_deep_ensemble_predictions.csv",
        "Subsample ensemble": "loco_subsample_ensemble_predictions.csv",
        "LLLA": "loco_llla_predictions.csv",
        "TabICLv2 (hidden)": "loco_tabicl_hidden_predictions.csv",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), constrained_layout=True)
    axes[0].plot([0, 1], [0, 1], "--", color="#68717a", label="Perfect calibration")
    axes[1].plot([0, 1], [0, 1], "--", color="#68717a", label="Random")
    for method, filename in files.items():
        frame = load_predictions(filename)
        x, y, ece = _reliability_points(frame)
        axes[0].plot(x, y, marker="o", color=COLORS[method], label=f"{method} (ECE={ece:.3f})")
        errors = 1 - frame.correct.astype(int).to_numpy()
        score = 1 - frame.confidence.to_numpy()
        fpr, tpr, _ = roc_curve(errors, score)
        axes[1].plot(fpr, tpr, color=COLORS[method], label=f"{method} (AUROC={roc_auc_score(errors, score):.3f})")
    axes[0].set(xlabel="Mean confidence per bin", ylabel="Empirical accuracy", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
    for axis, title in zip(axes, ["LOCO reliability", "LOCO error detection"]):
        axis.set_title(title, fontweight="bold")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    return fig, axes


def load_buffer_metrics() -> pd.DataFrame:
    return pd.read_csv(DATA / "buffer_uq_method_metrics.csv")


def load_perturbation_metrics() -> pd.DataFrame:
    return pd.read_csv(DATA / "feature_perturbation_all_methods.csv")


def borehole_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate token predictions to one spatial record per borehole."""
    return (
        frame.groupby("borehole", as_index=False)
        .agg(
            x=("x", "median"),
            y=("y", "median"),
            accuracy=("correct", "mean"),
            confidence=("confidence", "mean"),
            predictive_entropy=("total_uncertainty", "mean"),
            epistemic_proxy=("epistemic_uncertainty", "mean"),
            intervals=("depth", "size"),
            max_depth=("depth", "max"),
        )
    )


def plot_borehole_profile(frame: pd.DataFrame, borehole: str) -> tuple[plt.Figure, np.ndarray]:
    """Plot observed/predicted units, probabilities, and uncertainty with depth."""
    profile = frame.loc[frame["borehole"].astype(str) == str(borehole)].sort_values("depth").copy()
    if profile.empty:
        available = ", ".join(sorted(frame["borehole"].astype(str).unique())[:12])
        raise KeyError(f"Unknown borehole {borehole!r}. Examples: {available}")

    depth = profile["depth"].to_numpy()
    true_ids = profile["true_class_id"].astype(int).to_numpy()
    pred_ids = profile["pred_class_id"].astype(int).to_numpy()
    probability_columns = [f"prob_class_{index}" for index in range(6)]
    probabilities = profile[probability_columns].to_numpy()
    class_names = CLASS_NAMES.copy()
    class_names.update(dict(profile[["true_class_id", "true_class_name"]].drop_duplicates().values))
    class_names.update(dict(profile[["pred_class_id", "pred_class_name"]].drop_duplicates().values))

    fig, axes = plt.subplots(
        1, 4, figsize=(13.5, 7), sharey=True, constrained_layout=True,
        gridspec_kw={"width_ratios": [0.8, 0.8, 2.5, 1.8]},
    )
    for axis, values, title in zip(axes[:2], [true_ids, pred_ids], ["Observed", "Predicted"]):
        for row, value in enumerate(values):
            lower = depth[row - 1] / 2 + depth[row] / 2 if row else max(0, depth[row] - 0.8)
            upper = depth[row] / 2 + depth[row + 1] / 2 if row + 1 < len(depth) else depth[row] + 0.8
            axis.axhspan(lower, upper, color=CLASS_COLORS.get(value, "#bdbdbd"))
        axis.set(title=title, xticks=[])

    cumulative = np.zeros(len(profile))
    for class_id in range(6):
        upper = cumulative + probabilities[:, class_id]
        axes[2].fill_betweenx(depth, cumulative, upper, color=CLASS_COLORS[class_id], alpha=0.9,
                              label=class_names.get(class_id, f"Class {class_id}"))
        cumulative = upper
    axes[2].set(title="Class probabilities", xlabel="Probability", xlim=(0, 1))
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8)

    axes[3].plot(profile["confidence"], depth, color="#24476b", label="Confidence")
    axes[3].plot(profile["total_uncertainty"], depth, color="#d95f3d", label="Predictive entropy")
    axes[3].plot(profile["epistemic_uncertainty"], depth, color="#76559b", label="MI / epistemic proxy")
    axes[3].set(title="Confidence and uncertainty", xlabel="Score")
    axes[3].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.18)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Depth (m)")
    fig.suptitle(f"Borehole {borehole}: held-out prediction profile", fontweight="bold")
    return fig, axes


def plot_spatial_boreholes(frame: pd.DataFrame) -> tuple[plt.Figure, np.ndarray]:
    """Map held-out boreholes by predictive performance and uncertainty."""
    summary = borehole_summary(frame)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    specifications = [
        ("accuracy", "Held-out accuracy", "viridis", 0, 1),
        ("predictive_entropy", "Mean predictive entropy", "magma", None, None),
        ("epistemic_proxy", "Mean MI / epistemic proxy", "magma", None, None),
    ]
    for axis, (column, title, cmap, lower, upper) in zip(axes, specifications):
        points = axis.scatter(summary["x"], summary["y"], c=summary[column], cmap=cmap,
                              vmin=lower, vmax=upper, s=35 + summary["intervals"],
                              edgecolor="white", linewidth=0.5)
        axis.set(title=title, xlabel="X", ylabel="Y", aspect="equal")
        axis.ticklabel_format(style="plain", useOffset=False)
        axis.grid(alpha=0.18)
        fig.colorbar(points, ax=axis, shrink=0.82)
    fig.suptitle("Spatial distribution of held-out borehole results", fontweight="bold")
    return fig, axes
