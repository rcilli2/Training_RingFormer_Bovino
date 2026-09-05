"""Generate the conceptual figures used by the introductory Markdown chapters."""

from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOTS = (
    [ROOT / "bovino_full_reproduction", ROOT / "bovino_results_explorer"]
    if (ROOT / "bovino_full_reproduction").is_dir()
    else [ROOT]
)
ASSETS = PACKAGE_ROOTS[0] / "assets"
BLUE = "#234b68"
TEAL = "#25867c"
ORANGE = "#c97824"
GREY = "#56616b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})


def save(fig, name):
    fig.savefig(ASSETS / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    for package in PACKAGE_ROOTS[1:]:
        destination = package / "assets"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ASSETS / name, destination / name)


def box(ax, x, y, w, h, label, color=BLUE, size=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                               facecolor="#f4f7f9", edgecolor=color, linewidth=1.7))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            color=color, fontsize=size)


def arrow(ax, start, end, color=GREY, style="-", connection="arc3"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.8,
                            "linestyle": style, "connectionstyle": connection})


def geological_semantic_space():
    from matplotlib.patches import Ellipse

    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.set(xlim=(-4.8, 5.2), ylim=(-3.2, 4.1), xticks=[], yticks=[],
           xlabel="Illustrative embedding coordinate 1",
           ylabel="Illustrative embedding coordinate 2")
    ax.spines[["top", "right"]].set_visible(False)
    groups = [
        (-2.5, 1.8, 3.2, 2.7, BLUE, "Clay and silt", [
            (-3.1, 1.4, "Dry silty clay"),
            (-2.7, 2.2, "Moist silty clay"),
            (-1.6, 1.65, "Fine-grained silt")]),
        (2.5, 1.65, 3.7, 2.65, TEAL, "Sand and gravel", [
            (1.45, 1.3, "Fine sand"),
            (2.65, 2.25, "Dry coarse gravel"),
            (3.15, 1.3, "Wet coarse gravel")]),
        (.25, -1.4, 4.0, 2.2, ORANGE, "Limestone", [
            (-.65, -1.65, "Intact limestone"),
            (1.2, -1.3, "Fractured limestone")]),
    ]
    for x, y, width, height, color, title, points in groups:
        ax.add_patch(Ellipse((x, y), width, height, facecolor=color,
                             edgecolor="none", alpha=.065))
        ax.text(x, y + height/2 + .12, title, ha="center", color=color,
                weight="bold", fontsize=12)
        for px, py, label in points:
            ax.scatter(px, py, s=85, color=color, edgecolor="white", linewidth=.8, zorder=3)
            ax.annotate(label, (px, py), xytext=(0, -20), textcoords="offset points",
                        ha="center", fontsize=10, color=color)
    ax.set_title("Geological descriptions in a conceptual semantic space", fontsize=16,
                 weight="bold", pad=22)
    fig.text(.5, .02, "Hand-placed examples: not a MiniLM projection or measured geological distances.",
             ha="center", color=GREY, fontsize=10)
    fig.tight_layout(rect=(0, .055, 1, 1))
    save(fig, "geological_semantic_space.png")


def pca_geometry():
    rng = np.random.default_rng(13)
    angle = np.deg2rad(32)
    direction = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-direction[1], direction[0]])
    scores = rng.normal(0, 1.7, 55)
    points = scores[:, None] * direction + rng.normal(0, .38, (55, 1)) * normal
    points -= points.mean(axis=0)
    _, _, components = np.linalg.svd(points, full_matrices=False)
    direction = components[0]
    if direction[0] < 0:
        direction = -direction
    projection = (points @ direction)[:, None] * direction
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.scatter(*points.T, s=25, color=BLUE, alpha=.8)
    for p, q in zip(points, projection):
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#aeb9c0", lw=.7)
    ax.plot([-4 * direction[0], 4 * direction[0]],
            [-4 * direction[1], 4 * direction[1]], color=TEAL, lw=2.4,
            label="First principal component")
    ax.scatter(*projection.T, s=11, color=TEAL)
    ax.set(xlabel="Centered feature 1", ylabel="Centered feature 2",
           title="Two-dimensional vectors and their projections")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=10)
    ax = axes[1]
    ax.axhline(0, color=TEAL, lw=2)
    ax.scatter(points @ direction, np.zeros(len(points)), s=25, color=TEAL, alpha=.7)
    ax.set(xlabel="Coordinate on the first component", yticks=[], ylim=(-1, 1),
           title="One coordinate retained")
    ax.text(.5, .72, "Variation perpendicular to this axis\nis discarded.",
            transform=ax.transAxes, ha="center", color=GREY)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("PCA as a geometric projection", fontsize=17, weight="bold")
    fig.tight_layout()
    save(fig, "pca_geometry.png")


def encoder_block():
    fig, ax = plt.subplots(figsize=(10.5, 7.8))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(.5, .97, "One BOVINO Transformer encoder layer", ha="center", fontsize=18, weight="bold")
    stages = [(.82, "Encoder input: 128 features per position", BLUE),
              (.67, "Multihead self-attention: 16 heads x 8 dimensions\nOutput projection and dropout", TEAL),
              (.53, "Residual addition + LayerNorm", BLUE),
              (.33, "Token-wise feed-forward network\nLinear 128 -> 2048\nReLU + dropout\nLinear 2048 -> 128 + output dropout", ORANGE),
              (.16, "Residual addition + LayerNorm", BLUE),
              (.025, "Hidden output: 128 features per position", BLUE)]
    heights = [.075, .09, .07, .15, .07, .07]
    for (y, label, color), h in zip(stages, heights):
        box(ax, .2, y, .6, h, label, color)
    for i in range(len(stages) - 1):
        arrow(ax, (.5, stages[i][0] - .008), (.5, stages[i+1][0] + heights[i+1] + .008))
    for x, start, end in [(.10, .82, .565), (.9, .53, .195)]:
        edge = .2 if x < .5 else .8
        ax.plot([edge, x, x], [start + .02, start + .02, end], color=GREY, lw=1.4)
        arrow(ax, (x, end), (edge, end))
        ax.text(x - .018 if x < .5 else x + .018, (start + end) / 2, "Residual path",
                rotation=90, ha="center", va="center", color=GREY, fontsize=10)
    save(fig, "bovino_encoder_block.png")


def training_cycle():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(.5, .94, "How the model learns", ha="center", fontsize=18, weight="bold")
    box(ax, .02, .61, .18, .14, "Input features\nBorehole batch")
    box(ax, .28, .61, .23, .14, "Projection + encoder\n+ classification head", TEAL)
    box(ax, .59, .61, .17, .14, "Class\nprobabilities")
    box(ax, .83, .61, .15, .14, "Cross-entropy\nloss", ORANGE)
    for a, b in [(.20, .28), (.51, .59), (.76, .83)]:
        arrow(ax, (a + .008, .68), (b - .01, .68))
    box(ax, .81, .81, .18, .07, "Observed labels", ORANGE, 11)
    arrow(ax, (.9, .80), (.9, .76), ORANGE)
    box(ax, .64, .19, .32, .17, "Backpropagation\nCompute gradients through\nthe trainable network", ORANGE)
    box(ax, .27, .19, .25, .17, "Adam\nUpdate weights using\ngradient averages", TEAL)
    arrow(ax, (.90, .60), (.90, .37), ORANGE)
    arrow(ax, (.63, .275), (.53, .275), ORANGE)
    arrow(ax, (.395, .37), (.395, .60), TEAL)
    ax.text(.40, .49, "Updated weights", ha="left", color=TEAL, fontsize=10)
    ax.text(.02, .035, "Repeat over training batches and epochs.\nCached sentence embeddings and fitted PCA remain fixed.",
            fontsize=11, color=GREY)
    save(fig, "training_cycle.png")


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    geological_semantic_space()
    pca_geometry()
    encoder_block()
    training_cycle()
