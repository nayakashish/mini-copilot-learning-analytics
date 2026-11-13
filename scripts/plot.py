"""Make the charts for the report from results/metrics/.

Writes to results/figures/:
    accuracy_vs_n.png          top-1 and top-5 accuracy vs n, word vs letter
    edit_distance_vs_n.png     edit distance vs n (raw and normalized)
    accuracy_by_project.png    top-1 accuracy vs n for each test project

Run from the repo root after scripts/evaluate.py:
    python3 scripts/plot.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # just saving files, no window
import matplotlib.pyplot as plt  # noqa: E402

METRICS = Path("results/metrics")
FIGURES = Path("results/figures")

# colours picked to stay readable for colour-blind people; every line also
# gets its own marker and a label so colour is never the only clue
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
MODEL_STYLE = {"word": (BLUE, "o"), "letter": (ORANGE, "s")}
PROJECT_STYLE = [(BLUE, "o"), (ORANGE, "s"), (AQUA, "^"), (YELLOW, "D")]
NS = list(range(1, 11))


def read(name: str, targets: str = "words") -> list[dict]:
    with (METRICS / name).open() as f:
        return [r for r in csv.DictReader(f) if r["targets"] == targets]


def series(rows: list[dict], model: str, col: str, project: str | None = None) -> list[float]:
    by_n = {int(r["n"]): float(r[col]) for r in rows
            if r["model"] == model and (project is None or r["project"] == project)}
    return [by_n[n] for n in NS]


def style(ax, title: str, ylabel: str, percent: bool = False) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel("n", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_xticks(NS)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    if percent:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")


def end_label(ax, ys: list[float], text: str, dy: float = 0) -> None:
    ax.annotate(text, (NS[-1], ys[-1]), xytext=(6, dy), textcoords="offset points",
                va="center", fontsize=9, color=INK)


def line(ax, ys, color, marker, dashed=False, label=None):
    ax.plot(NS, ys, color=color, marker=marker, markersize=6, linewidth=2,
            linestyle="--" if dashed else "-", label=label,
            markeredgecolor="white", markeredgewidth=1)


def accuracy_vs_n(rows) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model, (color, marker) in MODEL_STYLE.items():
        top1 = series(rows, model, "accuracy")
        top5 = series(rows, model, "top5_accuracy")
        line(ax, top5, color, marker, dashed=True, label=f"{model} model, top-5")
        line(ax, top1, color, marker, label=f"{model} model, top-1")
        end_label(ax, top1, f"{model} top-1", dy=-6 if model == "letter" else 6)
        end_label(ax, top5, f"{model} top-5", dy=-6 if model == "letter" else 6)
    style(ax, "Accuracy vs n (real words only, 4,000 test spots)", "accuracy", percent=True)
    ax.set_ylim(0, 0.7)
    ax.set_xlim(0.6, 12.2)
    ax.legend(frameon=False, fontsize=9, loc="lower right", handlelength=4)
    fig.tight_layout()
    return fig


def edit_distance_vs_n(rows) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    panels = [("edit_distance", "Edit distance (letters off)", "letters"),
              ("normalized_edit_distance", "Normalized edit distance", "edits / length of the right word")]
    for ax, (col, title, ylabel) in zip(axes, panels):
        for model, (color, marker) in MODEL_STYLE.items():
            ys = series(rows, model, col)
            line(ax, ys, color, marker, label=f"{model} model")
            end_label(ax, ys, model)
        style(ax, f"{title}, lower is better", ylabel)
        ax.set_xlim(0.6, 11.6)
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    return fig


def accuracy_by_project(rows) -> plt.Figure:
    projects = sorted({r["project"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, model in zip(axes, MODEL_STYLE):
        for project, (color, marker) in zip(projects, PROJECT_STYLE):
            line(ax, series(rows, model, "accuracy", project), color, marker, label=project)
        style(ax, f"{model.capitalize()} model: top-1 accuracy by test project", "accuracy", percent=True)
        ax.set_ylim(0, 0.5)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


def main() -> int:
    if not (METRICS / "ngram.csv").exists():
        print("No results yet, run scripts/evaluate.py first", file=sys.stderr)
        return 1
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = read("ngram.csv")
    by_project = read("ngram_by_project.csv")
    figures = {
        "accuracy_vs_n.png": accuracy_vs_n(rows),
        "edit_distance_vs_n.png": edit_distance_vs_n(rows),
        "accuracy_by_project.png": accuracy_by_project(by_project),
    }
    for name, fig in figures.items():
        fig.savefig(FIGURES / name, dpi=200, facecolor="white")
        plt.close(fig)
        print(f"wrote {FIGURES / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
