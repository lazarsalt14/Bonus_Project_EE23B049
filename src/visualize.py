"""
Visualization utilities for L2RW experiments.

Generates all figures needed for the report:
  1. Imbalance experiment: F1 vs imbalance ratio
  2. Noise experiment: Accuracy vs noise rate
  3. Weight distribution histogram
  4. Training curves
  5. Confusion matrix comparison
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import json
from pathlib import Path


# ── Style ────────────────────────────────────────────────────

PALETTE = {
    "L2RW (ours)":               "#E85D24",   # coral
    "Vanilla MLP":               "#888780",   # gray
    "Logistic Reg":              "#378ADD",   # blue
    "Logistic Reg (balanced)":   "#185FA5",
    "Random Forest":             "#639922",   # green
    "Random Forest (balanced)":  "#3B6D11",
    "SVM":                       "#BA7517",   # amber
    "SVM (balanced)":            "#854F0B",
    "MLP + PROPORTION":          "#9F77DD",   # purple
    "MLP + RESAMPLE":            "#1D9E75",   # teal
}
MARKERS = {
    "L2RW (ours)":               "o",
    "Vanilla MLP":               "s",
    "Logistic Reg":              "^",
    "Logistic Reg (balanced)":   "v",
    "Random Forest":             "D",
    "Random Forest (balanced)":  "d",
    "SVM":                       "P",
    "SVM (balanced)":            "X",
    "MLP + PROPORTION":          "*",
    "MLP + RESAMPLE":            "h",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})


def _color(name):
    return PALETTE.get(name, "#333333")

def _marker(name):
    return MARKERS.get(name, "o")


# ─────────────────────────────────────────────────────────────
#  Figure 1: F1 vs Imbalance Ratio
# ─────────────────────────────────────────────────────────────

def plot_imbalance_results(results_dict, out_path, metric="f1_macro"):
    """
    results_dict: {ratio: [{"model": ..., "f1_macro": ...}, ...]}
    """
    ratios = sorted([int(k) for k in results_dict.keys()])
    all_models = [r["model"] for r in results_dict[str(ratios[0])]]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model in all_models:
        vals = [
            next(r[metric] for r in results_dict[str(ratio)]
                 if r["model"] == model)
            for ratio in ratios
        ]
        lw = 2.5 if "L2RW" in model else 1.2
        ax.plot(ratios, vals,
                label=model,
                color=_color(model),
                marker=_marker(model),
                linewidth=lw,
                markersize=6 if "L2RW" in model else 5,
                zorder=10 if "L2RW" in model else 5)

    ax.set_xlabel("Imbalance ratio (majority:minority)")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title("Class Imbalance Experiment – Credit Dataset")
    ax.set_xticks(ratios)
    ax.set_xticklabels([f"{r}:1" for r in ratios])
    ax.legend(fontsize=8, framealpha=0.9, loc="lower left",
               ncol=2, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
#  Figure 2: Accuracy vs Noise Rate
# ─────────────────────────────────────────────────────────────

def plot_noise_results(results_uniform, results_bg, out_path, metric="accuracy"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    for ax, results_dict, title in zip(
            axes,
            [results_uniform, results_bg],
            ["Uniform flip noise", "Background flip noise"]):

        rates = sorted([float(k) for k in results_dict.keys()])
        all_models = [r["model"] for r in results_dict[str(rates[0])]]

        for model in all_models:
            vals = []
            for rate in rates:
                row = next(r for r in results_dict[str(rate)]
                           if r["model"] == model)
                vals.append(row[metric])
            lw = 2.5 if "L2RW" in model else 1.2
            ax.plot([r * 100 for r in rates], vals,
                    label=model,
                    color=_color(model),
                    marker=_marker(model),
                    linewidth=lw,
                    markersize=6 if "L2RW" in model else 4,
                    zorder=10 if "L2RW" in model else 5)

        ax.set_xlabel("Noise rate (%)")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(title)
        ax.set_xticks([r * 100 for r in rates])

    axes[0].legend(fontsize=7.5, framealpha=0.9, loc="lower left",
                    ncol=1, columnspacing=0.8)
    fig.suptitle("Noisy Label Experiment – Phishing Dataset", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
#  Figure 3: Weight Distribution
# ─────────────────────────────────────────────────────────────

def plot_weight_distribution(l2rw_model, X_noisy, y_noisy,
                              noise_mask, out_path):
    """
    Reproduce Figure 3 from the paper:
    Weight histogram for clean vs noisy examples.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from l2rw import compute_meta_weights

    n_classes = len(np.unique(y_noisy))

    # Use the val set stored in the model
    scaler = l2rw_model.scaler_
    model  = l2rw_model.model_

    label_map = {c: i for i, c in enumerate(l2rw_model.classes_)}
    y_m = np.array([label_map[yi] for yi in y_noisy])

    X_s = scaler.transform(X_noisy)
    # Build a small clean val set from known clean examples
    clean_idx = np.where(~noise_mask)[0][:50]
    noisy_idx = np.where(noise_mask)[0][:50]

    if len(clean_idx) < 10 or len(noisy_idx) < 10:
        print("  Not enough clean/noisy examples for weight plot, skipping.")
        return

    # Compute weights for a batch containing clean and noisy examples
    batch_idx = np.concatenate([clean_idx[:25], noisy_idx[:25]])
    X_batch   = X_s[batch_idx]
    y_batch   = y_m[batch_idx]
    X_val_s   = X_s[clean_idx[:20]]
    y_val_s   = y_m[clean_idx[:20]]

    weights = compute_meta_weights(
        model, X_batch, y_batch, X_val_s, y_val_s,
        l2rw_model.lr, n_classes)

    w_clean = weights[:25]
    w_noisy = weights[25:]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.linspace(0, weights.max() * 1.1 + 1e-6, 30)
    ax.hist(w_clean, bins=bins, alpha=0.7, color="#378ADD",
             label="Noisy examples", density=True)
    ax.hist(w_noisy, bins=bins, alpha=0.7, color="#E85D24",
             label="Clean Examples", density=True)
    ax.set_xlabel("Example weight")
    ax.set_ylabel("Density")
    ax.set_title("Weight distribution: clean vs noisy examples")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
#  Figure 4: Training Curves
# ─────────────────────────────────────────────────────────────

def plot_training_curves(l2rw_model, vanilla_model, out_path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(l2rw_model.train_losses_,   color="#E85D24", lw=1.5,
             label="L2RW (train)")
    ax.plot(l2rw_model.val_losses_,     color="#E85D24", lw=1.5,
             linestyle="--", label="L2RW (val)")
    ax.plot(vanilla_model.train_losses_, color="#888780", lw=1.5,
             label="Vanilla MLP (train)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training curves: L2RW vs Vanilla MLP")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
#  Figure 5: Summary Bar Chart
# ─────────────────────────────────────────────────────────────

def plot_summary_bar(results_at_worst, out_path):
    """
    Bar chart comparing all models at the hardest setting
    (highest imbalance or highest noise).
    """
    models = [r["model"] for r in results_at_worst]
    f1s    = [r["f1_macro"] for r in results_at_worst]
    aucs   = [r["auc"] if not np.isnan(r["auc"]) else 0
               for r in results_at_worst]
    colors = [_color(m) for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width/2, f1s,  width, label="F1 Macro",  color=colors, alpha=0.85)
    ax.bar(x + width/2, aucs, width, label="AUC-ROC",   color=colors, alpha=0.5,
            edgecolor=colors, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (balanced)", "\n(balanced)") for m in models],
                        rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison at hardest setting")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
#  Generate all figures from saved results
# ─────────────────────────────────────────────────────────────

def generate_all_figures(results_path, figures_dir):
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        data = json.load(f)

    print("\nGenerating figures...")

    # Fig 1: Imbalance (F1)
    plot_imbalance_results(
        data["imbalance"],
        os.path.join(figures_dir, "fig1_imbalance_f1.pdf"),
        metric="f1_macro")

    # Fig 1b: Imbalance (AUC)
    plot_imbalance_results(
        data["imbalance"],
        os.path.join(figures_dir, "fig1b_imbalance_auc.pdf"),
        metric="auc")

    # Fig 2: Noise
    plot_noise_results(
        data["noise_uniform"],
        data["noise_background"],
        os.path.join(figures_dir, "fig2_noise_accuracy.pdf"),
        metric="accuracy")

    # Fig 2b: Noise (F1)
    plot_noise_results(
        data["noise_uniform"],
        data["noise_background"],
        os.path.join(figures_dir, "fig2b_noise_f1.pdf"),
        metric="f1_macro")

    # Fig 5: Summary bar (worst imbalance)
    worst_ratio = str(max(int(k) for k in data["imbalance"].keys()))
    plot_summary_bar(
        data["imbalance"][worst_ratio],
        os.path.join(figures_dir, "fig5_summary_bar.pdf"))

    print(f"All figures saved to {figures_dir}")


if __name__ == "__main__":
    results_path = "../results/all_results.json"
    figures_dir  = "../figures"
    if os.path.exists(results_path):
        generate_all_figures(results_path, figures_dir)
    else:
        print(f"Results not found at {results_path}. Run experiments.py first.")
