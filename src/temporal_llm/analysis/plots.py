"""
Plotting and analysis for paper figures.
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


# Paper-quality defaults
matplotlib.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

VARIANT_COLORS = {
    "baseline": "#d62728",
    "time_tokens": "#1f77b4",
    "time_memory": "#2ca02c",
    "text_time": "#9467bd",
}
VARIANT_LABELS = {
    "baseline": "Baseline (no time)",
    "time_tokens": "Time tokens",
    "time_memory": "Time + memory",
    "text_time": "Text time",
}

# Fallback palette for dynamically discovered variants
_FALLBACK_COLORS = ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _get_variant_color(variant: str, idx: int = 0) -> str:
    """Get color for a variant, falling back to palette for unknown variants."""
    if variant in VARIANT_COLORS:
        return VARIANT_COLORS[variant]
    return _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]


def _get_variant_label(variant: str) -> str:
    """Get label for a variant, falling back to cleaned-up name."""
    if variant in VARIANT_LABELS:
        return VARIANT_LABELS[variant]
    return variant.replace("_", " ").title()


def plot_results(results_dir: str | Path, output_dir: str | Path = None,
                 filter_variants: list[str] | None = None):
    """Generate all paper figures from saved results.

    Args:
        results_dir: Directory containing *_results.json files.
        output_dir: Where to save figures. Defaults to results_dir/figures.
        filter_variants: If set, only plot these variants. Otherwise discover all.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir or results_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all *_results.json files
    variant_results = {}
    for path in sorted(results_dir.glob("*_results.json")):
        variant = path.stem.replace("_results", "")
        if filter_variants and variant not in filter_variants:
            continue
        with open(path) as f:
            variant_results[variant] = json.load(f)

    if not variant_results:
        print("No results found to plot.")
        return

    # Generate figures
    _plot_overall_comparison(variant_results, output_dir)
    _plot_volatility_breakdown(variant_results, output_dir)
    _plot_stale_calibration(variant_results, output_dir)
    _plot_counterfactual_sensitivity(variant_results, output_dir)
    _plot_ablation_fake_time(variant_results, output_dir)

    print(f"Figures saved to {output_dir}")


def _plot_overall_comparison(variant_results: dict, output_dir: Path):
    """Figure 1: Overall accuracy and F1 across variants."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    variants = list(variant_results.keys())
    if not all("stale_calibration" in variant_results[v] for v in variants):
        return

    # Accuracy
    accs = [variant_results[v]["stale_calibration"]["overall_accuracy"] for v in variants]
    colors = [_get_variant_color(v, i) for i, v in enumerate(variants)]
    labels = [_get_variant_label(v) for v in variants]

    axes[0].bar(range(len(variants)), accs, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(range(len(variants)))
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Overall Action Accuracy")
    axes[0].set_ylim(0, 1)

    # F1
    f1s = [variant_results[v]["stale_calibration"]["f1_macro"] for v in variants]
    axes[1].bar(range(len(variants)), f1s, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(range(len(variants)))
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Action Classification F1")
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(output_dir / "fig1_overall_comparison.pdf")
    fig.savefig(output_dir / "fig1_overall_comparison.png")
    plt.close(fig)


def _plot_volatility_breakdown(variant_results: dict, output_dir: Path):
    """Figure 2: Accuracy broken down by fact volatility."""
    volatilities = ["high", "medium", "low", "stable"]
    variants = list(variant_results.keys())

    if not all("ablations" in variant_results[v] for v in variants):
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(volatilities))
    width = 0.25

    for i, variant in enumerate(variants):
        abl = variant_results[variant].get("ablations", {}).get("volatility_split", {})
        accs = [abl.get(vol, {}).get("accuracy", 0) for vol in volatilities]
        ax.bar(x + i * width, accs, width,
               label=_get_variant_label(variant),
               color=_get_variant_color(variant, i),
               edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Fact Volatility")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Fact Volatility")
    ax.set_xticks(x + width)
    ax.set_xticklabels([v.capitalize() for v in volatilities])
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(output_dir / "fig2_volatility_breakdown.pdf")
    fig.savefig(output_dir / "fig2_volatility_breakdown.png")
    plt.close(fig)


def _plot_stale_calibration(variant_results: dict, output_dir: Path):
    """Figure 3: False trust rate and unnecessary refresh rate."""
    variants = list(variant_results.keys())
    if not all("stale_calibration" in variant_results[v] for v in variants):
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = [_get_variant_label(v) for v in variants]
    colors = [_get_variant_color(v, i) for i, v in enumerate(variants)]

    # False trust rate
    ftr = [variant_results[v]["stale_calibration"]["false_trust_rate"] for v in variants]
    axes[0].bar(range(len(variants)), ftr, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(range(len(variants)))
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("Rate")
    axes[0].set_title("False Trust Rate (lower is better)")

    # Unnecessary refresh rate
    urr = [variant_results[v]["stale_calibration"]["unnecessary_refresh_rate"] for v in variants]
    axes[1].bar(range(len(variants)), urr, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(range(len(variants)))
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Unnecessary Refresh Rate (lower is better)")

    plt.tight_layout()
    fig.savefig(output_dir / "fig3_stale_calibration.pdf")
    fig.savefig(output_dir / "fig3_stale_calibration.png")
    plt.close(fig)


def _plot_counterfactual_sensitivity(variant_results: dict, output_dir: Path):
    """Figure 4: Counterfactual switch sensitivity and specificity."""
    variants = list(variant_results.keys())
    if not all("counterfactual_sensitivity" in variant_results[v] for v in variants):
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(variants))
    width = 0.35

    sens = [variant_results[v]["counterfactual_sensitivity"]["switch_sensitivity"]
            for v in variants]
    spec = [variant_results[v]["counterfactual_sensitivity"]["switch_specificity"]
            for v in variants]

    ax.bar(x - width / 2, sens, width, label="Switch sensitivity",
           color="#1f77b4", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, spec, width, label="Switch specificity",
           color="#ff7f0e", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Model Variant")
    ax.set_ylabel("Rate")
    ax.set_title("Counterfactual Time Sensitivity")
    ax.set_xticks(x)
    ax.set_xticklabels([_get_variant_label(v) for v in variants],
                       rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(output_dir / "fig4_counterfactual_sensitivity.pdf")
    fig.savefig(output_dir / "fig4_counterfactual_sensitivity.png")
    plt.close(fig)


def _plot_ablation_fake_time(variant_results: dict, output_dir: Path):
    """Figure 5: Real vs shuffled/fake time ablation."""
    # Only applies to time-aware variants
    time_variants = [v for v in variant_results
                     if "ablations" in variant_results[v]
                     and "fake_time" in variant_results[v].get("ablations", {})]
    if not time_variants:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(time_variants))
    width = 0.35

    real_accs = [variant_results[v]["ablations"]["fake_time"]["real_time_accuracy"]
                 for v in time_variants]
    fake_accs = [variant_results[v]["ablations"]["fake_time"]["shuffled_time_accuracy"]
                 for v in time_variants]

    ax.bar(x - width / 2, real_accs, width, label="Real time",
           color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, fake_accs, width, label="Shuffled time",
           color="#d62728", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Model Variant")
    ax.set_ylabel("Accuracy")
    ax.set_title("Ablation: Real vs Shuffled Time")
    ax.set_xticks(x)
    ax.set_xticklabels([_get_variant_label(v) for v in time_variants],
                       rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(output_dir / "fig5_ablation_fake_time.pdf")
    fig.savefig(output_dir / "fig5_ablation_fake_time.png")
    plt.close(fig)


def plot_volatility_temporal_gain(analysis_dir: str | Path, output_dir: str | Path = None):
    """Heatmap of accuracy delta (time-aware minus baseline) by volatility level."""
    analysis_dir = Path(analysis_dir)
    output_dir = Path(output_dir or analysis_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    vol_path = analysis_dir / "volatile_vs_stable.json"
    if not vol_path.exists():
        print(f"No volatile_vs_stable.json found in {analysis_dir}")
        return

    with open(vol_path) as f:
        results = json.load(f)

    # Find baselines
    baselines = {}
    for model_name in results:
        if "baseline" in model_name:
            baselines[model_name] = results[model_name]

    if not baselines:
        print("No baseline models found for temporal gain calculation")
        return

    volatilities = ["high", "medium", "low", "stable"]
    time_models = [m for m in results if "baseline" not in m]

    if not time_models:
        return

    # Build gain matrix
    gain_matrix = []
    model_labels = []
    for model_name in time_models:
        # Find matching baseline
        base_key = None
        for bname in baselines:
            prefix = bname.replace("_baseline", "").split("_")[0]
            if prefix in model_name:
                base_key = bname
                break
        if not base_key:
            base_key = list(baselines.keys())[0]

        gains = []
        for vol in volatilities:
            model_acc = results[model_name].get(vol, {}).get("accuracy", 0)
            base_acc = baselines[base_key].get(vol, {}).get("accuracy", 0)
            gains.append(model_acc - base_acc)
        gain_matrix.append(gains)
        model_labels.append(model_name.replace("_", " "))

    fig, ax = plt.subplots(figsize=(8, max(4, len(time_models) * 0.6 + 2)))
    data = np.array(gain_matrix)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-0.3, vmax=0.5)

    ax.set_xticks(range(len(volatilities)))
    ax.set_xticklabels([v.capitalize() for v in volatilities])
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels)

    # Annotate cells
    for i in range(len(model_labels)):
        for j in range(len(volatilities)):
            val = data[i, j]
            color = "white" if abs(val) > 0.25 else "black"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    color=color, fontsize=9)

    ax.set_title("Temporal Gain Over Baseline by Volatility")
    ax.set_xlabel("Fact Volatility")
    fig.colorbar(im, ax=ax, label="Accuracy Delta")

    plt.tight_layout()
    fig.savefig(output_dir / "volatility_temporal_gain.pdf")
    fig.savefig(output_dir / "volatility_temporal_gain.png")
    plt.close(fig)
    print(f"Saved volatility temporal gain heatmap to {output_dir}")


def plot_switch_sensitivity_by_volatility(analysis_dir: str | Path, output_dir: str | Path = None):
    """Bar chart: switch sensitivity per volatility level per model.

    This is the killer figure — shows time helps volatile facts
    without over-refreshing stable ones.
    """
    analysis_dir = Path(analysis_dir)
    output_dir = Path(output_dir or analysis_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # This requires per-volatility counterfactual data
    # Load predictions and compute per-volatility switch sensitivity
    pred_files = sorted(analysis_dir.glob("*_predictions.json"))
    if not pred_files:
        print("No prediction files found")
        return

    volatilities = ["high", "medium", "low", "stable"]
    model_data = {}

    for pred_path in pred_files:
        with open(pred_path) as f:
            data = json.load(f)

        model_name = data["model_name"]
        preds = data["predictions"]
        oracles = data["oracles"]
        vols = data["volatilities"]
        groups = data["counterfactual_groups"]

        # Compute switch sensitivity per volatility
        vol_sens = {}
        for vol in volatilities:
            # Filter to this volatility
            vol_indices = {i for i, v in enumerate(vols) if v == vol}

            # Group by counterfactual group
            group_data = defaultdict(list)
            for i in vol_indices:
                group_data[groups[i]].append(i)

            should_switch = 0
            did_switch = 0
            for gid, indices in group_data.items():
                if len(indices) < 2:
                    continue
                for a in range(len(indices)):
                    for b in range(a + 1, len(indices)):
                        ia, ib = indices[a], indices[b]
                        if oracles[ia] != oracles[ib]:
                            should_switch += 1
                            if preds[ia] != preds[ib]:
                                did_switch += 1

            vol_sens[vol] = did_switch / should_switch if should_switch > 0 else 0

        model_data[model_name] = vol_sens

    if not model_data:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(volatilities))
    n_models = len(model_data)
    width = 0.8 / n_models

    for i, (model_name, vol_sens) in enumerate(model_data.items()):
        vals = [vol_sens.get(v, 0) for v in volatilities]
        color = _get_variant_color(model_name, i)
        label = model_name.replace("_", " ")
        ax.bar(x + i * width - (n_models - 1) * width / 2, vals, width,
               label=label, color=color, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Fact Volatility")
    ax.set_ylabel("Switch Sensitivity")
    ax.set_title("Switch Sensitivity by Volatility Level")
    ax.set_xticks(x)
    ax.set_xticklabels([v.capitalize() for v in volatilities])
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(output_dir / "switch_sensitivity_by_volatility.pdf")
    fig.savefig(output_dir / "switch_sensitivity_by_volatility.png")
    plt.close(fig)
    print(f"Saved switch sensitivity by volatility to {output_dir}")


def plot_temporal_policy_curves(curves_data: dict, output_dir: str | Path):
    """
    2x2 grid (one per volatility), x=time (log), y=action probability.
    5 colored lines per panel (one per action).
    Expected: answer_directly decreases, refresh increases, crossover earlier for volatile.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    action_colors = {
        "answer_directly": "#2ca02c",
        "refresh": "#d62728",
        "retrieve_memory": "#1f77b4",
        "ask_clarify": "#ff7f0e",
        "abstain": "#9467bd",
    }

    volatilities = ["high", "medium", "low", "stable"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, vol in enumerate(volatilities):
        ax = axes[idx // 2][idx % 2]
        vol_data = curves_data.get(vol, {})

        if not vol_data:
            ax.set_title(f"{vol.capitalize()} Volatility (no data)")
            continue

        time_points = sorted(vol_data.keys(), key=float)
        times = [float(t) for t in time_points]

        for action, color in action_colors.items():
            probs = [vol_data[t].get(action, 0) for t in time_points]
            ax.plot(times, probs, color=color, label=action.replace("_", " "),
                    marker="o", markersize=3, linewidth=1.5)

        ax.set_xscale("log")
        ax.set_xlabel("Elapsed Time (seconds)")
        ax.set_ylabel("Action Probability")
        ax.set_title(f"{vol.capitalize()} Volatility")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Temporal Policy Curves by Volatility", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "temporal_policy_curves.pdf")
    fig.savefig(output_dir / "temporal_policy_curves.png")
    plt.close(fig)
    print(f"Saved temporal policy curves to {output_dir}")


def plot_prompt_robustness(robustness_data: dict, output_dir: str | Path):
    """Box/violin plot of accuracy distribution across prompt styles per model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = list(robustness_data.keys())
    conditions = set()
    for m in models:
        conditions.update(robustness_data[m].keys())
    conditions = sorted(conditions)

    if not models or not conditions:
        print("No robustness data to plot")
        return

    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 5),
                             squeeze=False)

    for cidx, condition in enumerate(conditions):
        ax = axes[0][cidx]
        data_for_plot = []
        labels = []

        for model in models:
            if condition in robustness_data[model]:
                m = robustness_data[model][condition]
                mean = m["mean_accuracy"]
                std = m["std_accuracy"]
                mn = m["min_accuracy"]
                mx = m["max_accuracy"]
                data_for_plot.append([mn, mean - std, mean, mean + std, mx])
                labels.append(model.replace("_", "\n"))

        if not data_for_plot:
            continue

        positions = range(len(labels))
        means = [d[2] for d in data_for_plot]
        mins = [d[0] for d in data_for_plot]
        maxs = [d[4] for d in data_for_plot]
        stds = [d[2] - d[1] for d in data_for_plot]

        ax.bar(positions, means, yerr=stds, capsize=5,
               color="#1f77b4", alpha=0.7, edgecolor="black", linewidth=0.5)

        # Mark min/max
        for i, (mn, mx) in enumerate(zip(mins, maxs)):
            ax.plot([i, i], [mn, mx], color="red", linewidth=1.5, zorder=5)
            ax.plot(i, mn, "v", color="red", markersize=6, zorder=5)
            ax.plot(i, mx, "^", color="red", markersize=6, zorder=5)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Condition: {condition}")
        ax.set_ylim(0, 1)

    plt.suptitle("Prompt Robustness: Accuracy Across Prompt Styles", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "prompt_robustness.pdf")
    fig.savefig(output_dir / "prompt_robustness.png")
    plt.close(fig)
    print(f"Saved prompt robustness plot to {output_dir}")


def generate_latex_table(results_dir: str | Path) -> str:
    """Generate LaTeX table for the paper."""
    results_dir = Path(results_dir)
    variant_results = {}
    for path in sorted(results_dir.glob("*_results.json")):
        variant = path.stem.replace("_results", "")
        with open(path) as f:
            variant_results[variant] = json.load(f)

    if not variant_results:
        return "% No results found"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main results across model variants.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Metric & Baseline & Time Tokens & Time + Memory \\",
        r"\midrule",
    ]

    metrics = [
        ("Accuracy", lambda v: v.get("stale_calibration", {}).get("overall_accuracy", 0)),
        ("Macro F1", lambda v: v.get("stale_calibration", {}).get("f1_macro", 0)),
        ("Refresh F1", lambda v: v.get("stale_calibration", {}).get("refresh_f1", 0)),
        ("False Trust Rate", lambda v: v.get("stale_calibration", {}).get("false_trust_rate", 0)),
        ("Switch Sensitivity", lambda v: v.get("counterfactual_sensitivity", {}).get("switch_sensitivity", 0)),
        ("Switch Specificity", lambda v: v.get("counterfactual_sensitivity", {}).get("switch_specificity", 0)),
    ]

    for name, fn in metrics:
        vals = []
        for v in ["baseline", "time_tokens", "time_memory"]:
            if v in variant_results:
                vals.append(f"{fn(variant_results[v]):.3f}")
            else:
                vals.append("--")
        lines.append(f"{name} & {' & '.join(vals)} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)
