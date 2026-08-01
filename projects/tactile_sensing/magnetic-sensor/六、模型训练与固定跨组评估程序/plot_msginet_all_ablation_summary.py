import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


GROUP_LABELS = {
    "benchmark": "Full",
    "task": "Task",
    "input": "Input",
    "multiscale": "Multiscale",
    "gating": "Gating",
    "depth": "Depth",
    "phy_weight": "Lambda",
    "stability": "Stability",
    "frontend": "Frontend",
    "pooling": "Pooling",
}


CONDITION_LABELS = {
    "full_msgi": "Full",
    "cls_only": "Cls only",
    "total4": "Total4",
    "xyz12": "XYZ12",
    "single_scale": "Single scale",
    "k3_only": "K3 only",
    "k7_only": "K7 only",
    "no_gate": "No gate",
    "layers_2": "2 blocks",
    "layers_6": "6 blocks",
    "phy0p00": "lambda=0",
    "phy0p05": "lambda=0.05",
    "phy0p20": "lambda=0.2",
    "phy0p50": "lambda=0.5",
    "no_residual": "No residual",
    "no_bn": "No BN",
    "no_resnorm": "No res+BN",
    "no_conv_stem": "No conv stem",
    "max_pool": "Max pool",
    "last_pool": "Last pool",
}


PLOT_ORDER = [
    "full_msgi",
    "cls_only",
    "total4",
    "xyz12",
    "single_scale",
    "k3_only",
    "k7_only",
    "no_gate",
    "layers_2",
    "layers_6",
    "phy0p00",
    "phy0p05",
    "phy0p20",
    "phy0p50",
    "no_residual",
    "no_bn",
    "no_resnorm",
    "no_conv_stem",
    "max_pool",
    "last_pool",
]


def _percent(series):
    return pd.to_numeric(series, errors="coerce") * 100.0


def _condition_label(condition):
    return CONDITION_LABELS.get(condition, condition.replace("_", " "))


def _sort_stats(stats):
    order = {name: idx for idx, name in enumerate(PLOT_ORDER)}
    stats = stats.copy()
    stats["_order"] = stats["condition"].map(lambda x: order.get(x, 10_000))
    return stats.sort_values(["_order", "group", "condition"]).drop(columns=["_order"])


def read_stats(results_dir):
    stats_path = results_dir / "all_ablation_stats.csv"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing stats file: {stats_path}")
    stats = pd.read_csv(stats_path, encoding="utf-8-sig")
    stats = _sort_stats(stats)
    stats["label"] = stats["condition"].map(_condition_label)
    stats["group_label"] = stats["group"].map(lambda x: GROUP_LABELS.get(x, x))
    return stats


def plot_acc_f1_bar(stats, out_dir):
    df = stats.copy()
    x = np.arange(len(df))
    width = 0.38
    acc = _percent(df["test_acc_mean"])
    acc_std = _percent(df["test_acc_std"])
    f1 = _percent(df["macro_f1_mean"])
    f1_std = _percent(df["macro_f1_std"])

    fig_width = max(12, len(df) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    ax.bar(x - width / 2, acc, width, yerr=acc_std, label="Test Acc", capsize=3, color="#4C78A8")
    ax.bar(x + width / 2, f1, width, yerr=f1_std, label="Macro-F1", capsize=3, color="#F58518")
    ax.set_ylabel("Score (%)")
    ax.set_title("Ablation Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "all_ablation_acc_f1_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_delta_acc(stats, out_dir):
    baseline = stats.loc[stats["condition"] == "full_msgi", "test_acc_mean"]
    if baseline.empty:
        return
    base = float(baseline.iloc[0])
    df = stats.copy()
    df["delta_acc"] = (pd.to_numeric(df["test_acc_mean"], errors="coerce") - base) * 100.0
    colors = ["#54A24B" if v >= 0 else "#E45756" for v in df["delta_acc"]]

    fig_width = max(12, len(df) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    ax.bar(np.arange(len(df)), df["delta_acc"], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Delta Test Acc (percentage points)")
    ax.set_title("Accuracy Change Relative to Full MSGI-Net")
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["label"], rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    fig.tight_layout()
    fig.savefig(out_dir / "all_ablation_delta_acc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_force_mae_bar(stats, out_dir):
    df = stats[pd.to_numeric(stats["force_mae_mean"], errors="coerce").notna()].copy()
    if df.empty:
        return
    x = np.arange(len(df))
    mae = pd.to_numeric(df["force_mae_mean"], errors="coerce")
    mae_std = pd.to_numeric(df["force_mae_std"], errors="coerce").fillna(0.0)

    fig_width = max(10, len(df) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    ax.bar(x, mae, yerr=mae_std, capsize=3, color="#72B7B2")
    ax.set_ylabel("Force MAE (N)")
    ax.set_title("Force Regression Error Across Ablations")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    fig.tight_layout()
    fig.savefig(out_dir / "force_mae_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_phy_weight_curve(stats, out_dir):
    rows = []
    mapping = {"phy0p00": 0.0, "phy0p05": 0.05, "full_msgi": 0.1, "phy0p20": 0.2, "phy0p50": 0.5}
    for condition, lam in mapping.items():
        subset = stats[stats["condition"] == condition]
        if subset.empty:
            continue
        row = subset.iloc[0].to_dict()
        row["lambda"] = lam
        rows.append(row)
    if len(rows) < 2:
        return
    df = pd.DataFrame(rows).sort_values("lambda")

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.errorbar(
        df["lambda"],
        _percent(df["test_acc_mean"]),
        yerr=_percent(df["test_acc_std"]),
        marker="o",
        capsize=3,
        color="#4C78A8",
        label="Test Acc",
    )
    ax1.set_xlabel("Physical supervision weight lambda")
    ax1.set_ylabel("Test Acc (%)", color="#4C78A8")
    ax1.tick_params(axis="y", labelcolor="#4C78A8")
    ax1.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.45)

    if pd.to_numeric(df["force_mae_mean"], errors="coerce").notna().any():
        ax2 = ax1.twinx()
        ax2.errorbar(
            df["lambda"],
            pd.to_numeric(df["force_mae_mean"], errors="coerce"),
            yerr=pd.to_numeric(df["force_mae_std"], errors="coerce").fillna(0.0),
            marker="s",
            capsize=3,
            color="#E45756",
            label="Force MAE",
        )
        ax2.set_ylabel("Force MAE (N)", color="#E45756")
        ax2.tick_params(axis="y", labelcolor="#E45756")

    ax1.set_title("Physical Supervision Weight Ablation")
    fig.tight_layout()
    fig.savefig(out_dir / "phy_weight_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_depth_curve(stats, out_dir):
    mapping = {"layers_2": 2, "full_msgi": 4, "layers_6": 6}
    rows = []
    for condition, depth in mapping.items():
        subset = stats[stats["condition"] == condition]
        if subset.empty:
            continue
        row = subset.iloc[0].to_dict()
        row["depth"] = depth
        rows.append(row)
    if len(rows) < 2:
        return
    df = pd.DataFrame(rows).sort_values("depth")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.errorbar(
        df["depth"],
        _percent(df["test_acc_mean"]),
        yerr=_percent(df["test_acc_std"]),
        marker="o",
        capsize=3,
        color="#4C78A8",
        label="Test Acc",
    )
    ax.errorbar(
        df["depth"],
        _percent(df["macro_f1_mean"]),
        yerr=_percent(df["macro_f1_std"]),
        marker="s",
        capsize=3,
        color="#F58518",
        label="Macro-F1",
    )
    ax.set_xlabel("Number of MSGIBlocks")
    ax.set_ylabel("Score (%)")
    ax.set_title("MSGIBlock Depth Ablation")
    ax.set_xticks(df["depth"])
    ax.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "depth_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def condition_meta(stats, condition):
    subset = stats[stats["condition"] == condition]
    if subset.empty:
        return None
    row = subset.iloc[0]
    return {
        "group": row["group"],
        "condition": row["condition"],
        "model_name": row["model_name"],
    }


def read_confusion(path):
    return pd.read_csv(path, index_col=0, encoding="utf-8-sig")


def mean_confusion_for_condition(results_dir, stats, condition, seeds):
    meta = condition_meta(stats, condition)
    if meta is None:
        return None
    matrices = []
    for seed in seeds:
        path = (
            results_dir
            / str(meta["group"])
            / f"{meta['condition']}_seed{seed}_details"
            / str(meta["model_name"])
            / "test"
            / "confusion_matrix.csv"
        )
        if path.exists():
            matrices.append(read_confusion(path))
    if not matrices:
        return None
    base_index = matrices[0].index
    base_columns = matrices[0].columns
    values = np.stack([m.loc[base_index, base_columns].values.astype(float) for m in matrices], axis=0)
    mean_values = values.mean(axis=0)
    return pd.DataFrame(mean_values, index=base_index, columns=base_columns)


def save_confusion_plots(results_dir, stats, out_dir, seeds, conditions):
    conf_dir = out_dir / "mean_confusion"
    conf_dir.mkdir(parents=True, exist_ok=True)
    for condition in conditions:
        cm = mean_confusion_for_condition(results_dir, stats, condition, seeds)
        if cm is None:
            continue
        label = _condition_label(condition).replace(" ", "_").lower()
        cm.to_csv(conf_dir / f"mean_confusion_{label}.csv", encoding="utf-8-sig")

        row_sums = cm.sum(axis=1).replace(0, np.nan)
        cm_norm = cm.div(row_sums, axis=0).fillna(0.0)
        cm_norm.to_csv(conf_dir / f"mean_confusion_{label}_normalized.csv", encoding="utf-8-sig")

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(cm_norm, cmap="Blues", vmin=0, vmax=1, ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Mean Normalized Confusion Matrix: {_condition_label(condition)}")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        fig.tight_layout()
        fig.savefig(conf_dir / f"mean_confusion_{label}_normalized.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def write_figure_index(out_dir):
    rows = []
    for path in sorted(out_dir.rglob("*.png")):
        rows.append({"figure": str(path.relative_to(out_dir)), "path": str(path)})
    if not rows:
        return
    with (out_dir / "figure_index.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["figure", "path"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Plot summary figures for MSGI-Net full ablation results.")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results/msginet_all_ablation_fixed_0112_val1314_test1520_ep60_3seeds"),
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[45, 46, 47])
    parser.add_argument(
        "--confusion_conditions",
        nargs="+",
        default=["full_msgi", "cls_only", "no_gate", "single_scale", "k3_only", "k7_only"],
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir = args.out_dir or (results_dir / "summary_figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper")
    stats = read_stats(results_dir)

    plot_acc_f1_bar(stats, out_dir)
    plot_delta_acc(stats, out_dir)
    plot_force_mae_bar(stats, out_dir)
    plot_phy_weight_curve(stats, out_dir)
    plot_depth_curve(stats, out_dir)
    save_confusion_plots(results_dir, stats, out_dir, args.seeds, args.confusion_conditions)
    write_figure_index(out_dir)

    print(f"Saved ablation summary figures to {out_dir}")


if __name__ == "__main__":
    main()
