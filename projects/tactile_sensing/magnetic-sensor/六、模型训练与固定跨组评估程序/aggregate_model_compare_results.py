import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PREFERRED_MEAN_ORDER = [
    "best_epoch",
    "best_val_loss",
    "best_val_acc",
    "test_loss",
    "test_acc",
    "test_ce",
    "test_phy",
    "eval_acc",
    "macro_f1",
    "weighted_f1",
    "n_samples",
]


def parse_numeric(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
        parsed = float(text)
        if not math.isfinite(parsed):
            return None
        return parsed
    except Exception:
        return None


def ordered_metric_names(metric_names):
    preferred = [name for name in PREFERRED_MEAN_ORDER if name in metric_names]
    remaining = sorted(name for name in metric_names if name not in preferred)
    return preferred + remaining


def load_confusion_matrix(confusion_csv):
    df = pd.read_csv(confusion_csv, index_col=0)
    df.index = [str(x) for x in df.index]
    df.columns = [str(x) for x in df.columns]
    return df


def save_heatmap(df, out_path, title, fmt, cmap):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=df.columns,
        yticklabels=df.index,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def aggregate_confusion_matrices(out_dir, models, seeds, eval_split):
    agg_root = out_dir / "aggregated_confusion"
    agg_root.mkdir(parents=True, exist_ok=True)

    for model in models:
        matrices = []
        labels = None
        for seed in seeds:
            cm_path = out_dir / f"model_compare_seed{seed}_details" / model / eval_split / "confusion_matrix.csv"
            if not cm_path.exists():
                raise FileNotFoundError(f"Missing confusion matrix CSV: {cm_path}")
            df = load_confusion_matrix(cm_path)
            if labels is None:
                labels = list(df.index)
            else:
                for label in df.index:
                    if label not in labels:
                        labels.append(label)
            matrices.append(df)

        aligned = []
        for df in matrices:
            aligned.append(df.reindex(index=labels, columns=labels, fill_value=0))

        mean_df = sum(aligned) / float(len(aligned))
        row_sums = mean_df.sum(axis=1).replace(0, 1.0)
        norm_df = mean_df.div(row_sums, axis=0)

        model_dir = agg_root / model
        model_dir.mkdir(parents=True, exist_ok=True)

        mean_csv = model_dir / "confusion_matrix_mean.csv"
        norm_csv = model_dir / "confusion_matrix_mean_normalized.csv"
        mean_png = model_dir / "confusion_matrix_mean.png"
        norm_png = model_dir / "confusion_matrix_mean_normalized.png"

        mean_df.to_csv(mean_csv, encoding="utf-8-sig")
        norm_df.to_csv(norm_csv, encoding="utf-8-sig")

        save_heatmap(
            mean_df,
            mean_png,
            title=f"{model} Mean Confusion Matrix ({eval_split})",
            fmt=".1f",
            cmap="Blues",
        )
        save_heatmap(
            norm_df,
            norm_png,
            title=f"{model} Mean Normalized Confusion Matrix ({eval_split})",
            fmt=".2f",
            cmap="Blues",
        )


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed model compare results and mean confusion matrices.")
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--eval_split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    metric_names = set()

    for seed in args.seeds:
        per_seed_csv = out_dir / f"model_compare_seed{seed}.csv"
        if not per_seed_csv.exists():
            raise FileNotFoundError(f"Missing per-seed CSV: {per_seed_csv}")
        with per_seed_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                record = {"model": row["model"], "seed": int(seed)}
                for key, value in row.items():
                    if key == "model":
                        continue
                    parsed = parse_numeric(value)
                    if parsed is not None:
                        record[key] = parsed
                        metric_names.add(key)

                detail_dir = out_dir / f"model_compare_seed{seed}_details" / row["model"] / args.eval_split
                cm_csv = detail_dir / "confusion_matrix.csv"
                summary_csv = detail_dir / "summary_metrics.csv"

                if cm_csv.exists():
                    cm_df = load_confusion_matrix(cm_csv)
                    total = float(cm_df.to_numpy().sum())
                    correct = float(cm_df.to_numpy().diagonal().sum())
                    record["eval_acc"] = (correct / total) if total > 0 else 0.0
                    record["n_samples"] = int(total)
                    metric_names.update({"eval_acc", "n_samples"})

                if summary_csv.exists():
                    summary_df = pd.read_csv(summary_csv)
                    if not summary_df.empty:
                        summary_row = summary_df.iloc[0].to_dict()
                        for key in ("macro_f1", "weighted_f1", "n_samples"):
                            parsed = parse_numeric(summary_row.get(key))
                            if parsed is not None:
                                record[key] = parsed
                                metric_names.add(key)

                run_rows.append(record)

    runs_csv = out_dir / "aggregated_runs.csv"
    stats_csv = out_dir / "aggregated_stats.csv"
    mean_csv = out_dir / "aggregated_mean.csv"

    ordered_metrics = ordered_metric_names(metric_names)

    with runs_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "seed"] + ordered_metrics)
        writer.writeheader()
        for row in run_rows:
            writer.writerow(row)

    grouped = {}
    for row in run_rows:
        grouped.setdefault(row["model"], []).append(row)

    stats_rows = []
    for model in args.models:
        items = grouped.get(model, [])
        if not items:
            continue
        row = {"model": model, "n_runs": len(items)}
        for metric in ordered_metrics:
            values = [float(item[metric]) for item in items if metric in item]
            if not values:
                continue
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        stats_rows.append(row)

    sort_key_name = "test_acc_mean" if any("test_acc_mean" in row for row in stats_rows) else "eval_acc_mean"
    if any(sort_key_name in row for row in stats_rows):
        stats_rows.sort(key=lambda r: (-r.get(sort_key_name, float("-inf")), r.get("best_val_loss_mean", float("inf"))))

    stats_fieldnames = ["model", "n_runs"]
    for metric in ordered_metrics:
        if any(f"{metric}_mean" in row for row in stats_rows):
            stats_fieldnames.extend([f"{metric}_mean", f"{metric}_std"])

    with stats_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stats_fieldnames)
        writer.writeheader()
        for row in stats_rows:
            writer.writerow(row)

    mean_fieldnames = ["model"] + [metric for metric in ordered_metrics if any(f"{metric}_mean" in row for row in stats_rows)]
    with mean_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=mean_fieldnames)
        writer.writeheader()
        for row in stats_rows:
            writer.writerow({"model": row["model"], **{metric: row.get(f"{metric}_mean") for metric in mean_fieldnames if metric != "model"}})

    aggregate_confusion_matrices(out_dir=out_dir, models=args.models, seeds=args.seeds, eval_split=args.eval_split)

    print(runs_csv)
    print(stats_csv)
    print(mean_csv)
    print(out_dir / "aggregated_confusion")


if __name__ == "__main__":
    main()
