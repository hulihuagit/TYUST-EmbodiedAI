import argparse
import copy
import csv
import math
import re
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESS_DIR = PROJECT_ROOT / "四、磁场信号预处理程序"
if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))

from preprocess import SequenceDataset
from train import eval_epoch, seed_worker, set_global_seed, train_epoch
from compare_model_zoo import (
    build_model,
    fit_evaluate_sklearn_model,
    is_sklearn_model,
    normalize_model_name,
    save_detailed_eval_artifacts,
)


SUFFIX_RE = re.compile(r"_(\d{2})\.(?:npy|csv)$", re.IGNORECASE)


def parse_suffix(path: Path):
    match = SUFFIX_RE.search(path.name)
    return match.group(1) if match else None


def collect_split_indices(dataset, train_suffixes, val_suffixes, test_suffixes):
    train_suffixes = set(train_suffixes)
    val_suffixes = set(val_suffixes)
    test_suffixes = set(test_suffixes)

    overlap = (train_suffixes & val_suffixes) | (train_suffixes & test_suffixes) | (val_suffixes & test_suffixes)
    if overlap:
        raise ValueError(f"Split suffixes overlap: {sorted(overlap)}")

    train_indices = []
    val_indices = []
    test_indices = []

    for idx, (path, _, _) in enumerate(dataset.items):
        suffix = parse_suffix(path)
        if suffix in train_suffixes:
            train_indices.append(idx)
        elif suffix in val_suffixes:
            val_indices.append(idx)
        elif suffix in test_suffixes:
            test_indices.append(idx)

    return train_indices, val_indices, test_indices


def summarize_split(dataset, indices):
    counts = {}
    for idx in indices:
        path, cls_idx, _ = dataset.items[idx]
        class_name = path.parent.name
        counts[class_name] = counts.get(class_name, 0) + 1
    return dict(sorted(counts.items()))


def main():
    parser = argparse.ArgumentParser(description="Compare model zoo models on a fixed suffix-based split.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_mode", type=str, default="multitask", choices=["multitask", "cls_only"])
    parser.add_argument("--phy_weight", type=float, default=1.0)
    parser.add_argument("--phy_loss", type=str, default="mae", choices=["mse", "mae", "huber", "relmse"])
    parser.add_argument("--calib_a", type=float, default=1.0)
    parser.add_argument("--calib_b", type=float, default=0.0)
    parser.add_argument("--channel_preset", type=str, default="all25", choices=["all25", "magnetic16", "xyz12", "total4"])
    parser.add_argument("--drop_axes", nargs="*", default=[], choices=["X", "Y", "Z", "total"])
    parser.add_argument("--drop_sensors", nargs="*", default=[], choices=["S1", "S2", "S3", "S4"])
    parser.add_argument("--save_detailed_eval", action="store_true")
    parser.add_argument("--eval_split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--no_val_split", action="store_true", help="skip validation split and keep the final epoch model")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", help="enable deterministic training")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false", help="disable deterministic training")
    parser.add_argument("--train_suffixes", nargs="+", default=["01", "02", "03", "04", "05", "06", "07"])
    parser.add_argument("--val_suffixes", nargs="*", default=["08"])
    parser.add_argument("--test_suffixes", nargs="+", default=["09", "10"])
    parser.add_argument(
        "--models",
        nargs="*",
        default=["baseline_1dcnn", "tcn", "mamba_like", "msgi_net"],
        choices=[
            "baseline_1dcnn",
            "tcn",
            "transformer",
            "lstm",
            "gru",
            "bilstm",
            "mamba_like",
            "msgi_net",
            "msgi_single_scale",
            "msgi_k3_only",
            "msgi_k7_only",
            "msgi_no_gate",
            "msgi_no_residual",
            "msgi_no_bn",
            "msgi_no_resnorm",
            "msgi_2layers",
            "msgi_6layers",
            "msgi_no_conv_stem",
            "msgi_max_pool",
            "msgi_last_pool",
            "ms_mamba_like",
            "knn",
            "random_forest",
            "linear_svm",
        ],
    )
    parser.add_argument("--out_csv", type=str, default="results/model_compare_fixed_suffix_split.csv")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="optional directory for saving each neural model best checkpoint",
    )
    parser.set_defaults(deterministic=True)
    args = parser.parse_args()

    normalized_models = []
    for model_name in args.models:
        normalized_name = normalize_model_name(model_name)
        if normalized_name not in normalized_models:
            normalized_models.append(normalized_name)
    args.models = normalized_models

    set_global_seed(int(args.seed), deterministic=bool(args.deterministic))
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested via --device cuda, but CUDA is not available.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_cuda = device.type == "cuda"
    pin_memory = bool(args.pin_memory and use_cuda)
    persistent_workers = bool(args.persistent_workers and args.num_workers > 0)

    ds = SequenceDataset(
        args.data,
        seq_len=args.seq_len,
        channel_preset=args.channel_preset,
        drop_axes=args.drop_axes,
        drop_sensors=args.drop_sensors,
    )
    if len(ds) == 0:
        raise RuntimeError(f"No samples found in {args.data}")

    train_indices, val_indices, test_indices = collect_split_indices(
        ds,
        train_suffixes=args.train_suffixes,
        val_suffixes=args.val_suffixes,
        test_suffixes=args.test_suffixes,
    )

    if not train_indices or not test_indices:
        raise RuntimeError(
            f"Invalid split sizes: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
        )
    if not args.no_val_split and not val_indices:
        raise RuntimeError(
            f"Invalid split sizes: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
        )
    if args.no_val_split and args.eval_split == "val":
        raise RuntimeError("Cannot use --eval_split val together with --no_val_split.")

    train_ds = Subset(ds, train_indices)
    val_ds = Subset(ds, val_indices) if val_indices else None
    test_ds = Subset(ds, test_indices)

    worker_init_fn = partial(seed_worker, base_seed=int(args.seed))

    def build_loaders():
        loader_gen = torch.Generator().manual_seed(int(args.seed))
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            worker_init_fn=worker_init_fn,
            generator=loader_gen,
        )
        val_loader = None
        if val_ds is not None:
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                worker_init_fn=worker_init_fn,
                generator=loader_gen,
            )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            worker_init_fn=worker_init_fn,
            generator=loader_gen,
        )
        return train_loader, val_loader, test_loader

    num_classes = len(ds.class_to_idx)
    in_channels = getattr(ds, "num_channels", 1)
    idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    effective_phy_weight = args.phy_weight if args.task_mode == "multitask" else 0.0

    print(f"Torch version: {torch.__version__}")
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {args.data}, samples={len(ds)}")
    print(f"Classes: {num_classes}, channels: {in_channels}")
    print(
        "Channel selection:",
        {
            "preset": args.channel_preset,
            "drop_axes": args.drop_axes,
            "drop_sensors": args.drop_sensors,
            "selected": getattr(ds, "selected_channel_names", []),
        },
    )
    print(f"Split suffixes: train={args.train_suffixes}, val={args.val_suffixes}, test={args.test_suffixes}")
    print(
        f"Split counts: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}, total={len(ds)}"
    )
    print("Train per class:", summarize_split(ds, train_indices))
    print("Val per class:", summarize_split(ds, val_indices) if val_indices else {})
    print("Test per class:", summarize_split(ds, test_indices))
    if args.no_val_split:
        print("Validation mode: disabled, final epoch model will be evaluated on the test split.")
    print(f"Models to compare: {args.models}")
    print(
        "DataLoader config:",
        {
            "num_workers": args.num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
            "deterministic": bool(args.deterministic),
        },
    )

    rows = []
    out_path = Path(args.out_csv)
    for model_name in args.models:
        set_global_seed(int(args.seed), deterministic=bool(args.deterministic))
        use_validation = val_ds is not None and not args.no_val_split

        if is_sklearn_model(model_name):
            print(f"\n=== Fitting {model_name} ===")
            detail_dir = out_path.parent / f"{out_path.stem}_details" / model_name / args.eval_split
            result = fit_evaluate_sklearn_model(
                model_name=model_name,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                seed=args.seed,
                idx_to_class=idx_to_class,
                save_detailed_eval=args.save_detailed_eval,
                eval_split=args.eval_split,
                detail_dir=detail_dir,
            )
            test_metrics = result["test_metrics"]
            if use_validation:
                print(
                    f"[{model_name}] Validation done: "
                    f"val_loss={result['best_val_loss']:.4f}, val_acc={result['best_val_acc']:.4f} | "
                    f"test_loss={test_metrics['test_loss']:.4f}, test_acc={test_metrics['test_acc']:.4f}"
                )
            else:
                print(
                    f"[{model_name}] Classical baseline fitted without epoch selection: "
                    f"test_loss={test_metrics['test_loss']:.4f}, test_acc={test_metrics['test_acc']:.4f}"
                )

            if result["saved"] is not None:
                print(f"[{model_name}] Saved detailed evaluation artifacts to {detail_dir}")
                for key, saved_path in result["saved"].items():
                    print(f"  - {key}: {saved_path}")

            rows.append(
                {
                    "model": model_name,
                    "best_epoch": result["best_epoch"],
                    "best_val_loss": result["best_val_loss"] if use_validation else "",
                    "best_val_acc": result["best_val_acc"] if use_validation else "",
                    "test_loss": test_metrics["test_loss"],
                    "test_acc": test_metrics["test_acc"],
                    "test_ce": "",
                    "test_phy": "",
                }
            )
            continue

        train_loader, val_loader, test_loader = build_loaders()
        model = build_model(
            name=model_name,
            num_classes=num_classes,
            seq_len=args.seq_len,
            in_channels=in_channels,
            task_mode=args.task_mode,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_epoch = 0
        best_state = None

        print(f"\n=== Training {model_name} ===")
        for epoch in range(1, args.epochs + 1):
            train_loss, train_ce, train_phy = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                ce_loss,
                mse_loss,
                effective_phy_weight,
                phy_loss_type=args.phy_loss,
                calib_a=args.calib_a,
                calib_b=args.calib_b,
                task_mode=args.task_mode,
            )
            if use_validation:
                val_loss, val_ce, val_phy, val_acc = eval_epoch(
                    model,
                    val_loader,
                    device,
                    ce_loss,
                    mse_loss,
                    effective_phy_weight,
                    phy_loss_type=args.phy_loss,
                    calib_a=args.calib_a,
                    calib_b=args.calib_b,
                    task_mode=args.task_mode,
                )
                print(
                    f"[{model_name}] Epoch {epoch}: "
                    f"Train {train_loss:.4f} (CE {train_ce:.4f}, PHY {train_phy:.4f}) | "
                    f"Val {val_loss:.4f} (CE {val_ce:.4f}, PHY {val_phy:.4f}) Acc {val_acc:.4f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
            else:
                print(
                    f"[{model_name}] Epoch {epoch}: "
                    f"Train {train_loss:.4f} (CE {train_ce:.4f}, PHY {train_phy:.4f}) | "
                    "Val disabled, keeping current epoch as final model"
                )
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

        if best_state is None:
            raise RuntimeError(f"Failed to capture best state for {model_name}")

        model.load_state_dict(best_state)

        if args.checkpoint_dir is not None:
            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"{model_name}_model.pth"
            cpu_state = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }
            torch.save(
                {
                    "model_state": cpu_state,
                    "model_name": model_name,
                    "model_config": {
                        "num_classes": num_classes,
                        "seq_len": args.seq_len,
                        "in_channels": in_channels,
                        "task_mode": args.task_mode,
                    },
                    "class_to_idx": ds.class_to_idx,
                    "selected_channel_names": getattr(
                        ds, "selected_channel_names", []
                    ),
                    "best_epoch": best_epoch,
                    "best_val_loss": (
                        float(best_val_loss) if use_validation else None
                    ),
                    "best_val_acc": (
                        float(best_val_acc) if use_validation else None
                    ),
                    "train_args": vars(args),
                },
                checkpoint_path,
            )
            print(f"[{model_name}] Saved best checkpoint to {checkpoint_path}")

        test_loss, test_ce, test_phy, test_acc = eval_epoch(
            model,
            test_loader,
            device,
            ce_loss,
            mse_loss,
            effective_phy_weight,
            phy_loss_type=args.phy_loss,
            calib_a=args.calib_a,
            calib_b=args.calib_b,
            task_mode=args.task_mode,
        )
        if use_validation:
            print(
                f"[{model_name}] Best val @ epoch {best_epoch}: "
                f"val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f} | "
                f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
            )
        else:
            print(
                f"[{model_name}] Final epoch {best_epoch}: "
                f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
            )

        if args.save_detailed_eval:
            eval_loader = test_loader if args.eval_split == "test" else val_loader
            detail_dir = out_path.parent / f"{out_path.stem}_details" / model_name / args.eval_split
            saved = save_detailed_eval_artifacts(
                model=model,
                loader=eval_loader,
                detail_dir=detail_dir,
                idx_to_class=idx_to_class,
                calib_a=args.calib_a,
                calib_b=args.calib_b,
            )
            print(f"[{model_name}] Saved detailed evaluation artifacts to {detail_dir}")
            for key, saved_path in saved.items():
                print(f"  - {key}: {saved_path}")

        rows.append(
            {
                "model": model_name,
                "best_epoch": best_epoch,
                "best_val_loss": float(best_val_loss) if use_validation else "",
                "best_val_acc": float(best_val_acc) if use_validation else "",
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "test_ce": float(test_ce),
                "test_phy": float(test_phy),
            }
        )

    def sort_key(row):
        best_val_acc = row["best_val_acc"]
        if isinstance(best_val_acc, (int, float)) and math.isfinite(best_val_acc):
            best_val_rank = -best_val_acc
        else:
            best_val_rank = float("inf")
        return (-row["test_acc"], row["test_loss"], best_val_rank)

    rows = sorted(rows, key=sort_key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "best_epoch",
                "best_val_loss",
                "best_val_acc",
                "test_loss",
                "test_acc",
                "test_ce",
                "test_phy",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("\n=== Comparison Result (sorted by test_acc desc) ===")
    for i, row in enumerate(rows, 1):
        best_val_display = row["best_val_acc"] if row["best_val_acc"] != "" else "N/A"
        print(
            f"{i}. {row['model']:<15} "
            f"test_acc={row['test_acc']:.4f} "
            f"test_loss={row['test_loss']:.4f} "
            f"best_val_acc={best_val_display if isinstance(best_val_display, str) else f'{best_val_display:.4f}'} "
            f"epoch={row['best_epoch']}"
        )
    print("Saved csv to", out_path)


if __name__ == "__main__":
    main()
