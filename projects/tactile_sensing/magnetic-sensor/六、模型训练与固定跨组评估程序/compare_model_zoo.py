import argparse
import copy
import csv
import math
import sys
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESS_DIR = PROJECT_ROOT / "四、磁场信号预处理程序"
MODEL_DIR = PROJECT_ROOT / "五、多尺度门控交互网络及对比模型"
for search_path in (PREPROCESS_DIR, MODEL_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from preprocess import SequenceDataset
from train import SqueezeTaCo, train_epoch, eval_epoch, set_global_seed, seed_worker
from recurrent_model import BiLSTMClassifier, GRUClassifier, LSTMClassifier
from mamba_like_model import MambaLikeClassifier, MSGINetClassifier
from tcn_model import TCNClassifier
from transformer_model import TransformerTimeSeriesClassifier


MODEL_NAME_ALIASES = {
    'ms_mamba_like': 'msgi_net',
}

SKLEARN_MODEL_NAMES = {
    'knn',
    'random_forest',
    'linear_svm',
}

CLASSICAL_FEATURE_POOL = 64


def normalize_model_name(name):
    return MODEL_NAME_ALIASES.get(name, name)


def is_sklearn_model(name):
    return normalize_model_name(name) in SKLEARN_MODEL_NAMES


def build_sklearn_model(name, seed):
    name = normalize_model_name(name)
    if name == 'knn':
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5, weights='distance'),
        )
    if name == 'random_forest':
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=int(seed),
            n_jobs=-1,
        )
    if name == 'linear_svm':
        return make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=1.0,
                class_weight='balanced',
                dual=False,
                max_iter=5000,
                random_state=int(seed),
            ),
        )
    raise ValueError(f'Unknown sklearn model name: {name}')


def subset_to_classical_arrays(subset, feature_pool_len=CLASSICAL_FEATURE_POOL):
    features = []
    labels = []
    forces = []
    sample_ids = []
    for idx in range(len(subset)):
        sample = subset[idx]
        x = sample['x'].float()
        dx = torch.diff(x, dim=1, prepend=x[:, :1])
        pooled_x = F.adaptive_avg_pool1d(x.unsqueeze(0), output_size=feature_pool_len).squeeze(0)
        pooled_dx = F.adaptive_avg_pool1d(dx.unsqueeze(0), output_size=feature_pool_len).squeeze(0)
        feat = torch.cat([pooled_x.flatten(), pooled_dx.flatten()], dim=0)
        features.append(feat.cpu().numpy())
        labels.append(int(sample['cls'].item()))
        force = sample.get('force')
        forces.append(float(force.item()) if force is not None else 0.0)
        sample_ids.append(idx)
    return {
        'X': np.asarray(features, dtype=np.float32),
        'y': np.asarray(labels, dtype=np.int64),
        'force': np.asarray(forces, dtype=np.float32),
        'sample_id': sample_ids,
    }


def collect_sklearn_outputs(model, X, y, sample_ids=None, force_true=None):
    y_pred = model.predict(X)
    outputs = {
        'sample_id': sample_ids if sample_ids is not None else list(range(len(y))),
        'y_true': y.tolist() if isinstance(y, np.ndarray) else list(y),
        'y_pred': y_pred.tolist() if isinstance(y_pred, np.ndarray) else list(y_pred),
    }
    if force_true is not None:
        outputs['force_true'] = force_true.tolist() if isinstance(force_true, np.ndarray) else list(force_true)
    return outputs


def compute_classical_loss(model, X, y, y_pred=None):
    if y_pred is None:
        y_pred = model.predict(X)
    acc = float(np.mean(np.asarray(y_pred) == np.asarray(y)))
    if hasattr(model, 'predict_proba'):
        try:
            probs = model.predict_proba(X)
            labels = getattr(model, 'classes_', None)
            if labels is None and hasattr(model, 'steps') and len(model.steps) > 0:
                labels = getattr(model.steps[-1][1], 'classes_', None)
            if labels is None:
                labels = np.unique(y)
            return float(log_loss(y, probs, labels=labels)), acc
        except Exception:
            pass
    return float(1.0 - acc), acc


def collect_eval_outputs(model, loader, device, calib_a=1.0, calib_b=0.0):
    model.eval()
    y_true = []
    y_pred = []
    force_true = []
    force_pred = []
    sample_idx = 0
    sample_ids = []
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            cls = batch['cls'].to(device)
            force = batch.get('force')
            if force is not None:
                force = force.to(device)
            cls_out, reg_out = model(x)
            preds = cls_out.argmax(dim=1)
            y_true.extend(cls.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            batch_size = x.size(0)
            sample_ids.extend(range(sample_idx, sample_idx + batch_size))
            sample_idx += batch_size

            if force is not None and reg_out is not None:
                reg = calib_a * reg_out.squeeze(1) + calib_b
                force_true.extend(force.squeeze(1).cpu().tolist())
                force_pred.extend(reg.cpu().tolist())

    outputs = {
        'sample_id': sample_ids,
        'y_true': y_true,
        'y_pred': y_pred,
    }
    if len(force_true) == len(y_true) and len(force_pred) == len(y_true):
        outputs['force_true'] = force_true
        outputs['force_pred'] = force_pred
    return outputs


def save_detailed_eval_artifacts_from_outputs(outputs, detail_dir, idx_to_class):
    detail_dir.mkdir(parents=True, exist_ok=True)
    y_true = outputs['y_true']
    y_pred = outputs['y_pred']
    class_names = [idx_to_class[i] for i in sorted(idx_to_class)]

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(sorted(idx_to_class)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv = detail_dir / 'classification_report.csv'
    report_df.to_csv(report_csv, encoding='utf-8-sig')

    cm = confusion_matrix(y_true, y_pred, labels=list(sorted(idx_to_class)))
    cm_csv = detail_dir / 'confusion_matrix.csv'
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_csv, encoding='utf-8-sig')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    fig.savefig(detail_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    pred_df = pd.DataFrame(
        {
            'sample_id': outputs['sample_id'],
            'true_idx': y_true,
            'true_label': [idx_to_class[i] for i in y_true],
            'pred_idx': y_pred,
            'pred_label': [idx_to_class[i] for i in y_pred],
        }
    )

    force_predictions_csv = None
    force_summary_csv = None
    force_by_true_csv = None
    force_by_class_force_csv = None
    force_scatter_png = None
    if 'force_true' in outputs and 'force_pred' in outputs:
        pred_df['true_force'] = outputs['force_true']
        pred_df['pred_force'] = outputs['force_pred']
        pred_df['force_abs_error'] = (pred_df['pred_force'] - pred_df['true_force']).abs()
        pred_df['force_signed_error'] = pred_df['pred_force'] - pred_df['true_force']

        force_predictions_csv = detail_dir / 'force_predictions.csv'
        pred_df[
            [
                'sample_id',
                'true_label',
                'pred_label',
                'true_force',
                'pred_force',
                'force_abs_error',
                'force_signed_error',
            ]
        ].to_csv(force_predictions_csv, index=False, encoding='utf-8-sig')

        force_summary = {
            'mae': float(pred_df['force_abs_error'].mean()),
            'rmse': float(np.sqrt(np.mean((pred_df['pred_force'] - pred_df['true_force']) ** 2))),
            'mean_signed_error': float(pred_df['force_signed_error'].mean()),
            'max_abs_error': float(pred_df['force_abs_error'].max()),
            'n_samples': int(len(pred_df)),
            'true_force_min': float(pred_df['true_force'].min()),
            'true_force_max': float(pred_df['true_force'].max()),
            'pred_force_min': float(pred_df['pred_force'].min()),
            'pred_force_max': float(pred_df['pred_force'].max()),
        }
        ss_res = float(np.sum((pred_df['true_force'] - pred_df['pred_force']) ** 2))
        ss_tot = float(np.sum((pred_df['true_force'] - pred_df['true_force'].mean()) ** 2))
        force_summary['r2'] = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

        force_summary_csv = detail_dir / 'force_regression_summary.csv'
        pd.DataFrame([force_summary]).to_csv(force_summary_csv, index=False, encoding='utf-8-sig')

        force_by_true_csv = detail_dir / 'force_regression_by_true_force.csv'
        (
            pred_df.groupby('true_force', as_index=False)
            .agg(
                n_samples=('sample_id', 'count'),
                pred_force_mean=('pred_force', 'mean'),
                pred_force_std=('pred_force', 'std'),
                abs_error_mean=('force_abs_error', 'mean'),
                signed_error_mean=('force_signed_error', 'mean'),
            )
            .sort_values('true_force')
            .fillna(0.0)
            .to_csv(force_by_true_csv, index=False, encoding='utf-8-sig')
        )

        force_by_class_force_csv = detail_dir / 'force_regression_by_class_force.csv'
        (
            pred_df.groupby(['true_label', 'true_force'], as_index=False)
            .agg(
                n_samples=('sample_id', 'count'),
                pred_force_mean=('pred_force', 'mean'),
                pred_force_std=('pred_force', 'std'),
                abs_error_mean=('force_abs_error', 'mean'),
                signed_error_mean=('force_signed_error', 'mean'),
            )
            .sort_values(['true_label', 'true_force'])
            .fillna(0.0)
            .to_csv(force_by_class_force_csv, index=False, encoding='utf-8-sig')
        )

        force_scatter_png = detail_dir / 'force_scatter.png'
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(
            pred_df['true_force'],
            pred_df['pred_force'],
            s=12,
            alpha=0.55,
            edgecolors='none',
        )
        lo = float(min(pred_df['true_force'].min(), pred_df['pred_force'].min()))
        hi = float(max(pred_df['true_force'].max(), pred_df['pred_force'].max()))
        ax.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1.5, color='tab:red')
        ax.set_xlabel('True Force')
        ax.set_ylabel('Predicted Force')
        ax.set_title('Force Regression: True vs Predicted')
        plt.tight_layout()
        fig.savefig(force_scatter_png, dpi=300, bbox_inches='tight')
        plt.close(fig)

    predictions_csv = detail_dir / 'predictions.csv'
    pred_df.to_csv(predictions_csv, index=False, encoding='utf-8-sig')

    summary_csv = detail_dir / 'summary_metrics.csv'
    pd.DataFrame(
        [
            {
                'macro_f1': f1_score(y_true, y_pred, average='macro'),
                'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
                'n_samples': len(y_true),
            }
        ]
    ).to_csv(summary_csv, index=False, encoding='utf-8-sig')

    return {
        'classification_report_csv': report_csv,
        'confusion_matrix_csv': cm_csv,
        'confusion_matrix_png': detail_dir / 'confusion_matrix.png',
        'predictions_csv': predictions_csv,
        'summary_metrics_csv': summary_csv,
        'force_predictions_csv': force_predictions_csv,
        'force_summary_csv': force_summary_csv,
        'force_by_true_csv': force_by_true_csv,
        'force_by_class_force_csv': force_by_class_force_csv,
        'force_scatter_png': force_scatter_png,
    }


def save_detailed_eval_artifacts(model, loader, detail_dir, idx_to_class, calib_a=1.0, calib_b=0.0):
    outputs = collect_eval_outputs(
        model,
        loader,
        next(model.parameters()).device,
        calib_a=calib_a,
        calib_b=calib_b,
    )
    return save_detailed_eval_artifacts_from_outputs(
        outputs=outputs,
        detail_dir=detail_dir,
        idx_to_class=idx_to_class,
    )


def split_lengths(n, val_frac, test_frac):
    train_frac = 1.0 - val_frac - test_frac
    ratios = [train_frac, val_frac, test_frac]
    ideal = [r * n for r in ratios]
    counts = [int(math.floor(v)) for v in ideal]
    remain = n - sum(counts)
    frac_order = sorted(
        [(ideal[i] - counts[i], i) for i in range(3)],
        key=lambda x: x[0],
        reverse=True,
    )
    for j in range(remain):
        counts[frac_order[j % 3][1]] += 1
    return counts


def fit_evaluate_sklearn_model(
    model_name,
    train_ds,
    val_ds,
    test_ds,
    seed,
    idx_to_class,
    save_detailed_eval=False,
    eval_split='val',
    detail_dir=None,
):
    train_arr = subset_to_classical_arrays(train_ds)
    test_arr = subset_to_classical_arrays(test_ds) if test_ds is not None else None
    val_arr = subset_to_classical_arrays(val_ds) if val_ds is not None else None

    model = build_sklearn_model(model_name, seed=seed)
    model.fit(train_arr['X'], train_arr['y'])

    val_outputs = None
    val_loss = float('nan')
    val_acc = float('nan')
    if val_arr is not None:
        val_outputs = collect_sklearn_outputs(
            model,
            val_arr['X'],
            val_arr['y'],
            sample_ids=val_arr['sample_id'],
            force_true=val_arr['force'],
        )
        val_loss, val_acc = compute_classical_loss(
            model,
            val_arr['X'],
            val_arr['y'],
            y_pred=np.asarray(val_outputs['y_pred']),
        )

    test_metrics = None
    test_outputs = None
    if test_arr is not None:
        test_outputs = collect_sklearn_outputs(
            model,
            test_arr['X'],
            test_arr['y'],
            sample_ids=test_arr['sample_id'],
            force_true=test_arr['force'],
        )
        test_loss, test_acc = compute_classical_loss(
            model,
            test_arr['X'],
            test_arr['y'],
            y_pred=np.asarray(test_outputs['y_pred']),
        )
        test_metrics = {
            'test_loss': float(test_loss),
            'test_acc': float(test_acc),
        }

    saved = None
    if save_detailed_eval:
        if detail_dir is None:
            raise ValueError('detail_dir is required when save_detailed_eval=True for sklearn models')
        if eval_split == 'test':
            if test_outputs is None:
                raise RuntimeError('Detailed test evaluation was requested, but no test split exists.')
            outputs = test_outputs
        else:
            if val_outputs is None:
                raise RuntimeError('Detailed val evaluation was requested, but no validation split exists.')
            outputs = val_outputs
        saved = save_detailed_eval_artifacts_from_outputs(
            outputs=outputs,
            detail_dir=detail_dir,
            idx_to_class=idx_to_class,
        )

    return {
        'model': model,
        'best_epoch': 0,
        'best_val_loss': float(val_loss),
        'best_val_acc': float(val_acc),
        'test_metrics': test_metrics,
        'saved': saved,
    }


def build_model(name, num_classes, seq_len, in_channels, task_mode):
    name = normalize_model_name(name)
    if name == 'baseline_1dcnn':
        return SqueezeTaCo(
            num_classes=num_classes,
            seq_len=seq_len,
            in_channels=in_channels,
            use_attention=True,
            task_mode=task_mode,
        )
    if name == 'tcn':
        return TCNClassifier(
            num_classes=num_classes,
            seq_len=seq_len,
            in_channels=in_channels,
            task_mode=task_mode,
        )
    if name == 'transformer':
        return TransformerTimeSeriesClassifier(
            num_classes=num_classes,
            seq_len=seq_len,
            in_channels=in_channels,
            task_mode=task_mode,
        )
    if name == 'lstm':
        return LSTMClassifier(
            num_classes=num_classes,
            seq_len=seq_len,
            in_channels=in_channels,
            task_mode=task_mode,
        )
    if name == 'gru':
        return GRUClassifier(
            num_classes=num_classes,
            seq_len=seq_len,
            in_channels=in_channels,
            task_mode=task_mode,
        )
    if name == 'bilstm':
        return BiLSTMClassifier(
            num_classes=num_classes,
            seq_len=seq_len,
            in_channels=in_channels,
            task_mode=task_mode,
        )
    if name == 'mamba_like':
        return MambaLikeClassifier(
            num_classes=num_classes,
            seq_len=seq_len,
            in_channels=in_channels,
            task_mode=task_mode,
        )
    if name == 'msgi_net':
        return MSGINetClassifier(
            num_classes=num_classes,
            seq_len=seq_len,
            in_channels=in_channels,
            task_mode=task_mode,
        )
    if name == 'msgi_single_scale':
        return MSGINetClassifier(
            num_classes=num_classes,
            seq_len=seq_len,
            in_channels=in_channels,
            task_mode=task_mode,
            kernel_sizes=(5,),
        )
    if name == 'msgi_k3_only':
        return MSGINetClassifier(
            num_classes=num_classes,
            seq_len=seq_len,
            in_channels=in_channels,
            task_mode=task_mode,
            kernel_sizes=(3,),
        )
    if name == 'msgi_k7_only':
        return MSGINetClassifier(
            num_classes=num_classes,
            seq_len=seq_len,
            in_channels=in_channels,
            task_mode=task_mode,
            kernel_sizes=(7,),
        )
    if name == 'msgi_no_gate':
        return MSGINetClassifier(
            num_classes=num_classes,
            seq_len=seq_len,
            in_channels=in_channels,
            task_mode=task_mode,
            use_gating=False,
        )
    if name == 'msgi_no_residual':
        return MSGINetClassifier(
            num_classes=num_classes,
            seq_len=seq_len,
            in_channels=in_channels,
            task_mode=task_mode,
            use_residual=False,
        )
    if name == 'msgi_no_bn':
        return MSGINetClassifier(
            num_classes=num_classes,
            seq_len=seq_len,
            in_channels=in_channels,
            task_mode=task_mode,
            use_norm=False,
        )
    if name == 'msgi_no_resnorm':
        return MSGINetClassifier(
            num_classes=num_classes,
            seq_len=seq_len,
            in_channels=in_channels,
            task_mode=task_mode,
            use_residual_norm=False,
        )
    if name == 'msgi_2layers':
        return MSGINetClassifier(
            num_classes=num_classes,
            seq_len=seq_len,
            in_channels=in_channels,
            task_mode=task_mode,
            num_layers=2,
        )
    if name == 'msgi_6layers':
        return MSGINetClassifier(
            num_classes=num_classes,
            seq_len=seq_len,
            in_channels=in_channels,
            task_mode=task_mode,
            num_layers=6,
        )
    if name == 'msgi_no_conv_stem':
        return MSGINetClassifier(
            num_classes=num_classes,
            seq_len=seq_len,
            in_channels=in_channels,
            task_mode=task_mode,
            stem_type='pointwise',
        )
    if name == 'msgi_max_pool':
        return MSGINetClassifier(
            num_classes=num_classes,
            seq_len=seq_len,
            in_channels=in_channels,
            task_mode=task_mode,
            pool_type='max',
        )
    if name == 'msgi_last_pool':
        return MSGINetClassifier(
            num_classes=num_classes,
            seq_len=seq_len,
            in_channels=in_channels,
            task_mode=task_mode,
            pool_type='last',
        )
    raise ValueError(f'Unknown model name: {name}')


def main():
    parser = argparse.ArgumentParser(description='Compare baseline 1DCNN with model_zoo models on same split.')
    parser.add_argument('--data', type=str, default='data_new_processed')
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--test_frac', type=float, default=0.0)
    parser.add_argument('--task_mode', type=str, default='multitask', choices=['multitask', 'cls_only'])
    parser.add_argument('--phy_weight', type=float, default=1.0)
    parser.add_argument('--phy_loss', type=str, default='mae', choices=['mse', 'mae', 'huber', 'relmse'])
    parser.add_argument('--calib_a', type=float, default=1.0)
    parser.add_argument('--calib_b', type=float, default=0.0)
    parser.add_argument('--channel_preset', type=str, default='all25', choices=['all25', 'magnetic16', 'xyz12', 'total4'])
    parser.add_argument('--drop_axes', nargs='*', default=[], choices=['X', 'Y', 'Z', 'total'])
    parser.add_argument('--drop_sensors', nargs='*', default=[], choices=['S1', 'S2', 'S3', 'S4'])
    parser.add_argument('--save_detailed_eval', action='store_true')
    parser.add_argument('--eval_split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--persistent_workers', action='store_true')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--deterministic', dest='deterministic', action='store_true', help='enable deterministic training')
    parser.add_argument('--no-deterministic', dest='deterministic', action='store_false', help='disable deterministic training to improve speed')
    parser.add_argument(
        '--models',
        nargs='*',
        default=['baseline_1dcnn', 'tcn', 'transformer', 'mamba_like', 'msgi_net'],
        choices=[
            'baseline_1dcnn',
            'tcn',
            'transformer',
            'lstm',
            'gru',
            'bilstm',
            'mamba_like',
            'msgi_net',
            'msgi_single_scale',
            'msgi_k3_only',
            'msgi_k7_only',
            'msgi_no_gate',
            'msgi_no_residual',
            'msgi_no_bn',
            'msgi_no_resnorm',
            'msgi_2layers',
            'msgi_6layers',
            'msgi_no_conv_stem',
            'msgi_max_pool',
            'msgi_last_pool',
            'ms_mamba_like',
            'knn',
            'random_forest',
            'linear_svm',
        ],
    )
    parser.add_argument('--out_csv', type=str, default='results/model_compare.csv')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=None,
        help='optional directory for saving each neural model best checkpoint',
    )
    parser.set_defaults(deterministic=True)
    args = parser.parse_args()

    if not (0.0 <= args.val_frac <= 1.0):
        raise ValueError(f'val_frac must be in [0,1], got {args.val_frac}')
    if not (0.0 <= args.test_frac <= 1.0):
        raise ValueError(f'test_frac must be in [0,1], got {args.test_frac}')
    if args.val_frac + args.test_frac > 1.0:
        raise ValueError(f'val_frac + test_frac must be <= 1, got {args.val_frac + args.test_frac}')

    normalized_models = []
    for model_name in args.models:
        normalized_name = normalize_model_name(model_name)
        if normalized_name not in normalized_models:
            normalized_models.append(normalized_name)
    args.models = normalized_models

    set_global_seed(int(args.seed), deterministic=bool(args.deterministic))
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested via --device cuda, but this Python environment does not have CUDA available.')
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = device.type == 'cuda'
    pin_memory = bool(args.pin_memory and use_cuda)
    persistent_workers = bool(args.persistent_workers and args.num_workers > 0)

    ds = SequenceDataset(
        args.data,
        seq_len=args.seq_len,
        channel_preset=args.channel_preset,
        drop_axes=args.drop_axes,
        drop_sensors=args.drop_sensors,
    )
    n = len(ds)
    if n == 0:
        raise RuntimeError(f'No samples found in {args.data}')

    n_train, n_val, n_test = split_lengths(n, args.val_frac, args.test_frac)
    if n_train <= 0 or n_val <= 0:
        raise RuntimeError(f'Invalid split sizes train={n_train}, val={n_val}, test={n_test}')

    split_gen = torch.Generator().manual_seed(int(args.seed))
    test_ds = None
    if n_test > 0:
        train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [n_train, n_val, n_test], generator=split_gen)
    else:
        train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=split_gen)

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
        test_loader = None
        if test_ds is not None:
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
    in_channels = getattr(ds, 'num_channels', 1)
    idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    effective_phy_weight = args.phy_weight if args.task_mode == 'multitask' else 0.0

    print(f'Torch version: {torch.__version__}')
    print(f'Device: {device}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if use_cuda:
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('Note: current environment is not using CUDA, so transformer-like models may train much more slowly.')
    print(f'Dataset: {args.data}, samples={n}, train={n_train}, val={n_val}, test={n_test}')
    print(f'Classes: {num_classes}, channels: {in_channels}')
    print(
        'Channel selection:',
        {
            'preset': args.channel_preset,
            'drop_axes': args.drop_axes,
            'drop_sensors': args.drop_sensors,
            'selected': getattr(ds, 'selected_channel_names', []),
        }
    )
    print(f'Models to compare: {args.models}')
    print(
        'DataLoader config:',
        {
            'num_workers': args.num_workers,
            'pin_memory': pin_memory,
            'persistent_workers': persistent_workers,
            'deterministic': bool(args.deterministic),
        }
    )

    rows = []
    has_test_split = test_ds is not None
    out_path = Path(args.out_csv)

    for model_name in args.models:
        set_global_seed(int(args.seed), deterministic=bool(args.deterministic))

        if is_sklearn_model(model_name):
            print(f'\n=== Fitting {model_name} ===')
            detail_dir = out_path.parent / f'{out_path.stem}_details' / model_name / args.eval_split
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

            test_metrics = result['test_metrics']
            if test_metrics is not None:
                print(
                    f'[{model_name}] Validation done: '
                    f'val_loss={result["best_val_loss"]:.4f}, val_acc={result["best_val_acc"]:.4f} | '
                    f'test_loss={test_metrics["test_loss"]:.4f}, test_acc={test_metrics["test_acc"]:.4f}'
                )
            else:
                print(
                    f'[{model_name}] Validation done: '
                    f'val_loss={result["best_val_loss"]:.4f}, val_acc={result["best_val_acc"]:.4f}'
                )

            rows.append(
                {
                    'model': model_name,
                    'best_epoch': result['best_epoch'],
                    'best_val_loss': result['best_val_loss'],
                    'best_val_acc': result['best_val_acc'],
                    'test_loss': test_metrics['test_loss'] if test_metrics is not None else '',
                    'test_acc': test_metrics['test_acc'] if test_metrics is not None else '',
                    'test_ce': '',
                    'test_phy': '',
                }
            )

            if result['saved'] is not None:
                print(f'[{model_name}] Saved detailed evaluation artifacts to {detail_dir}')
                for key, saved_path in result['saved'].items():
                    print(f'  - {key}: {saved_path}')
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

        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_epoch = 0
        best_state = None

        print(f'\n=== Training {model_name} ===')
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
                f'[{model_name}] Epoch {epoch}: '
                f'Train {train_loss:.4f} (CE {train_ce:.4f}, PHY {train_phy:.4f}) | '
                f'Val {val_loss:.4f} (CE {val_ce:.4f}, PHY {val_phy:.4f}) Acc {val_acc:.4f}'
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

        if best_state is not None:
            model.load_state_dict(best_state)

        if args.checkpoint_dir is not None:
            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f'{model_name}_model.pth'
            cpu_state = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }
            torch.save(
                {
                    'model_state': cpu_state,
                    'model_name': model_name,
                    'model_config': {
                        'num_classes': num_classes,
                        'seq_len': args.seq_len,
                        'in_channels': in_channels,
                        'task_mode': args.task_mode,
                    },
                    'class_to_idx': ds.class_to_idx,
                    'selected_channel_names': getattr(
                        ds, 'selected_channel_names', []
                    ),
                    'best_epoch': best_epoch,
                    'best_val_loss': float(best_val_loss),
                    'best_val_acc': float(best_val_acc),
                    'train_args': vars(args),
                },
                checkpoint_path,
            )
            print(f'[{model_name}] Saved best checkpoint to {checkpoint_path}')

        test_loss = ''
        test_ce = ''
        test_phy = ''
        test_acc = ''
        if test_loader is not None:
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
            print(
                f'[{model_name}] Best val @ epoch {best_epoch}: '
                f'val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f} | '
                f'test_loss={test_loss:.4f}, test_acc={test_acc:.4f}'
            )
        else:
            print(
                f'[{model_name}] Best val @ epoch {best_epoch}: '
                f'val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}'
            )

        rows.append(
            {
                'model': model_name,
                'best_epoch': best_epoch,
                'best_val_loss': float(best_val_loss),
                'best_val_acc': float(best_val_acc),
                'test_loss': float(test_loss) if test_loader is not None else '',
                'test_acc': float(test_acc) if test_loader is not None else '',
                'test_ce': float(test_ce) if test_loader is not None else '',
                'test_phy': float(test_phy) if test_loader is not None else '',
            }
        )

        if args.save_detailed_eval:
            if args.eval_split == 'test' and test_loader is None:
                raise RuntimeError('Detailed test evaluation was requested, but test_frac=0 so no test split exists.')
            eval_loader = test_loader if args.eval_split == 'test' else val_loader
            detail_dir = out_path.parent / f'{out_path.stem}_details' / model_name / args.eval_split
            saved = save_detailed_eval_artifacts(
                model=model,
                loader=eval_loader,
                detail_dir=detail_dir,
                idx_to_class=idx_to_class,
                calib_a=args.calib_a,
                calib_b=args.calib_b,
            )
            print(f'[{model_name}] Saved detailed evaluation artifacts to {detail_dir}')
            for key, saved_path in saved.items():
                print(f'  - {key}: {saved_path}')

    if has_test_split:
        rows = sorted(rows, key=lambda r: (-r['test_acc'], r['test_loss'], -r['best_val_acc']))
    else:
        rows = sorted(rows, key=lambda r: (-r['best_val_acc'], r['best_val_loss']))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['model', 'best_epoch', 'best_val_loss', 'best_val_acc', 'test_loss', 'test_acc', 'test_ce', 'test_phy'],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    if has_test_split:
        print('\n=== Comparison Result (sorted by test_acc desc) ===')
    else:
        print('\n=== Comparison Result (sorted by best_val_acc desc) ===')
    for i, r in enumerate(rows, 1):
        if has_test_split:
            print(
                f"{i}. {r['model']:<15} "
                f"test_acc={r['test_acc']:.4f} "
                f"test_loss={r['test_loss']:.4f} "
                f"best_val_acc={r['best_val_acc']:.4f} "
                f"epoch={r['best_epoch']}"
            )
        else:
            print(
                f"{i}. {r['model']:<15} "
                f"acc={r['best_val_acc']:.4f} "
                f"val_loss={r['best_val_loss']:.4f} "
                f"epoch={r['best_epoch']}"
            )
    print('Saved csv to', out_path)


if __name__ == '__main__':
    main()
