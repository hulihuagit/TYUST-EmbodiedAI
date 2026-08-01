import argparse
import os
import math
import random
import re
import sys
from functools import partial
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESS_DIR = PROJECT_ROOT / "四、磁场信号预处理程序"
if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))

from preprocess import SequenceDataset

# --- Merged model classes (from models.py) ---
import torch.nn as nn


class ChannelAttention1D(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(in_planes, in_planes // ratio)
        self.fc2 = nn.Linear(in_planes // ratio, in_planes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        maxv = self.max_pool(x).view(b, c)
        avg_out = self.fc2(self.relu(self.fc1(avg)))
        max_out = self.fc2(self.relu(self.fc1(maxv)))
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1)
        return x * out


class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(concat)
        out = self.sigmoid(out)
        return x * out


class CBAM1D(nn.Module):
    def __init__(self, channels, ratio=8, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention1D(channels, ratio)
        self.sa = SpatialAttention1D(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class Backbone1D(nn.Module):
    def __init__(self, in_channels=1, channels=(32, 64, 128)):
        super().__init__()
        layers = []
        prev = in_channels
        for ch in channels:
            layers += [
                nn.Conv1d(prev, ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2)
            ]
            prev = ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SqueezeTaCo(nn.Module):
    def __init__(self, num_classes, seq_len=1024, in_channels=1, backbone_channels=(32, 64, 128), use_attention=True, task_mode='multitask'):
        super().__init__()
        self.backbone = Backbone1D(in_channels=in_channels, channels=backbone_channels)
        self.use_attention = bool(use_attention)
        self.task_mode = task_mode
        self.att = CBAM1D(backbone_channels[-1]) if self.use_attention else nn.Identity()
        self.seq_len = seq_len
        downscale = 2 ** len(backbone_channels)
        reduced_len = max(1, seq_len // downscale)
        feat_dim = backbone_channels[-1] * reduced_len
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(backbone_channels[-1], 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        if self.task_mode == 'multitask':
            self.reg_head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(backbone_channels[-1], 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        else:
            self.reg_head = None

    def forward(self, x):
        f = self.backbone(x)
        f = self.att(f)
        cls_out = self.cls_head(f)
        reg_out = self.reg_head(f) if self.reg_head is not None else None
        return cls_out, reg_out


def phy_loss_fn(pred, target, loss_type='mse'):
    """支持多种物理监督损失：mse, mae, huber, relmse(相对MSE)。"""
    if loss_type == 'mse':
        return F.mse_loss(pred, target)
    if loss_type == 'mae':
        return F.l1_loss(pred, target)
    if loss_type == 'huber':
        return F.smooth_l1_loss(pred, target)
    if loss_type == 'relmse':
        eps = 1e-6
        rel = (pred - target) / (torch.abs(target) + eps)
        return torch.mean(rel ** 2)
    return F.mse_loss(pred, target)


def train_epoch(model, loader, optimizer, device, ce_loss, mse_loss, phy_weight, phy_loss_type='mse', calib_a=1.0, calib_b=0.0, task_mode='multitask'):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_phy = 0.0
    for batch in loader:
        x = batch['x'].to(device)
        cls = batch['cls'].to(device)
        force = batch['force'].to(device)
        optimizer.zero_grad()
        cls_out, reg_out = model(x)
        loss_cls = ce_loss(cls_out, cls)
        if task_mode == 'multitask' and reg_out is not None:
            reg = calib_a * reg_out.squeeze(1) + calib_b
            loss_phy = phy_loss_fn(reg, force.squeeze(1), loss_type=phy_loss_type)
            loss = loss_cls + phy_weight * loss_phy
            phy_value = loss_phy.item()
        else:
            loss = loss_cls
            phy_value = 0.0
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_ce += loss_cls.item() * x.size(0)
        total_phy += phy_value * x.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_ce / n, total_phy / n


def eval_epoch(model, loader, device, ce_loss, mse_loss, phy_weight, phy_loss_type='mse', calib_a=1.0, calib_b=0.0, task_mode='multitask'):
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_phy = 0.0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            cls = batch['cls'].to(device)
            force = batch['force'].to(device)
            cls_out, reg_out = model(x)
            loss_cls = ce_loss(cls_out, cls)
            if task_mode == 'multitask' and reg_out is not None:
                reg = calib_a * reg_out.squeeze(1) + calib_b
                loss_phy = phy_loss_fn(reg, force.squeeze(1), loss_type=phy_loss_type)
                loss = loss_cls + phy_weight * loss_phy
                phy_value = loss_phy.item()
            else:
                loss = loss_cls
                phy_value = 0.0
            total_loss += loss.item() * x.size(0)
            total_ce += loss_cls.item() * x.size(0)
            total_phy += phy_value * x.size(0)
            preds = cls_out.argmax(dim=1)
            correct += (preds == cls).sum().item()
    n = len(loader.dataset)
    acc = correct / n if n > 0 else 0.0
    return total_loss / n, total_ce / n, total_phy / n, acc


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


SUFFIX_RE = re.compile(r'_(\d{2})\.(?:npy|csv)$', re.IGNORECASE)


def parse_suffix_from_path(path: Path):
    match = SUFFIX_RE.search(path.name)
    return match.group(1) if match else None


def collect_suffix_split_indices(dataset, train_suffixes, val_suffixes, test_suffixes):
    train_suffixes = set(train_suffixes or [])
    val_suffixes = set(val_suffixes or [])
    test_suffixes = set(test_suffixes or [])

    overlap = (train_suffixes & val_suffixes) | (train_suffixes & test_suffixes) | (val_suffixes & test_suffixes)
    if overlap:
        raise ValueError(f'split suffixes overlap: {sorted(overlap)}')

    train_indices = []
    val_indices = []
    test_indices = []

    for idx, (path, _, _) in enumerate(dataset.items):
        suffix = parse_suffix_from_path(path)
        if suffix in train_suffixes:
            train_indices.append(idx)
        elif suffix in val_suffixes:
            val_indices.append(idx)
        elif suffix in test_suffixes:
            test_indices.append(idx)

    return train_indices, val_indices, test_indices


def summarize_subset_by_class(dataset, indices):
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    counts = {}
    for idx in indices:
        _, cls_idx, _ = dataset.items[idx]
        class_name = idx_to_class.get(cls_idx, str(cls_idx))
        counts[class_name] = counts.get(class_name, 0) + 1
    return dict(sorted(counts.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data_new')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--phy_weight', type=float, default=1.0)
    parser.add_argument('--phy_loss', type=str, default='mse', choices=['mse','mae','huber','relmse'], help='物理损失类型')
    parser.add_argument('--disable_attention', action='store_true', help='消融：关闭CBAM注意力模块')
    parser.add_argument('--task_mode', type=str, default='multitask', choices=['multitask', 'cls_only'], help='任务头模式：multitask=分类+回归, cls_only=仅分类头')
    parser.add_argument('--calib_a', type=float, default=1.0, help='预测校准线性比例 a')
    parser.add_argument('--calib_b', type=float, default=0.0, help='预测校准线性偏置 b')
    parser.add_argument('--save', type=str, default='models/squeezetaco.pth')
    parser.add_argument('--seed', type=int, default=42, help='random seed for splits')
    parser.add_argument('--deterministic', dest='deterministic', action='store_true', help='enable deterministic training for reproducibility')
    parser.add_argument('--no-deterministic', dest='deterministic', action='store_false', help='disable deterministic training to improve speed')
    parser.add_argument('--val_frac', type=float, default=0.2, help='fraction of data for validation')
    parser.add_argument('--test_frac', type=float, default=0.0, help='fraction of data for test (optional)')
    parser.add_argument('--train_suffixes', nargs='*', default=None, help='fixed train split by filename suffix, e.g. 01 02 03')
    parser.add_argument('--val_suffixes', nargs='*', default=None, help='fixed validation split by filename suffix')
    parser.add_argument('--test_suffixes', nargs='*', default=None, help='fixed test split by filename suffix')
    parser.add_argument('--num_workers', type=int, default=2, help='number of DataLoader workers')
    parser.add_argument('--pin_memory', action='store_true', help='pin DataLoader memory when using CUDA')
    parser.add_argument('--persistent_workers', action='store_true', help='keep DataLoader workers alive between epochs')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='execution device')
    parser.set_defaults(deterministic=True)
    args = parser.parse_args()

    set_global_seed(int(args.seed), deterministic=bool(args.deterministic))

    # 参数校验：val/test 比例必须在 [0,1]，且二者和不超过 1
    if not (0.0 <= args.val_frac <= 1.0):
        raise ValueError(f'val_frac must be in [0,1], got {args.val_frac}')
    if not (0.0 <= args.test_frac <= 1.0):
        raise ValueError(f'test_frac must be in [0,1], got {args.test_frac}')
    if args.val_frac + args.test_frac > 1.0:
        raise ValueError(f'val_frac + test_frac must be <= 1, got {args.val_frac + args.test_frac}')

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

    print(f'Torch version: {torch.__version__}')
    print(f'Device: {device}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if use_cuda:
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    elif os.name == 'nt':
        print('Note: current environment is not using CUDA. If you have an NVIDIA GPU, check whether this venv has a CPU-only PyTorch build.')
    print(
        'DataLoader config:',
        {
            'num_workers': args.num_workers,
            'pin_memory': pin_memory,
            'persistent_workers': persistent_workers,
            'deterministic': bool(args.deterministic),
        }
    )

    ds = SequenceDataset(args.data, seq_len=args.seq_len)
    # deterministic split into train/val/(test) with saved file lists
    n = len(ds)
    if n == 0:
        print('No data found in', args.data)
        return

    # 严格按比例分配（整数样本不可避免有舍入）：
    # 1) 先对 train/val/test 的期望数量取 floor
    # 2) 剩余样本按小数部分从大到小分配
    # 3) 保证 n_train + n_val + n_test == n
    fixed_split_requested = any(
        value is not None and len(value) > 0
        for value in (args.train_suffixes, args.val_suffixes, args.test_suffixes)
    )

    if fixed_split_requested:
        if not args.train_suffixes or not args.val_suffixes:
            raise ValueError('Fixed suffix split requires both --train_suffixes and --val_suffixes')
        train_indices, val_indices, test_indices = collect_suffix_split_indices(
            ds,
            train_suffixes=args.train_suffixes,
            val_suffixes=args.val_suffixes,
            test_suffixes=args.test_suffixes,
        )
        n_train = len(train_indices)
        n_val = len(val_indices)
        n_test = len(test_indices)
        if n_train <= 0 or n_val <= 0:
            raise ValueError(f'Invalid fixed split sizes: train={n_train}, val={n_val}, test={n_test}')
        train_ds = Subset(ds, train_indices)
        val_ds = Subset(ds, val_indices)
        test_ds = Subset(ds, test_indices) if n_test > 0 else None
        print('Split mode: fixed suffix')
        print(f'Split suffixes: train={args.train_suffixes}, val={args.val_suffixes}, test={args.test_suffixes or []}')
        print(f'Split counts exact : train={n_train}, val={n_val}, test={n_test}, total={n}')
        print(f'Split ratios actual: train={n_train / n:.4f}, val={n_val / n:.4f}, test={n_test / n:.4f}')
        print('Train per class:', summarize_subset_by_class(ds, train_indices))
        print('Val per class:', summarize_subset_by_class(ds, val_indices))
        if n_test > 0:
            print('Test per class:', summarize_subset_by_class(ds, test_indices))
        if args.val_frac != 0.2 or args.test_frac != 0.0:
            print('Note: --val_frac/--test_frac are ignored when fixed suffix split is enabled.')
    else:
        train_frac = 1.0 - args.val_frac - args.test_frac
        ratios = [train_frac, args.val_frac, args.test_frac]
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

        n_train, n_val, n_test = counts
        if n_train <= 0 or n_val < 0 or n_test < 0:
            raise ValueError('Invalid split fractions resulting in non-positive sizes')
        if n_train + n_val + n_test != n:
            raise RuntimeError('Split count mismatch')

        print('Split mode: random sample split')
        print(f'Split ratios target: train={train_frac:.4f}, val={args.val_frac:.4f}, test={args.test_frac:.4f}')
        print(f'Split counts exact : train={n_train}, val={n_val}, test={n_test}, total={n}')
        print(f'Split ratios actual: train={n_train / n:.4f}, val={n_val / n:.4f}, test={n_test / n:.4f}')
        gen = torch.Generator()
        gen.manual_seed(int(args.seed))
        if n_test > 0:
            train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [n_train, n_val, n_test], generator=gen)
        else:
            train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=gen)
            test_ds = None
    # save file lists to splits/ for reproducible evaluation
    try:
        splits_dir = Path(args.save).parent / 'splits'
        splits_dir.mkdir(parents=True, exist_ok=True)
        def _write_list(name, subset):
            path = splits_dir / name
            inds = getattr(subset, 'indices', None)
            if inds is None:
                inds = []
                for i in range(len(subset)):
                    pass
            with open(path, 'w', encoding='utf-8') as f:
                for i in inds:
                    f.write(str(ds.files[i]) + '\n')
            return path
        train_list = _write_list('train_files.txt', train_ds)
        val_list = _write_list('val_files.txt', val_ds)
        print('Saved split lists to', splits_dir)
        if n_test > 0:
            test_list = _write_list('test_files.txt', test_ds)
    except Exception:
        pass
    loader_gen = torch.Generator()
    loader_gen.manual_seed(int(args.seed))
    worker_init_fn = partial(seed_worker, base_seed=int(args.seed))

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

    num_classes = len(ds.class_to_idx)
    in_channels = getattr(ds, 'num_channels', 1)
    print('Dataset channels:', in_channels)
    print('Found classes:', ds.class_to_idx)
    print('Ablation config:', {'use_attention': not args.disable_attention, 'task_mode': args.task_mode})

    model = SqueezeTaCo(
        num_classes=num_classes,
        seq_len=args.seq_len,
        in_channels=in_channels,
        use_attention=not args.disable_attention,
        task_mode=args.task_mode,
    )
    model.to(device)

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    effective_phy_weight = args.phy_weight if args.task_mode == 'multitask' else 0.0

    best_val = float('inf')

    def _save_checkpoint_with_fallback(state: dict, save_path_str: str) -> str:
        target = Path(save_path_str).expanduser()
        if not target.is_absolute():
            target = Path(__file__).resolve().parent / target

        def _save_via_file_handle(path_obj: Path) -> str:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(path_obj, 'wb') as f:
                torch.save(state, f)
            return str(path_obj)

        try:
            return _save_via_file_handle(target)
        except Exception as e:
            fallback_dir = Path(__file__).resolve().parent / 'models'
            fallback_path = fallback_dir / target.name
            print(f'Warning: save to {target} failed: {e}')
            print(f'Fallback saving to {fallback_path}')
            return _save_via_file_handle(fallback_path)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_ce, train_phy = train_epoch(model, train_loader, optimizer, device, ce_loss, mse_loss, effective_phy_weight, phy_loss_type=args.phy_loss, calib_a=args.calib_a, calib_b=args.calib_b, task_mode=args.task_mode)
        val_loss, val_ce, val_phy, val_acc = eval_epoch(model, val_loader, device, ce_loss, mse_loss, effective_phy_weight, phy_loss_type=args.phy_loss, calib_a=args.calib_a, calib_b=args.calib_b, task_mode=args.task_mode)
        print(f'Epoch {epoch}: Train loss {train_loss:.4f} (CE {train_ce:.4f}, PHY {train_phy:.4f}) | Val loss {val_loss:.4f} (CE {val_ce:.4f}, PHY {val_phy:.4f}) Acc {val_acc:.3f}')
        if val_loss < best_val:
            best_val = val_loss
            saved_to = _save_checkpoint_with_fallback(
                {
                    'model_state': model.state_dict(),
                    'class_to_idx': ds.class_to_idx,
                    'train_args': vars(args),
                    'model_config': {
                        'seq_len': args.seq_len,
                        'in_channels': in_channels,
                        'use_attention': (not args.disable_attention),
                        'task_mode': args.task_mode,
                    }
                },
                args.save,
            )
            print('Saved best model to', saved_to)


if __name__ == '__main__':
    main()
