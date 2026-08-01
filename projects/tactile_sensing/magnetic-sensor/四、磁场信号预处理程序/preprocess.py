
"""数据预处理脚本
功能：

用法示例：

"""
"""数据预处理脚本（已实现）

功能（已实现）:
- 递归遍历 `data/` 下样本 CSV 并忽略 gt/模板等元文件。
- 将多列保留为多通道 (C, L)，并同时向后兼容按列平均为单通道的行为。
- 支持带通/低通/高通滤波（优先使用 `scipy`，没有时回退到可用实现）。
- 去趋势（线性去趋势）已实现。
- 重采样到固定长度 `seq_len`（默认 512）已实现。
- 归一化（`zscore` 或 `minmax`）已实现。
- 将处理后数组保存为 `data_processed/<same_subdir>/*.npy`，并写入 `data_processed/manifest.csv`（字段: `orig`, `processed`, `class`, `force`, `shape`）。

用法示例（已实现）:
python preprocess.py --data_root data --out_root data_processed --seq_len 512 --band 5 150 --norm zscore --keep_channels --dry

"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import re
import os

try:
    from scipy.signal import butter, filtfilt, detrend
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _parse_force_from_name(name: str):
    m = re.search(r"(\d+(?:\.\d+)?)N", name)
    if m:
        return float(m.group(1))
    return None


def _parse_class_from_name(name_or_path):
    """从文件名或路径推断类别。

    优先使用父目录名（例如 data_new/01_GlueStick_data -> GlueStick），
    退回到文件名前缀（下划线前）如果无法从目录名提取。
    """
    try:
        # 如果传入的是 Path 或包含路径分隔符，解析为 Path
        p = Path(name_or_path) if not isinstance(name_or_path, Path) else name_or_path
    except Exception:
        p = None

    def _clean_name(s: str) -> str:
        s0 = str(s)
        # 去掉前导序号和下划线，如 01_GlueStick -> GlueStick
        s1 = re.sub(r'^\d+[_-]*', '', s0)
        # 去掉常见的后缀如 _data/_dataset/_files
        s1 = re.sub(r'[_-]*(data|dataset|files|datafolder)$', '', s1, flags=re.IGNORECASE)
        # 取第一个字母开头的单词
        m = re.search(r'[A-Za-z][A-Za-z0-9]*', s1)
        if m:
            return m.group(0)
        # 如果没有字母，退回用原始字符串的第一个部分
        return s1.split('_')[0] if '_' in s1 else s1

    # 优先使用父目录名
    try:
        if p is not None and p.parent is not None and p.parent.name:
            cand = _clean_name(p.parent.name)
            if cand and cand.lower() not in ('', '.', '..'):
                return cand
    except Exception:
        pass

    # 回退到文件名本身
    try:
        if p is not None:
            return _clean_name(p.stem)
    except Exception:
        pass

    # 最后退回到简单分割
    try:
        s = str(name_or_path)
        parts = s.split('_')
        if parts:
            return parts[0]
    except Exception:
        pass
    return 'unknown'


def _is_metadata(p: Path):
    name = p.name.lower()
    stem = p.stem.lower()
    if name in ('gt.csv', 'labels.csv', 'gt_template.csv', 'manifest.csv'):
        return True
    if stem.endswith('_template') or stem == 'template':
        return True
    return False


KNOWN_CHANNEL_PRESETS = ('all25', 'magnetic16', 'xyz12', 'total4')


def infer_channel_names(num_channels: int):
    sensors = ('S1', 'S2', 'S3', 'S4')
    if num_channels == 25:
        names = ['time']
        for sensor in sensors:
            names.extend([f'{sensor}_Traw', f'{sensor}_temp', f'{sensor}_X', f'{sensor}_Y', f'{sensor}_Z', f'{sensor}_total'])
        return names
    if num_channels == 16:
        names = []
        for sensor in sensors:
            names.extend([f'{sensor}_X', f'{sensor}_Y', f'{sensor}_Z', f'{sensor}_total'])
        return names
    if num_channels == 12:
        names = []
        for sensor in sensors:
            names.extend([f'{sensor}_X', f'{sensor}_Y', f'{sensor}_Z'])
        return names
    if num_channels == 4:
        return [f'{sensor}_total' for sensor in sensors]
    return [f'ch_{i:02d}' for i in range(num_channels)]


def parse_channel_meta(channel_name: str):
    if channel_name == 'time':
        return {'sensor': None, 'axis': None, 'kind': 'aux'}
    parts = str(channel_name).split('_', 1)
    if len(parts) != 2:
        return {'sensor': None, 'axis': None, 'kind': 'unknown'}
    sensor, suffix = parts
    if sensor not in {'S1', 'S2', 'S3', 'S4'}:
        return {'sensor': None, 'axis': None, 'kind': 'unknown'}
    if suffix in {'X', 'Y', 'Z', 'total'}:
        return {'sensor': sensor, 'axis': suffix, 'kind': 'magnetic'}
    return {'sensor': sensor, 'axis': None, 'kind': 'aux'}


def resolve_channel_selection(channel_names, preset='all25', drop_axes=None, drop_sensors=None):
    preset = str(preset or 'all25')
    if preset not in KNOWN_CHANNEL_PRESETS:
        raise ValueError(f'Unknown channel preset: {preset}')

    drop_axes = {str(x) for x in (drop_axes or [])}
    drop_sensors = {str(x) for x in (drop_sensors or [])}
    if not drop_axes.issubset({'X', 'Y', 'Z', 'total'}):
        raise ValueError(f'Unsupported drop_axes: {sorted(drop_axes)}')
    if not drop_sensors.issubset({'S1', 'S2', 'S3', 'S4'}):
        raise ValueError(f'Unsupported drop_sensors: {sorted(drop_sensors)}')

    indices = []
    selected_names = []
    for idx, channel_name in enumerate(channel_names):
        meta = parse_channel_meta(channel_name)
        if preset == 'magnetic16' and meta['kind'] != 'magnetic':
            continue
        if preset == 'xyz12' and meta['axis'] not in {'X', 'Y', 'Z'}:
            continue
        if preset == 'total4' and meta['axis'] != 'total':
            continue
        if meta['sensor'] in drop_sensors:
            continue
        if meta['axis'] in drop_axes:
            continue
        indices.append(idx)
        selected_names.append(channel_name)

    if not indices:
        raise ValueError(
            f'Channel selection is empty for preset={preset}, '
            f'drop_axes={sorted(drop_axes)}, drop_sensors={sorted(drop_sensors)}'
        )
    return indices, selected_names

def read_csv_to_multi(path: Path):
    """Read numeric columns and return array shape (C, L) where C is channels.
    If file has single numeric column, returns 1D array (L,) for compatibility.
    """
    try:
        df = pd.read_csv(path)
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] == 0:
            df2 = pd.read_csv(path, header=None)
            arr = df2.values.astype(np.float32)
            if arr.ndim == 2 and arr.shape[1] > 1:
                # arr (L, C) -> transpose to (C, L)
                return arr.T
            else:
                return arr.reshape(-1)
        arr = num.values.astype(np.float32)  # (L, C) or (L,)
        if arr.ndim == 2 and arr.shape[1] > 1:
            return arr.T
        else:
            return arr.reshape(-1)
    except Exception:
        return np.zeros((1,), dtype=np.float32)


def bandpass_filter(x, fs, lowcut=None, highcut=None, order=4):
    if not _HAS_SCIPY:
        return x
    nyq = 0.5 * fs
    btype = None
    Wn = None
    if lowcut is not None and highcut is not None:
        Wn = [lowcut / nyq, highcut / nyq]
        btype = 'band'
    elif lowcut is not None:
        Wn = lowcut / nyq
        btype = 'high'
    elif highcut is not None:
        Wn = highcut / nyq
        btype = 'low'
    else:
        return x
    b, a = butter(order, Wn, btype=btype)
    y = filtfilt(b, a, x)
    return y


def resample_to_len(x, target_len):
    if len(x) == target_len:
        return x
    if len(x) == 0:
        return np.zeros((target_len,), dtype=np.float32)
    old_idx = np.linspace(0, 1, num=len(x))
    new_idx = np.linspace(0, 1, num=target_len)
    new_x = np.interp(new_idx, old_idx, x)
    return new_x.astype(np.float32)


def normalize(x, method='zscore'):
    if method == 'zscore':
        mu = x.mean() if x.size else 0.0
        s = x.std() if x.size else 1.0
        s = s if s > 1e-6 else 1.0
        return (x - mu) / s
    if method == 'minmax':
        mn = x.min() if x.size else 0.0
        mx = x.max() if x.size else 1.0
        rng = mx - mn if (mx - mn) > 1e-6 else 1.0
        return (x - mn) / rng
    return x


def process_file(p: Path, seq_len=512, fs=1000.0, lowcut=None, highcut=None, detr=True, norm='zscore'):
    # default single-channel pipeline
    x = read_csv_to_1d(p)
    if detr and _HAS_SCIPY:
        x = detrend(x)
    elif detr:
        # simple linear detrend
        if x.size > 1:
            t = np.arange(x.size)
            A = np.vstack([t, np.ones_like(t)]).T
            m, c = np.linalg.lstsq(A, x, rcond=None)[0]
            x = x - (m * t + c)
    if lowcut is not None or highcut is not None:
        x = bandpass_filter(x, fs=fs, lowcut=lowcut, highcut=highcut)
    x = resample_to_len(x, seq_len)
    x = normalize(x, method=norm)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data_new')
    parser.add_argument('--out_root', type=str, default=None, help='output root; if omitted will use <data_root>_processed')
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--fs', type=float, default=1000.0, help='assumed sampling rate for filtering')
    parser.add_argument('--band', nargs='*', type=float, default=None, help='low high in Hz, e.g. --band 5 150')
    parser.add_argument('--detrend', action='store_true', help='apply detrending')
    parser.add_argument('--norm', choices=['zscore','minmax','none'], default='zscore')
    parser.add_argument('--keep_channels', action='store_true', help='preserve multi-column CSV as multi-channel and save npy shape (C,L)')
    parser.add_argument('--dry', action='store_true', help='dry run: do not write files')
    parser.add_argument('--limit', type=int, default=0, help='process only N files for testing')
    args = parser.parse_args()

    root = Path(args.data_root)
    # derive out_root: if user didn't pass --out_root, use data_root + '_processed'
    if args.out_root:
        out_root = Path(args.out_root)
    else:
        out_root = Path(str(root).rstrip('/\\') + '_processed')
    rows = []
    files = [p for p in root.rglob('*.csv') if not _is_metadata(p)]
    total = len(files)
    print('Found', total, 'csv files (excluding metadata)')
    if args.limit > 0:
        files = files[: args.limit]
    for i, p in enumerate(files, 1):
        try:
            rel = p.relative_to(root)
        except Exception:
            rel = p.name
        cls = _parse_class_from_name(p)
        force = _parse_force_from_name(p.name)
        out_dir = out_root / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.keep_channels:
            # multi-channel path: process each numeric column independently
            try:
                arr = read_csv_to_multi(p)
                if arr.ndim == 1:
                    # single column file fallback
                    arr = arr[np.newaxis, :]
                # arr shape (C, L)
                C, L0 = arr.shape
                proc = np.zeros((C, args.seq_len), dtype=np.float32)
                for ci in range(C):
                    col = arr[ci, :]
                    if args.detrend and _HAS_SCIPY:
                        col = detrend(col)
                    elif args.detrend and col.size > 1:
                        t = np.arange(col.size)
                        A = np.vstack([t, np.ones_like(t)]).T
                        m, c = np.linalg.lstsq(A, col, rcond=None)[0]
                        col = col - (m * t + c)
                    if args.band and (args.band[0] is not None or (len(args.band) >= 2 and args.band[1] is not None)):
                        low = args.band[0] if args.band and len(args.band) >= 1 else None
                        high = args.band[1] if args.band and len(args.band) >= 2 else None
                        col = bandpass_filter(col, fs=args.fs, lowcut=low, highcut=high)
                    col = resample_to_len(col, args.seq_len)
                    if args.norm == 'zscore':
                        mu = col.mean() if col.size else 0.0
                        s = col.std() if col.size else 1.0
                        s = s if s > 1e-6 else 1.0
                        col = (col - mu) / s
                    elif args.norm == 'minmax':
                        mn = col.min() if col.size else 0.0
                        mx = col.max() if col.size else 1.0
                        rng = mx - mn if (mx - mn) > 1e-6 else 1.0
                        col = (col - mn) / rng
                    proc[ci, :] = col
                x = proc
            except Exception:
                x = process_file(p, seq_len=args.seq_len, fs=args.fs,
                                 lowcut=(args.band[0] if args.band and len(args.band) >= 1 else None),
                                 highcut=(args.band[1] if args.band and len(args.band) >= 2 else None),
                                 detr=args.detrend, norm=args.norm)
        else:
            x = process_file(p, seq_len=args.seq_len, fs=args.fs,
                             lowcut=(args.band[0] if args.band and len(args.band) >= 1 else None),
                             highcut=(args.band[1] if args.band and len(args.band) >= 2 else None),
                             detr=args.detrend, norm=args.norm)
        out_path = out_dir / (p.stem + '.npy')
        if not args.dry:
            np.save(str(out_path), x)
        shape_info = None
        try:
            shape_info = tuple(x.shape)
        except Exception:
            shape_info = None
        rows.append({'orig': str(p), 'processed': str(out_path), 'class': cls, 'force': force, 'shape': shape_info})
        if i % 100 == 0 or i == len(files):
            print(f'Processed {i}/{len(files)}')

    manifest = out_root / 'manifest.csv'
    df = pd.DataFrame(rows)
    if not args.dry:
        df.to_csv(manifest, index=False)
    print('Wrote manifest rows:', len(rows))
    if not _HAS_SCIPY:
        print('Note: scipy not available, filtering/detrend used fallbacks where possible.')


if __name__ == '__main__':
    main()


# --- Merged SequenceDataset (originally in dataset.py) ---
import re
from pathlib import Path
import torch
from torch.utils.data import Dataset


def _parse_force_from_name(name: str):
    m = re.search(r"(\d+(?:\.\d+)?)N", name)
    if m:
        return float(m.group(1))
    return None


def _parse_class_from_name(name_or_path):
    """从文件名或路径推断类别。

    优先使用父目录名（例如 data_new/01_GlueStick_data -> GlueStick），
    退回到文件名前缀（下划线前）如果无法从目录名提取。
    """
    try:
        # 如果传入的是 Path 或包含路径分隔符，解析为 Path
        p = Path(name_or_path) if not isinstance(name_or_path, Path) else name_or_path
    except Exception:
        p = None

    def _clean_name(s: str) -> str:
        s0 = str(s)
        # 去掉前导序号和下划线，如 01_GlueStick -> GlueStick
        s1 = re.sub(r'^\d+[_-]*', '', s0)
        # 去掉常见的后缀如 _data/_dataset/_files
        s1 = re.sub(r'[_-]*(data|dataset|files|datafolder)$', '', s1, flags=re.IGNORECASE)
        # 取第一个字母开头的单词
        m = re.search(r'[A-Za-z][A-Za-z0-9]*', s1)
        if m:
            return m.group(0)
        # 如果没有字母，退回用原始字符串的第一个部分
        return s1.split('_')[0] if '_' in s1 else s1

    # 优先使用父目录名
    try:
        if p is not None and p.parent is not None and p.parent.name:
            cand = _clean_name(p.parent.name)
            if cand and cand.lower() not in ('', '.', '..'):
                return cand
    except Exception:
        pass

    # 回退到文件名本身
    try:
        if p is not None:
            return _clean_name(p.stem)
    except Exception:
        pass

    # 最后退回到简单分割
    try:
        s = str(name_or_path)
        parts = s.split('_')
        if parts:
            return parts[0]
    except Exception:
        pass
    return 'unknown'


class SequenceDataset(Dataset):
    """Load 1D sequences from CSVs under a root folder.

    - Classification label inferred from filename prefix (before first underscore).
    - Regression target inferred from filename pattern like '10N'.
    """

    def __init__(
        self,
        root_dir,
        seq_len=1024,
        transform=None,
        normalize=True,
        normalize_mode='per_sample',
        channel_preset='all25',
        drop_axes=None,
        drop_sensors=None,
    ):
        self.root_dir = Path(root_dir)
        # collect both .npy and .csv processed files so dataset can load preprocessed outputs
        all_csv = list(self.root_dir.rglob("*.csv"))
        all_npy = list(self.root_dir.rglob("*.npy"))

        def _is_metadata(p: Path):
            name = p.name.lower()
            stem = p.stem.lower()
            if name in ('gt.csv', 'labels.csv', 'gt_template.csv', 'manifest.csv'):
                return True
            if stem.endswith('_template') or stem == 'template':
                return True
            return False

        # prefer npy files when both exist; include csv files otherwise
        npy_stems = {p.with_suffix('') for p in all_npy}
        files = []
        for p in all_npy:
            if not _is_metadata(p):
                files.append(p)
        for p in all_csv:
            if not _is_metadata(p):
                if p.with_suffix('') in npy_stems:
                    continue
                files.append(p)

        self.files = files
        self.seq_len = seq_len
        self.transform = transform
        self.normalize = normalize
        self.normalize_mode = normalize_mode
        self.channel_preset = channel_preset
        self.drop_axes = list(drop_axes or [])
        self.drop_sensors = list(drop_sensors or [])

        # read gt map if exists
        self.gt_map = {}
        gt_file1 = self.root_dir / 'gt.csv'
        gt_file2 = self.root_dir / 'labels.csv'
        gt_path = None
        if gt_file1.exists():
            gt_path = gt_file1
        elif gt_file2.exists():
            gt_path = gt_file2
        if gt_path is not None:
            try:
                import pandas as _pd
                df = _pd.read_csv(gt_path, header=0)
                if 'filename' in df.columns and 'force' in df.columns:
                    for _, r in df.iterrows():
                        self.gt_map[str(r['filename'])] = float(r['force'])
                else:
                    for _, r in df.iterrows():
                        key = str(r.iloc[0])
                        val = float(r.iloc[1])
                        self.gt_map[key] = val
            except Exception:
                self.gt_map = {}

        classes = {}
        items = []
        for p in self.files:
            cls = _parse_class_from_name(p)
            if cls not in classes:
                classes[cls] = len(classes)
            force = None
            if self.gt_map:
                if p.name in self.gt_map:
                    force = self.gt_map[p.name]
                elif p.stem in self.gt_map:
                    force = self.gt_map[p.stem]
            if force is None:
                force = _parse_force_from_name(p.name)
            items.append((p, classes[cls], force))

        self.items = items
        self.class_to_idx = {k: v for k, v in classes.items()}

        # detect channels (handle both .npy and .csv robustly)
        self.num_channels = 1
        if len(self.items) > 0:
            p0 = self.items[0][0]
            try:
                # if it's a preprocessed .npy, load with numpy
                if p0.suffix.lower() == '.npy':
                    a0 = np.load(str(p0))
                    if a0.ndim == 2:
                        # stored as (C, L)
                        self.num_channels = int(a0.shape[0])
                    else:
                        self.num_channels = 1
                else:
                    # csv path: try reading as CSV and inspect columns
                    df0 = pd.read_csv(p0, header=None)
                    arr0 = df0.values.astype(np.float32)
                    if arr0.ndim == 2 and arr0.shape[1] > 1:
                        # CSV loaded as (L, C)
                        self.num_channels = int(arr0.shape[1])
                    else:
                        self.num_channels = 1
            except Exception:
                # fallback: use dataset loader helper which already handles .npy/.csv
                try:
                    arr0 = self._load_csv(p0)
                    if arr0 is not None and hasattr(arr0, 'ndim') and arr0.ndim == 2:
                        self.num_channels = int(arr0.shape[0])
                    else:
                        self.num_channels = 1
                except Exception:
                    self.num_channels = 1

        self.channel_names = infer_channel_names(int(self.num_channels))
        self.selected_channel_indices, self.selected_channel_names = resolve_channel_selection(
            self.channel_names,
            preset=self.channel_preset,
            drop_axes=self.drop_axes,
            drop_sensors=self.drop_sensors,
        )
        self.num_channels = len(self.selected_channel_indices)

        # global stats
        self.global_mean = None
        self.global_std = None
        if self.normalize and self.normalize_mode == 'global' and len(self.items) > 0:
            sums = None
            sumsqs = None
            count = 0
            for p, _, _ in self.items:
                arr = self._load_csv(p)
                arr = self._pad_or_trim(arr)
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
                arr = self._select_channels(arr)
                if sums is None:
                    sums = arr.sum(axis=1)
                    sumsqs = (arr ** 2).sum(axis=1)
                else:
                    sums += arr.sum(axis=1)
                    sumsqs += (arr ** 2).sum(axis=1)
                count += arr.shape[1]
            try:
                mean = sums / float(count)
                var = (sumsqs / float(count)) - (mean ** 2)
                std = np.sqrt(np.maximum(var, 1e-6))
                self.global_mean = mean
                self.global_std = std
            except Exception:
                self.global_mean = None
                self.global_std = None

    def __len__(self):
        return len(self.items)

    def _load_csv(self, path: Path):
        try:
            # support loading preprocessed .npy files as well as CSV
            if path.suffix.lower() == '.npy':
                a = np.load(str(path))
                a = a.astype(np.float32)
                if a.ndim == 2:
                    return a
                else:
                    return a.reshape(-1)
            df = pd.read_csv(path, header=None)
            arr = df.values.astype(np.float32)
            if arr.ndim == 2 and arr.shape[1] > 1:
                return arr.T
            else:
                return arr.reshape(-1)
        except Exception:
            return np.zeros((self.seq_len,), dtype=np.float32)

    def _pad_or_trim(self, arr):
        if arr.ndim == 1:
            L = arr.shape[0]
            if L == self.seq_len:
                return arr
            if L > self.seq_len:
                return arr[: self.seq_len]
            out = np.zeros((self.seq_len,), dtype=arr.dtype)
            out[:L] = arr
            return out
        else:
            C, L = arr.shape
            if L == self.seq_len:
                return arr
            if L > self.seq_len:
                return arr[:, : self.seq_len]
            out = np.zeros((C, self.seq_len), dtype=arr.dtype)
            out[:, :L] = arr
            return out

    def _select_channels(self, arr):
        if arr.ndim == 1:
            return arr
        return arr[self.selected_channel_indices, :]

    def __getitem__(self, idx):
        p, cls_idx, force = self.items[idx]
        arr = self._load_csv(p)
        arr = self._pad_or_trim(arr)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        arr = self._select_channels(arr)
        if self.normalize:
            if self.normalize_mode == 'per_sample' or self.global_mean is None:
                mean = arr.mean(axis=1, keepdims=True)
                std = arr.std(axis=1, keepdims=True)
                std[std < 1e-6] = 1.0
                arr = (arr - mean) / std
            else:
                gm = self.global_mean[:, None]
                gs = self.global_std[:, None]
                gs[gs < 1e-6] = 1.0
                arr = (arr - gm) / gs
        x = torch.from_numpy(arr.astype(np.float32))
        sample = {"x": x, "cls": torch.tensor(cls_idx, dtype=torch.long)}
        if force is not None:
            sample["force"] = torch.tensor([force], dtype=torch.float32)
        else:
            sample["force"] = torch.tensor([0.0], dtype=torch.float32)
        if self.transform:
            sample = self.transform(sample)
        return sample

