import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.signal import detrend, butter, filtfilt
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ========================================================
# 🌟 第一部分：物理友好的预处理 (修复了幅值抹杀Bug)
# ========================================================
def apply_mild_filter(data, fs=36.0, cutoff=0.1, order=2):
    """降低了截止频率(0.5->0.1)，保留更多滑动过程中的低频趋势力"""
    b, a = butter(order, cutoff / (0.5 * fs), btype='high', analog=False)
    return filtfilt(b, a, data, axis=0)


def interpolate_to_fixed_length(data, target_length=400):
    rows, cols = data.shape
    if rows <= 1: return None
    x_old, x_new = np.linspace(0, rows - 1, rows), np.linspace(0, rows - 1, target_length)
    res = np.zeros((target_length, cols))
    for c in range(cols):
        res[:, c] = interp1d(x_old, data[:, c], kind='linear')(x_new)
    return res


def process_csv_to_physical_sequence(csv_path):
    try:
        df = pd.read_csv(csv_path, header=0, encoding='utf-8')
    except:
        df = pd.read_csv(csv_path, header=0, encoding='gbk')

    cols = ['S1_X', 'S1_Y', 'S1_Z', 'S2_X', 'S2_Y', 'S2_Z', 'S3_X', 'S3_Y', 'S3_Z', 'S4_X', 'S4_Y', 'S4_Z']
    if any(c not in df.columns for c in cols): return None
    d12 = df[cols].values
    if len(d12) > 20: d12 = d12[int(len(d12) * 0.15):int(len(d12) * 0.90)]

    s1, s2, s3, s4 = [np.linalg.norm(d12[:, i:i + 3], axis=1, keepdims=True) for i in (0, 3, 6, 9)]
    d16 = np.hstack((d12, s1, s2, s3, s4))

    # 去趋势与温和滤波
    detrended = detrend(d16, axis=0, type='linear')
    filtered = apply_mild_filter(detrended)

    # ⚠️ 核心修复：使用固定比例缩放，而不是逐文件 RobustScaler
    # 假设 MLX90393 的波动幅度在 ±500 左右，这里统一除以 500 进行粗略归一化
    # 这样能绝对保留“重摩擦力”和“轻摩擦力”材质在振幅上的物理差异！
    normalized = filtered / 500.0

    return interpolate_to_fixed_length(normalized, target_length=400)


def generate_sliding_windows(sequence_400, window_size=100, step=20):
    windows = []
    for start in range(0, 400 - window_size + 1, step):
        win = sequence_400[start: start + window_size, :]
        windows.append(np.transpose(win))  # (16, 100)
    return windows


# ========================================================
# 🌟 第二部分：时频双流融合网络 (Dual-Stream Fusion)
# ========================================================

class SEBlock1D(nn.Module):
    def __init__(self, channel, reduction=4):  # ⚠️ 降低了 reduction，增强注意力表达
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.fc(nn.AdaptiveAvgPool1d(1)(x).view(b, c)).view(b, c, 1)
        return x * y.expand_as(x)


# 分支1：时域流 (提取滑动形态特征)
class TimeStream(nn.Module):
    def __init__(self, in_channels=16, out_features=128):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv1d(in_channels, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv1d(in_channels, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU())
        self.b3 = nn.Sequential(nn.Conv1d(in_channels, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU())
        self.se = SEBlock1D(192)
        # 将三尺度特征降维到 out_features
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(192, out_features)

    def forward(self, x):
        out = self.se(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))
        out = self.pool(out).squeeze(-1)
        return self.fc(out)  # (Batch, 128)


# 分支2：频域流 (提取高频纹理与摩擦振动)
class FreqStream(nn.Module):
    def __init__(self, seq_len=100, in_channels=16, out_features=64):
        super().__init__()
        # 长度为100的实数FFT，输出频点数为 100//2 + 1 = 51
        freq_bins = (seq_len // 2) + 1

        self.fc_net = nn.Sequential(
            nn.Linear(in_channels * freq_bins, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_features),
            nn.ReLU()
        )

    def forward(self, x):
        # 核心魔法：利用 PyTorch 内置的 FFT 将时域张量转为频域频谱
        # x shape: (Batch, 16, 100)
        fft_complex = torch.fft.rfft(x, dim=2)
        fft_mag = torch.abs(fft_complex)  # 取模值，得到振幅谱 (Batch, 16, 51)

        # 展平后喂入全连接层提取频域高阶特征
        b = fft_mag.size(0)
        fft_flat = fft_mag.view(b, -1)
        return self.fc_net(fft_flat)  # (Batch, 64)


# 主模型：双流融合架构
class DualStreamFusionNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=16, seq_len=100):
        super().__init__()
        # 实例化双流
        self.time_stream = TimeStream(in_channels=in_channels, out_features=128)
        self.freq_stream = FreqStream(seq_len=seq_len, in_channels=in_channels, out_features=64)

        # 融合后的决策层
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        t_feat = self.time_stream(x)  # 时域特征 (B, 128)
        f_feat = self.freq_stream(x)  # 频域特征 (B, 64)

        # 将时域和频域特征在通道维度上拼接 (Concat)
        fused_feat = torch.cat((t_feat, f_feat), dim=1)  # (B, 192)

        return self.classifier(fused_feat)


class TactileDataset(Dataset):
    def __init__(self, X, Y): self.X, self.Y = X, Y

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx],
                                                                                                    dtype=torch.long)


# ========================================================
# 🌟 第三部分：控制流
# ========================================================
def train_thesis_master():
    data_dir = r"D:\pycharmproject\pythonProject\dataset"
    class_map = {"zaojin": 0, "pige": 1, "A4": 2, "shazhi_40": 3, "shazhi_120": 4,
                 "shazhi_240": 5, "shazhi_400": 6, "shazhi_600": 7, "shazhi_800": 8, "shazhi_1000": 9}

    file_list = []
    for name, lab in class_map.items():
        for f in glob.glob(os.path.join(data_dir, name, "*.csv")):
            file_list.append((f, lab))

    print(f"📦 扫描到原始 CSV 文件总数: {len(file_list)}")
    if len(file_list) == 0: return

    labels = [item[1] for item in file_list]
    tr_files, te_files, tr_labels, te_labels = train_test_split(file_list, labels, test_size=0.3, random_state=42,
                                                                stratify=labels)
    va_files, ts_files = train_test_split(te_files, test_size=0.33, random_state=42, stratify=te_labels)

    def build_dataset(files, step_size, desc):
        X, Y = [], []
        for f_path, label in files:
            seq_400 = process_csv_to_physical_sequence(f_path)
            if seq_400 is not None:
                windows = generate_sliding_windows(seq_400, window_size=100, step=step_size)
                X.extend(windows);
                Y.extend([label] * len(windows))
        print(f"[{desc}] 包含文件数: {len(files)} -> 裂变后样本数: {len(X)}")
        return np.array(X), np.array(Y)

    print("\n🚀 正在执行核爆级数据扩增 (滑窗)...")
    X_tr, Y_tr = build_dataset(tr_files, step_size=15, desc="训练集")
    X_va, Y_va = build_dataset(va_files, step_size=30, desc="验证集")

    train_loader = DataLoader(TactileDataset(X_tr, Y_tr), batch_size=64, shuffle=True)
    val_loader = DataLoader(TactileDataset(X_va, Y_va), batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 🌟 初始化双流网络
    model = DualStreamFusionNet(num_classes=len(class_map), in_channels=16, seq_len=100).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)  # 增加了少许L2正则化防过拟合

    best_name, patience, wait, best_acc = "best_dual_stream.pth", 40, 0, 0.0

    print("\n🔥 开始训练 时频双流融合网络 ...")
    for epoch in range(500):
        model.train()
        for inp, lab in train_loader:
            inp, lab = inp.to(device), lab.to(device)
            optimizer.zero_grad()
            criterion(model(inp), lab).backward()
            optimizer.step()

        model.eval()
        cor, tot = 0, 0
        with torch.no_grad():
            for inp, lab in val_loader:
                inp, lab = inp.to(device), lab.to(device)
                _, pred = torch.max(model(inp), 1)
                tot += lab.size(0)
                cor += (pred == lab).sum().item()

        acc = 100 * cor / tot
        if (epoch + 1) % 5 == 0: print(f"Epoch {epoch + 1:03d} | Val Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc, wait = acc, 0
            torch.save(model.state_dict(), best_name)
        else:
            wait += 1
            if wait >= patience:
                print(f"🌟 早停触发。最佳验证集准确率: {best_acc:.2f}%")
                break

    # ========================================================
    # 测试集投票验证
    # ========================================================
    print("\n================ 正在进行测试集级联投票验证 ================")
    model.load_state_dict(torch.load(best_name))
    model.eval()
    test_cor, test_tot, all_true, all_pred = 0, 0, [], []

    with torch.no_grad():
        for f_path, true_label in ts_files:
            seq_400 = process_csv_to_physical_sequence(f_path)
            if seq_400 is None: continue

            test_windows = generate_sliding_windows(seq_400, window_size=100, step=20)
            inp_tensor = torch.tensor(np.array(test_windows), dtype=torch.float32).to(device)

            outputs = model(inp_tensor)
            _, preds = torch.max(outputs, 1)

            pred_list = preds.cpu().tolist()
            final_pred = max(set(pred_list), key=pred_list.count)

            all_true.append(true_label);
            all_pred.append(final_pred)
            test_tot += 1
            if true_label == final_pred: test_cor += 1

    final_acc = 100 * test_cor / test_tot
    print(f"测试集准确率: {final_acc:.2f}%")

    cm = confusion_matrix(all_true, all_pred)
    names = [k for k, v in sorted(class_map.items(), key=lambda i: i[1])]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=names, yticklabels=names, annot_kws={"size": 14})
    plt.title(f'测试集混淆矩阵 (Acc: {final_acc:.2f}%)', fontsize=16, fontweight='bold')
    plt.ylabel('真实材质', fontsize=14);
    plt.xlabel('预测材质', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix_dual_stream.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    train_thesis_master()