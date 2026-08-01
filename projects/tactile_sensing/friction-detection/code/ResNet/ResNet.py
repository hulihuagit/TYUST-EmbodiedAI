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
# 🌟 第一部分：物理友好的预处理 (保持绝对不变)
# ========================================================
def apply_mild_filter(data, fs=36.0, cutoff=0.1, order=2):
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

    detrended = detrend(d16, axis=0, type='linear')
    filtered = apply_mild_filter(detrended)
    normalized = filtered / 500.0

    return interpolate_to_fixed_length(normalized, target_length=400)


def generate_sliding_windows(sequence_400, window_size=100, step=20):
    windows = []
    for start in range(0, 400 - window_size + 1, step):
        win = sequence_400[start: start + window_size, :]
        windows.append(np.transpose(win))  # (16, 100)
    return windows


# ========================================================
# 🌟 第二部分：1D-ResNet 网络模型 (核心替换部分)
# ========================================================
class ResidualBlock1D(nn.Module):
    """
    1D 残差基础块 (BasicBlock)
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        # 主干路径：两个 3x3 的卷积层
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 捷径路径 (Shortcut / Skip Connection)
        # 如果通道数改变或者进行了下采样(stride>1)，捷径也需要用 1x1 卷积调整形状，才能做加法
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # ⚠️ ResNet 的核心魔法：F(x) + x
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """
    针对 16通道、长度100 定制的 1D-ResNet
    """

    def __init__(self, in_channels=16, num_classes=10):
        super(ResNet1D, self).__init__()

        # 1. 初始卷积层 (提取浅层特征，不改变序列长度)
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        # 第一次降采样：长度 100 -> 50
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 2. 残差层堆叠 (每个 layer 包含 2 个 ResidualBlock)
        self.layer1 = self._make_layer(in_channels=64, out_channels=64, stride=1)  # 长度保持 50
        self.layer2 = self._make_layer(in_channels=64, out_channels=128, stride=2)  # 长度降至 25
        self.layer3 = self._make_layer(in_channels=128, out_channels=256, stride=2)  # 长度降至 13

        # 3. 全局平均池化与分类器
        # ResNet 原生放弃了 VGG 庞大的全连接层，改用全局池化直接压平
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, stride):
        layers = []
        # 第一个块：负责改变通道数和降采样
        layers.append(ResidualBlock1D(in_channels, out_channels, stride))
        # 第二个块：维持通道数和形状不变，加深网络
        layers.append(ResidualBlock1D(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)  # (Batch, 256, 1)
        x = torch.flatten(x, 1)  # (Batch, 256)
        x = self.fc(x)  # (Batch, 10)
        return x


class TactileDataset(Dataset):
    def __init__(self, X, Y): self.X, self.Y = X, Y

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx],
                                                                                                    dtype=torch.long)


# ========================================================
# 🌟 第三部分：控制流 (保持逻辑不变)
# ========================================================
def train_thesis_master_resnet():
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
                X.extend(windows)
                Y.extend([label] * len(windows))
        print(f"[{desc}] 包含文件数: {len(files)} -> 裂变后样本数: {len(X)}")
        return np.array(X), np.array(Y)

    print("\n🚀 正在执行核爆级数据扩增 (滑窗)...")
    X_tr, Y_tr = build_dataset(tr_files, step_size=15, desc="训练集")
    X_va, Y_va = build_dataset(va_files, step_size=30, desc="验证集")

    train_loader = DataLoader(TactileDataset(X_tr, Y_tr), batch_size=64, shuffle=True)
    val_loader = DataLoader(TactileDataset(X_va, Y_va), batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🌟 实例化 ResNet-1D
    model = ResNet1D(in_channels=16, num_classes=len(class_map)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)

    best_name, patience, wait, best_acc = "best_resnet_model.pth", 40, 0, 0.0

    print("\n🔥 开始训练 1D-ResNet 深度残差网络 ...")
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

            all_true.append(true_label)
            all_pred.append(final_pred)
            test_tot += 1
            if true_label == final_pred: test_cor += 1

    final_acc = 100 * test_cor / test_tot
    print(f"测试集准确率: {final_acc:.2f}%")

    cm = confusion_matrix(all_true, all_pred)
    names = [k for k, v in sorted(class_map.items(), key=lambda i: i[1])]
    plt.figure(figsize=(10, 8))
    # 调整了热力图颜色为 Reds 方便区分这是 ResNet 的结果
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=names, yticklabels=names, annot_kws={"size": 14})
    plt.title(f'1D-ResNet 测试集混淆矩阵 (Acc: {final_acc:.2f}%)', fontsize=16, fontweight='bold')
    plt.ylabel('真实材质', fontsize=14)
    plt.xlabel('预测材质', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix_resnet.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    train_thesis_master_resnet()