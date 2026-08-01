import os
import glob
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.signal import detrend, butter, filtfilt
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score
import warnings

warnings.filterwarnings('ignore')


# =====================================================================
# 📦 第一部分：统一的物理数据预处理
# =====================================================================
def apply_mild_filter(data, fs=36.0, cutoff=0.1, order=2):
    b, a = butter(order, cutoff / (0.5 * fs), btype='high', analog=False)
    return filtfilt(b, a, data, axis=0)


def interpolate_to_fixed_length(data, target_length=400):
    rows, cols = data.shape
    if rows <= 1: return None
    x_old, x_new = np.linspace(0, rows - 1, rows), np.linspace(0, rows - 1, target_length)
    res = np.zeros((target_length, cols))
    for c in range(cols): res[:, c] = interp1d(x_old, data[:, c], kind='linear')(x_new)
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
    return interpolate_to_fixed_length(filtered / 500.0, target_length=400)


def generate_sliding_windows(sequence_400, window_size=100, step=20):
    windows = []
    for start in range(0, 400 - window_size + 1, step):
        windows.append(np.transpose(sequence_400[start: start + window_size, :]))
    return windows


class TactileDataset(Dataset):
    def __init__(self, X, Y): self.X, self.Y = X, Y

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx],
                                                                                                    dtype=torch.long)


# =====================================================================
# 🧠 第二部分：消融实验核心网络定义 (步步递进)
# =====================================================================

# 【公共组件】 SE注意力模块
class SEBlock1D(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.fc(nn.AdaptiveAvgPool1d(1)(x).view(b, c)).view(b, c, 1)
        return x * y.expand_as(x)


# ----------------------------------------------------
# 模块 1：Baseline (MultiScale 1D-CNN) - 只有多尺度卷积
# ----------------------------------------------------
class M1_BaseCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv1d(16, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv1d(16, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU())
        self.b3 = nn.Sequential(nn.Conv1d(16, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(192, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)).squeeze(-1))


# ----------------------------------------------------
# 模块 2：MultiScale + SE - 加入了通道注意力
# ----------------------------------------------------
class M2_SE_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv1d(16, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv1d(16, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU())
        self.b3 = nn.Sequential(nn.Conv1d(16, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU())
        self.se = SEBlock1D(192)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(192, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.se(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))).squeeze(-1))


# ----------------------------------------------------
# 模块 3：双流 (1D-CNN+SE + FFT) - 没有MLP，直接拼接原始频谱
# ----------------------------------------------------
class M3_Dual_NoMLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 时域分支 (输出 128)
        self.time_net = M2_SE_CNN(num_classes=128)  # 复用M2作为提特征器
        # 频域分支：FFT 后展平 (16通道 * 51个频点 = 816维特征)
        # 融合层 (128 + 816 = 944)
        self.classifier = nn.Sequential(
            nn.Linear(944, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        t_feat = self.time_net(x)  # (B, 128)
        # 直接使用 FFT 幅值，没有 MLP 提取特征
        f_feat = torch.abs(torch.fft.rfft(x, dim=2)).view(x.size(0), -1)  # (B, 816)
        return self.classifier(torch.cat((t_feat, f_feat), dim=1))


# ----------------------------------------------------
# 模块 4：全功能双流 (1D-CNN+SE + FFT+MLP) - 最终方案
# ----------------------------------------------------
class M4_Dual_Full(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.time_net = M2_SE_CNN(num_classes=128)
        # 频域分支增加了 MLP 层进行深层映射
        self.freq_mlp = nn.Sequential(
            nn.Linear(816, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 64), nn.ReLU()
        )
        # 融合层 (128 + 64 = 192)
        self.classifier = nn.Sequential(
            nn.Linear(192, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        t_feat = self.time_net(x)  # (B, 128)
        f_feat_raw = torch.abs(torch.fft.rfft(x, dim=2)).view(x.size(0), -1)
        f_feat_mlp = self.freq_mlp(f_feat_raw)  # (B, 64)
        return self.classifier(torch.cat((t_feat, f_feat_mlp), dim=1))


# =====================================================================
# 🛠️ 第三部分：公共评估函数 (包含级联投票)
# =====================================================================
def get_model_params(model): return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def evaluate_ablation(model, ts_files, device):
    all_true, all_pred = [], []
    model.eval()
    for f_path, true_label in ts_files:
        seq = process_csv_to_physical_sequence(f_path)
        if seq is None: continue
        windows = generate_sliding_windows(seq, window_size=100, step=20)
        with torch.no_grad():
            out = model(torch.tensor(np.array(windows), dtype=torch.float32).to(device))
            preds = torch.max(out, 1)[1].cpu().tolist()
        all_true.append(true_label)
        all_pred.append(max(set(preds), key=preds.count))  # 文件级多数投票

    acc = np.mean(np.array(all_true) == np.array(all_pred))
    mac_f1 = f1_score(all_true, all_pred, average='macro', zero_division=0)
    mac_prec = precision_score(all_true, all_pred, average='macro', zero_division=0)
    return acc, mac_f1, mac_prec


# =====================================================================
# 👑 第四部分：消融实验主控台
# =====================================================================
def run_ablation_study():
    data_dir = r"D:\pycharmproject\pythonProject\dataset"
    class_map = {"zaojin": 0, "pige": 1, "A4": 2, "shazhi_40": 3, "shazhi_120": 4, "shazhi_240": 5, "shazhi_400": 6,
                 "shazhi_600": 7, "shazhi_800": 8, "shazhi_1000": 9}
    file_list = [(f, lab) for name, lab in class_map.items() for f in glob.glob(os.path.join(data_dir, name, "*.csv"))]
    if not file_list: return print("❌ 未找到数据！")
    labels = [item[1] for item in file_list]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4个消融模块的字典
    ablation_models = {
        "1. Base_CNN": M1_BaseCNN,
        "2. CNN + SE": M2_SE_CNN,
        "3. Dual_No_MLP": M3_Dual_NoMLP,
        "4. Dual_Full (Ours)": M4_Dual_Full
    }

    num_runs = 5  # 正式跑论文数据时建议改为 3 或 5
    epochs = 80  # 正式出图建议设为 40-50
    results = {}

    print(f"🚀 开始执行消融实验 (Ablation Study)！每个模块重复 {num_runs} 次，训练 {epochs} 轮...")

    for model_name, ModelClass in ablation_models.items():
        print(f"\n" + "=" * 50 + f"\n🔥 评估消融模块: {model_name}\n" + "=" * 50)
        m_acc, m_f1, m_prec, m_param = [], [], [], 0

        for run in range(num_runs):
            print(f"  👉 第 {run + 1}/{num_runs} 次独立实验...")
            tr_files, te_files, tr_labels, te_labels = train_test_split(file_list, labels, test_size=0.3,
                                                                        random_state=run, stratify=labels)
            va_files, ts_files = train_test_split(te_files, test_size=0.33, random_state=run, stratify=te_labels)

            X_tr, Y_tr = [], []
            for f, l in tr_files:
                s = process_csv_to_physical_sequence(f)
                if s is not None:
                    w = generate_sliding_windows(s, 100, 15)
                    X_tr.extend(w);
                    Y_tr.extend([l] * len(w))
            X_tr, Y_tr = np.array(X_tr), np.array(Y_tr)

            model = ModelClass(num_classes=10).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.0003)
            train_loader = DataLoader(TactileDataset(X_tr, Y_tr), batch_size=64, shuffle=True)

            model.train()
            for _ in range(epochs):
                for inp, lab in train_loader:
                    optimizer.zero_grad()
                    nn.CrossEntropyLoss()(model(inp.to(device)), lab.to(device)).backward()
                    optimizer.step()

            acc, f1, prec = evaluate_ablation(model, ts_files, device)
            if run == 0: m_param = get_model_params(model)

            m_acc.append(acc * 100)
            m_f1.append(f1)
            m_prec.append(prec)

        results[model_name] = {
            "Acc (%)": f"{np.mean(m_acc):.2f} ± {np.std(m_acc):.2f}",
            "Macro-F1": f"{np.mean(m_f1):.4f} ± {np.std(m_f1):.4f}",
            "Precision": f"{np.mean(m_prec):.4f} ± {np.std(m_prec):.4f}",
            "Params(M)": f"{m_param:.3f}"
        }

    print("\n\n" + "★" * 80)
    print("🏆 时频双流网络 —— 消融实验 (Ablation Study) 终极报告 🏆")
    print("★" * 80)
    print(pd.DataFrame(results).T.to_markdown())


if __name__ == "__main__":
    run_ablation_study()