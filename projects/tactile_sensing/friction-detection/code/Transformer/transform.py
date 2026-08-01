import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt, detrend
from scipy.interpolate import interp1d
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================================
# 解决画图时中文字体显示问题
# ========================================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ========================================================
# 第一部分：黄金 16通道 预处理 (保持物理真实感)
# ========================================================
def apply_highpass_filter(data, fs=36.0, cutoff=0.5, order=4):
    nyquist = 0.5 * fs
    b, a = butter(order, cutoff / nyquist, btype='high', analog=False)
    return filtfilt(b, a, data, axis=0)


def dynamic_resample_16ch(normalized_data, target_length=100):
    original_rows = normalized_data.shape[0]
    num_channels = normalized_data.shape[1]
    if original_rows <= 1: return None
    x_old = np.linspace(0, original_rows - 1, num=original_rows)
    x_new = np.linspace(0, original_rows - 1, num=target_length)
    resampled_matrix = np.zeros((target_length, num_channels))
    for channel in range(num_channels):
        f_interpolator = interp1d(x_old, normalized_data[:, channel], kind='linear')
        resampled_matrix[:, channel] = f_interpolator(x_new)
    return resampled_matrix


def process_single_csv(csv_path, target_length=100):
    try:
        df = pd.read_csv(csv_path, header=0, encoding='utf-8')
    except:
        df = pd.read_csv(csv_path, header=0, encoding='gbk')

    target_columns = ['S1_X', 'S1_Y', 'S1_Z', 'S2_X', 'S2_Y', 'S2_Z', 'S3_X', 'S3_Y', 'S3_Z', 'S4_X', 'S4_Y', 'S4_Z']
    mag_data_12 = df[target_columns].values
    L = len(mag_data_12)
    if L > 20: mag_data_12 = mag_data_12[int(L * 0.15):int(L * 0.90)]

    s1_t = np.linalg.norm(mag_data_12[:, 0:3], axis=1, keepdims=True)
    s2_t = np.linalg.norm(mag_data_12[:, 3:6], axis=1, keepdims=True)
    s3_t = np.linalg.norm(mag_data_12[:, 6:9], axis=1, keepdims=True)
    s4_t = np.linalg.norm(mag_data_12[:, 9:12], axis=1, keepdims=True)

    mag_data_16 = np.hstack((mag_data_12, s1_t, s2_t, s3_t, s4_t))
    data = detrend(mag_data_16, axis=0)
    data = apply_highpass_filter(data)
    data = RobustScaler().fit_transform(data)
    res = dynamic_resample_16ch(data, target_length)
    return np.transpose(res) if res is not None else None


# ==========================================
# 第二部分：数据增强与 Dataset
# ==========================================
def augment_data(data):
    # 加噪
    if np.random.rand() < 0.5:
        data += np.random.normal(0, 0.02, data.shape)
    # 拉伸
    if np.random.rand() < 0.5:
        c, l = data.shape
        crop = np.random.uniform(0.92, 1.0)
        new_l = int(l * crop)
        start = np.random.randint(0, l - new_l + 1)
        f = interp1d(np.linspace(0, new_l - 1, new_l), data[:, start:start + new_l], kind='linear', axis=1)
        data = f(np.linspace(0, new_l - 1, l))
    return data


class EnsembleDataset(Dataset):
    def __init__(self, X, Y, is_train=False):
        self.X, self.Y, self.is_train = X, Y, is_train

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = augment_data(self.X[idx].copy()) if self.is_train else self.X[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.long)


# ==========================================
# 第三部分：模型 (黄金 1D-CNN)
# ==========================================
class MultiScaleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MultiScaleCNN, self).__init__()
        self.b1 = nn.Sequential(nn.Conv1d(16, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv1d(16, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU())
        self.b3 = nn.Sequential(nn.Conv1d(16, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU())
        self.drop = nn.Dropout(0.5)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return self.fc(self.gap(self.drop(out)).squeeze(-1))


# ==========================================
# 第四部分：五折集成流水线
# ==========================================
def run_ensemble_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = r"D:\pycharmproject\pythonProject\dataset"
    class_map = {"zaojin": 0, "pige": 1, "A4": 2, "shazhi_40": 3, "shazhi_120": 4,
                 "shazhi_240": 5, "shazhi_400": 6, "shazhi_600": 7, "shazhi_800": 8, "shazhi_1000": 9}

    # 1. 提取所有数据
    X_all, Y_all = [], []
    for name, label in class_map.items():
        files = glob.glob(os.path.join(data_dir, name, "*.csv"))
        for f in files:
            win = process_single_csv(f)
            if win is not None: X_all.append(win); Y_all.append(label)

    X_all, Y_all = np.array(X_all), np.array(Y_all)

    # 2. 先切出 10% 的“终极黑盒测试集”，坚决不参与训练
    X_dev, X_test, Y_dev, Y_test = train_test_split(X_all, Y_all, test_size=0.1, stratify=Y_all, random_state=42)

    # 3. 准备五折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_models = []

    print(f"--- 开始五折集成 (总样本: {len(X_all)}, 测试集: {len(X_test)}) ---")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev, Y_dev)):
        print(f"\n🔥 正在训练第 {fold + 1} 个集成子模型...")
        x_tr, x_va = X_dev[train_idx], X_dev[val_idx]
        y_tr, y_va = Y_dev[train_idx], Y_dev[val_idx]

        train_loader = DataLoader(EnsembleDataset(x_tr, y_tr, is_train=True), batch_size=32, shuffle=True)
        val_loader = DataLoader(EnsembleDataset(x_va, y_va, is_train=False), batch_size=32, shuffle=False)

        model = MultiScaleCNN(num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        best_acc, patience, wait = 0, 80, 0
        save_path = f"ensemble_fold_{fold}.pth"

        for epoch in range(1000):
            model.train()
            for inp, lab in train_loader:
                inp, lab = inp.to(device), lab.to(device)
                optimizer.zero_grad();
                loss = criterion(model(inp), lab);
                loss.backward();
                optimizer.step()

            model.eval()
            cor, tot = 0, 0
            with torch.no_grad():
                for inp, lab in val_loader:
                    inp, lab = inp.to(device), lab.to(device)
                    _, pred = torch.max(model(inp).data, 1)
                    tot += lab.size(0);
                    cor += (pred == lab).sum().item()

            acc = 100 * cor / tot
            if acc > best_acc:
                best_acc = acc;
                wait = 0;
                torch.save(model.state_dict(), save_path)
            else:
                wait += 1
                if wait >= patience: break

        print(f"Fold {fold + 1} 完成，最高验证集准确率: {best_acc:.2f}%")
        fold_models.append(save_path)

    # 4. 🌟 集成测试阶段 (民主投票)
    print("\n--- 5个模型一起 ---")
    models = []
    for path in fold_models:
        m = MultiScaleCNN(num_classes=10).to(device)
        m.load_state_dict(torch.load(path))
        m.eval()
        models.append(m)

    test_loader = DataLoader(EnsembleDataset(X_test, Y_test, is_train=False), batch_size=1, shuffle=False)

    t1_cor, tot = 0, 0
    y_preds, y_true = [], []

    with torch.no_grad():
        for inp, lab in test_loader:
            inp, lab = inp.to(device), lab.to(device)

            # 核心：收集所有模型的“柔性分数”并取平均
            ensemble_output = torch.zeros((1, 10)).to(device)
            for m in models:
                ensemble_output += m(inp)
            ensemble_output /= len(models)  # 取均值，即集成预测

            # 纯净版：只取绝对命中的预测结果
            _, p1 = torch.max(ensemble_output.data, 1)

            tot += 1
            t1_cor += (p1 == lab).sum().item()

            y_preds.append(p1.item())
            y_true.append(lab.item())

    final_acc = 100 * t1_cor / tot
    print(f"\n测试集最终准确率: {final_acc:.2f}%")

    # 画混淆矩阵
    names = [k for k, v in sorted(class_map.items(), key=lambda i: i[1])]
    cm = confusion_matrix(y_true, y_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=names, yticklabels=names)
    # 标题直接带上最终分数，方便插论文
    plt.title(f'五折集成最终混淆矩阵 (准确率: {final_acc:.2f}%)', fontsize=16, fontweight='bold')
    plt.xlabel('预测材质 (Predicted)', fontsize=12)
    plt.ylabel('真实材质 (True)', fontsize=12)
    plt.tight_layout()
    plt.savefig('ensemble_confusion_matrix.png', dpi=300)
    plt.close()  # 绝对不弹窗报错

    print("🎉 混淆矩阵已生成并保存为: ensemble_confusion_matrix.png")


if __name__ == "__main__":
    run_ensemble_benchmark()