import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import detrend, butter, filtfilt
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler  # ⚠️ 新增：特征标准化
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ========================================================
# 🌟 第一部分：物理友好的预处理 (保持不变)
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
# 🌟 新增：特征工程提取器
# ========================================================
def extract_statistical_features(windows):
    """
    将 (Batch, 16, 100) 的滑窗数据转化为 (Batch, 48) 的统计特征向量
    """
    features = []
    for win in windows:
        # win 的形状是 (16, 100)
        mean_vals = np.mean(win, axis=1)  # 均值 (16维)
        std_vals = np.std(win, axis=1)  # 标准差 (16维)
        ptp_vals = np.ptp(win, axis=1)  # 峰峰值 (Max - Min) (16维)

        # 将三种特征拼接起来，形成 16*3 = 48 维特征向量
        win_features = np.hstack((mean_vals, std_vals, ptp_vals))
        features.append(win_features)

    return np.array(features)


# ========================================================
# 🌟 第二部分：主控流
# ========================================================
def train_thesis_master_knn_features():
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

    print("\n🚀 正在执行数据扩增...")
    X_tr, Y_tr = build_dataset(tr_files, step_size=15, desc="训练集")
    X_va, Y_va = build_dataset(va_files, step_size=30, desc="验证集")

    # ⚠️ 核心步骤 1：特征提取
    print("\n⚙️ 正在进行特征工程 (提取均值、标准差、峰峰值)...")
    X_tr_feats = extract_statistical_features(X_tr)
    X_va_feats = extract_statistical_features(X_va)

    # ⚠️ 核心步骤 2：特征标准化 (这对基于距离的 KNN 极其重要)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_feats)
    X_va_scaled = scaler.transform(X_va_feats)

    print(f"📊 降维完成！数据形状从 {X_tr.shape} 变为了 {X_tr_scaled.shape}")

    # 训练 KNN 模型
    print("\n🔥 开始训练 K-Nearest Neighbors (KNN) 模型...")
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
    knn.fit(X_tr_scaled, Y_tr)
    print("✅ KNN 训练完成！")

    val_acc = knn.score(X_va_scaled, Y_va) * 100
    print(f"🌟 验证集准确率: {val_acc:.2f}%")

    # ========================================================
    # 测试集投票验证
    # ========================================================
    print("\n================ 正在进行测试集级联投票验证 ================")
    test_cor, test_tot, all_true, all_pred = 0, 0, [], []

    for f_path, true_label in ts_files:
        seq_400 = process_csv_to_physical_sequence(f_path)
        if seq_400 is None: continue

        test_windows = np.array(generate_sliding_windows(seq_400, window_size=100, step=20))

        # ⚠️ 测试集也需要提取特征并标准化
        test_feats = extract_statistical_features(test_windows)
        test_scaled = scaler.transform(test_feats)

        preds = knn.predict(test_scaled)

        # 多数投票
        pred_list = preds.tolist()
        final_pred = max(set(pred_list), key=pred_list.count)

        all_true.append(true_label)
        all_pred.append(final_pred)
        test_tot += 1
        if true_label == final_pred: test_cor += 1

    final_acc = 100 * test_cor / test_tot
    print(f"🎯 测试集最终准确率: {final_acc:.2f}%")

    cm = confusion_matrix(all_true, all_pred)
    names = [k for k, v in sorted(class_map.items(), key=lambda i: i[1])]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=names, yticklabels=names, annot_kws={"size": 14})
    plt.title(f'KNN (特征工程版) 测试集混淆矩阵 (Acc: {final_acc:.2f}%)', fontsize=16, fontweight='bold')
    plt.ylabel('真实材质', fontsize=14)
    plt.xlabel('预测材质', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix_knn_features.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    train_thesis_master_knn_features()