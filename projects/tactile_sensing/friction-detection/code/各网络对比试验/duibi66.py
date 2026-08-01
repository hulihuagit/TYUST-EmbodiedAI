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
from sklearn.metrics import classification_report, f1_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


# =====================================================================
# 📦 第一部分：统一的数据预处理与特征工程
# =====================================================================
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
        windows.append(np.transpose(win))
    return windows


def extract_statistical_features(windows):
    """专为 KNN 准备的特征工程"""
    features = []
    for win in windows:
        mean_vals, std_vals, ptp_vals = np.mean(win, axis=1), np.std(win, axis=1), np.ptp(win, axis=1)
        features.append(np.hstack((mean_vals, std_vals, ptp_vals)))
    return np.array(features)


class TactileDataset(Dataset):
    def __init__(self, X, Y): self.X, self.Y = X, Y

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx],
                                                                                                    dtype=torch.long)


# =====================================================================
# 🧠 第二部分：9 大核心算法模型库
# =====================================================================
# 1. Linear
class LinearClassifier(nn.Module):
    def __init__(self, in_channels=16, seq_len=100, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_channels * seq_len, num_classes)

    def forward(self, x): return self.fc(torch.flatten(x, 1))


# 2. VGG-1D
class VGG1D(nn.Module):
    def __init__(self, in_channels=16, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.avgpool = nn.AdaptiveAvgPool1d(6)
        self.classifier = nn.Sequential(nn.Linear(256 * 6, 1024), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(1024, num_classes))

    def forward(self, x): return self.classifier(torch.flatten(self.avgpool(self.features(x)), 1))


# 3. ResNet-1D
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                                          nn.BatchNorm1d(out_channels))

    def forward(self, x): return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))


class ResNet1D(nn.Module):
    def __init__(self, in_channels=16, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels, 64, 7, 1, 3, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
                                   nn.MaxPool1d(3, 2, 1))
        self.layer1 = self._make_layer(64, 64, 1)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_c, out_c, stride): return nn.Sequential(ResidualBlock1D(in_c, out_c, stride),
                                                                     ResidualBlock1D(out_c, out_c, 1))

    def forward(self, x): return self.fc(
        torch.flatten(self.avgpool(self.layer3(self.layer2(self.layer1(self.conv1(x))))), 1))


# 4. MultiScale CNN (Baseline)
class Baseline_MultiScale1DCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=16):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv1d(in_channels, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv1d(in_channels, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU())
        self.b3 = nn.Sequential(nn.Conv1d(in_channels, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(192, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, x): return self.classifier(
        self.pool(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)).squeeze(-1))


# 5. MultiScale CNN + SE Attention
class SEBlock1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(),
                                nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _ = x.size()
        y = self.fc(nn.AdaptiveAvgPool1d(1)(x).view(b, c)).view(b, c, 1)
        return x * y.expand_as(x)


class Thesis_MultiScale1DCNN_Attention(nn.Module):
    def __init__(self, num_classes=10, in_channels=16):
        super().__init__()
        self.b1, self.b2, self.b3 = nn.Sequential(nn.Conv1d(in_channels, 64, 3, padding=1), nn.BatchNorm1d(64),
                                                  nn.ReLU()), nn.Sequential(nn.Conv1d(in_channels, 64, 5, padding=2),
                                                                            nn.BatchNorm1d(64),
                                                                            nn.ReLU()), nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU())
        self.se = SEBlock1D(192)
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(192, num_classes))

    def forward(self, x): return self.classifier(
        nn.AdaptiveAvgPool1d(1)(self.se(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))).squeeze(-1))


# 6. Transformer
class TimeSeriesTransformer(nn.Module):
    def __init__(self, in_channels=16, seq_len=100, num_classes=10, d_model=64, nhead=8, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.1,
                                       batch_first=True), num_layers=num_layers)
        self.classifier = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(0.1),
                                        nn.Linear(128, num_classes))

    def forward(self, x): return self.classifier(
        self.transformer_encoder(self.input_proj(x.permute(0, 2, 1)) + self.pos_encoder).mean(dim=1))


# 7. Dual-Stream (时频双流)
class FreqStream(nn.Module):
    def __init__(self, seq_len=100, in_channels=16, out_features=64):
        super().__init__()
        self.fc_net = nn.Sequential(nn.Linear(in_channels * ((seq_len // 2) + 1), 256), nn.BatchNorm1d(256), nn.ReLU(),
                                    nn.Dropout(0.3), nn.Linear(256, out_features), nn.ReLU())

    def forward(self, x): return self.fc_net(torch.abs(torch.fft.rfft(x, dim=2)).view(x.size(0), -1))


class DualStreamFusionNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=16, seq_len=100):
        super().__init__()
        self.time_stream = Thesis_MultiScale1DCNN_Attention(out_features=128) if False else nn.Sequential(
            Thesis_MultiScale1DCNN_Attention(128, 16).b1, Thesis_MultiScale1DCNN_Attention(128, 16).b2)  # 简写，直接使用你原版的组合
        # 恢复你原版的双流结构：
        self.b1 = nn.Sequential(nn.Conv1d(16, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv1d(16, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU())
        self.b3 = nn.Sequential(nn.Conv1d(16, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU())
        self.se = SEBlock1D(192)
        self.time_fc = nn.Linear(192, 128)
        self.freq_stream = FreqStream(seq_len, 16, 64)
        self.classifier = nn.Sequential(nn.Linear(128 + 64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
                                        nn.Linear(128, num_classes))

    def forward(self, x):
        t_feat = self.time_fc(
            nn.AdaptiveAvgPool1d(1)(self.se(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))).squeeze(-1))
        return self.classifier(torch.cat((t_feat, self.freq_stream(x)), dim=1))


# 8. CNN + E-ELM (核心提取与集成)
class CNN_FeatureExtractor(nn.Module):
    def __init__(self, in_channels=16):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv1d(in_channels, 64, 3, padding=1), nn.BatchNorm1d(64), nn.GELU())
        self.b2 = nn.Sequential(nn.Conv1d(in_channels, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU())
        self.b3 = nn.Sequential(nn.Conv1d(in_channels, 64, 7, padding=3), nn.BatchNorm1d(64), nn.GELU())
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x): return self.pool(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)).squeeze(-1)


class ExtremeLearningMachine:
    def __init__(self, input_dim, hidden_dim, num_classes, device='cpu', C=1e-3):
        self.input_dim, self.hidden_dim, self.num_classes, self.device, self.C = input_dim, hidden_dim, num_classes, device, C
        self.W = torch.randn(input_dim, hidden_dim).to(device)
        self.b = torch.randn(hidden_dim).to(device)

    def fit(self, H_features, labels):
        H = torch.tanh(torch.matmul(H_features, self.W) + self.b)
        Y = torch.zeros(labels.size(0), self.num_classes).to(self.device).scatter_(1, labels.unsqueeze(1), 1.0)
        self.beta = torch.matmul(
            torch.matmul(torch.inverse(torch.matmul(H.t(), H) + self.C * torch.eye(self.hidden_dim).to(self.device)),
                         H.t()), Y)

    def predict(self, H_features): return torch.matmul(torch.tanh(torch.matmul(H_features, self.W) + self.b), self.beta)


class EnsembleELM:
    def __init__(self, num_models, input_dim, hidden_dim, num_classes, device='cpu'):
        self.models = [ExtremeLearningMachine(input_dim, hidden_dim, num_classes, device) for _ in range(num_models)]

    def fit(self, H_features, labels):
        for model in self.models: model.fit(H_features, labels)

    def predict(self, H_features):
        preds = torch.stack([torch.max(model.predict(H_features), 1)[1] for model in self.models], dim=0)
        return torch.mode(preds, dim=0)[0]


# =====================================================================
# 🛠️ 第三部分：公共训练与评估引擎
# =====================================================================
def get_model_params(model):
    """计算模型参数量 (Millions)"""
    if isinstance(model, KNeighborsClassifier): return 0.0
    if isinstance(model, dict) and 'elm' in model:  # ELM Special Case
        cnn_p = sum(p.numel() for p in model['extractor'].parameters())
        elm_p = (192 * 1500 + 1500 + 1500 * 10) * 10  # 10个ELM实例的权重
        return (cnn_p + elm_p) / 1e6
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def evaluate_model_pipeline(model_name, model, ts_files, device, scaler=None):
    """测试集投票验证与指标计算"""
    all_true, all_pred = [], []

    # 1. 计算 Latency (毫秒)
    dummy_input = torch.randn(1, 16, 100).to(device) if model_name != "KNN" else np.random.randn(1, 16, 100)
    start_time = time.time()
    if model_name == "KNN":
        dummy_feat = scaler.transform(extract_statistical_features(dummy_input))
        for _ in range(100): model.predict(dummy_feat)
        latency = ((time.time() - start_time) / 100) * 1000
    elif model_name == "ELM":
        with torch.no_grad():
            for _ in range(100): model['elm'].predict(model['extractor'](dummy_input))
        latency = ((time.time() - start_time) / 100) * 1000
    else:
        model.eval()
        with torch.no_grad():
            for _ in range(100): model(dummy_input)
        latency = ((time.time() - start_time) / 100) * 1000

    # 2. 级联投票验证 (Accuracy, F1, Precision)
    for f_path, true_label in ts_files:
        seq_400 = process_csv_to_physical_sequence(f_path)
        if seq_400 is None: continue
        windows = generate_sliding_windows(seq_400, window_size=100, step=20)

        if model_name == "KNN":
            feats = scaler.transform(extract_statistical_features(np.array(windows)))
            preds = model.predict(feats).tolist()
        elif model_name == "ELM":
            with torch.no_grad():
                feats = model['extractor'](torch.tensor(np.array(windows), dtype=torch.float32).to(device))
                preds = model['elm'].predict(feats).cpu().tolist()
        else:
            with torch.no_grad():
                out = model(torch.tensor(np.array(windows), dtype=torch.float32).to(device))
                preds = torch.max(out, 1)[1].cpu().tolist()

        all_true.append(true_label)
        all_pred.append(max(set(preds), key=preds.count))  # 多数投票

    acc = np.mean(np.array(all_true) == np.array(all_pred))
    mac_f1 = f1_score(all_true, all_pred, average='macro', zero_division=0)
    mac_prec = precision_score(all_true, all_pred, average='macro', zero_division=0)

    return acc, mac_f1, mac_prec, latency


# =====================================================================
# 👑 第四部分：核心实验控制大循环
# =====================================================================
def run_benchmark():
    data_dir = r"D:\pycharmproject\pythonProject\dataset"
    class_map = {"zaojin": 0, "pige": 1, "A4": 2, "shazhi_40": 3, "shazhi_120": 4, "shazhi_240": 5, "shazhi_400": 6,
                 "shazhi_600": 7, "shazhi_800": 8, "shazhi_1000": 9}
    file_list = [(f, lab) for name, lab in class_map.items() for f in glob.glob(os.path.join(data_dir, name, "*.csv"))]
    if not file_list: return print("❌ 未找到数据！")
    labels = [item[1] for item in file_list]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🌟 想要测试的模型列表 (全家福)
    models_to_test = [
        "Linear", "KNN", "VGG1D", "ResNet1D",
        "Baseline_MultiScale", "Attention_MultiScale",
        "ELM", "Transformer", "DualStream"
    ]
    num_runs = 5  # 测稳定性，建议设为 3 到 5
    results = {}

    print(f" 开超级自动化对比实验！共 {len(models_to_test)} 个模型，每个重复 {num_runs} 次...")

    for model_name in models_to_test:
        print(f"\n" + "=" * 50 + f"\n 评估模型: {model_name}\n" + "=" * 50)
        m_acc, m_f1, m_prec, m_lat, m_param = [], [], [], [], 0

        for run in range(num_runs):
            print(f"   正在运行第 {run + 1}/{num_runs} 次实验...")
            tr_files, te_files, tr_labels, te_labels = train_test_split(file_list, labels, test_size=0.3,
                                                                        random_state=run, stratify=labels)
            va_files, ts_files = train_test_split(te_files, test_size=0.33, random_state=run, stratify=te_labels)

            # 数据提取
            X_tr, Y_tr = [], []
            for f, l in tr_files:
                s = process_csv_to_physical_sequence(f)
                if s is not None:
                    w = generate_sliding_windows(s, 100, 15)
                    X_tr.extend(w);
                    Y_tr.extend([l] * len(w))
            X_tr, Y_tr = np.array(X_tr), np.array(Y_tr)

            # ----- KNN 专属逻辑 -----
            if model_name == "KNN":
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(extract_statistical_features(X_tr))
                model = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
                model.fit(X_tr_scaled, Y_tr)
                acc, f1, prec, lat = evaluate_model_pipeline(model_name, model, ts_files, device, scaler)

            # ----- ELM 专属逻辑 -----
            elif model_name == "ELM":
                extractor = CNN_FeatureExtractor(16).to(device)
                head = nn.Linear(192, 10).to(device)
                optimizer = optim.Adam(list(extractor.parameters()) + list(head.parameters()), lr=0.001)
                train_loader = DataLoader(TactileDataset(X_tr, Y_tr), batch_size=128, shuffle=True)
                for _ in range(20):  # 预热 5 Epochs 提速实验
                    extractor.train()
                    for inp, lab in train_loader:
                        optimizer.zero_grad()
                        nn.CrossEntropyLoss()(head(extractor(inp.to(device))), lab.to(device)).backward()
                        optimizer.step()
                extractor.eval()
                H_train = []
                with torch.no_grad():
                    for inp, _ in DataLoader(TactileDataset(X_tr, Y_tr), batch_size=512): H_train.append(
                        extractor(inp.to(device)))
                elm = EnsembleELM(10, 192, 1500, 10, device)
                elm.fit(torch.cat(H_train, dim=0), torch.tensor(Y_tr).to(device))
                model_pack = {'extractor': extractor, 'elm': elm}
                acc, f1, prec, lat = evaluate_model_pipeline(model_name, model_pack, ts_files, device)

            # ----- 标准深度学习模型 -----
            else:
                if model_name == "Linear":
                    model = LinearClassifier(16, 100, 10).to(device)
                elif model_name == "VGG1D":
                    model = VGG1D(16, 10).to(device)
                elif model_name == "ResNet1D":
                    model = ResNet1D(16, 10).to(device)
                elif model_name == "Baseline_MultiScale":
                    model = Baseline_MultiScale1DCNN(10, 16).to(device)
                elif model_name == "Attention_MultiScale":
                    model = Thesis_MultiScale1DCNN_Attention(10, 16).to(device)
                elif model_name == "Transformer":
                    model = TimeSeriesTransformer(16, 100, 10).to(device)
                elif model_name == "DualStream":
                    model = DualStreamFusionNet(10, 16, 100).to(device)

                optimizer = optim.Adam(model.parameters(), lr=0.0005)
                train_loader = DataLoader(TactileDataset(X_tr, Y_tr), batch_size=64, shuffle=True)
                model.train()
                for _ in range(50):  # 为节约基准测试时间，统一跑 15 轮 (实际论文可改 40 轮)
                    for inp, lab in train_loader:
                        optimizer.zero_grad()
                        nn.CrossEntropyLoss()(model(inp.to(device)), lab.to(device)).backward()
                        optimizer.step()
                acc, f1, prec, lat = evaluate_model_pipeline(model_name, model, ts_files, device)

            if run == 0: m_param = get_model_params(model if model_name != "ELM" else model_pack)
            m_acc.append(acc * 100);
            m_f1.append(f1);
            m_prec.append(prec);
            m_lat.append(lat)

        # 汇总当前模型成绩
        results[model_name] = {
            "Acc (%)": f"{np.mean(m_acc):.2f} ± {np.std(m_acc):.2f}",
            "Macro-F1": f"{np.mean(m_f1):.4f} ± {np.std(m_f1):.4f}",
            "Precision": f"{np.mean(m_prec):.4f} ± {np.std(m_prec):.4f}",
            "Latency (ms)": f"{np.mean(m_lat):.2f}",
            "Params (M)": f"{m_param:.3f}"
        }

    # ========================================================
    # 对比表格
    # ========================================================
    print("\n\n" + "★" * 80)
    print("触觉传感器材质识别 —— 算法基准测试综合报告")
    print("★" * 80)
    df_results = pd.DataFrame(results).T
    print(df_results.to_markdown())
    print("★" * 80)


if __name__ == "__main__":
    run_benchmark()