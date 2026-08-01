import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerTimeSeriesClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        seq_len=1024,
        in_channels=1,
        d_model=128,
        nhead=8,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        task_mode='multitask',
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, d_model, kernel_size=1)
        self.pos = PositionalEncoding(d_model=d_model, max_len=max(seq_len, 1024))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.task_mode = task_mode

        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        if self.task_mode == 'multitask':
            self.reg_head = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
        else:
            self.reg_head = None

    def forward(self, x):
        feat = self.input_proj(x)
        feat = feat.transpose(1, 2)
        feat = self.pos(feat)
        feat = self.encoder(feat)
        feat = self.norm(feat)
        pooled = feat.mean(dim=1)

        cls_out = self.cls_head(pooled)
        reg_out = self.reg_head(pooled) if self.reg_head is not None else None
        return cls_out, reg_out
