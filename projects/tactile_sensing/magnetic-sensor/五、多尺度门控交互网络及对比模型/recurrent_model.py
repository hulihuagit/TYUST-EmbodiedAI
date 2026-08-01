import torch
import torch.nn as nn


class RecurrentTimeSeriesClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        seq_len=1024,
        in_channels=1,
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
        bidirectional=False,
        cell_type='lstm',
        task_mode='multitask',
    ):
        super().__init__()
        self.task_mode = task_mode
        self.bidirectional = bool(bidirectional)
        self.input_proj = nn.Linear(in_channels, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)

        recurrent_dropout = dropout if num_layers > 1 else 0.0
        if cell_type == 'lstm':
            recurrent_cls = nn.LSTM
        elif cell_type == 'gru':
            recurrent_cls = nn.GRU
        else:
            raise ValueError(f'Unsupported recurrent cell type: {cell_type}')

        self.encoder = recurrent_cls(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        feat_dim = hidden_size * (2 if self.bidirectional else 1)
        self.dropout = nn.Dropout(dropout)

        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        if self.task_mode == 'multitask':
            self.reg_head = nn.Sequential(
                nn.Linear(feat_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
        else:
            self.reg_head = None

    def forward(self, x):
        # x: (B, C, L) -> (B, L, C)
        seq = x.transpose(1, 2)
        seq = self.input_proj(seq)
        seq = self.input_norm(seq)
        seq, _ = self.encoder(seq)
        pooled = self.dropout(seq.mean(dim=1))
        cls_out = self.cls_head(pooled)
        reg_out = self.reg_head(pooled) if self.reg_head is not None else None
        return cls_out, reg_out


class LSTMClassifier(RecurrentTimeSeriesClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, cell_type='lstm', bidirectional=False, **kwargs)


class GRUClassifier(RecurrentTimeSeriesClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, cell_type='gru', bidirectional=False, **kwargs)


class BiLSTMClassifier(RecurrentTimeSeriesClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, cell_type='lstm', bidirectional=True, **kwargs)
