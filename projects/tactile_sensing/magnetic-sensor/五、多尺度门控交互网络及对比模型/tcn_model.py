import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.drop(self.act(self.bn1(self.conv1(x))))
        out = self.drop(self.act(self.bn2(self.conv2(out))))
        return out + res


class TCNClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        seq_len=1024,
        in_channels=1,
        hidden_channels=(32, 64, 128),
        kernel_size=5,
        dropout=0.1,
        task_mode='multitask',
    ):
        super().__init__()
        blocks = []
        prev = in_channels
        for idx, ch in enumerate(hidden_channels):
            dilation = 2 ** idx
            blocks.append(TemporalBlock(prev, ch, kernel_size=kernel_size, dilation=dilation, dropout=dropout))
            prev = ch
        self.backbone = nn.Sequential(*blocks)
        self.task_mode = task_mode

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

        if self.task_mode == 'multitask':
            self.reg_head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(hidden_channels[-1], 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )
        else:
            self.reg_head = None

    def forward(self, x):
        feat = self.backbone(x)
        cls_out = self.cls_head(feat)
        reg_out = self.reg_head(feat) if self.reg_head is not None else None
        return cls_out, reg_out
