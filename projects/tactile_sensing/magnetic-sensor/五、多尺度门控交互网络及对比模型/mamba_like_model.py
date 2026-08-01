import torch
import torch.nn as nn


class MambaLikeBlock(nn.Module):
    def __init__(self, d_model, expansion=2, kernel_size=5, dropout=0.1):
        super().__init__()
        hidden = d_model * expansion
        self.in_proj = nn.Conv1d(d_model, hidden * 2, kernel_size=1)
        self.dw_conv = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden,
        )
        self.act = nn.SiLU()
        self.out_proj = nn.Conv1d(hidden, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x):
        residual = x
        proj = self.in_proj(x)
        u, gate = proj.chunk(2, dim=1)
        u = self.dw_conv(u)
        u = self.act(u)
        u = u * torch.sigmoid(gate)
        y = self.out_proj(u)
        y = self.dropout(y)
        return self.norm(y + residual)


class MSGIBlock(nn.Module):
    def __init__(
        self,
        d_model,
        expansion=2,
        kernel_sizes=(3, 7),
        dropout=0.1,
        use_gating=True,
        use_residual=True,
        use_norm=True,
        use_residual_norm=None,
    ):
        super().__init__()
        hidden = d_model * expansion
        if use_residual_norm is not None:
            use_residual = bool(use_residual_norm)
            use_norm = bool(use_residual_norm)
        self.use_gating = bool(use_gating)
        self.use_residual = bool(use_residual)
        self.use_norm = bool(use_norm)
        self.in_proj = nn.Conv1d(d_model, hidden * 2, kernel_size=1)
        self.dw_convs = nn.ModuleList([
            nn.Conv1d(
                hidden,
                hidden,
                kernel_size=k,
                padding=k // 2,
                groups=hidden,
            )
            for k in kernel_sizes
        ])
        self.mix = nn.Conv1d(hidden * len(kernel_sizes), hidden, kernel_size=1)
        self.act = nn.SiLU()
        self.out_proj = nn.Conv1d(hidden, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x):
        residual = x
        proj = self.in_proj(x)
        u, gate = proj.chunk(2, dim=1)
        multi_scale = [conv(u) for conv in self.dw_convs]
        u = torch.cat(multi_scale, dim=1)
        u = self.mix(u)
        u = self.act(u)
        if self.use_gating:
            u = u * torch.sigmoid(gate)
        y = self.out_proj(u)
        y = self.dropout(y)
        if self.use_residual:
            y = y + residual
        if self.use_norm:
            y = self.norm(y)
        return y


class MambaLikeClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        seq_len=1024,
        in_channels=1,
        d_model=128,
        num_layers=4,
        dropout=0.1,
        task_mode='multitask',
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
        )

        self.layers = nn.Sequential(*[
            MambaLikeBlock(d_model=d_model, expansion=2, kernel_size=5, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.task_mode = task_mode

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        if self.task_mode == 'multitask':
            self.reg_head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(d_model, 64),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
        else:
            self.reg_head = None

    def forward(self, x):
        feat = self.stem(x)
        feat = self.layers(feat)
        cls_out = self.cls_head(feat)
        reg_out = self.reg_head(feat) if self.reg_head is not None else None
        return cls_out, reg_out


class MSGINetClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        seq_len=1024,
        in_channels=1,
        d_model=128,
        num_layers=4,
        dropout=0.1,
        task_mode='multitask',
        kernel_sizes=(3, 7),
        use_gating=True,
        use_residual=True,
        use_norm=True,
        use_residual_norm=None,
        stem_type='conv7',
        pool_type='avg',
    ):
        super().__init__()
        if use_residual_norm is not None:
            use_residual = bool(use_residual_norm)
            use_norm = bool(use_residual_norm)
        if stem_type == 'conv7':
            stem_kernel_size = 7
        elif stem_type == 'pointwise':
            stem_kernel_size = 1
        else:
            raise ValueError(f'Unknown MSGI-Net stem_type: {stem_type}')
        self.pool_type = pool_type
        if self.pool_type not in {'avg', 'max', 'last'}:
            raise ValueError(f'Unknown MSGI-Net pool_type: {pool_type}')

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=stem_kernel_size, padding=stem_kernel_size // 2),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
        )
        self.layers = nn.Sequential(*[
            MSGIBlock(
                d_model=d_model,
                expansion=2,
                kernel_sizes=kernel_sizes,
                dropout=dropout,
                use_gating=use_gating,
                use_residual=use_residual,
                use_norm=use_norm,
            )
            for _ in range(num_layers)
        ])
        self.task_mode = task_mode
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        if self.task_mode == 'multitask':
            self.reg_head = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
        else:
            self.reg_head = None

    def _pool(self, feat):
        if self.pool_type == 'avg':
            return feat.mean(dim=-1)
        if self.pool_type == 'max':
            return feat.amax(dim=-1)
        return feat[..., -1]

    def forward(self, x):
        feat = self.stem(x)
        feat = self.layers(feat)
        pooled = self._pool(feat)
        cls_out = self.cls_head(pooled)
        reg_out = self.reg_head(pooled) if self.reg_head is not None else None
        return cls_out, reg_out


# Backward-compatible aliases for existing imports and result files.
MultiScaleMambaBlock = MSGIBlock
MultiScaleMambaLikeClassifier = MSGINetClassifier
