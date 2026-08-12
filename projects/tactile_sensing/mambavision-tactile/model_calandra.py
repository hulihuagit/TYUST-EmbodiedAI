import torch
import torch.nn as nn
from mamba_ssm import Mamba
from modeling_mambavision import MambaVision


class MambaVisionEncoderWrapper(nn.Module):
    def __init__(self, in_chans=3):
        super().__init__()
        dim = 80
        in_dim = 32
        depths = [1, 3, 8, 4]
        num_heads = [2, 4, 8, 16]
        window_size = [8, 8, 14, 7]

        self.model = MambaVision(
            dim=dim,
            in_dim=in_dim,
            depths=depths,
            window_size=window_size,
            mlp_ratio=4.0,
            num_heads=num_heads,
            drop_path_rate=0.2,
            in_chans=in_chans,
            num_classes=1000,
        )
        self.out_dim_stage3 = int(dim * 2 ** (len(depths) - 2))
        self.out_dim = int(dim * 2 ** (len(depths) - 1))

    def load_pretrained(self, ckpt_path, strict=False):
        self.model._load_state_dict(pretrained=ckpt_path, strict=strict)

    def forward_features(self, x):
        global_feat, _ = self.model.forward_features(x)
        return global_feat

    def forward_features_with_spatial(self, x):
        global_feat, outs = self.model.forward_features(x)
        spatial_map = outs[-1]
        if spatial_map.dim() == 4:
            spatial_feat = spatial_map.flatten(2).transpose(1, 2)
        else:
            spatial_feat = spatial_map
        return global_feat, spatial_feat

    def forward_features_multi(self, x):
        global_feat, outs = self.model.forward_features(x)
        stage3_map, stage4_map = outs[-2], outs[-1]

        if stage3_map.dim() == 4:
            stage3 = stage3_map.flatten(2).transpose(1, 2)
        else:
            stage3 = stage3_map

        if stage4_map.dim() == 4:
            stage4 = stage4_map.flatten(2).transpose(1, 2)
        else:
            stage4 = stage4_map

        return global_feat, stage3, stage4


class TactileGuidedSpatialAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

    def forward(self, v_spatial, t_global):
        B, N, D = v_spatial.shape

        Q = self.q_proj(t_global).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(v_spatial).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(v_spatial).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale
        attn_weights = attn_scores.softmax(dim=-1)

        v_attended = (attn_weights @ V).transpose(1, 2).reshape(B, 1, D).squeeze(1)
        v_attended = self.out_proj(v_attended)

        spatial_heatmap = attn_weights.mean(dim=1).squeeze(1)
        return v_attended, spatial_heatmap


class HierarchicalTactileAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.stage3_attn = TactileGuidedSpatialAttention(d_model, num_heads, dropout)
        self.stage4_attn = TactileGuidedSpatialAttention(d_model, num_heads, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, v3, v4, tactile):
        v3_feat, heat3 = self.stage3_attn(v3, tactile)
        v4_feat, heat4 = self.stage4_attn(v4, tactile)
        fused = self.fusion(torch.cat([v3_feat, v4_feat], dim=-1))
        return fused, heat3, heat4


class DynamicCrossModalFusion(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dynamic_gate = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, v_feat, t_feat, v_global_ctx, t_global_ctx):
        gate = self.dynamic_gate(
            torch.cat([v_feat, t_feat, v_global_ctx, t_global_ctx], dim=-1)
        )
        fused = gate * v_feat + (1 - gate) * t_feat
        return fused, gate


class TemporalAttentionPooling(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_weights = torch.softmax(self.attn_net(x), dim=1)
        pooled = (x * attn_weights).sum(dim=1)
        return self.dropout(pooled), attn_weights


class DynamicTwoStreamTemporalFusion(nn.Module):
    def __init__(self, d_model, d_state=16, dropout=0.2):
        super().__init__()
        self.vision_temporal_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
        self.vision_norm = nn.LayerNorm(d_model)

        self.tactile_temporal_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
        self.tactile_norm = nn.LayerNorm(d_model)

        self.cross_modal_fusion = DynamicCrossModalFusion(d_model, dropout)
        self.temporal_pooling = TemporalAttentionPooling(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def compute_global_context(self, seq):
        if seq.size(1) == 1:
            return seq.squeeze(1)
        return seq.mean(dim=1) + seq.std(dim=1, unbiased=False)

    def forward(self, v_seq, t_seq):
        B, T, D = v_seq.shape

        v_temporal = v_seq + self.dropout(
            self.vision_temporal_mamba(self.vision_norm(v_seq))
        )
        t_temporal = t_seq + self.dropout(
            self.tactile_temporal_mamba(self.tactile_norm(t_seq))
        )

        v_global_ctx = self.compute_global_context(v_temporal)
        t_global_ctx = self.compute_global_context(t_temporal)

        fused_list, gate_list = [], []
        for i in range(T):
            fused_t, gate_t = self.cross_modal_fusion(
                v_temporal[:, i, :],
                t_temporal[:, i, :],
                v_global_ctx,
                t_global_ctx
            )
            fused_list.append(fused_t)
            gate_list.append(gate_t)

        fused_seq = torch.stack(fused_list, dim=1)
        gate_seq = torch.stack(gate_list, dim=1)

        pooled, temporal_attn = self.temporal_pooling(fused_seq)
        return pooled, {
            "temporal_attention": temporal_attn,
            "dynamic_gates": gate_seq
        }


class SpatioTemporalDynamicClassifier(nn.Module):

    def __init__(
        self,
        num_classes=2,
        d_model=256,
        d_state=16,
        dropout=0.2,
        pretrained_vision_path=None,
        hierarchical=False,
        no_tgsa=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.hierarchical = hierarchical
        self.no_tgsa = no_tgsa

        self.vision_encoder = MambaVisionEncoderWrapper(in_chans=3)
        self.tactile_encoder = MambaVisionEncoderWrapper(in_chans=3)

        if pretrained_vision_path is not None:
            self.vision_encoder.load_pretrained(pretrained_vision_path, strict=False)
            self.tactile_encoder.load_pretrained(pretrained_vision_path, strict=False)

        encoder_out_dim = self.vision_encoder.out_dim
        encoder_out_dim_s3 = self.vision_encoder.out_dim_stage3


        self.tac_fusion = nn.Sequential(
            nn.Linear(encoder_out_dim * 2, encoder_out_dim),
            nn.LayerNorm(encoder_out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.v_spatial_proj = nn.Sequential(
            nn.Linear(encoder_out_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.t_proj = nn.Sequential(
            nn.Linear(encoder_out_dim, d_model),
            nn.LayerNorm(d_model)
        )

        if not no_tgsa:
            if hierarchical:
                self.v3_spatial_proj = nn.Sequential(
                    nn.Linear(encoder_out_dim_s3, d_model),
                    nn.LayerNorm(d_model)
                )
                self.spatial_attention = HierarchicalTactileAttention(
                    d_model, num_heads=8, dropout=dropout
                )
            else:
                self.spatial_attention = TactileGuidedSpatialAttention(
                    d_model, num_heads=8, dropout=dropout
                )

        self.temporal_fusion = DynamicTwoStreamTemporalFusion(
            d_model=d_model, d_state=d_state, dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, rgb, tac, return_analysis=False):
        if rgb.ndim == 4:
            rgb = rgb.unsqueeze(1)
            tac = tac.unsqueeze(1)

        assert rgb.ndim == 5, "rgb 需要 [B, T, 3, H, W]"
        assert tac.ndim == 5, "tac 需要 [B, T, 6, H, W]"

        b, t_len, _, h, w = rgb.shape
        rgb_flat = rgb.reshape(b * t_len, 3, h, w)

        tac_flat = tac.reshape(b * t_len, 6, h, w)
        gelA_flat = tac_flat[:, :3, :, :]
        gelB_flat = tac_flat[:, 3:, :, :]


        featA = self.tactile_encoder.forward_features(gelA_flat)
        featB = self.tactile_encoder.forward_features(gelB_flat)

        t_global_flat = self.tac_fusion(torch.cat([featA, featB], dim=-1))
        t_global_flat = self.t_proj(t_global_flat)
        t_seq = t_global_flat.reshape(b, t_len, -1)


        if self.no_tgsa:
            v_global_flat, v_spatial_flat = self.vision_encoder.forward_features_with_spatial(rgb_flat)
            v_attended_flat = self.v_spatial_proj(v_global_flat)

            n_stage4 = v_spatial_flat.shape[1]

            try:
                _, v3_flat, _ = self.vision_encoder.forward_features_multi(rgb_flat)
                n_stage3 = v3_flat.shape[1]
            except Exception:
                n_stage3 = n_stage4

            spatial_heatmaps = {
                "stage3": torch.ones(b, t_len, n_stage3, device=rgb.device) / n_stage3,
                "stage4": torch.ones(b, t_len, n_stage4, device=rgb.device) / n_stage4,
                "uniform": True,
            }

        elif self.hierarchical:
            _, v3_flat, v4_flat = self.vision_encoder.forward_features_multi(rgb_flat)

            v3_flat = self.v3_spatial_proj(v3_flat)
            v4_flat = self.v_spatial_proj(v4_flat)

            v_attended_flat, heat3_flat, heat4_flat = self.spatial_attention(
                v3_flat, v4_flat, t_global_flat
            )

            spatial_heatmaps = {
                "stage3": heat3_flat.reshape(b, t_len, -1),
                "stage4": heat4_flat.reshape(b, t_len, -1),
                "uniform": False,
            }

        else:
            _, v_spatial_flat = self.vision_encoder.forward_features_with_spatial(rgb_flat)
            v_spatial_flat = self.v_spatial_proj(v_spatial_flat)

            v_attended_flat, spatial_heatmaps_flat = self.spatial_attention(
                v_spatial_flat, t_global_flat
            )

            spatial_heatmaps = {
                "stage3": spatial_heatmaps_flat.reshape(b, t_len, -1),
                "stage4": spatial_heatmaps_flat.reshape(b, t_len, -1),
                "uniform": False,
            }

        v_seq = v_attended_flat.reshape(b, t_len, -1)

        fused_pooled, temp_analysis = self.temporal_fusion(v_seq, t_seq)
        logits = self.classifier(fused_pooled)

        if return_analysis:
            temp_analysis["spatial_heatmaps"] = spatial_heatmaps
            return logits, temp_analysis

        return logits
