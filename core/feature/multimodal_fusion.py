# coding:utf-8
"""Cross-modal Transformer fusion: visual ⊕ spatial ⊕ motion → unified embedding.

Fuses three modalities using a lightweight Transformer encoder:
  Visual  [B, 512]  — RGB appearance features (MobileNetV3)
  Spatial [B, 256]  — hand skeleton topology (SpatialGCN)
  Motion  [B, 128]  — optical flow + keypoint deltas (MotionEncoder)

Output: [B, 256] unified per-frame embedding.
Parameters: ~0.8M
"""
import torch
import torch.nn as nn


class CrossModalFusion(nn.Module):
    """Transformer-based cross-modal fusion with learned modality embeddings.

    Input:  visual [B, 512], spatial [B, 256], motion [B, 128]
    Output: fused [B, 256]
    Params: ~0.8M
    """

    def __init__(self, visual_dim=512, spatial_dim=256, motion_dim=128,
                 d_model=256, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()

        # Modality-specific projections to d_model
        self.vis_proj = nn.Sequential(
            nn.Linear(visual_dim, d_model),
            nn.LayerNorm(d_model))
        self.spa_proj = nn.Sequential(
            nn.Linear(spatial_dim, d_model),
            nn.LayerNorm(d_model))
        self.mot_proj = nn.Sequential(
            nn.Linear(motion_dim, d_model),
            nn.LayerNorm(d_model))

        # Learnable modality type embeddings
        self.modality_embed = nn.Parameter(torch.randn(3, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, visual, spatial, motion):
        """Fuse three modalities with Transformer cross-attention.

        visual:  [B, 512] or zeros if unavailable
        spatial: [B, 256]  always available (from keypoints)
        motion:  [B, 128]  always available (from keypoint deltas)
        Returns: [B, 256]
        """
        B = spatial.shape[0]
        device = spatial.device

        # Project each modality to d_model
        v = self.vis_proj(visual).unsqueeze(1)   # [B, 1, 256]
        s = self.spa_proj(spatial).unsqueeze(1)   # [B, 1, 256]
        m = self.mot_proj(motion).unsqueeze(1)    # [B, 1, 256]

        # Stack as token sequence: [visual, spatial, motion]
        tokens = torch.cat([v, s, m], dim=1)      # [B, 3, 256]

        # Add modality type embeddings
        tokens = tokens + self.modality_embed.unsqueeze(0)

        # Cross-modal attention
        fused = self.transformer(tokens)           # [B, 3, 256]

        # Global pooling (mean over modality tokens)
        pooled = fused.mean(dim=1)                 # [B, 256]

        return self.output_proj(pooled)            # [B, 256]
