# coding:utf-8
"""BimanualCrossAttention: left↔right hand cross-attention module.

Lightweight multi-head cross-attention between left and right hand features.
Design follows Cross Attentive Multi-Cue Fusion (Park et al., IEEE TPAMI 2025) [24].

Single-hand data (missing hand = zeros):
  cross-attn(Q_left, K=V=zeros) ≈ 0 → residual → output ≈ input (identity)
Dual-hand data (both hands present):
  cross-attn learns meaningful inter-hand spatial relationships.

Params: ~0.3M
"""
import torch
import torch.nn as nn


class BimanualCrossAttention(nn.Module):
    """Cross-attention between left and right hand fused features.

    Each hand attends to the other's features, then adds residual.
    When one hand is absent (zeros), the module gracefully degrades to
    identity (residual connection preserves the present hand's features).

    Input:  left [B, D], right [B, D]  — D=256 (or D=128 for motion slot)
    Output: left [B, D], right [B, D]  — same shape, cross-enhanced
    """

    def __init__(self, dim=256, heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True)
        self.norm_left = nn.LayerNorm(dim)
        self.norm_right = nn.LayerNorm(dim)

    def forward(self, left_feat, right_feat):
        """Cross-attention with residual connection.

        Args:
            left_feat:  [B, D] or [B*T, D]  left hand features
            right_feat: [B, D] or [B*T, D]  right hand features

        Returns:
            (left_out, right_out): both [B, D] or [B*T, D]
        """
        # Add seq dim for MHA: [B, D] → [B, 1, D]
        l = left_feat.unsqueeze(1)
        r = right_feat.unsqueeze(1)

        # Left attends to Right
        l_ctx, _ = self.cross_attn(query=l, key=r, value=r)
        l_out = self.norm_left((l + l_ctx).squeeze(1))

        # Right attends to Left
        r_ctx, _ = self.cross_attn(query=r, key=l, value=l)
        r_out = self.norm_right((r + r_ctx).squeeze(1))

        return l_out, r_out
