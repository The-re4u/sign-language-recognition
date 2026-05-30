# coding:utf-8
"""GCAR: Graph Convolutional Attention Recurrent network (baseline model).

Lightweight two-stream separable TCN with channel attention.
Target: ~0.69M parameters for fair comparison with SlowFastTCN.

Architecture:
  Stream A (sep-TCN, deep) — captures long-range dependencies
  Stream B (sep-TCN, shallow) — fine-grained temporal details
  Channel attention (SE block) — adaptive feature reweighting
  Late fusion — concatenate + projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, reduced, 1),
            nn.ReLU(),
            nn.Conv1d(reduced, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: [B, C, T] → [B, C, T]"""
        scale = self.fc(x)
        return x * scale


class SepTCNBlock(nn.Module):
    """Depthwise-separable TCN residual block with channel attention."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        # Depthwise
        self.dw_conv = nn.utils.weight_norm(
            nn.Conv1d(in_ch, in_ch, kernel_size, padding=pad, dilation=dilation, groups=in_ch))
        # Pointwise
        self.pw_conv = nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, 1))
        self.attention = ChannelAttention(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.dw_conv(x)
        out = self.pw_conv(out)
        out = self.attention(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        if out.shape[2] != res.shape[2]:
            min_len = min(out.shape[2], res.shape[2])
            out, res = out[:, :, :min_len], res[:, :, :min_len]
        return self.relu(out + res)


class GCAR(nn.Module):
    """Graph Convolutional Attention Recurrent network.

    Input:  [B, T, C] fused multimodal features
    Output: [B, T, num_classes] frame-wise logits

    Two-stream sep-TCN with channel attention:
      Stream A: deep, large kernels, dilated (semantics)
      Stream B: shallow, small kernels (detail)
    """

    def __init__(self, input_dim=256,
                 stream_a_channels=(64, 128, 256),
                 stream_b_channels=(32, 64, 128),
                 stream_a_kernel=7, stream_b_kernel=3,
                 stream_a_dilations=(1, 2, 4),
                 stream_b_dilations=(1, 1, 2),
                 num_classes=10, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim

        # Stream A: deep semantic pathway
        a_layers = []
        ch_in = input_dim
        for i, ch_out in enumerate(stream_a_channels):
            d = stream_a_dilations[i] if i < len(stream_a_dilations) else 1
            a_layers.append(SepTCNBlock(ch_in, ch_out, stream_a_kernel, d, dropout))
            ch_in = ch_out
        self.stream_a = nn.Sequential(*a_layers)

        # Stream B: shallow detail pathway
        b_layers = []
        ch_in = input_dim
        for i, ch_out in enumerate(stream_b_channels):
            d = stream_b_dilations[i] if i < len(stream_b_dilations) else 1
            b_layers.append(SepTCNBlock(ch_in, ch_out, stream_b_kernel, d, dropout))
            ch_in = ch_out
        self.stream_b = nn.Sequential(*b_layers)

        # Fusion
        fusion_dim = stream_a_channels[-1] + stream_b_channels[-1]
        self.fusion_attention = ChannelAttention(fusion_dim)
        self.output_proj = nn.Sequential(
            nn.Conv1d(fusion_dim, fusion_dim // 2, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(fusion_dim // 2, num_classes, 1),
        )

    def forward(self, x):
        """x: [B, T, C] → [B, T, num_classes]"""
        x_t = x.transpose(1, 2)  # [B, C, T]

        a_out = self.stream_a(x_t)
        b_out = self.stream_b(x_t)

        # Align time dimensions
        if a_out.shape[2] != b_out.shape[2]:
            min_t = min(a_out.shape[2], b_out.shape[2])
            a_out = a_out[:, :, :min_t]
            b_out = b_out[:, :, :min_t]

        fused = torch.cat([a_out, b_out], dim=1)
        fused = self.fusion_attention(fused)
        logits = self.output_proj(fused)  # [B, num_classes, T]
        return logits.transpose(1, 2)     # [B, T, num_classes]
