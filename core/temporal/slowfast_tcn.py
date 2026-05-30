# coding:utf-8
"""SlowFast Temporal Convolutional Network for sign language recognition.

Two-pathway design:
  Slow path — deep, large kernels, dilated convolutions for long-range semantics
  Fast path — shallow, small kernels for fine-grained motion details
  Lateral fusion — fast→slow at each stage

Reference: config/model_config.yaml slowfast_tcn section.
Total parameters: ~1.5M (well within 10M budget).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    """Single TCN residual block with dilation and optional downsampling."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation))
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation))
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.conv2(out)
        res = x if self.downsample is None else self.downsample(x)
        # Trim to match if padding produced different lengths
        if out.shape[2] != res.shape[2]:
            min_len = min(out.shape[2], res.shape[2])
            out, res = out[:, :, :min_len], res[:, :, :min_len]
        return self.relu(out + res)


class SlowFastTCN(nn.Module):
    """SlowFast TCN for gesture sequence classification.

    Input:  [B, T, 256]  fused multimodal features per frame
    Output: [B, T, C]    frame-wise class logits (C=num_classes)
    Params: ~1.5M
    """

    def __init__(self, input_dim=256,
                 slow_channels=(64, 128, 128),
                 fast_channels=(32, 32, 64, 64),
                 slow_kernel=7, fast_kernel=3,
                 slow_dilations=(1, 2, 4),
                 fast_dilations=(1, 1, 2, 2),
                 num_classes=200, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # --- Slow pathway (captures long-range semantics) ---
        slow_layers = []
        ch_in = input_dim
        for i, ch_out in enumerate(slow_channels):
            d = slow_dilations[i] if i < len(slow_dilations) else 1
            slow_layers.append(TCNBlock(ch_in, ch_out, slow_kernel, d, dropout))
            ch_in = ch_out
        self.slow_path = nn.Sequential(*slow_layers)

        # --- Fast pathway (captures fine-grained motion) ---
        fast_layers = []
        ch_in = input_dim
        self.fast_fusions = nn.ModuleList()
        slow_ch_idx = 0
        for i, ch_out in enumerate(fast_channels):
            d = fast_dilations[i] if i < len(fast_dilations) else 1
            fast_layers.append(TCNBlock(ch_in, ch_out, fast_kernel, d, dropout))
            # Lateral connection: fast → slow at matching resolution points
            if i < len(slow_channels):
                self.fast_fusions.append(
                    nn.Conv1d(ch_out, slow_channels[i], 1))
            ch_in = ch_out
        self.fast_path = nn.Sequential(*fast_layers)

        # --- Output head ---
        final_dim = slow_channels[-1]
        self.output_proj = nn.Sequential(
            nn.Conv1d(final_dim, final_dim, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(final_dim, num_classes, 1),
        )

    def forward(self, x):
        """x: [B, T, C] → logits: [B, T, num_classes]"""
        x_t = x.transpose(1, 2)  # [B, C, T]

        slow = x_t
        fast = x_t

        # Interleave slow/fast processing with lateral fusion at each stage
        max_stages = max(len(self.slow_path), len(self.fast_path))
        for i in range(max_stages):
            if i < len(self.slow_path):
                slow = self.slow_path[i](slow)
            if i < len(self.fast_path):
                fast = self.fast_path[i](fast)
            if i < len(self.fast_fusions):
                lateral = self.fast_fusions[i](fast)
                if lateral.shape[2] != slow.shape[2]:
                    lateral = F.interpolate(lateral, size=slow.shape[2], mode='nearest')
                slow = slow + lateral

        logits = self.output_proj(slow)  # [B, num_classes, T]
        return logits.transpose(1, 2)     # [B, T, num_classes]
