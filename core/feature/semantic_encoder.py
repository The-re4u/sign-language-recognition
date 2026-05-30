# coding:utf-8
"""Semantic modality encoder — bridges visual features with semantic prototypes.

Projects fused 256-dim features into a shared embedding space where each
gesture class has a learnable prototype. Supervised contrastive (InfoNCE)
loss pulls features toward their class prototype and pushes them away from
others, structuring the feature space with semantic regularization.

Design follows SignCLIP (EMNLP 2024) and Sigma (2025): visual-text
alignment via contrastive learning, adapted here with learnable prototypes
instead of frozen CLIP text embeddings.

Reference: docs/语义模态扩展方案.md
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticProjector(nn.Module):
    """Project fused features to semantic embedding space.

    Input:  [B, 256] fused multimodal features
    Output: [B, semantic_dim] projected embeddings

    Params: ~400K (with semantic_dim=256)
    """

    def __init__(self, feat_dim=256, semantic_dim=256, num_classes=10,
                 dropout=0.1):
        super().__init__()
        self.semantic_dim = semantic_dim

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, semantic_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(semantic_dim * 2, semantic_dim),
        )

        # Learnable class prototypes (analogous to CLIP text embeddings)
        self.class_prototypes = nn.Parameter(
            torch.randn(num_classes, semantic_dim) * 0.02)

    def forward(self, fused_feat):
        """Project fused features to semantic space.

        fused_feat: [N, 256]
        Returns: [N, semantic_dim]
        """
        return F.normalize(self.proj(fused_feat), dim=-1)

    def get_prototypes(self):
        """Return normalized class prototypes [num_classes, semantic_dim]."""
        return F.normalize(self.class_prototypes, dim=-1)


def semantic_contrastive_loss(projected, labels, prototypes, temperature=0.07):
    """Supervised InfoNCE loss between projected features and class prototypes.

    Args:
        projected:  [N, D] L2-normalized feature embeddings
        labels:     [N]   integer class labels
        prototypes: [C, D] L2-normalized class prototypes
        temperature: float (default 0.07, following CLIP/SimCLR)

    Returns:
        scalar loss
    """
    # Cosine similarity logits: [N, C]
    logits = projected @ prototypes.T / temperature

    # InfoNCE with class labels as positives
    loss = F.cross_entropy(logits, labels)

    return loss
