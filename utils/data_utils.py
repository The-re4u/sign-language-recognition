# coding:utf-8
"""Data loading utilities for sign language recognition training.

Loads keypoint sequences + ROI images from .npy files, gesture labels from
annotation JSON. Returns PyTorch DataLoader yielding (keypoints, roi, label, length).
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SignLanguageDataset(Dataset):
    """Dataset for sign language keypoint sequences + ROI images with gesture labels.

    Each sample: keypoints [T, 21, 3] + roi_images [T, 3, 96, 96] + integer label.
    ROI images fallback to zeros when roi_path is missing.
    """

    def __init__(self, annotation_path, data_dir='data/sequences',
                 transform=None, min_frames=8, max_frames=64):
        with open(annotation_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        self.data_dir = data_dir
        self.transform = transform
        self.min_frames = min_frames
        self.max_frames = max_frames

        # Build label vocabulary
        self.labels = sorted(set(a.get('label', a.get('gesture', 'unknown'))
                                 for a in self.annotations))
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}
        self.num_classes = len(self.labels)

        # Filter: keep only samples with valid .npy files
        self.samples = []
        for ann in self.annotations:
            seq_path = os.path.join(data_dir, os.path.basename(
                ann.get('frames_path', '')))
            if not os.path.exists(seq_path):
                seq_path = ann.get('frames_path', '')
            if os.path.exists(seq_path):
                label = ann.get('label', ann.get('gesture', 'unknown'))
                # ROI path (optional — fallback to zeros if missing)
                roi_path = ann.get('roi_path', '')
                if roi_path:
                    full_roi = os.path.join(data_dir, os.path.basename(roi_path))
                    if not os.path.exists(full_roi):
                        full_roi = roi_path if os.path.exists(roi_path) else ''
                else:
                    full_roi = ''
                self.samples.append({
                    'path': seq_path,
                    'roi_path': full_roi,
                    'label': self.label_to_idx[label],
                    'label_name': label,
                })

        print(f'[DataLoader] {len(self.samples)} samples, '
              f'{self.num_classes} classes')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        kp_seq = np.load(sample['path']).astype(np.float32)  # [T, 21, 3]

        # Load ROI images [T, 96, 96, 3] or zeros
        T = kp_seq.shape[0]
        if sample['roi_path']:
            roi_seq = np.load(sample['roi_path']).astype(np.float32)  # [T, 96, 96, 3]
            roi_seq = roi_seq / 255.0  # normalize to [0, 1]
            # HWC -> CHW
            roi_seq = np.transpose(roi_seq, (0, 3, 1, 2))  # [T, 3, 96, 96]
        else:
            roi_seq = np.zeros((T, 3, 96, 96), dtype=np.float32)

        # Pad or truncate to fixed length (sync keypoints and ROI)
        if T < self.min_frames:
            kp_pad = np.zeros((self.min_frames - T, 21, 3), dtype=np.float32)
            roi_pad = np.zeros((self.min_frames - T, 3, 96, 96), dtype=np.float32)
            kp_seq = np.concatenate([kp_seq, kp_pad], axis=0)
            roi_seq = np.concatenate([roi_seq, roi_pad], axis=0)
            T = self.min_frames
        elif T > self.max_frames:
            indices = np.linspace(0, T - 1, self.max_frames).astype(int)
            kp_seq = kp_seq[indices]
            roi_seq = roi_seq[indices]
            T = self.max_frames

        if self.transform:
            kp_seq = self.transform(kp_seq)

        return (
            torch.from_numpy(kp_seq),           # [T, 21, 3]
            torch.from_numpy(roi_seq),           # [T, 3, 96, 96]
            torch.tensor(sample['label']).long(), # scalar
            torch.tensor(T).long(),               # actual length
        )


def create_data_loader(annotation_path, batch_size=16, data_dir='data/sequences',
                       shuffle=True, num_workers=0, **kwargs):
    dataset = SignLanguageDataset(annotation_path, data_dir=data_dir, **kwargs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, drop_last=True,
                        collate_fn=collate_fn)
    return loader


def collate_fn(batch):
    """Collate variable-length sequences with padding (keypoints + ROI)."""
    keypoints, rois, labels, lengths = zip(*batch)
    max_len = max(kp.shape[0] for kp in keypoints)
    B = len(keypoints)
    N = keypoints[0].shape[1]   # 21
    C = keypoints[0].shape[2]   # 3

    padded_kp = torch.zeros(B, max_len, N, C)
    padded_roi = torch.zeros(B, max_len, 3, 96, 96)
    for i in range(B):
        padded_kp[i, :keypoints[i].shape[0]] = keypoints[i]
        padded_roi[i, :rois[i].shape[0]] = rois[i]

    return (
        padded_kp,               # [B, max_T, 21, 3]
        padded_roi,              # [B, max_T, 3, 96, 96]
        torch.stack(labels),     # [B]
        torch.stack(lengths),    # [B]
    )
