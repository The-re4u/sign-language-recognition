# coding:utf-8
"""Data loading utilities for sign language recognition training.

Loads keypoint sequences + ROI images from .npy files, gesture labels from
annotation JSON. Returns PyTorch DataLoader yielding (keypoints, roi, label, length).

v5.3: Added KeypointAugmentation, ROI transforms, WeightedRandomSampler support.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class KeypointAugmentation:
    """Online keypoint augmentation for training — no disk overhead.

    Applies random geometric + temporal transforms to keypoint sequences.
    All transforms are wrist-anchored to preserve hand structure.

    Returns (augmented_kp, time_indices) where time_indices can be used
    to apply the same temporal warp to ROI and flow sequences.
    """

    def __init__(self, rotation_range=15.0, scale_range=0.1, shift_range=0.03,
                 time_warp_range=0.2, noise_std=0.005, apply_prob=0.8,
                 part_mix_prob=0.3):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.time_warp_range = time_warp_range
        self.noise_std = noise_std
        self.apply_prob = apply_prob
        self.part_mix_prob = part_mix_prob
        self._part_mix_buffer = None  # stores the other sample for mixing

    def set_part_mix_pair(self, kp_other):
        """Store another sample as the Part Mixing swap source (SKIM [35])."""
        self._part_mix_buffer = kp_other

    def __call__(self, kp_seq):
        """kp_seq: numpy [T, 21, 3] or bimanual [T, 2, 21, 3] → (augmented, indices)"""
        if np.random.random() > self.apply_prob:
            return kp_seq, None

        is_bimanual = (kp_seq.ndim == 4)  # [T, 2, 21, 3]
        if is_bimanual:
            T, H, N, C = kp_seq.shape
            kp = kp_seq.copy()
        else:
            T, N, C = kp_seq.shape
            kp = kp_seq.copy()

        # Anchor at wrist (landmark 0)
        wrist = kp[..., 0:1, :2].copy()  # [T, 1, 2] or [T, 2, 1, 2]

        # 1. Random rotation around wrist
        angle = np.radians(np.random.uniform(-self.rotation_range, self.rotation_range))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        centered = kp[..., :2] - wrist
        kp[..., 0] = centered[..., 0] * cos_a - centered[..., 1] * sin_a + wrist[..., 0]
        kp[..., 1] = centered[..., 0] * sin_a + centered[..., 1] * cos_a + wrist[..., 1]

        # 2. Random scaling
        scale = 1.0 + np.random.uniform(-self.scale_range, self.scale_range)
        kp[..., :2] = (kp[..., :2] - wrist) * scale + wrist

        # 3. Random translation
        shift_x = np.random.uniform(-self.shift_range, self.shift_range)
        shift_y = np.random.uniform(-self.shift_range, self.shift_range)
        kp[..., 0] += shift_x
        kp[..., 1] += shift_y

        # 4. Time warping
        time_indices = None
        if T > 8 and np.random.random() < 0.5:
            warp_factor = 1.0 + np.random.uniform(-self.time_warp_range, self.time_warp_range)
            new_T = max(8, int(T * warp_factor))
            old_indices = np.linspace(0, T - 1, T)
            new_indices = np.linspace(0, T - 1, new_T)
            # Determine target shape
            if is_bimanual:
                warped = np.zeros((new_T, H, N, C), dtype=kp.dtype)
                for h in range(H):
                    for j in range(N):
                        for c in range(C):
                            warped[:, h, j, c] = np.interp(new_indices, old_indices, kp[:, h, j, c])
            else:
                warped = np.zeros((new_T, N, C), dtype=kp.dtype)
                for j in range(N):
                    for c in range(C):
                        warped[:, j, c] = np.interp(new_indices, old_indices, kp[:, j, c])
            kp = warped
            time_indices = np.clip(np.floor(new_indices).astype(int), 0, T - 1)

        # 5. Gaussian noise
        kp += np.random.randn(*kp.shape).astype(np.float32) * self.noise_std
        kp[..., :2] = np.clip(kp[..., :2], -0.5, 1.5)

        # 6. Part Mixing (SKIM [35]): only for single-hand data
        if not is_bimanual and self._part_mix_buffer is not None and np.random.random() < self.part_mix_prob:
            kp = self._apply_part_mixing(kp)
        self._part_mix_buffer = None

        return kp, time_indices

    def _apply_part_mixing(self, kp):
        """Swap a random finger chain between this sample and the stored pair."""
        if self._part_mix_buffer is None:
            return kp
        other = self._part_mix_buffer
        # Align time dimensions
        min_t = min(kp.shape[0], other.shape[0])
        if min_t < 2:
            return kp
        finger_chains = [
            [1, 2, 3, 4],      # Thumb: CMC→MCP→IP→TIP
            [5, 6, 7, 8],      # Index: MCP→PIP→DIP→TIP
            [9, 10, 11, 12],   # Middle
            [13, 14, 15, 16],  # Ring
            [17, 18, 19, 20],  # Pinky
        ]
        chain = finger_chains[np.random.randint(len(finger_chains))]
        t_start = np.random.randint(0, min_t)
        t_end = min(t_start + max(1, min_t // 4), min_t)
        kp[t_start:t_end, chain, :] = other[t_start:t_end, chain, :]
        return kp


def _get_roi_transforms():
    """Build torchvision ROI augmentation pipeline."""
    try:
        import torchvision.transforms as T
        return T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomErasing(scale=(0.02, 0.1), p=0.3),
        ])
    except ImportError:
        return None


class SignLanguageDataset(Dataset):
    """Dataset for sign language keypoint sequences with gesture labels.

    Each sample: keypoints [T, 21, 3] + optional ROI [T, 3, 96, 96] + integer label.
    Flow is deprecated (always zeros). ROI is only loaded when use_cnn=True.
    """

    def __init__(self, annotation_path, data_dir='data/sequences',
                 transform=None, min_frames=8, max_frames=64,
                 augment=False, use_cnn=False, part_mixing=False,
                 part_mix_prob=0.3):
        with open(annotation_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        self.data_dir = data_dir
        self.transform = transform
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.use_cnn = use_cnn
        self.part_mixing = part_mixing

        # Build label vocabulary
        self.labels = sorted(set(a.get('label', a.get('gesture', 'unknown'))
                                 for a in self.annotations))
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}
        self.num_classes = len(self.labels)

        # Augmentation (training only)
        self.augment = augment
        self.kp_aug = KeypointAugmentation(part_mix_prob=part_mix_prob) if augment else None
        self.roi_transforms = _get_roi_transforms() if augment else None

        # Filter: keep only samples with valid .npy files
        self.samples = []
        for ann in self.annotations:
            seq_path = self._resolve_path(ann.get('frames_path', ''), data_dir, ann)
            if seq_path:
                label = ann.get('label', ann.get('gesture', 'unknown'))
                full_roi = ''
                if use_cnn:
                    full_roi = self._resolve_path(ann.get('roi_path', ''), data_dir, ann) or ''
                self.samples.append({
                    'path': seq_path,
                    'roi_path': full_roi,
                    'label': self.label_to_idx[label],
                    'label_name': label,
                })

        print(f'[DataLoader] {len(self.samples)} samples, '
              f'{self.num_classes} classes'
              f'{", augment=True" if augment else ""}'
              f'{", CNN=on" if use_cnn else ""}')

    @staticmethod
    def _resolve_path(path, data_dir, ann=None):
        """Resolve a file path: try as-is, then data_dir, then _src_dir."""
        if not path:
            return ''
        if os.path.exists(path):
            return path
        basename = os.path.basename(path)
        # Try data_dir
        candidate = os.path.join(data_dir, basename)
        if os.path.exists(candidate):
            return candidate
        # Try _src_dir (set during multi-source merge)
        src_dir = ann.get('_src_dir', '') if ann else ''
        if src_dir:
            candidate = os.path.join(src_dir, basename)
            if os.path.exists(candidate):
                return candidate
        return ''

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        kp_seq = np.load(sample['path']).astype(np.float32)  # [T, 21, 3] or [T, 2, 21, 3]
        # Bimanual data: pick one hand randomly for training (P2 not yet activated)
        if kp_seq.ndim == 4:
            kp_seq = kp_seq[:, np.random.randint(0, 2), :, :]  # [T, 21, 3]

        # Load ROI images [T, 3, 96, 96] or zeros
        if sample['roi_path']:
            roi_seq = np.load(sample['roi_path']).astype(np.float32)  # [T, 96, 96, 3] or [T, 2, 96, 96, 3]
            # Bimanual ROI: pick same hand as kp_seq
            if roi_seq.ndim == 5:  # [T, 2, 96, 96, 3]
                roi_seq = roi_seq[:, np.random.randint(0, 2), :, :, :]
            roi_seq = roi_seq / 255.0
            roi_seq = np.transpose(roi_seq, (0, 3, 1, 2))  # [T, 3, 96, 96]
        else:
            roi_seq = np.zeros((kp_seq.shape[0], 3, 96, 96), dtype=np.float32)

        # Flow is deprecated — always zeros
        flow_seq = np.zeros((kp_seq.shape[0], 128), dtype=np.float32)

        # Apply keypoint augmentation during training
        time_idx = None
        if self.augment and self.kp_aug is not None:
            # Part Mixing pairing (SKIM [35])
            if self.part_mixing:
                if not hasattr(self, '_pair_buffer'):
                    self._pair_buffer = None
                if self._pair_buffer is not None and np.random.random() < 0.5:
                    self.kp_aug.set_part_mix_pair(self._pair_buffer)
                self._pair_buffer = kp_seq.copy()

            kp_seq, time_idx = self.kp_aug(kp_seq)
            if time_idx is not None:
                roi_seq = roi_seq[time_idx]
                flow_seq = flow_seq[time_idx]

        # Apply ROI augmentation during training (only when CNN is active)
        if self.augment and self.use_cnn and self.roi_transforms is not None and roi_seq.shape[0] > 0:
            roi_tensor = torch.from_numpy(roi_seq)
            roi_tensor = self.roi_transforms(roi_tensor)
            roi_seq = roi_tensor.numpy()

        # Pad or truncate to fixed length
        T = kp_seq.shape[0]
        if T < self.min_frames:
            kp_pad = np.zeros((self.min_frames - T, 21, 3), dtype=np.float32)
            roi_pad = np.zeros((self.min_frames - T, 3, 96, 96), dtype=np.float32)
            flow_pad = np.zeros((self.min_frames - T, 128), dtype=np.float32)
            kp_seq = np.concatenate([kp_seq, kp_pad], axis=0)
            roi_seq = np.concatenate([roi_seq, roi_pad], axis=0)
            flow_seq = np.concatenate([flow_seq, flow_pad], axis=0)
            T = self.min_frames
        elif T > self.max_frames:
            indices = np.linspace(0, T - 1, self.max_frames).astype(int)
            kp_seq = kp_seq[indices]
            roi_seq = roi_seq[indices]
            flow_seq = flow_seq[indices]
            T = self.max_frames

        if self.transform:
            kp_seq = self.transform(kp_seq)

        return (
            torch.from_numpy(kp_seq),
            torch.from_numpy(roi_seq),
            torch.from_numpy(flow_seq),
            torch.tensor(sample['label']).long(),
            torch.tensor(T).long(),
        )


def create_data_loader(annotation_path, batch_size=16, data_dir='data/sequences',
                       shuffle=True, num_workers=0, class_weight=False,
                       augment=False, use_cnn=False, part_mixing=False,
                       part_mix_prob=0.3, **kwargs):
    """Create a DataLoader with optional class balancing and augmentation.

    Args:
        part_mixing: enable SKIM Part Mixing augmentation (default False)
        part_mix_prob: probability of applying Part Mixing per sample (default 0.3)
    """
    dataset = SignLanguageDataset(annotation_path, data_dir=data_dir,
                                  augment=augment, use_cnn=use_cnn,
                                  part_mixing=part_mixing,
                                  part_mix_prob=part_mix_prob, **kwargs)

    if class_weight:
        # Build class weights: 1/frequency per sample
        label_counts = np.zeros(dataset.num_classes, dtype=np.float32)
        for s in dataset.samples:
            label_counts[s['label']] += 1
        sample_weights = np.zeros(len(dataset.samples), dtype=np.float32)
        for i, s in enumerate(dataset.samples):
            sample_weights[i] = 1.0 / max(label_counts[s['label']], 1.0)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(dataset.samples),
            replacement=True
        )
        print(f'[DataLoader] class_weight enabled — '
              f'label range: {int(label_counts.min())}-{int(label_counts.max())}')
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=num_workers, drop_last=True,
                            collate_fn=collate_fn)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, drop_last=True,
                            collate_fn=collate_fn)
    return loader


def collate_fn(batch):
    """Collate variable-length sequences with padding (keypoints + ROI + flow)."""
    keypoints, rois, flows, labels, lengths = zip(*batch)
    max_len = max(max(kp.shape[0] for kp in keypoints),
                  max(r.shape[0] for r in rois),
                  max(f.shape[0] for f in flows))
    B = len(keypoints)
    N = keypoints[0].shape[1]
    C = keypoints[0].shape[2]

    padded_kp = torch.zeros(B, max_len, N, C)
    padded_roi = torch.zeros(B, max_len, 3, 96, 96)
    padded_flow = torch.zeros(B, max_len, 128)
    for i in range(B):
        n_kp = min(keypoints[i].shape[0], max_len)
        n_roi = min(rois[i].shape[0], max_len)
        n_flow = min(flows[i].shape[0], max_len)
        padded_kp[i, :n_kp] = keypoints[i][:n_kp]
        padded_roi[i, :n_roi] = rois[i][:n_roi]
        padded_flow[i, :n_flow] = flows[i][:n_flow]

    return (
        padded_kp,
        padded_roi,
        padded_flow,
        torch.stack(labels),
        torch.stack(lengths),
    )
