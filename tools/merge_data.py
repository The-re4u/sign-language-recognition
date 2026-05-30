# coding:utf-8
"""Merge all data sources and create stratified train/val/test splits.

Usage:
  python tools/merge_data.py
"""
import json, os, sys
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# ── Data sources ──
SOURCES = {
    'my_right': ('data/my_annotations.json', 'data', 'self', 'Right'),
    'my_left':  ('data/my_annotations_left.json', 'data', 'self', 'Left'),
}

# Cross-user data (all in separate dirs)
CROSS_USER = {
    'slx_old':  ('data/multiuser_annotations.json', 'data', 'slx'),
    'slx_nine': ('data/slx_nine.json', 'data', 'slx'),
    'qyt_right':('data/qyt_right.json', 'data', 'qyt'),
    'qyt_left': ('data/qyt_left.json', 'data', 'qyt'),
    'shh_right':('data/shh_right.json', 'data', 'shh'),
    'shh_left': ('data/shh_left.json', 'data', 'shh'),
}

# Bimanual data
BIMANUAL = {
    'self_both': ('data/bimanual_self.json', 'data', 'self'),
    'slx_both':  ('data/bimanual_slx.json', 'data', 'slx'),
    'qyt_both':  ('data/bimanual_qyt.json', 'data', 'qyt'),
    'shh_both':  ('data/bimanual_shh.json', 'data', 'shh'),
}

# Confusion pairs
CONFUSION = ('data/confusion_v6.json', 'data')


def load_source(ann_path, data_dir, user, hand=None):
    """Load annotations, resolve paths, add user/hand tags."""
    if not os.path.exists(ann_path):
        print(f'  WARNING: {ann_path} not found')
        return []
    with open(ann_path, encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for d in data:
        # slx old data had 16 variants per hand (incl. front Nine, now deleted).
        # Remove ALL old slx Nine records; replaced by slx_nine.json (side only).
        if user == 'slx' and ann_path == 'data/multiuser_annotations.json' and d.get('label') == 'Nine':
            continue

        kp_path = d.get('frames_path', '')
        roi_path = d.get('roi_path', '')
        # Resolve paths relative to data_dir
        if not os.path.exists(kp_path):
            basename = os.path.basename(kp_path)
            candidate = os.path.join(data_dir, basename)
            if os.path.exists(candidate):
                kp_path = candidate
            else:
                continue  # skip if can't find kp

        if roi_path and not os.path.exists(roi_path):
            basename = os.path.basename(roi_path)
            candidate = os.path.join(data_dir, basename)
            if os.path.exists(candidate):
                roi_path = candidate
            else:
                roi_path = ''  # mark as missing

        samples.append({
            'label': d['label'],
            'user': user,
            'hand': hand or d.get('hand', '?'),
            'frames_path': kp_path,
            'roi_path': roi_path,
            'num_frames': d.get('num_frames', 0),
            'variant': d.get('variant', ''),
        })
    return samples


def main():
    all_samples = []

    print('Loading sources...')

    # 1. Self single-hand (main training data)
    for name, (ann, ddir, user, hand) in SOURCES.items():
        samples = load_source(ann, ddir, user, hand)
        print(f'  {name}: {len(samples)} samples (user={user}, hand={hand})')
        all_samples.extend(samples)

    # 2. Cross-user single-hand
    for name, (ann, ddir, user) in CROSS_USER.items():
        samples = load_source(ann, ddir, user)
        print(f'  {name}: {len(samples)} samples (user={user})')
        all_samples.extend(samples)

    # 3. Bimanual
    bimanual_samples = []
    for name, (ann, ddir, user) in BIMANUAL.items():
        samples = load_source(ann, ddir, user, hand='Both')
        print(f'  {name}: {len(samples)} samples (bimanual)')
        bimanual_samples.extend(samples)

    # 4. Confusion pairs
    conf_ann, conf_dir = CONFUSION
    conf_samples = load_source(conf_ann, conf_dir, 'self', hand='Right')
    for s in conf_samples:
        s['is_confusion'] = True
    print(f'  confusion: {len(conf_samples)} samples')
    # Confusion pairs go to test set, not merged here
    with open('data/test_confusion.json', 'w', encoding='utf-8') as f:
        json.dump(conf_samples, f, ensure_ascii=False, indent=2)
    print(f'  -> data/test_confusion.json')

    print(f'\nTotal single-hand: {len(all_samples)}')
    print(f'Total bimanual: {len(bimanual_samples)}')

    # Split self data from cross-user data
    self_samples = [s for s in all_samples if s['user'] == 'self']
    cross_samples = [s for s in all_samples if s['user'] != 'self']

    print(f'  Self: {len(self_samples)}, Cross-user: {len(cross_samples)}')

    # Stratified split on self data
    labels = [s['label'] for s in self_samples]
    label_counts = Counter(labels)
    print(f'\nSelf label distribution:')
    for lbl in sorted(label_counts):
        print(f'  {lbl}: {label_counts[lbl]}')

    # Split: 70% train, 15% val, 15% test_inuser
    train_self, tmp = train_test_split(
        self_samples, test_size=0.30, stratify=labels, random_state=42)

    tmp_labels = [s['label'] for s in tmp]
    val_self, test_self = train_test_split(
        tmp, test_size=0.50, stratify=tmp_labels, random_state=42)

    # Train = self_train + all bimanual
    train_all = train_self + bimanual_samples

    # Cross-user test = slx + qyt + shh single-hand
    # (slx has some ROI, qyt/shh don't — graceful degradation)
    test_cross = cross_samples

    # Save splits
    splits = {
        'data/train.json': train_all,
        'data/val.json': val_self,
        'data/test_inuser.json': test_self,
        'data/test_crossuser.json': test_cross,
    }

    for path, samples in splits.items():
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        labels = Counter(s['label'] for s in samples)
        users = Counter(s['user'] for s in samples)
        has_roi = sum(1 for s in samples if s.get('roi_path'))
        print(f'\n{path}: {len(samples)} samples')
        print(f'  Users: {dict(users)}')
        print(f'  ROI: {has_roi}/{len(samples)}')
        print(f'  Labels: {dict(sorted(labels.items()))}')

    print('\nDone! Data splits ready.')


if __name__ == '__main__':
    main()
