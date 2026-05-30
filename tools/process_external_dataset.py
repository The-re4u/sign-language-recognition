# coding:utf-8
"""Process external hand gesture dataset for cross-dataset evaluation.

Reads images from a folder-per-class dataset, runs MediaPipe to extract
21 landmarks + 96x96 ROI, saves as .npy in our standard format.

Usage:
  python tools/process_external_dataset.py \
      --input C:/path/to/dataset --output data/external_processed \
      --classes palm,fist --map palm=Open_Palm,fist=Closed_Fist
"""
import argparse, os, sys, json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Root dir of external dataset (subdirs per class)')
    p.add_argument('--output', required=True, help='Output dir for processed .npy files')
    p.add_argument('--classes', required=True, help='Comma-sep list of class folder names to process')
    p.add_argument('--map', required=True, help='Comma-sep mapping: src_class=dst_label (e.g. palm=Open_Palm,fist=Closed_Fist)')
    p.add_argument('--max_per_class', type=int, default=100,
                   help='Max samples per class (default: 100)')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    class_list = [c.strip() for c in args.classes.split(',')]
    mapping = {}
    for pair in args.map.split(','):
        k, v = pair.strip().split('=')
        mapping[k] = v

    # Our model class index mapping (10 CSL digit classes)
    label_to_idx = {
        'Closed_Fist': 0, 'One': 1, 'Two': 2, 'Three': 3, 'Four': 4,
        'Open_Palm': 5, 'Six': 6, 'Seven': 7, 'Eight': 8, 'Nine': 9,
    }
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                           min_detection_confidence=0.5)

    manifest = []
    stats = {'total': 0, 'success': 0, 'no_hand': 0, 'multi_hand': 0}

    for src_class in class_list:
        class_dir = os.path.join(args.input, src_class)
        if not os.path.isdir(class_dir):
            print(f'WARNING: {class_dir} not found, skipping')
            continue

        dst_label = mapping.get(src_class, src_class)
        dst_idx = label_to_idx.get(dst_label)
        if dst_idx is None:
            print(f'WARNING: unknown mapping {src_class} -> {dst_label}, skipping')
            continue

        out_dir = os.path.join(args.output, f'cls_{dst_idx:02d}')
        os.makedirs(out_dir, exist_ok=True)

        images = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
        if args.max_per_class and len(images) > args.max_per_class:
            images = images[:args.max_per_class]
        print(f'Processing {src_class} -> {dst_label} (idx={dst_idx}): {len(images)} images')

        for fname in images:
            img_path = os.path.join(class_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            stats['total'] += 1
            if not result.multi_hand_landmarks:
                stats['no_hand'] += 1
                continue
            if len(result.multi_hand_landmarks) > 1:
                stats['multi_hand'] += 1
                continue

            lm = result.multi_hand_landmarks[0]
            h, w = img.shape[:2]

            # Extract 21 keypoints in Image Landmarks (x, y, z normalized)
            kp = np.zeros((21, 3), dtype=np.float32)
            for i in range(21):
                kp[i, 0] = lm.landmark[i].x  # normalized [0, 1]
                kp[i, 1] = lm.landmark[i].y
                kp[i, 2] = lm.landmark[i].z

            # Crop ROI: bounding box around hand with padding
            xs = [lm.landmark[i].x for i in range(21)]
            ys = [lm.landmark[i].y for i in range(21)]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            pad = 0.3
            bx = max(0, x_min - pad)
            by = max(0, y_min - pad)
            bw = min(1.0, x_max + pad) - bx
            bh = min(1.0, y_max + pad) - by

            # Ensure square-ish
            if bw > bh:
                by = max(0, by - (bw - bh) / 2)
                bh = bw
            else:
                bx = max(0, bx - (bh - bw) / 2)
                bw = bh

            px1 = int(bx * w)
            py1 = int(by * h)
            px2 = int((bx + bw) * w)
            py2 = int((by + bh) * h)

            roi = img[py1:py2, px1:px2] if px2 > px1 and py2 > py1 else img
            roi = cv2.resize(roi, (96, 96))
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            base_name = os.path.splitext(fname)[0]
            np.save(os.path.join(out_dir, f'{base_name}_kp.npy'), kp)
            np.save(os.path.join(out_dir, f'{base_name}_roi.npy'), roi_rgb)
            stats['success'] += 1

            manifest.append({
                'class_src': src_class,
                'class_dst': dst_label,
                'label_idx': dst_idx,
                'file': base_name,
            })

    hands.close()

    manifest_path = os.path.join(args.output, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f'\nDone: {stats}')
    print(f'Manifest saved to {manifest_path}')


if __name__ == '__main__':
    main()
