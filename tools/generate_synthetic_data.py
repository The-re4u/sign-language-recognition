# coding:utf-8
"""
Synthetic training data generator for sign language recognition.

Generates labeled keypoint sequences by:
1. Creating canonical keypoint patterns for each gesture using hand skeleton model
2. Adding spatial noise (jitter, rotation, scaling)
3. Simulating temporal dynamics (transition between keypoints)
4. Outputting annotation JSON + .npy sequence files

Usage: python generate_synthetic_data.py --output data/annotations.json --data_dir data/sequences --samples 100
"""
import numpy as np
import json
import os
import argparse


# MediaPipe hand skeleton bone lengths (normalized, approx)
# Indices: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
BONE_LINKS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (0, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
]

# Finger tip indices for each finger (thumb, index, middle, ring, pinky)
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_MCP = [2, 5, 9, 13, 17]
FINGER_PIP = [3, 6, 10, 14, 18]


def create_canonical_hand(finger_states, palm_angle=0.0):
    """
    Create canonical 21-keypoint hand given finger states.

    finger_states: tuple of 5 bools (thumb, index, middle, ring, pinky)
                   True = extended, False = curled
    palm_angle: in radians, 0 = palm facing camera

    Returns: [21, 3] array of (x, y, z) normalized to [0,1]
    """
    kp = np.zeros((21, 3), dtype=np.float32)

    # Wrist at center
    kp[0] = [0.5, 0.7, 0.0]

    # Base positions for each finger
    finger_bases = [
        np.array([0.45, 0.55, 0.0]),  # Thumb MCP
        np.array([0.40, 0.45, 0.0]),  # Index MCP
        np.array([0.50, 0.42, 0.0]),  # Middle MCP
        np.array([0.58, 0.45, 0.0]),  # Ring MCP
        np.array([0.63, 0.50, 0.0]),  # Pinky MCP
    ]

    # Direction vectors (extended)
    directions = [
        np.array([-0.05, -0.15, 0.0]),  # Thumb points up-left
        np.array([-0.02, -0.18, 0.0]),  # Index points up
        np.array([0.0, -0.18, 0.0]),    # Middle points up
        np.array([0.02, -0.18, 0.0]),   # Ring points up
        np.array([0.02, -0.15, 0.0]),   # Pinky points up-right
    ]

    # Curled directions fold toward palm
    curl_directions = [
        np.array([0.03, 0.0, 0.05]),
        np.array([0.0, 0.05, 0.05]),
        np.array([0.0, 0.05, 0.05]),
        np.array([0.0, 0.05, 0.05]),
        np.array([-0.02, 0.02, 0.05]),
    ]

    kp[1] = finger_bases[0] + np.array([0.02, -0.02, 0.0])

    for finger_idx in range(5):
        # MCP joint
        mcp_idx = FINGER_MCP[finger_idx]
        kp[mcp_idx] = finger_bases[finger_idx].copy()

        # PIP and DIP joints
        pip_idx = mcp_idx + 1
        dip_idx = mcp_idx + 2
        tip_idx = FINGER_TIPS[finger_idx]

        if finger_states[finger_idx]:
            # Extended: joints follow direction
            kp[pip_idx] = kp[mcp_idx] + directions[finger_idx] * 0.35
            kp[dip_idx] = kp[pip_idx] + directions[finger_idx] * 0.30
            kp[tip_idx] = kp[dip_idx] + directions[finger_idx] * 0.25
        else:
            # Curled: joints fold inward
            mid_dir = directions[finger_idx] * 0.6 + curl_directions[finger_idx] * 0.4
            kp[pip_idx] = kp[mcp_idx] + mid_dir * 0.20
            curl_dir = curl_directions[finger_idx]
            kp[dip_idx] = kp[pip_idx] + curl_dir * 0.18
            kp[tip_idx] = kp[dip_idx] + curl_dir * 0.12

    # Apply palm rotation
    if palm_angle != 0:
        cos_a, sin_a = np.cos(palm_angle), np.sin(palm_angle)
        wrist = kp[0, :2].copy()
        translated = kp[:, :2] - wrist
        kp[:, 0] = translated[:, 0] * cos_a - translated[:, 1] * sin_a + wrist[0]
        kp[:, 1] = translated[:, 0] * sin_a + translated[:, 1] * cos_a + wrist[1]

    return kp


# Gesture definitions: (finger states, palm angle, label)
GESTURE_DEFS = [
    ([False, False, False, False, False], 0.0, 'Closed_Fist'),
    ([True,  False, False, False, False], -0.3, 'Good'),
    ([False, True,  False, False, False], 0.0, 'One'),
    ([False, True,  True,  False, False], 0.0, 'Two'),
    ([False, True,  True,  False, False], 0.0, 'Victory'),     # Same fingers as Two
    ([True,  True,  False, False, False], 0.0, 'Eight'),
    ([True,  False, False, False, True],  0.0, 'Six'),
    ([True,  True,  True,  False, False], 0.0, 'Seven'),
    ([False, False, True,  True,  True],  0.0, 'Three'),
    ([False, True,  True,  True,  True],  0.0, 'Four'),
    ([True,  False, True,  True,  True],  0.0, 'Nine'),
    ([True,  True,  True,  True,  True],  0.0, 'Open_Palm'),
]

# Semantic chain gestures (sequence-based)
CHAIN_GESTURES = {
    'I_love_you': {
        'sequence': ['One', 'Good', 'Eight'],
        'label': '我爱你',
    },
    'hello': {
        'sequence': ['Open_Palm'],
        'label': '你好',
    },
    'goodbye': {
        'sequence': ['Open_Palm', 'Closed_Fist'],
        'label': '再见',
    },
    'light_on': {
        'sequence': ['Open_Palm', 'Good'],
        'label': '开灯',
    },
    'light_off': {
        'sequence': ['Closed_Fist', 'Good'],
        'label': '关灯',
    },
    'help': {
        'sequence': ['Open_Palm', 'Open_Palm'],
        'label': '帮助',
    },
    'drink': {
        'sequence': ['Three', 'Closed_Fist'],
        'label': '喝水',
    },
    'thanks': {
        'sequence': ['Good'],
        'label': '谢谢',
    },
    'victory_': {
        'sequence': ['Victory'],
        'label': '胜利',
    },
    'ok': {
        'sequence': ['Good'],
        'label': '好的',
    },
    'no': {
        'sequence': ['Seven'],
        'label': '不行',
    },
}


def get_keypoints_for_gesture(gesture_name):
    """Get canonical keypoint for a gesture."""
    for finger_states, palm_angle, name in GESTURE_DEFS:
        if name == gesture_name:
            return create_canonical_hand(finger_states, palm_angle)
    return create_canonical_hand([False]*5, 0.0)


def add_noise(kp, jitter=0.008, rot_range=3, scale_range=0.03):
    """Add random noise to keypoints."""
    kp = kp.copy()
    # Gaussian jitter
    kp[:, :2] += np.random.randn(*kp[:, :2].shape) * jitter
    # Slight random rotation
    angle = np.random.uniform(-rot_range, rot_range)
    theta = np.radians(angle)
    cos_a, sin_a = np.cos(theta), np.sin(theta)
    wrist = kp[0, :2].copy()
    translated = kp[:, :2] - wrist
    kp[:, 0] = translated[:, 0] * cos_a - translated[:, 1] * sin_a + wrist[0]
    kp[:, 1] = translated[:, 0] * sin_a + translated[:, 1] * cos_a + wrist[1]
    # Slight scaling
    scale = 1.0 + np.random.uniform(-scale_range, scale_range)
    kp[:, :2] = (kp[:, :2] - wrist) * scale + wrist
    # Clamp
    kp[:, :2] = np.clip(kp[:, :2], 0.0, 1.0)
    return kp


def generate_sequence(gesture_name, num_frames=32, noise_level=0.008):
    """Generate a temporal sequence for a single gesture."""
    base_kp = get_keypoints_for_gesture(gesture_name)
    frames = []

    for t in range(num_frames):
        # Start from neutral and transition to gesture
        if t < 5:
            # Transition in
            alpha = t / 5.0
            neutral = create_canonical_hand([False]*5, 0.0)  # Closed fist as neutral
            kp = neutral * (1 - alpha) + base_kp * alpha
        elif t > num_frames - 5:
            # Transition out
            alpha = (t - (num_frames - 5)) / 5.0
            neutral = create_canonical_hand([False]*5, 0.0)
            kp = base_kp * (1 - alpha) + neutral * alpha
        else:
            kp = base_kp.copy()

        # Add micro-movement
        kp = add_noise(kp, jitter=noise_level * (1 + 0.5 * np.sin(t * 0.5)))

        frames.append(kp)

    return np.array(frames, dtype=np.float32)


def add_occlusion(kp, max_hidden=5):
    """Simulate partial hand occlusion by zeroing random landmarks."""
    n_hidden = np.random.randint(1, max_hidden + 1)
    indices = np.random.choice(21, n_hidden, replace=False)
    kp_occ = kp.copy()
    kp_occ[indices, :2] = 0.0  # zero out x,y
    return kp_occ


def add_lighting_noise(kp, lux_level='normal'):
    """Simulate low-light conditions by increasing keypoint jitter.
    MediaPipe jitter increases from ~0.005 (normal) to ~0.030 (dim) at <100 lux.
    """
    levels = {'bright': 0.002, 'normal': 0.006, 'dim': 0.015, 'dark': 0.030}
    sigma = levels.get(lux_level, 0.006)
    kp_lit = kp.copy()
    kp_lit[:, :2] += np.random.randn(*kp[:, :2].shape) * sigma
    kp_lit[:, :2] = np.clip(kp_lit[:, :2], 0.0, 1.0)
    return kp_lit


def generate_static_gesture_data(output_path, data_dir, samples_per_gesture=50):
    """Generate static gesture training data with domain randomization."""
    os.makedirs(data_dir, exist_ok=True)
    annotations = []
    lighting_levels = ['bright', 'normal', 'normal', 'normal', 'dim', 'dark']  # weighted

    for finger_states, palm_angle, gesture_name in GESTURE_DEFS:
        base_name = gesture_name
        for i in range(samples_per_gesture):
            frames = generate_sequence(gesture_name, num_frames=32,
                                       noise_level=0.008 + np.random.random() * 0.012)

            # Domain randomization: 30% chance occlusion, 40% chance low-light
            if np.random.random() < 0.3:
                frames = np.array([add_occlusion(f) for f in frames])
            if np.random.random() < 0.4:
                lux = lighting_levels[np.random.randint(0, len(lighting_levels))]
                frames = np.array([add_lighting_noise(f, lux) for f in frames])

            seq_name = f'{base_name}_static_{i:04d}.npy'
            seq_path = os.path.join(data_dir, seq_name)
            np.save(seq_path, frames)
            annotations.append({
                'video_id': f'{base_name}_static_{i:04d}',
                'label': gesture_name,
                'frames_path': seq_name
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    print(f'Static gestures: {len(annotations)} sequences -> {output_path}')
    return annotations


def generate_chain_data(output_path, data_dir, samples_per_chain=30):
    """Generate gesture chain training data."""
    os.makedirs(data_dir, exist_ok=True)
    annotations = []

    for chain_id, chain_info in CHAIN_GESTURES.items():
        seq_gestures = chain_info['sequence']
        label = chain_info['label']

        for i in range(samples_per_chain):
            all_frames = []

            for g_idx, gesture_name in enumerate(seq_gestures):
                # Generate transition + hold for this gesture
                base_kp = get_keypoints_for_gesture(gesture_name)

                # Hold gesture for some frames
                hold_frames = 12 + np.random.randint(0, 8)
                for t in range(hold_frames):
                    kp = base_kp.copy()
                    kp = add_noise(kp, jitter=0.006)
                    all_frames.append(kp)

                # Transition to neutral between gestures
                if g_idx < len(seq_gestures) - 1:
                    next_kp = get_keypoints_for_gesture(seq_gestures[g_idx + 1])
                    trans_frames = 6
                    for t in range(trans_frames):
                        alpha = (t + 1) / trans_frames
                        kp = base_kp * (1 - alpha) + next_kp * alpha
                        kp = add_noise(kp, jitter=0.004)
                        all_frames.append(kp)

            frames_arr = np.array(all_frames, dtype=np.float32)
            seq_name = f'chain_{chain_id}_{i:04d}.npy'
            seq_path = os.path.join(data_dir, seq_name)
            np.save(seq_path, frames_arr)
            annotations.append({
                'video_id': f'chain_{chain_id}_{i:04d}',
                'label': label,
                'frames_path': seq_name
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    print(f'Chain gestures: {len(annotations)} sequences -> {output_path}')
    return annotations


def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic sign language training data')
    parser.add_argument('--output', default='data/annotations.json',
                        help='Output annotation JSON path')
    parser.add_argument('--data_dir', default='data/sequences',
                        help='Directory for .npy sequence files')
    parser.add_argument('--samples_per_gesture', type=int, default=50,
                        help='Number of samples per static gesture')
    parser.add_argument('--samples_per_chain', type=int, default=30,
                        help='Number of samples per chain sequence')
    parser.add_argument('--static_only', action='store_true',
                        help='Generate only static gesture data')
    return parser.parse_args()


def main():
    args = parse_args()

    print('=' * 60)
    print('  SYNTHETIC TRAINING DATA GENERATOR')
    print('=' * 60)

    # Generate static gesture data
    static_annotations = generate_static_gesture_data(
        args.output, args.data_dir,
        samples_per_gesture=args.samples_per_gesture
    )

    # Generate chain data
    if not args.static_only:
        chain_output = args.output.replace('.json', '_chains.json')
        chain_annotations = generate_chain_data(
            chain_output, args.data_dir,
            samples_per_chain=args.samples_per_chain
        )

        # Merge all annotations
        all_annotations = static_annotations + chain_annotations
        merged_output = args.output.replace('.json', '_all.json')
        with open(merged_output, 'w', encoding='utf-8') as f:
            json.dump(all_annotations, f, ensure_ascii=False, indent=2)
        print(f'\nMerged: {len(all_annotations)} total sequences -> {merged_output}')

    print('\nDone! Run training with:')
    print(f'  python train.py --data {args.output} --epochs 50')


if __name__ == '__main__':
    main()
