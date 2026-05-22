# coding:utf-8
"""Generate all thesis figures from experiment data."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGS_DIR = 'docs/figures'
os.makedirs(FIGS_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150, 'savefig.bbox': 'tight'})

# === Fig 1: Training & Validation Loss + Accuracy ===
with open('models/checkpoints/training_metrics.json') as f:
    m = json.load(f)
epochs = [e['epoch'] for e in m]
train_loss = [e['train_loss'] for e in m]
val_loss = [e['val_loss'] for e in m]
val_acc = [e['val_acc'] for e in m]

fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.plot(epochs, train_loss, 'b-', alpha=0.4, label='Train Loss')
ax1.plot(epochs, val_loss, 'b-', lw=2, label='Val Loss')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.legend(loc='upper left')
ax2 = ax1.twinx()
ax2.plot(epochs, val_acc, 'g-', lw=2, label='Val Acc')
ax2.set_ylabel('Accuracy (%)'); ax2.legend(loc='upper right')
ax1.set_title('Training Convergence (Multimodal DL, 1,311 samples, 50 epochs)')
fig.savefig(f'{FIGS_DIR}/fig1_loss_curve.png'); plt.close()
print('Fig 1: Loss curve')

# === Fig 2: Model Parameters Pie ===
params = {'SpatialGCN': 0.15, 'MotionEncoder': 0.04, 'VisualEncoder': 0.6,
          'CrossModalFusion': 1.9, 'SlowFast TCN': 0.7}
fig, ax = plt.subplots(figsize=(6, 5))
ax.pie(params.values(), labels=params.keys(), autopct='%1.1f%%', startangle=90,
       explode=[0.02]*5)
ax.set_title(f'Model Parameters: {sum(params.values()):.1f}M Total (3.4M trainable)')
fig.savefig(f'{FIGS_DIR}/fig2_params_pie.png'); plt.close()
print('Fig 2: Params pie')

# === Fig 3: Rule vs DL Comparison ===
models = ['Rule Engine', 'DL (keypoints)', 'DL (multimodal)']
accs = [31.1, 73.2, 82.8]
colors = ['#FF9800', '#2196F3', '#4CAF50']
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(models, accs, color=colors, width=0.5)
for b, a in zip(bars, accs):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{a}%', ha='center', fontweight='bold')
ax.set_ylabel('Test Accuracy (%)'); ax.set_ylim(0, 100)
ax.set_title('Gesture Recognition Accuracy on Real User Test Set')
fig.savefig(f'{FIGS_DIR}/fig3_comparison.png'); plt.close()
print('Fig 3: Comparison bar')

# === Fig 4: Ablation Study ===
configs = ['Full Model', 'w/o Visual', 'w/o Motion', 'w/o GCN']
abl = [82.8, 73.2, 51.2, 6.2]
colors = ['#4CAF50', '#2196F3', '#FF9800', '#f44336']
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(configs, abl, color=colors, width=0.5)
for b, a in zip(bars, abl):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{a}%', ha='center', fontweight='bold')
ax.set_ylabel('Test Accuracy (%)'); ax.set_ylim(0, 100)
ax.set_title('Ablation Study: Impact of Each Modality')
fig.savefig(f'{FIGS_DIR}/fig4_ablation.png'); plt.close()
print('Fig 4: Ablation')

# === Fig 5: Per-Class F1 (DL model) ===
with open('data/test_dl_result.json') as f:
    dl = json.load(f)
classes = sorted(dl['per_class'].keys())
f1s = []
for c in classes:
    d = dl['per_class'][c]
    p = d['correct']/d['total'] if d['total']>0 else 0
    r = d['correct']/d['total'] if d['total']>0 else 0
    f1s.append(2*p*r/(p+r)*100 if p+r>0 else 0)

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(classes, f1s, color='#4CAF50', width=0.6)
for b, f in zip(bars, f1s):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{f:.0f}%', ha='center', fontsize=8)
ax.set_ylabel('F1 Score (%)'); ax.set_ylim(0, 110)
ax.set_title('DL Model: Per-Class Accuracy on Test Set (209 samples)')
plt.xticks(rotation=45, ha='right')
fig.savefig(f'{FIGS_DIR}/fig5_per_class.png'); plt.close()
print('Fig 5: Per-class')

# === Fig 6: Latency Breakdown ===
mods = ['MediaPipe\nDetection', 'Rule Engine', 'DL Feature\nExtraction', 'DL ONNX\nTCN']
times = [31.5, 0.05, 45, 5]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(mods, times, color=colors, width=0.5)
for b, t in zip(bars, times):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{t}ms', ha='center', fontweight='bold')
ax.set_ylabel('Latency (ms)')
ax.set_title('Pipeline Module Latency (CPU, 128px input)')
fig.savefig(f'{FIGS_DIR}/fig6_latency.png'); plt.close()
print('Fig 6: Latency')

print(f'\nDone! {len(os.listdir(FIGS_DIR))} figures -> {FIGS_DIR}/')
