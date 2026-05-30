# coding:utf-8
"""Generate v6.0 thesis figures from training metrics and evaluation results.

Output: docs/figures/ (all PNG files)
  - fig_ablation_v6.png      消融柱状图 (E1-E6+GCAR)
  - fig_training_curves_v6.png 训练曲线叠图 (7 models)
  - fig_confusion_e4_v6.png   E4混淆矩阵
  - fig_confusion_e6_v6.png   E6混淆矩阵
  - fig_crossuser_degradation_v6.png 跨用户泛化衰减
  - fig_params_accuracy_v6.png 参数-准确率散点图

Usage: python tools/plot_thesis_figures_v6.py
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from collections import Counter

os.makedirs('docs/figures', exist_ok=True)

# ── Experiment data ──
# (name, in_user, cross_user, confusion, params_m, color, marker, multimodal)
EXPERIMENTS = [
    ('E1\nGCN',         0.902, 0.878, 0.750, 0.35, '#e74c3c', 'o', False),
    ('E2\nGCN+CNN',     0.943, 0.678, 0.708, 4.40, '#e67e22', 's', True),
    ('E3\nGCN+Geo',     0.967, 0.956, 0.896, 2.80, '#2ecc71', 'D', False),
    ('E4\nGCN+CNN+Geo', 0.992, 0.944, 0.875, 6.89, '#3498db', '^', True),
    ('E5\n+PartMix',    0.984, 0.956, 0.771, 6.89, '#9b59b6', 'v', True),
    ('E6\n+Semantic',   0.992, 0.978, 0.896, 7.30, '#1abc9c', 'P', True),
    ('GCAR\nsep-TCN',   0.992, 0.911, 0.875, 0.36, '#95a5a6', '*', True),
]

METRICS_PATHS = {
    'E1': 'models/checkpoints/E1_spatial/training_metrics.json',
    'E2': 'models/checkpoints/E2_spatial_cnn/training_metrics.json',
    'E3': 'models/checkpoints/E3_spatial_geo/training_metrics.json',
    'E4': 'models/checkpoints/E4_spatial_cnn_geo/training_metrics.json',
    'E5': 'models/checkpoints/E5_partmix/training_metrics.json',
    'E6': 'models/checkpoints/E6_semantic/training_metrics.json',
    'GCAR': 'models/baseline_gcar/training_metrics.json',
}

EVAL_PATHS = {
    'E4_in': 'data/test_inuser_dl_result.json',
    'E4_cross': 'data/test_crossuser_dl_result.json',
    'E6_in': 'data/test_inuser_dl_result.json',
    'E6_cross': 'data/test_crossuser_dl_result.json',
}

LABEL_NAMES = ['Closed_Fist','Eight','Four','Nine','One','Open_Palm','Seven','Six','Three','Two']


def fig_ablation():
    """Figure: ablation bar chart — 7 models × 3 test sets."""
    print('Generating: Ablation bar chart...')
    fig, ax = plt.subplots(figsize=(14, 6))
    names = [e[0].replace('\n',' ') for e in EXPERIMENTS]
    x = np.arange(len(names))
    w = 0.25
    for i, (label, color) in enumerate([
        ('In-User', '#2ecc71'), ('Cross-User', '#3498db'), ('Confusion', '#e74c3c')]):
        vals = [e[i+1] for e in EXPERIMENTS]
        bars = ax.bar(x + (i-1)*w, [v*100 for v in vals], w, label=label,
                      color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                    f'{v*100:.1f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3-9: Ablation Study — Model Performance Across Test Splits',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.set_ylim(50, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=90, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    plt.tight_layout()
    plt.savefig('docs/figures/fig_ablation_v6.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  → docs/figures/fig_ablation_v6.png')


def fig_training_curves():
    """Figure: training curves overlay — val_acc per epoch for all 7 models."""
    print('Generating: Training curves overlay...')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = ['#e74c3c','#e67e22','#2ecc71','#3498db','#9b59b6','#1abc9c','#95a5a6']
    labels = ['E1 GCN','E2 GCN+CNN','E3 GCN+Geo','E4 GCN+CNN+Geo','E5 +PartMix','E6 +Semantic','GCAR']
    for (key, path), color, label in zip(METRICS_PATHS.items(), colors, labels):
        if not os.path.exists(path):
            print(f'  SKIP: {path} not found')
            continue
        with open(path) as f:
            data = json.load(f)
        epochs = [d['epoch'] for d in data]
        val_acc = [d['val_acc']*100 for d in data]
        train_loss = [d['train_loss'] for d in data]
        ax1.plot(epochs, val_acc, color=color, linewidth=1.8, label=label, alpha=0.9)
        ax1.scatter(epochs, val_acc, color=color, s=12, alpha=0.3)
        ax2.plot(epochs, train_loss, color=color, linewidth=1.5, label=label, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=11); ax1.set_ylabel('Validation Accuracy (%)', fontsize=11)
    ax1.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=7, ncol=2, framealpha=0.8)
    ax1.grid(alpha=0.2)
    ax2.set_xlabel('Epoch', fontsize=11); ax2.set_ylabel('Training Loss', fontsize=11)
    ax2.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=7, ncol=2, framealpha=0.8)
    ax2.grid(alpha=0.2)
    fig.suptitle('Figure 3-10: Training Dynamics — Validation Accuracy & Training Loss',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('docs/figures/fig_training_curves_v6.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  → docs/figures/fig_training_curves_v6.png')


def fig_confusion_matrix(result_path, output_name, title):
    """Generate a single confusion matrix figure from eval result JSON."""
    if not os.path.exists(result_path):
        print(f'  SKIP: {result_path} not found')
        return
    with open(result_path) as f:
        data = json.load(f)
    num_classes = data.get('num_classes', 10)
    labels = data.get('labels', [])
    preds = data.get('predictions', [])
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    for i in range(num_classes):
        for j in range(num_classes):
            if cm[i,j] > 0:
                color = 'white' if cm[i,j] > cm.max()/2 else 'black'
                ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=9,
                        fontweight='bold', color=color)
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xticklabels(LABEL_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(LABEL_NAMES, fontsize=8)
    ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax.set_ylabel('True', fontsize=11, fontweight='bold')
    acc = data['accuracy']
    ax.set_title(f'{title}\nAccuracy: {acc*100:.1f}%', fontsize=13, fontweight='bold', pad=12)
    plt.colorbar(im, ax=ax, shrink=0.85, label='Count')
    plt.tight_layout()
    plt.savefig(f'docs/figures/{output_name}', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  → docs/figures/{output_name}')


def fig_crossuser_degradation():
    """Figure: cross-user degradation — in-user vs cross-user gap per model."""
    print('Generating: Cross-user degradation...')
    fig, ax = plt.subplots(figsize=(10, 5.5))
    names_short = ['E1','E2','E3','E4','E5','E6','GCAR']
    colors = ['#e74c3c','#e67e22','#2ecc71','#3498db','#9b59b6','#1abc9c','#95a5a6']
    in_vals = [e[1]*100 for e in EXPERIMENTS]
    cross_vals = [e[2]*100 for e in EXPERIMENTS]
    x = np.arange(len(names_short))
    ax.bar(x - 0.15, in_vals, 0.3, label='In-User', color='#2ecc71', alpha=0.85, edgecolor='white')
    ax.bar(x + 0.15, cross_vals, 0.3, label='Cross-User', color='#3498db', alpha=0.85, edgecolor='white')
    # Degradation arrows
    for i in range(len(names_short)):
        gap = in_vals[i] - cross_vals[i]
        if gap > 3:
            ax.annotate(f'-{gap:.1f}pp', xy=(i+0.15, cross_vals[i]),
                        xytext=(i+0.55, cross_vals[i]-5 if gap>10 else cross_vals[i]-3),
                        fontsize=8, color='#e74c3c', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.2))
    ax.set_xticks(x); ax.set_xticklabels(names_short, fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Figure 3-13: Cross-User Generalization Gap', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.2)
    ax.set_ylim(60, 105)
    plt.tight_layout()
    plt.savefig('docs/figures/fig_crossuser_degradation_v6.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  → docs/figures/fig_crossuser_degradation_v6.png')


def fig_params_accuracy():
    """Figure: parameter count vs accuracy scatter plot."""
    print('Generating: Params-Accuracy scatter...')
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, exp in enumerate(EXPERIMENTS):
        name, in_u, cross_u, conf, params, color, marker, multi = exp
        ax.scatter(params, in_u*100, s=200, c=color, marker=marker, zorder=5,
                   edgecolors='white', linewidth=1.2)
        ax.annotate(exp[0].replace('\n',' '),
                    (params, in_u*100), textcoords="offset points",
                    xytext=(8, 4), fontsize=8, fontweight='bold', color=color)
    ax.set_xlabel('Parameters (Millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('In-User Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3-14: Parameter Efficiency — Accuracy vs Model Size',
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.2, linestyle='--')
    # Annotate E3 as optimal
    ax.annotate('E3: Optimal\n2.80M / 96.7%',
                xy=(2.80, 96.7), xytext=(1.0, 88),
                fontsize=10, color='#2ecc71', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.5))
    plt.tight_layout()
    plt.savefig('docs/figures/fig_params_accuracy_v6.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  → docs/figures/fig_params_accuracy_v6.png')


def print_results_table():
    """Print LaTeX-friendly results table."""
    print('\n=== Ablation Table (LaTeX-ready) ===')
    print(f'{"Experiment":<22} {"In-User":>8} {"Cross-User":>10} {"Confusion":>9} {"Params":>7} {"Multi?":>6}')
    print('-' * 68)
    for exp in EXPERIMENTS:
        name = exp[0].replace('\n',' ')
        print(f'{name:<22} {exp[1]*100:>6.1f}%  {exp[2]*100:>8.1f}%  {exp[3]*100:>8.1f}%  {exp[4]:>5.2f}M  {"Yes" if exp[7] else "No":>6}')


def main():
    print('=' * 60)
    print('Thesis Figure Generation v6.0')
    print('=' * 60)
    fig_ablation()
    fig_training_curves()
    fig_confusion_matrix('data/test_inuser_e4_result.json', 'fig_confusion_e4_v6.png',
                         'Figure 3-11: E4 Confusion Matrix (In-User)')
    fig_confusion_matrix('data/test_inuser_e6_result.json', 'fig_confusion_e6_v6.png',
                         'Figure 3-12: E6 Confusion Matrix (In-User)')
    fig_crossuser_degradation()
    fig_params_accuracy()
    print_results_table()
    print('\nDone. All figures saved to docs/figures/')


if __name__ == '__main__':
    main()
