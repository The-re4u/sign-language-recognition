"""Restore missing Chapter 4 experiments content."""
with open('docs/毕业论文_完整英文版.md', 'r', encoding='utf-8') as f:
    content = f.read()

idx1 = content.find('## 3.8 Chapter Summary')
idx2 = content.find('# CHAPTER 5: SYSTEM DESIGN')

ch4 = '''
## 3.8 Chapter Summary

The proposed method spans seven processing layers, each contributing a distinct function to the gesture-to-language pipeline. The perception layer provides robust, real-time hand keypoint detection with clinical-grade normalization. The rule engine provides XAI-compliant, instantaneous classification with clinical traceability, while the dual-hand combinatorial encoding protocol achieves 4.7-fold semantic efficiency through compositional mode+content separation.

---

# CHAPTER 4: EXPERIMENTS AND ANALYSIS

This chapter presents comprehensive experimental evaluation of the proposed system, covering dataset construction, gesture classification performance, ablation studies, training dynamics, system benchmarks, and discussion.

## 4.1 Dataset Construction

The dataset combines self-collected real samples and synthetic augmentation. Public datasets (WLASL [29], MS-ASL, AUTSL) use single-hand annotation schemes incompatible with the dual-hand combinatorial encoding protocol.

**Self-Collected Data:** 351 real sessions (13 classes x 27 samples) across 3 lighting x 3 angle x 3 distance = 27 controlled conditions. Each 2-second recording yields synchronized keypoints (.npy) and 96x96 ROI crops (_roi.npy).

**Synthetic Data Augmentation:** 960 samples (80 per class, 12 classes) with domain randomization: keypoint jitter (sigma=0.008-0.012, 100%, calibrated to Hamaguchi [4] MAD 2.46 degrees), finger occlusion (1-5 keypoints zeroed, 30%), low-light simulation (jitter increased to 0.015-0.030, 40%), and rotation plus scaling (plus or minus 3 degrees and 3%, 100%).

**Dataset Split:** 1,311 samples split 70/15/15 (906/196/209) via stratified sampling.

> **[Figure 4-1: Dataset Composition]**
> *Insert: docs/figures/fig10_dataset.png*

## 4.2 Experimental Setup

The DL model was trained for 50 epochs with AdamW (lr=1e-4, weight_decay=1e-4), batch size 8, 5-epoch linear warmup + cosine annealing, and CE loss with temporal mean pooling. Mixed precision was disabled for CPU numerical stability. MobileNetV3-Small backbone remained frozen. The rule engine was evaluated directly without training.

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Batch Size | 8 |
| Epochs | 50 |
| Warmup Epochs | 5 |
| LR Schedule | Cosine Annealing |
| Loss Function | CE with Temporal Mean Pooling |

> **[Figure 4-2: Learning Rate Schedule]**
> *Insert: docs/figures/fig8_lr_schedule.png*

## 4.3 Gesture Classification Performance

The multimodal DL model achieved 82.8% test accuracy [95% CI: 77.4%-87.6%, Bootstrap 1,000 resamples], substantially outperforming the rule engine (31.1% [95% CI: 24.9%-37.8%]). A 5-fold stratified cross-validation confirmed these findings with mean accuracy 84.1% (plus or minus 1.5% std, range 81.7%-85.5%).

| Model | Test Accuracy | 95% CI | Parameters | Inference |
|-------|--------------|--------|------------|-----------|
| Rule Engine | 31.1% | [24.9%, 37.8%] | 0 | <1 ms |
| DL (Keypoints Only) | 73.2% | [66.8%, 79.1%] | ~0.9M | ~30 ms |
| DL (Multimodal) | 82.8% | [77.4%, 87.6%] | 3.4M | ~50 ms |

The rule-engine gap (97.6% synthetic to 31.1% real) reveals the limitation of hard-coded thresholds. Per-class analysis reveals complementary error patterns. DL achieves 100% on Eight, Good, and Nine. The weakest DL performers are Open_Palm (41.2%) and Victory (47.1%). Error analysis: Open_Palm misclassified as Nine (6/10 errors) and Four (3/10); Victory misclassified as Two (6/9 errors) and Seven (3/9).

> **[Figure 4-3: Rule vs DL Overall Comparison]**
> *Insert: docs/figures/fig3_comparison.png*
>
> **[Figure 4-4: DL Confusion Matrix]**
> *Insert: docs/figures/fig7_confusion_matrix.png*
>
> **[Figure 4-5: Per-Class Rule vs DL Comparison]**
> *Insert: docs/figures/fig9_per_class_comparison.png*

## 4.4 Ablation Study

Systematic modality removal quantifies contribution:

| Configuration | Test Accuracy | Delta |
|--------------|---------------|-------|
| Full Model (GCN + Visual + Motion + Fusion + TCN) | 82.8% | Baseline |
| Without Visual Encoder | 73.2% | -9.6 pp |
| Without Motion Encoder | 51.2% | -31.6 pp |
| Without GCN | 6.2% | -76.6 pp |

GCN removal causes catastrophic collapse -- spatial topology is foundational. Motion contributes 31.6pp -- temporal dynamics matter even for static gestures (captures subtle pose transitions, tremors, velocity patterns). Visual adds 9.6pp -- RGB texture provides complementary appearance information.

> **[Figure 4-6: Ablation Study Results]**
> *Insert: docs/figures/fig4_ablation.png*

## 4.5 Training Convergence

Training loss: 2.62 (epoch 1) to 0.15 (epoch 50), with validation loss minimum 0.32 at epoch 37. Validation accuracy: 8.3% to peak 90.6% (epoch 37), stabilizing at 85-90% for final epochs. Two learning phases: rapid discrimination (epochs 1-13, 8.3% to 55.7%) and gradual refinement (epochs 14-50) under cosine annealing. Mild overfitting (train 0.15 vs. val 0.43 at epoch 50) is expected given 906 training samples.

> **[Figure 4-7: Training Convergence -- Loss + Accuracy]**
> *Insert: docs/figures/fig1_loss_curve.png*

## 4.6 System Performance Benchmarks

92 recording sessions measured. Detection: 31.5 ms (31.7 FPS) exceeding 25 FPS requirement. Rule engine: <0.05 ms. DL feature extraction: approx 45 ms (async daemon, non-blocking). ONNX TCN: approx 5 ms. End-to-end: 32.1 ms. Frame rate: 22.7 FPS stable (SD <2 FPS).

| Pipeline Stage | Latency | Mode |
|---------------|---------|------|
| MediaPipe Detection | 31.5 ms | Synchronous |
| Rule Engine | <0.05 ms | Synchronous |
| DL Feature Extraction | approx 45 ms | Async daemon |
| ONNX TCN Inference | approx 5 ms | Async daemon |
| End-to-End Processing | 32.1 ms | Per-frame average |

> **[Figure 4-8: Pipeline Module Latency Breakdown]**
> *Insert: docs/figures/fig6_latency.png*
>
> **[Figure 4-9: System FPS Stability Across 92 Sessions]**
> *Insert: docs/figures/fig11_fps_stability.png*

## 4.7 Discussion

**Rule-DL Complementarity:** The rule engine embodies explicit anatomical knowledge (97.6% on idealized synthetic data, 31.1% on real data). The DL model learns user-specific features (82.8%, 51.7pp improvement). This complementarity motivates the dual-mode architecture.

**Limitations:** The dataset of 1,311 samples is modest; no formal usability testing with deaf participants has been conducted; DL inference latency of approximately 2 seconds limits instantaneous feedback; challenging classes (Open_Palm 41.2%, Victory 47.1%) need targeted augmentation. The scene-transferable architecture enables migration across hospital, banking, airport, and smart home scenarios without code modification.

## 4.8 Chapter Summary

Comprehensive experiments on 1,311 samples demonstrated: 82.8% DL test accuracy (51.7pp over rule engine), ablation quantifying modality contributions (GCN 76.6pp, motion 31.6pp, visual 9.6pp), 90.6% peak validation accuracy, and 31.7 FPS production performance across 92 sessions. The complementary relationship between rule-based and DL approaches motivates the dual-mode architecture.

---

'''

content = content[:idx1] + ch4 + content[idx2:]
words = len(content.split())
print(f'Chapter 4 restored. Total: {words:,} words')

with open('docs/毕业论文_完整英文版.md', 'w', encoding='utf-8') as f:
    f.write(content)
