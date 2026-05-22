"""Aggressive trim to ~15,000 words. Uses regex for robustness."""
import re

with open('docs/毕业论文_完整英文版.md', 'r', encoding='utf-8') as f:
    content = f.read()

start_words = len(content.split())
print(f'Start: {start_words:,} words')

# Helper: replace if found, report cut size
def cut(old, new, label=''):
    global content
    if old in content:
        n = len(old.split()) - len(new.split())
        content = content.replace(old, new)
        return n
    # Try em-dash variant
    alt = old.replace('--', '—')
    if alt in content:
        n = len(alt.split()) - len(new.split())
        content = content.replace(alt, new)
        return n
    return 0

total_cut = 0

# ====== CH1: 2,692 → 1,200 (cut ~1,500) ======
# Remove entire verbose gap analysis subsections, keep only positioning
total_cut += cut(
    '### 1.2.1 Gap Analysis of Existing Solutions\n\n'
    'To contextualize the contribution of this work, it is necessary to examine the existing landscape of assistive communication technologies available to deaf individuals in China and identify their fundamental limitations. Five categories of existing solutions are analyzed below.',
    '### 1.2.1 Existing Solutions and Their Limitations\n\n'
    'Five categories of assistive communication technologies are examined below.',
    'Ch1-gap-intro')

# Cut Professional Interpreters paragraph aggressively
old_ip = re.search(r'\*\*Professional Sign Language Interpreters:\*\*.*?accessibility system\.', content, re.DOTALL)
if old_ip:
    total_cut += cut(old_ip.group(),
        '**Professional Sign Language Interpreters:** China has ~20M hearing-impaired but only ~10,000 certified interpreters (2,000:1 ratio). Advance booking (24-48h) and costs (300-800 RMB/session) make interpreter access impractical for urgent hospital communication.',
        'Ch1-interpreters')

# Cut Mobile Apps paragraph aggressively
old_ma = re.search(r'\*\*Mobile Translation Applications:\*\*.*?customization capability\.', content, re.DOTALL)
if old_ma:
    total_cut += cut(old_ma.group(),
        '**Mobile Translation Applications:** Existing apps operate on isolated word recognition, function as passive dictionaries without sentence composition, and use fixed vocabularies tied to specific sign languages without customization capability.',
        'Ch1-mobile')

# Cut Text-Based paragraph
old_tb = re.search(r'\*\*Text-Based Communication \(Typing/Note-Taking\):\*\*.*?medical consultation\.', content, re.DOTALL)
if old_tb:
    total_cut += cut(old_tb.group(),
        '**Text-Based Communication:** Mobile typing (20-30 wpm vs. 120-180 signs/min for natural signing) imposes the double conversion cognitive burden and undermines face-to-face consultation quality.',
        'Ch1-text')

# Cut Glove paragraph
old_gl = re.search(r'\*\*Wearable Glove-Based Systems:\*\*.*?sign language expressions\.', content, re.DOTALL)
if old_gl:
    total_cut += cut(old_gl.group(),
        '**Wearable Glove-Based Systems:** Despite high laboratory accuracy (>95%), glove-based systems face prohibitive barriers: expensive hardware ($100s-$1000s), physical discomfort, visible social stigma, and calibration/maintenance requirements.',
        'Ch1-glove')

# Cut Academic Cameras paragraph
old_ac = re.search(r'\*\*Camera-Based Academic Recognition Systems:\*\*.*?no extensibility\.', content, re.DOTALL)
if old_ac:
    total_cut += cut(old_ac.group(),
        '**Camera-Based Academic Systems:** Recent deep learning systems (SignGemma 4B, ST-GCN, Transformer) are GPU-dependent, output class labels rather than communicable sentences, and use fixed vocabularies without scene customization.',
        'Ch1-academic')

# Compress 1.2.2 Positioning
old_pos = re.search(r'### 1\.2\.2 Positioning of This Work.*?recognition code\.', content, re.DOTALL)
if old_pos:
    total_cut += cut(old_pos.group(),
        '### 1.2.2 Positioning of This Work\n\n'
        'The proposed system addresses gaps left by each existing category: 24/7 availability vs. interpreter scheduling; continuous sentence-level expression vs. isolated word lookup in mobile apps; zero additional hardware vs. expensive gloves; CPU-only 3.4M-parameter model vs. GPU-dependent academic systems; and scene-transferable JSON configuration vs. fixed-vocabulary benchmarks.\n\n'
        'The primary objective is a lightweight, real-time system satisfying four constraints: commodity RGB webcam operation, DL parameters below 10M for CPU inference, end-to-end latency below 500ms, and graceful degradation. The significance spans three dimensions: technical (LLM-enhanced gesture-to-language pipeline), social (accessible communication for 20M hearing-impaired in China), and methodological (JSON-configurable scene portability).',
        'Ch1-positioning')

# ====== CH3: 6,389 → 4,900 (cut ~1,500) ======
# Remove verbose normalization formula text
old_norm = re.search(r'palm_scale = \|\| kp_9 - kp_0 \|\|.*?operating range\.', content, re.DOTALL)
if old_norm:
    total_cut += cut(old_norm.group(),
        'palm_scale = ||kp_9 - kp_0||; kp\'_i = (kp_i - kp_0) / palm_scale. This canonical normalization eliminates camera distance and user position variation.',
        'Ch3-norm')

# Remove verbose visual encoder zero-input paragraph
old_zi = re.search(r'\*\*Zero-Input Handling:\*\*.*?visual information is absent\.', content, re.DOTALL)
if old_zi:
    total_cut += cut(old_zi.group(),
        'Synthetic samples without ROI receive zero tensors; an early-exit path prevents BatchNorm NaN. The model learns to rely more heavily on geometric and motion features when visual information is absent.',
        'Ch3-zeroinput')

# Remove long CrossModalFusion "Why" paragraph
old_wm = re.search(r'\*\*Why Learnable Modality Embeddings\*\*:.*?reliable geometric and motion tokens\.', content, re.DOTALL)
if old_wm:
    total_cut += cut(old_wm.group(),
        'Learnable modality embeddings enable attention heads to dynamically weight modalities — favoring geometric features under poor lighting where visual features are degraded.',
        'Ch3-why-modal')

# Compress Ch3.5 long stage descriptions (Stage 1-4)
old_s1 = re.search(r'\*\*Stage 1.*?dual-channel scrutiny\.', content, re.DOTALL)
if old_s1:
    total_cut += cut(old_s1.group(),
        '**Stage 1:** Finger extension angle theta_f = angle(kp_MCP, kp_PIP, kp_DIP); extended if >165deg (Hamaguchi [4], r=0.78).\n'
        '**Stage 2:** 5-bit pattern (32 combinations) mapped to 15 gesture classes. Two vs. Victory disambiguated by index-middle fingertip Euclidean distance (threshold 0.06 normalized units).\n'
        '**Stage 3:** Dual-channel PIP/MCP verification — PIP joints exhibit the highest MediaPipe error (20-30deg [2][3]). Per-finger disagreement applies 12% confidence penalty.\n'
        '**Stage 4:** 5-bit pattern maps to gesture name. PIP-sensitive classes (Victory, Six, Three, Four, Seven, Nine [3]) receive additional scrutiny.',
        'Ch3-stages')

# ====== CH4: 3,952 → 2,500 (cut ~1,400) ======
# Cut verbose dataset intro
old_ds = re.search(r'\*\*Self-Collected Data:\*\*.*?gesture poses\.', content, re.DOTALL)
if old_ds:
    total_cut += cut(old_ds.group(),
        '**Self-Collected Data:** 351 real sessions (13 classes x 27 samples) across 3 lighting x 3 angle x 3 distance = 27 controlled conditions. Each 2-second recording yields synchronized keypoints (.npy) and 96x96 ROI crops (_roi.npy).',
        'Ch4-selfdata')

# Cut verbose training rationale table
old_tr = re.search(r'\*\*Training Configuration:\*\*.*?without any training.*?production system\.', content, re.DOTALL)
if old_tr:
    total_cut += cut(old_tr.group(),
        'The DL model was trained for 50 epochs with AdamW (lr=1e-4, weight_decay=1e-4), batch size 8, 5-epoch linear warmup + cosine annealing, and CE loss with temporal mean pooling. Mixed precision was disabled for CPU numerical stability. MobileNetV3-Small backbone remained frozen. The rule engine was evaluated directly without training.',
        'Ch4-training')

# Cut long per-class table (keep table, cut verbose intro)
old_pc = re.search(r'The per-class accuracy reveals.*?decision boundary\.', content, re.DOTALL)
if old_pc:
    total_cut += cut(old_pc.group(),
        'Per-class analysis reveals complementary error patterns across approaches (detailed in the table below).',
        'Ch4-perclass-intro')

# Cut verbose training dynamics paragraphs (keep concise)
old_td1 = re.search(r'Training loss trajectory:.*?fine-grained convergence.*?later training stages\.', content, re.DOTALL)
if old_td1:
    total_cut += cut(old_td1.group(),
        'Training loss: 2.62 (epoch 1) to 0.15 (epoch 50), with validation loss minimum 0.32 at epoch 37. Validation accuracy: 8.3% to peak 90.6% (epoch 37), stabilizing at 85-90% for final epochs. Two learning phases observed: rapid discrimination learning (epochs 1-13) and gradual refinement (epochs 14-50) under cosine annealing.',
        'Ch4-training-dynamics')

old_td2 = re.search(r'The gap between training.*?sufficient for the 3\.4M-parameter model employed\.', content, re.DOTALL)
if old_td2:
    total_cut += cut(old_td2.group(),
        'Mild overfitting (train 0.15 vs. val 0.43 at epoch 50) is expected given 906 training samples for a 3.4M-parameter model.',
        'Ch4-overfit')

# Cut verbose complementarity paragraph
old_cp = re.search(r'\*\*The Complementary Nature of Rule and DL:\*\*.*?motivates the dual-mode architecture.*?prioritizing accuracy\.', content, re.DOTALL)
if old_cp:
    total_cut += cut(old_cp.group(),
        '**Rule-DL Complementarity:** The rule engine embodies explicit anatomical knowledge (97.6% on idealized synthetic data, 31.1% on real data with anatomical variation). The DL model learns user-specific features from data (82.8% on real data, 51.7pp improvement). This complementarity motivates the dual-mode architecture: rule for XAI-guaranteed production, DL for accuracy-optimized enhancement.',
        'Ch4-complement')

# ====== CH5: 2,335 → 1,300 (cut ~1,000) ======
# Compress FR list
old_fr = re.search(r'\*\*FR1.*?\*\*FR9.*?recognition code\.', content, re.DOTALL)
if old_fr:
    total_cut += cut(old_fr.group(),
        '**FR1:** Real-time dual-hand detection via standard RGB webcam. **FR2:** 15-gesture recognition (7 mode + 10 content, 4 dual-role). **FR3:** Dual-mode recognition with one-click rule/DL switching. **FR4:** Dual-hand combinatorial encoding (70 semantic slots). **FR5:** Natural Chinese sentence output via chain matching + DeepSeek polishing. **FR6:** AI triage with multi-turn follow-up and department recommendation. **FR7:** Web GUI with real-time skeleton, gesture cards, FPS monitoring. **FR8:** Video upload for offline testing. **FR9:** Scene migration via JSON configuration replacement.',
        'Ch5-fr')

# Compress NFR list
old_nfr = re.search(r'\*\*NFR1.*?\*\*NFR6.*?without plugin installation.*?scenarios\.', content, re.DOTALL)
if old_nfr:
    total_cut += cut(old_nfr.group(),
        '**NFR1:** Detection >=25 FPS (achieved 31.7 FPS). **NFR2:** DL parameters <10M (3.4M, 66% margin). **NFR3:** CPU-only operation. **NFR4:** End-to-end latency <500ms. **NFR5:** Four-level graceful degradation. **NFR6:** Cross-platform browser accessibility.',
        'Ch5-nfr')

# ====== CH6: 1,617 → 1,000 (cut ~600) ======
# Compress future work items
old_fw = re.search(r'1\. \*\*Deaf User Testing.*?for DeepSeek API access\.', content, re.DOTALL)
if old_fw:
    total_cut += cut(old_fw.group(),
        '1. **User Testing:** Formal SUS studies with deaf participants to validate gesture vocabulary and identify ergonomic improvements.\n'
        '2. **Dynamic Gestures:** Extend from static poses to dynamic motions (circles, swipes) for trajectory-based semantic expansion.\n'
        '3. **Multi-Camera Triangulation:** Per Maggioni et al. [2], dual-camera setup could reduce PIP error from 20-30deg to ~10.9deg.\n'
        '4. **Kalman Filtering:** Per Hamaguchi et al. [4], keypoint filtering could reduce MAD from 2.46deg to 0.81deg.\n'
        '5. **Parameter Optimization:** CrossModalFusion (1.9M, 56% of parameters) could be replaced with lightweight gating [22]; INT8 quantization (4x compression to ~0.7MB ONNX). With MediaPipe Lite (~2MB), total footprint could drop from ~10.5MB to <3MB, enabling Raspberry Pi deployment.\n'
        '6. **Continuous SLR:** CTC/RNN-T for variable-length gesture sequences, replacing SentenceRecorder segmentation.\n'
        '7. **Mobile Deployment:** Smartphone packaging of the complete CPU pipeline.',
        'Ch6-future')

# ====== FINAL COUNT ======
words = len(content.split())
print(f'Total cuts: {total_cut:,} words')
print(f'Final: {words:,} words (from {start_words:,})')

with open('docs/毕业论文_完整英文版.md', 'w', encoding='utf-8') as f:
    f.write(content)
