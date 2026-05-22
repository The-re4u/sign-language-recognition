"""Trim thesis from 21,770 to ~15,000 words following reference thesis proportions."""
with open('docs/毕业论文_完整英文版.md', 'r', encoding='utf-8') as f:
    content = f.read()

# ====== Ch1: 2,692→1,200 (cut ~1,500) ======
# Strategy: Compress the verbose gap analysis, merge tech stack justification

# Cut: long interpreter paragraph → 1 sentence
old = '''**Professional Sign Language Interpreters:** China has approximately 20 million hearing-impaired individuals but only about 10,000 certified sign language interpreters nationwide — a ratio of roughly 2,000 to 1. The shortage is particularly acute in public service settings such as community hospitals, where dedicated interpreters are rarely available outside of major urban centers. Even when interpreters can be scheduled, the need for advance booking (typically 24-48 hours) eliminates spontaneity entirely — a deaf patient experiencing sudden acute symptoms cannot communicate effectively with triage staff without waiting hours or days for interpreter availability. The cost of professional interpretation services (typically 300-800 RMB per session in urban areas) further limits accessibility for economically disadvantaged deaf individuals, creating a two-tier system where those who can afford interpreters receive adequate healthcare communication while those who cannot are left to struggle with makeshift alternatives.'''
new = '''**Professional Sign Language Interpreters:** China has approximately 20 million hearing-impaired individuals but only about 10,000 certified interpreters — a 2,000:1 ratio. The shortage in public hospitals means deaf patients experiencing acute symptoms cannot communicate effectively without hours of waiting. Costs (300-800 RMB per session) create a two-tier accessibility system.'''
content = content.replace(old, new)

# Cut: long mobile apps paragraph
old = '''**Mobile Translation Applications:** Existing smartphone-based sign language translation apps (e.g., SignAll, HandTalk, various CSL dictionary applications) share three fundamental limitations. First, they operate on isolated word recognition — the user signs a single word, and the app displays a text equivalent. This word-by-word interaction model fundamentally breaks the natural flow of language, requiring the user to artificially segment their communication into discrete dictionary lookups rather than expressing complete thoughts. Second, these apps function as passive dictionaries rather than active communication agents — they output a word but do not compose sentences, maintain conversational context, interpret pragmatic intent, or adapt to the communicative norms of specific settings (e.g., the polite register and structured symptom description expected in medical consultations). Third, they are typically designed for one specific sign language (ASL, CSL) with fixed vocabularies and cannot be customized to novel gesture sets, domain-specific terminology, or application-specific workflows — a hospital cannot adapt a generic CSL app to include local medical vocabulary or triage-specific question-answer protocols.'''
new = '''**Mobile Translation Applications:** Existing apps share three limitations: (1) isolated word recognition breaks natural language flow; (2) passive dictionary functionality without sentence composition or conversational context; (3) fixed vocabulary tied to specific sign languages without customization capability.'''
content = content.replace(old, new)

# Cut: long typing paragraph
old = '''**Text-Based Communication (Typing/Note-Taking):** The most commonly used makeshift solution in practice — deaf patients typing on mobile phones or writing on paper — imposes the double conversion cognitive burden described in Section 1.1. Beyond cognitive load, this approach introduces significant throughput limitations: typical mobile typing speeds of 20-30 words per minute are far below natural sign language expression rates, which are comparable to speech at 120-180 signs per minute. The resulting communication asymmetry — where the deaf person spends the majority of the interaction looking down at a screen rather than engaging face-to-face with the hearing participant — undermines the interpersonal rapport and non-verbal cue exchange essential to effective medical consultation. Studies in clinical communication have consistently shown that patient outcomes are correlated with perceived quality of doctor-patient interaction, not merely with information transfer accuracy.'''
new = '''**Text-Based Communication:** Typing on mobile phones (20-30 words/minute vs. 120-180 signs/minute for natural sign language) imposes the double conversion burden and undermines face-to-face engagement essential to medical consultation.'''
content = content.replace(old, new)

# Cut: long glove paragraph
old = '''**Wearable Glove-Based Systems:** Academic and commercial research has produced glove-based sign language recognition systems using flex sensors, inertial measurement units (IMUs), and haptic feedback mechanisms. While these systems can achieve high raw classification accuracy (>95% in controlled laboratory settings), they face insurmountable practical barriers to real-world adoption: (a) the specialized hardware costs hundreds to thousands of dollars per unit, placing them beyond the economic reach of the majority of deaf individuals, particularly in developing regions; (b) wearing sensor-equipped gloves for extended periods is physically uncomfortable and visibly stigmatizing in public settings, creating a conspicuous \"disability marker\" that many deaf individuals actively prefer to avoid in social interactions; (c) the hardware requires regular calibration, charging, and maintenance, introducing reliability concerns for continuous daily use; and (d) the physical interface constrains natural hand movement and precludes the fine-grained finger articulation essential to many sign language expressions.'''
new = '''**Wearable Glove-Based Systems:** Despite high lab accuracy (>95%), glove-based systems face practical barriers: expensive hardware, physical discomfort, social stigma as a visible \"disability marker,\" and maintenance requirements.'''
content = content.replace(old, new)

# Cut: long academic cameras paragraph - keep but compress
old = '''**Camera-Based Academic Recognition Systems:** The recent wave of deep learning-based camera systems — including Google's SignGemma (4B parameters, ASL-to-English), various ST-GCN architectures, and Transformer-based approaches surveyed in Chapter 2 — has demonstrated impressive classification accuracy on benchmark datasets. However, these systems universally share three deployment-limiting characteristics: (a) they require GPU hardware for inference, with model sizes ranging from tens of millions to billions of parameters, preventing deployment on commodity laptops, embedded devices, or smartphones; (b) they treat gesture recognition as an end in itself — the system output is a class label or word, not a communicable sentence, leaving the crucial step of compositional language generation unaddressed; and (c) they are trained on fixed-vocabulary benchmark datasets (WLASL, AUTSL, CSL) with no architectural mechanism for scene-specific customization, domain adaptation, or post-deployment vocabulary extension — the vocabulary learned during training is the vocabulary available at deployment, with no extensibility.'''
new = '''**Camera-Based Academic Systems:** Recent systems (SignGemma 4B parameters, various ST-GCN/Transformer architectures) share three limitations: GPU-dependent inference, output limited to class labels rather than communicable sentences, and fixed vocabularies without scene customization mechanisms.'''
content = content.replace(old, new)

# ====== Ch3: 6,389→4,500 (cut ~1,900) ======
# Strategy: Remove verbose "rationale" paragraphs, keep the method description

# Cut: long normalization paragraph → keep formula only
old = '''palm_scale = || kp_9 - kp_0 ||

All 21 keypoints are then normalized:

kp'_i = (kp_i - kp_0) / palm_scale

This normalization maps the hand to a canonical coordinate frame where the wrist is at the origin and the palm size is unit length. This eliminates variations due to camera distance (a hand closer to the camera appears larger but has the same normalized keypoint positions) and provides a consistent input space for downstream models regardless of the user's position within the recommended 30-80cm operating range.'''
new = '''palm_scale = || kp_9 - kp_0 ||; kp'_i = (kp_i - kp_0) / palm_scale. This maps the hand to a canonical coordinate frame invariant to camera distance and user position.'''
content = content.replace(old, new)

# Cut: long auxiliary angle encoder explanation
old = '''The auxiliary angle encoder addresses a fundamental limitation of pure GCN approaches: geometric relationships that are trivial for human anatomy (e.g., \"the index finger is extended when its MCP-PIP-DIP angle exceeds 165 degrees\") must be learned from data in a standard GCN, requiring substantial training samples. By explicitly computing and encoding these angles, the model receives strong geometric priors that accelerate learning and improve performance on limited data. This design choice was directly motivated by the clinical literature [4] that established the 165-degree threshold as a clinically validated criterion for finger extension.'''
new = '''The auxiliary angle encoder addresses a limitation of pure GCNs: geometric relationships that are anatomically obvious must otherwise be learned from data. By explicitly encoding joint angles, the model receives strong geometric priors that accelerate learning on limited data [4].'''
content = content.replace(old, new)

# Cut: long visual encoder explanation
old = '''**Freezing Strategy**: The backbone's 2.5M parameters are frozen during training — their weights remain at the ImageNet pre-trained values. This decision was motivated by the limited dataset size (1,311 samples). Training a 2.5M-parameter CNN from scratch, or even fine-tuning it, on such limited data would lead to severe overfitting where the model memorizes the training samples rather than learning generalizable visual features. The frozen backbone provides robust, general-purpose visual features that transfer well to the hand ROI domain, while the small trainable projection head (0.15M parameters) adapts these features to the specific gesture classification task.'''
new = '''The backbone's 2.5M parameters are frozen at ImageNet pre-trained values to prevent overfitting on the limited 1,311-sample dataset. The trainable projection head (0.15M) adapts generic visual features to the gesture classification task.'''
content = content.replace(old, new)

# Cut: long input preparation for visual encoder
old = '''**Input Preparation**: A 96x96 pixel ROI is cropped around the detected hand keypoints. The crop region is computed as the bounding box of all 21 keypoints expanded by a 25-pixel margin in each direction. The crop is resized to exactly 96x96 pixels using bilinear interpolation. Pixel values are normalized from [0, 255] to [0, 1].

**Zero-Input Handling**: Synthetic training samples lack corresponding ROI images (the data generator produces only keypoint sequences). For these samples, the encoder receives a tensor of zeros. A dedicated early-exit path checks `x.abs().sum() == 0` at the start of the forward pass and returns a zero vector immediately, bypassing the MobileNetV3 backbone entirely. This prevents BatchNorm layers in the backbone from producing NaN values due to zero-variance input (a numerical stability issue encountered during development). The model is trained with a mixture of real ROI samples and zero-input synthetic samples, learning to rely more heavily on geometric (GCN) and motion features when visual information is absent.'''
new = '''A 96x96 pixel ROI is cropped around keypoints with a 25-pixel margin and resized. Synthetic samples without ROI images receive zero tensors; a dedicated early-exit path prevents BatchNorm NaN from zero-variance input. The model learns to rely on geometric and motion features when visual information is absent.'''
content = content.replace(old, new)

# Cut: long motion encoder explanation
old = '''Two input streams are computed from consecutive frame keypoints:

1. **Keypoint Velocity Field (63 dimensions):** For each of the 21 keypoints, the 2D displacement vector (dx, dy) between frames t-1 and t is computed. Additionally, the magnitude of the displacement (sqrt(dx^2 + dy^2)) is included for each keypoint, yielding 21x3 = 63 dimensions. This captures both the direction and speed of each joint's movement.

2. **Optical Flow Histogram (128 dimensions):** The 21 keypoint displacement vectors are aggregated into a flow histogram inspired by optical flow representations in video analysis. The first 21 bins accumulate displacement magnitudes per keypoint, capturing which joints moved most. The remaining 107 bins form a 42-bin directional histogram (0-360 degrees in 8.57-degree increments) accumulated across all keypoints, with each keypoint's contribution weighted by its displacement magnitude. This captures the dominant motion direction and distribution.

The two streams are processed through separate small fully-connected networks (63 to 64 to 64 and 128 to 64 to 64), then fused through a learned softmax-weighted combination:'''
new = '''Two input streams from consecutive frames: (1) 63-dim keypoint velocity field — 21 keypoints x (dx, dy, magnitude); (2) 128-dim optical flow histogram — 21 magnitude bins + 42-bin directional histogram. Separate small FC networks (63/128 to 64 to 64) with learned softmax-weighted fusion:'''
content = content.replace(old, new)

# Cut: long XAI explanation
old = '''**Why XAI matters**: Every classification traces to specific joint angles and clinical thresholds. A \"Victory\" classification can be explained as: \"Index and middle MCP-PIP-DIP angles > 165deg (extended), other three fingers < 165deg (flexed), index-middle tip distance > 0.06 normalized units (separated).\"'''
new = '''Every classification traces to specific joint angles and clinical thresholds — for example, \"Victory: index and middle MCP-PIP-DIP > 165deg, others < 165deg, tip distance > 0.06.\"'''
content = content.replace(old, new)

# Cut: long TCN rationale
old = '''**Selection rationale over Transformer**: A 10M+ parameter Transformer would overfit severely on 1,311 samples. The 0.7M TCN's causal convolutions are naturally suited for real-time inference without future-frame leakage, and ONNX-exportable convolutions are more deployment-friendly than attention operations.'''
new = '''A 10M+ Transformer would overfit on 1,311 samples; the 0.7M TCN provides causal, ONNX-friendly convolutions suitable for real-time deployment.'''
content = content.replace(old, new)

# ====== Ch4: 3,952→2,500 (cut ~1,400) ======
# Cut verbose dataset description
old = '''The dataset was constructed through a combination of self-collected real samples and synthetically augmented data, following the task book's provision ("collect or use public sign language datasets") that explicitly permits self-collection. This choice was further motivated by the unavailability of public datasets supporting the dual-hand combinatorial encoding protocol — existing datasets (WLASL [29], MS-ASL, AUTSL) use single-hand, single-gesture-to-single-word annotation schemes that cannot represent the mode+content gesture combinations central to this system.'''
new = '''The dataset combines self-collected real samples and synthetic augmentation. Public datasets (WLASL [29], MS-ASL, AUTSL) use single-hand annotation schemes incompatible with the dual-hand combinatorial encoding protocol.'''
content = content.replace(old, new)

# Cut verbose synthetic data description
old = '''**Synthetic Data Augmentation:** The synthetic data generator (generate_synthetic_data.py) produced 960 additional samples (80 per class for 12 classes) with domain randomization designed to simulate the challenging conditions identified in the clinical MediaPipe validation literature [2][3]:

- **Keypoint Jitter:** Gaussian noise (sigma = 0.008-0.012 normalized units) added to all 21 keypoint coordinates with 100% probability. This simulates the inherent measurement noise of single-camera MediaPipe tracking, with sigma values calibrated to the 2.46deg MAD reported by Hamaguchi et al. [4].
- **Finger Occlusion:** 1-5 randomly selected keypoints zeroed (30% probability), simulating partial hand occlusion where MediaPipe loses tracking on individual joints.
- **Low-Light Simulation:** Jitter magnitude increased to 0.015-0.030 (40% probability), modeling the accuracy degradation under poor lighting conditions documented by Sprague et al. [3].
- **Rotation and Scaling:** Random rotation (plus or minus 3 degrees) and scaling (plus or minus 3%) applied to all keypoints (100% probability), simulating variations in hand orientation and camera distance.'''
new = '''Synthetic augmentation produced 960 samples (80 per class, 12 classes) with domain randomization: keypoint jitter (sigma=0.008-0.012, 100%, calibrated to Hamaguchi [4] MAD 2.46deg), finger occlusion (1-5 keypoints zeroed, 30%), low-light simulation (jitter increased to 0.015-0.030, 40%), and rotation plus scaling (plus/minus 3deg and 3%, 100%).'''
content = content.replace(old, new)

# Cut verbose CE vs CTC section in Ch3
old = '''An alternative approach would be Connectionist Temporal Classification (CTC) [32], widely used in speech recognition and continuous sign language recognition. CTC introduces a blank token and learns an alignment between input frames and output labels, handling variable-length output sequences. However, CTC is designed for **sequence-to-sequence** tasks (T input frames to L output labels, L no more than T), such as recognizing a continuous stream of multiple gestures from a long video. For single-label static gesture sequences, CTC would introduce unnecessary complexity:
- The blank token mechanism would need to suppress all but one label, turning the alignment learning problem into a degenerate case
- Training would be less stable as the model learns to balance blank vs. non-blank outputs
- The additional hyperparameters (blank index, CTC beam width) provide no benefit for single-label prediction

The CE with temporal pooling approach, by contrast, directly supervises the sequence-level prediction and converges more stably on limited data. The comparison is illustrated in Figure 2-5 (Chapter 2).'''
new = '''CTC [32], while standard for continuous sign language recognition, is designed for sequence-to-sequence tasks. For single-label static gesture sequences, it introduces unnecessary complexity through blank tokens, alignment learning, and additional hyperparameters with no benefit for single-label prediction. CE with temporal pooling directly supervises sequence-level prediction and converges more stably (see Figure 2-5).'''
content = content.replace(old, new)

# ====== Count result ======
words = len(content.split())
print(f'After trimming: {words:,} words (cut {21770-words:,})')
print('Target: ~15,000 (' + ('OK' if abs(words-15000) < 1000 else 'need more' if words > 15000 else 'over-cut') + ')')

# ====== ROUND 2: More aggressive cuts ======

# Ch3: Cut long SlowFast pathway detail
old = '''**Slow Pathway** (kernel size 7, dilations [1, 2, 4], channel progression [256 to 64 to 128 to 128]):
- Purpose: Capture long-range semantic context spanning complete gesture phrases.
- Receptive field: RF = 1 + (7-1)x(1+2+4) = 43 frames approximately 1.7 seconds at 25 FPS.
- The 1.7-second window covers: hand entry into the gesture pose (0.2s), stable hold (1.0-1.5s), and gesture release/transition (0.2-0.5s) — the complete lifecycle of a single gesture.
- Large kernel (7) and increasing dilations provide exponentially growing temporal context without increasing parameter count.'''
new = '''**Slow Pathway** (kernel 7, dilations [1, 2, 4], channels 256-64-128-128): receptive field 43 frames (approx1.7s at 25 FPS), covering complete gesture lifecycle.'''
content = content.replace(old, new)

old = '''**Fast Pathway** (kernel size 3, dilations [1, 1, 2, 2], channel progression [256 to 32 to 32 to 64 to 64]):
- Purpose: Preserve fine-grained temporal dynamics at high temporal resolution.
- Smaller kernel (3) retains sensitivity to rapid finger movements and brief pose adjustments.
- Lower channel dimensions keep the parameter count minimal — the Fast pathway's job is to capture high-frequency temporal detail, not to model complex semantic relationships (which is the Slow pathway's role).'''
new = '''**Fast Pathway** (kernel 3, dilations [1, 1, 2, 2], channels 256-32-32-64-64): preserves fine-grained temporal dynamics with minimal parameters.'''
content = content.replace(old, new)

old = '''**Lateral Connections**: After each stage in the Fast pathway (after dilations 1 and 2), a 1x1 convolution projects Fast features to the Slow pathway's channel dimension and adds them to the corresponding Slow stage. These lateral connections enable the Slow pathway to benefit from the Fast pathway's fine-grained temporal information without increasing its own temporal resolution, maintaining the computational efficiency of the two-pathway design.

After both pathways complete their processing, the Slow and Fast output features are concatenated and projected through a final output layer that produces per-frame class logits: [B, T, num_classes].'''
new = '''Lateral 1x1 convolutions from Fast to Slow stages enable cross-scale information flow. After dual-pathway processing, features are concatenated and projected to per-frame class logits [B, T, num_classes].'''
content = content.replace(old, new)

# Ch3: Cut long design rationale (4 points)
old = '''The choice was driven by four considerations:

1. **Parameter Efficiency**: A standard Transformer encoder for sequence modeling (e.g., ViT-Base) contains 10M+ parameters just for the self-attention and feed-forward layers. The SlowFast TCN achieves comparable temporal modeling capacity at 0.7M parameters. On a dataset of 1,311 samples, a 10M-parameter Transformer would overfit severely — memorizing training samples rather than learning generalizable temporal features.

2. **Causal Inference**: Sign language recognition is inherently a real-time, causal task — the system must predict the current gesture from past and present frames, without access to future frames that have not yet been captured. TCNs with causal convolutions (padding only on the left/ past side) naturally satisfy this constraint. Transformers without causal masking can attend to future frames, creating an unrealistic advantage during offline evaluation that does not transfer to real-time deployment.

3. **ONNX Export**: The TCN's purely convolutional architecture exports cleanly to ONNX format for cross-platform inference. Transformer models, particularly those using multi-head attention, have historically had more complex ONNX export paths with operator compatibility considerations.

4. **Receptive Field Control**: The TCN's receptive field can be precisely engineered through kernel size and dilation rate selection. The Slow pathway's 43-frame (1.7-second) window was deliberately designed to match the temporal extent of a complete gesture phrase. Transformer models typically have global receptive fields (each token can attend to all others), which is unnecessary for this task where the relevant temporal context is bounded by the gesture duration.

The literature supports this choice. GCAR [13] achieved 90.31% on WLASL with only 0.69M TCN parameters — competitive with much larger Transformer models. MSE-GCN [17] and AM-GCN [8] further validated TCN-based architectures as the state-of-the-art for efficient skeleton-based recognition. The dual-pathway SlowFast design, adapted from Feichtenhofer et al. [11], provides the multi-scale temporal analysis capability that a single-pathway TCN lacks, without the parameter explosion of Transformer attention.'''
new = '''Four considerations drove this choice: (1) Parameter efficiency — 0.7M vs. 10M+ for Transformer, critical on 1,311 samples; (2) Causal convolutions naturally prevent future-frame leakage in real-time inference; (3) Pure convolutional architecture exports cleanly to ONNX; (4) Receptive field precisely engineerable (43 frames for 1.7s gesture phrases). GCAR [13] validated TCN efficiency at 0.69M on WLASL; MSE-GCN [17] and AM-GCN [8] further support this design choice.'''
content = content.replace(old, new)

# Ch3: Cut long Stage 1-4 of rule engine
old = '''**Stage 1 — Finger Extension Detection:** For each of the five fingers, the extension angle is measured at the MCP-PIP-DIP joint chain. The angle formed by the three joint positions (e.g., points 5-6-7 for the index finger MCP-PIP-DIP) is computed:

theta_f = angle(kp_MCP, kp_PIP, kp_DIP)

A finger is classified as \"extended\" if theta_f > 165 degrees. This threshold was originally validated by Hamaguchi et al. [4] in their criterion-related validity study of markerless hand tracking, where they found that a 165-degree criterion produced the highest agreement (r = 0.78) with goniometric measurements across Brunnstrom recovery stages in stroke patients. The threshold represents a compromise: lower values increase sensitivity but produce false positives (slightly bent fingers classified as extended), while higher values increase specificity but miss genuinely extended fingers with natural joint flexion.

**Stage 2 — Finger Count Mapping:** The five binary extension states form a 5-bit pattern with 2^5 = 32 possible combinations. These map to 15 gesture classes as follows:'''
new = '''**Stage 1 — Finger Extension:** theta_f = angle(kp_MCP, kp_PIP, kp_DIP). Extended if theta_f > 165 degrees, validated by Hamaguchi et al. [4] (r = 0.78 agreement with goniometry).

**Stage 2 — Mapping:** The 5-bit pattern (32 combinations) maps to 15 gesture classes:'''
content = content.replace(old, new)

# Ch3: Cut verbose control gesture explanation
old = '''The tip distance criterion for Two vs. Victory is a critical disambiguation mechanism. In both gestures, the index and middle fingers are extended while other fingers are flexed. The distinguishing feature is whether the two extended fingers are held together (Two, representing the number 2 in Chinese number gestures) or spread apart (Victory, representing the V-sign). The Euclidean distance between the index fingertip (point 8) and middle fingertip (point 12) is computed in normalized coordinates. A threshold of 0.06 normalized units separates the two classes — this value was determined empirically from the recorded data and represents approximately 6% of palm size.

**Stage 3 — PIP/MCP Dual-Channel Verification:** Clinical studies [2][3] have identified the PIP joint as the primary source of MediaPipe tracking error, with angular errors of 20-30 degrees compared to 10-15 degrees for MCP joints. To mitigate this, a dual-channel verification mechanism is implemented:

For each finger, the extension state is independently computed from:
1. PIP-based measurement: MCP-PIP-DIP angle (standard method)
2. MCP-based measurement: wrist-MCP-PIP angle (alternative, more reliable)

When the two channels disagree on a finger's extension state, a 12% confidence penalty is applied to the final gesture classification confidence. The penalty value was calibrated such that a single-finger disagreement (12% penalty) does not change the classification outcome for unambiguous cases, while multi-finger disagreements (e.g., 3 inconsistent fingers = 36% penalty) significantly reduce confidence in borderline classifications.

**Stage 4 — Final Classification:** The 5-bit finger pattern maps to the gesture name. Gestures involving PIP-sensitive finger configurations (Victory, Six, Three, Four, Seven, Nine — identified by Sprague et al. [3] as having high PIP tracking error) receive additional scrutiny through the dual-channel mechanism.'''
new = '''Two vs. Victory disambiguation uses Euclidean distance between index and middle fingertips (threshold 0.06 normalized units).

**Stage 3 — PIP/MCP Dual-Channel Verification:** PIP joints exhibit the highest MediaPipe error (20-30 degrees [2][3]). Each finger's extension is independently computed from PIP (MCP-PIP-DIP) and MCP (wrist-MCP-PIP) measurements. Disagreement applies a 12% confidence penalty per finger.

**Stage 4 — Final Classification:** The 5-bit pattern maps to the gesture. PIP-sensitive classes (Victory, Six, Three, Four, Seven, Nine [3]) receive additional dual-channel scrutiny.'''
content = content.replace(old, new)

# Ch3: Cut long innovation paragraph
old = '''**Innovation over existing encoding schemes**: Traditional SLR maps gestures 1:1 to words. The combinatorial protocol achieves 70 slots from 15 gestures — a 4.7x encoding efficiency. Mode/content separation mirrors linguistic compositionality (function words + content words). The auto-separator and symmetric control gestures minimize explicit meta-communication overhead: the user focuses on WHAT to say, not HOW to operate the system.'''
new = '''The protocol achieves 4.7x encoding efficiency over 1:1 gesture-to-word mapping. Mode/content separation mirrors linguistic compositionality; auto-separator and symmetric controls minimize meta-communication overhead.'''
content = content.replace(old, new)

# Ch4: Cut long training config paragraph
old = '''**Training Configuration:** The multimodal DL model was trained for 50 epochs with the following hyperparameters, selected through preliminary experiments balancing convergence speed, stability, and generalization:

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Optimizer | AdamW | Adaptive learning rates + decoupled weight decay for regularization |
| Learning Rate (initial) | 1e-4 | Empirically determined to provide stable convergence without divergence |
| Weight Decay | 1e-4 | L2 regularization strength, calibrated to dataset size |
| Batch Size | 8 | Maximum batch size fitting in GPU memory with T approx 50-frame sequences |
| Epochs | 50 | Sufficient for convergence; validation metrics stabilized by epoch 37 |
| Warmup Epochs | 5 | Linear LR increase from 0 to 1e-4; prevents early training instability |
| LR Schedule | Cosine Annealing | Smooth LR decay from 1e-4 to approx 1e-7 over 45 epochs post-warmup |
| Loss Function | Cross-Entropy with Temporal Mean Pooling | Sequence to label classification (see Section 3.4.2) |
| Mixed Precision | Disabled (FP32) | Ensures numerical stability on CPU; AMP caused NaN with Transformer attention |

The MobileNetV3-Small backbone was frozen throughout training. The rule engine was evaluated directly on the test set without any training, using the same geometric thresholds as the production system.'''
new = '''The DL model was trained for 50 epochs with AdamW (lr=1e-4, weight_decay=1e-4), batch size 8, 5-epoch linear warmup followed by cosine annealing, and CE loss with temporal mean pooling. Mixed precision was disabled for CPU numerical stability. The MobileNetV3-Small backbone remained frozen. The rule engine was evaluated directly without training.'''
content = content.replace(old, new)

# Ch5: Cut long operational flow descriptions
old = '''**Operational Flow:**
1. User activates webcam, establishing WebSocket connection at 25 FPS frame streaming.
2. Recording initiation: both-hands Closed_Fist to Open_Palm transition (natural \"preparing to speak\" motion) or sustained both-Open_Palm for 1.0s (fallback). SentenceRecorder: IDLE to RECORDING.
3. During RECORDING, gesture tokens are accumulated with engineering parameters: minimum hold duration 0.3s (filters transient gestures caused by frame-to-frame fluctuation), inter-gesture cooldown 0.35s (prevents double-registration of the same gesture), idle timeout 15s (auto-ends recording when no gesture is added). Mode transitions trigger automatic comma insertion into the token buffer. Mode persistence for 1.5s after hand loss provides robustness against brief MediaPipe tracking dropouts.
4. Recording termination: both-hands Closed_Fist sustained 1.5s (intentional end signal) or idle timeout. SentenceRecorder: RECORDING to OUTPUT, auto-reset to IDLE after output generation.
5. Token sequence processing: longest-suffix semantic chain matching against 40 general-purpose and 65 hospital-specific chains to produce a base sentence. DeepSeek API polishing (1.5s connect timeout, 2.0s read timeout) for word reordering, sentence completion, and tone customization. Graceful fallback to chain-matched base sentence on API timeout (Level 2 degradation) or raw token concatenation when no chain matches (Level 3). Polished sentence displayed in recognition history with timestamp and raw tokens.'''
new = '''The translation workflow proceeds through five stages: (1) WebSocket connection at 25 FPS; (2) recording initiation via both-hands transition (Closed_Fist to Open_Palm) or sustained hold (1.0s); (3) gesture accumulation with 0.3s min hold, 0.35s cooldown, 15s idle timeout, and auto-separator on mode transitions; (4) termination via sustained Closed_Fist (1.5s) or timeout; (5) token processing through longest-suffix chain matching (40 general + 65 hospital chains) and DeepSeek polishing (1.5s connect, 2.0s read timeout) with graceful degradation.'''
content = content.replace(old, new)

with open('docs/毕业论文_完整英文版.md', 'w', encoding='utf-8') as f:
    f.write(content)
