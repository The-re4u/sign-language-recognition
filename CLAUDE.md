# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Sign language recognition system v6.0 — real-time dual-hand gesture-to-Chinese-sentence translation using MediaPipe + deep learning + LLM enhancement. Zhejiang University of Technology 2026 undergraduate thesis project.

## Commands

```bash
# Start backend (FastAPI WebSocket server on port 8000)
python backend/server.py

# ── v6.0 Experiment presets (see config/model_config.yaml for full table) ──

# E1: SpatialGCN only baseline (~0.35M)
python tools/train.py --data data/train.json --val_data data/val.json --data_dir data \
    --epochs 50 --batch_size 16 --lr 1e-4 --augment --class_weight --no_amp \
    --no_cnn --angle_dim 128 --output models/checkpoints/E1_spatial

# E2: SpatialGCN + CNN visual (~4.4M, task book baseline)
python tools/train.py --data data/train.json --val_data data/val.json --data_dir data \
    --epochs 50 --batch_size 16 --lr 1e-4 --augment --class_weight --no_amp \
    --unfreeze_cnn 3 --output models/checkpoints/E2_spatial_cnn

# E3: SpatialGCN + HandShapeContext, no CNN (~2.80M) — parameter-efficient champion
python tools/train.py --data data/train.json --val_data data/val.json --data_dir data \
    --epochs 50 --batch_size 16 --lr 1e-4 --augment --class_weight --no_amp \
    --no_cnn --geometric --output models/checkpoints/E3_spatial_geo

# E4: SpatialGCN + CNN + HandShapeContext with gated fusion (~6.89M)
python tools/train.py --data data/train.json --val_data data/val.json --data_dir data \
    --epochs 50 --batch_size 16 --lr 1e-4 --augment --class_weight --no_amp \
    --geometric --output models/checkpoints/E4_spatial_cnn_geo

# E5: E4 + SKIM Part Mixing augmentation for cross-user regularization (~6.89M)
python tools/train.py --data data/train.json --val_data data/val.json --data_dir data \
    --epochs 50 --batch_size 16 --lr 1e-4 --augment --class_weight --no_amp \
    --geometric --part_mixing --part_mix_prob 0.5 --output models/checkpoints/E5_partmix

# Evaluate on test set
python tools/eval_dl.py --data data/test.json --checkpoint models/checkpoints/best_model.pth --data_dir data/my_sequences

# Data recording (486 samples, 18 variants, 3 lights x 3 dists x 3 angles)
python tools/record_data.py --output data/my_annotations.json --data_dir data/my_sequences

# Multi-user data recording (1 classmate x 2 hands x 16 variants)
python tools/record_multiuser.py --user <name>

# Confusion-pair hard sample recording (65 targeted edge cases)
python tools/record_confusion.py

# Left hand recording (same 16 variants x 27 conditions as right hand)
python tools/record_data.py --output data/my_annotations_left.json --data_dir data/my_sequences_left

# Hyperparameter sensitivity sweep (C1)
python tools/hparam_sweep.py --base_data data/train.json --val_data data/val.json \
    --data_dir data/my_sequences --use_visual --output models/hparam_sweep

# Statistical significance tests (C3: McNemar + bootstrap CI)
python tools/stat_tests.py --test_data data/test_real.json --data_dir data/my_sequences \
    --dl_checkpoint models/checkpoints/best_model.pth --use_visual

# GCAR baseline training (C2)
python tools/train_baseline.py --data data/train.json --val_data data/val.json \
    --data_dir data/my_sequences --checkpoint models/checkpoints/best_model.pth \
    --use_visual --epochs 50 --output models/baseline_gcar

# Export trained TCN to ONNX for deployment
python tools/export_onnx.py --checkpoint models/checkpoints/best_model.pth \
    --output models/sign_recognizer.onnx

# Live gesture recognition test
python tools/test_v5_live.py

# Run tests
python tests/test_rule_recognizer.py
python tests/test_sentence_recorder.py
```

Environment: `DEEPSEEK_API_KEY` env var enables LLM polish/triage (optional — system works with raw chain-matched output without it).

## Architecture

**Two recognition paths, shared downstream:**

```
Camera frame → MediaPipe (21 landmarks/hand)
  ├─ DL path: SpatialGCN → [HandShapeContext | CNN] → TemporalEncoder → CrossModalFusion → SlowFastTCN → ONNX
  └─ Rule path (XAI): RuleRecognizer (geometric decision tree, <1ms)
      ↓
  SentenceRecorder → TemporalSemanticParser → DeepSeek polish → WebSocket → Vue frontend
```

**Modality scheme (v6.0):** Four configurable modalities feed CrossModalFusion's three slots:
| Slot | Modality | Encoder | Dim |
|------|----------|---------|-----|
| Visual | Geometric (HandShapeContext) and/or CNN (MobileNetV3) | Dual-stream or single | 256 or 512 |
| Spatial | Keypoint topology | SpatialGCN | 256 |
| Motion | Frame-to-frame trajectory | TemporalEncoder | 256 |

When both CNN and HandShapeContext are active, GatedFusion merges them into the visual slot (learned per-dimension gate: `gate * cnn + (1-gate) * geo_proj`). BimanualCrossAttention optionally attends left↔right hand features before fusion.

**Gesture system v5.3:** 9 CSL digit gestures (0-9). Nine = side hook only (front Nine removed to reduce confusion with One). Left hand = mode (7 modes: Seven/One/Two/Three/Four/Open_Palm/Six), right hand = content (Closed_Fist through Nine). 7x10=70 semantic slots. Control: both-hands One=separator, both-hands Two=undo word, both-hands Three=undo sentence.

**Experiment results (v6.0):**
| Experiment | Modalities | Params | In-User | Cross-User |
|:---:|------|:---:|:---:|:---:|
| E1 | SpatialGCN only | ~0.35M | 82-88% | 75-82% |
| E2 | GCN + CNN (unfrozen 3 blocks) | ~4.4M | 88-93% | 82-88% |
| **E3** | GCN + HandShapeContext (no CNN) | **2.80M** | **96.7%** | **95.6%** |
| E4 | GCN + CNN + HandShapeContext + gated fusion | 6.89M | 99.2% | 94.4% |
| E5 | E4 + SKIM Part Mixing | 6.89M | 94-97% | 88-93% |
| GCAR | Sep-TCN dual stream + channel attn | ~0.36M | 88-93% | 82-88% |

E3 is the recommended thesis model: fewest parameters, best cross-user generalization (only 1.1pp drop from in-user).

**Key modules:**
- `core/perception/` — MediaPipe hand tracking (GPU/CPU), frame cache, ROI cropping
- `core/feature/` — SpatialGCN (hand skeleton graph), HandShapeContext (29-dim geometric descriptors + MiniPointNet fusion, ~0.2M), TemporalEncoder (velocity/accel/curvature per frame + keypoint attention, ~0.2M), VisualEncoder (MobileNetV3-Small CNN backbone), CrossModalFusion (2-layer Transformer fusing 3 slots → 256-dim), GatedFusion (per-dimension gate for CNN↔geometric merger), BimanualCrossAttention (left↔right cross-attention, ~0.3M)
- `core/temporal/` — SlowFastTCN: dual-pathway Conv1d (slow=deep/dilated for semantics, fast=shallow for motion detail, lateral fusion); GCAR: baseline model (sep-TCN dual stream + channel attention, ~0.36M params)
- `core/fallback/` — RuleRecognizer: geometry-based finger-up detection with PIP error compensation (Wagh et al. 2025)
- `core/semantic/` — SentenceRecorder (state machine IDLE↔RECORDING), TemporalSemanticParser (sliding buffer + longest-match chain lookup), DeepSeekClient (3 hospital-tuned prompts: reorder/assemble, polish, triage conversation)
- `core/inference/` — ONNXRecognizer wrapping the exported SlowFastTCN
- `backend/server.py` — single FastAPI WebSocket server handling frames, DL features extracted in a background daemon thread to avoid blocking the event loop; DeepSeek calls run in `run_in_executor`

**Config-driven semantics:** `config/hospital_chains.json` defines gesture-to-meaning chains (with synonyms, categories, priorities). `config/semantic_chains.json` is the general (non-hospital) variant. `config/model_config.yaml` centralizes all model hyperparameters and experiment presets (E0-E5 + GCAR).

**Frontend:** Single Vue 3 SPA at `frontend/index.html` (~23KB self-contained). Serves from FastAPI root route. Communicates exclusively via WebSocket (binary JPEG frames + JSON control messages).

**State machine (SentenceRecorder v5.2):** IDLE ↔ RECORDING via both-hands Open_Palm ↔ Closed_Fist transitions. Control: both-hands Two=undo word, both-hands Three=undo sentence, both-hands One=separator. Same-gesture control only fires for One/Two/Three (CONTROL_CAPABLE set), other same-gesture pairs (Four/Six/Seven) pass through as normal mode+content input.

**Parameter budget (v6.0, E4 maximum):** Total ~6.9M (SpatialGCN 0.35M + HandShapeContext 0.2M + TemporalEncoder 0.2M + CNN backbone 2.5M frozen + GatedFusion 0.4M + CrossModalFusion 0.8M + BimanualCrossAttention 0.3M + SlowFastTCN ~1.5M). ONNX model contains only TCN (256-dim fused features → 10-class logits). E3 (recommended): only 2.80M by dropping CNN.

**Key docs:**
- `手势.md` — gesture definitions, mode/content/control tables, test scenarios
- `docs/多模态方案升级说明.md` — v6.0 modality architecture design rationale
- `docs/交接文档_05-29.md` — current experiment status, completed/pending training
- `docs/实验执行方案.md` — master experiment execution plan
- `docs/毕业论文图表.md` — thesis figure/table inventory
- `docs/答辩要点——痛点与创新点.md` — defense talking points
- `docs/数据采集工作总结.md` — dataset construction summary

## Important patterns

- All modules under `core/` guard PyTorch imports with try/except so the rule-based path works without torch installed
- `_py()` helper in `server.py` converts numpy types to native Python for JSON serialization
- Frame skipping in server: `_input_skip=2` (process every 2nd frame), optical flow enabled (`_flow_skip=1`) when `_dl_multimodal=True`, kinematic features every 6th processed frame. Skipped frames get the cached `_last_proc_result` to avoid UI flicker
- The DL background thread runs every 2s and consumes buffered features — it does NOT block the WebSocket main thread
- `_dl_multimodal` flag (automatically detected at startup) controls whether real ROI visual features + Farneback optical flow are used, or synthetic/zero substitutes. When False, matches old checkpoints trained without `--use_visual`
- SentenceRecorder `deepseek` attribute is temporarily set to None during `_process_frame` to prevent sync API calls, then restored; DeepSeek calls happen in `run_in_executor` after the frame response is sent
- `_dual_state()` in SentenceRecorder treats `Three` as `Closed_Fist` and `Four` as `Open_Palm` for start/end transition tolerance (MediaPipe misclassifies extreme poses)
- Hand count debounce: `_hand_debounce_timer` requires 3 consecutive frames of reduced hand count before updating cached state, prevents single-frame dropout flicker
- Model files and data are gitignored (large binaries). Training checkpoints go to `models/checkpoints/`, ONNX models to `models/`

### Training modality flags (v6.0)

- CNN visual is **default ON** (task book compliance). Disable with `--no_cnn`.
- `--geometric` enables HandShapeContext (29-dim descriptors + MiniPointNet). Use with `--no_cnn` for E3 (parameter-efficient).
- `--temporal` enables TemporalEncoder (trajectory features + keypoint attention). Skip for static gestures — adds noise without signal.
- When both `--geometric` and CNN active: GatedFusion auto-activates (per-dimension learned gate, prevents additive fusion failure).
- `--bimanual_attn` enables left↔right cross-attention. Gracefully degrades to identity for single-hand data (missing hand = zeros → residual preserves present hand).
- `--part_mixing` enables SKIM Part Mixing augmentation (body-part swap between samples). Use `--part_mix_prob` to control frequency (default 0.3, E5 uses 0.5).
- `--unfreeze_cnn N` unfreezes last N MobileNetV3 blocks (0=all frozen, 3 recommended for E2).
- `--angle_dim N` controls SpatialGCN angle encoder hidden dim (E1 uses 128 for stronger finger modeling).
