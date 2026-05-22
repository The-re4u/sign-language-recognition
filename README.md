# Hand Sign Language Recognition System v3.4

Lightweight multimodal real-time sign language recognition system based on MediaPipe. Converts hand gestures to natural Chinese sentences using dual-hand combinatorial encoding, GCN + SlowFast TCN, and LLM enhancement.

> Zhejiang University of Technology 2026 Undergraduate Thesis
> Advisor: Chen Bo | Student: Zhang Miuqi

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download MediaPipe Model (~7.5MB)

```bash
# Download hand_landmarker.task to project root:
# https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### 3. Set DeepSeek API Key (optional)

Without it, the system works with chain-matched output (no LLM polish).

```bash
# Windows
set DEEPSEEK_API_KEY=sk-your-key-here

# Mac/Linux
export DEEPSEEK_API_KEY=sk-your-key-here
```

### 4. Start Backend

```bash
python backend/server.py
```

### 5. Open Frontend

Open `frontend/index.html` in a browser (tested on Chrome/Edge), or visit `http://localhost:8000`.

## Features

| Feature | Description |
|---------|-------------|
| Real-time dual-hand tracking | MediaPipe, 31.7 FPS on CPU |
| Dual-hand encoding | 15 gestures → 70 semantic slots (7 modes × 10 content) |
| Rule engine (XAI) | Geometric decision tree, <1ms, traceable |
| Deep Learning model | SpatialGCN + Motion + Visual + SlowFast TCN, 82.8% accuracy, 2.8M params |
| DL/Rule one-click switch | Frontend toggle, both share downstream pipeline |
| AI Triage | Multi-turn symptom inquiry → department recommendation (DeepSeek) |
| Scene transfer | JSON config for hospital / banking / airport / smart home |

## Architecture

```
Camera → MediaPipe (21 kp) → SpatialGCN + Motion + Visual
       → CrossModalFusion (Transformer) → SlowFast TCN
       → Rule Engine (XAI) → Semantic Parser → DeepSeek → UI
```

| Module | Trainable Params |
|--------|-----------------|
| SpatialGCN + Angle Encoder | 0.15M |
| MotionEncoder | 0.04M |
| LightweightVisualEncoder (proj head) | 0.15M |
| CrossModalFusion (2-layer Transformer) | 1.90M |
| SlowFast TCN (dual-pathway) | 0.70M |
| MobileNetV3-Small backbone | 2.50M (frozen) |
| **Total Trainable** | **2.80M** |

## Gesture Guide

| Action | Gesture |
|--------|---------|
| Start recording | Both hands: Closed_Fist → Open_Palm |
| Mode hand (left) | Good(number)/Seven(body)/Victory(hospital)/Eight(symptom)/Two(severity)/Pinky_Up(time)/One(phrase) |
| Content hand (right) | Closed_Fist(0)~Open_Palm(5), Six~Nine |
| Undo last word | Both hands Victory (hold 0.5s) |
| Undo sentence | Both hands Seven (hold 0.5s) |
| Insert separator | Both hands One (hold 0.5s) |
| End & output | Both hands Closed_Fist (hold 1.5s) |

## Project Structure

```
backend/server.py          — FastAPI WebSocket server
core/perception/           — MediaPipe hand tracking
core/feature/              — SpatialGCN, Motion, Visual, CrossModalFusion
core/temporal/             — SlowFast TCN
core/fallback/             — Rule-based XAI recognizer
core/semantic/             — SentenceRecorder, TemporalParser, DeepSeekClient
core/inference/            — ONNX Runtime inference
config/                    — Semantic chains (JSON) & model config (YAML)
frontend/index.html        — Vue 3 single-page application
tools/                     — Training, evaluation, plotting, data generation
tests/                     — Unit tests (pytest)
docs/                      — Thesis and documentation
```
