"""Round 3: Cut verbose paragraphs across all chapters."""
with open('docs/毕业论文_完整英文版.md', 'r', encoding='utf-8') as f:
    content = f.read()

replacements = [
    # Ch2: clinical validation summary
    ('Subsequent clinical validation studies have systematically quantified MediaPipe hand tracking accuracy across different populations, movement types, and hardware configurations. The table below summarizes the key findings that inform the system design choices in this thesis.',
     'Key clinical findings informing system design are summarized below.'),

    # Ch2: design implication
    ('The key design implication from these clinical studies is the identification of the PIP joint as the primary source of tracking error in single-camera MediaPipe setups. This finding directly motivated the dual-channel PIP/MCP verification mechanism in the rule-based recognition layer (Section 3.5), where PIP-based classifications are cross-validated against more reliable MCP-based measurements with confidence penalties applied for inconsistent detections.',
     'The key design implication is the identification of the PIP joint as the primary tracking error source, directly motivating the dual-channel PIP/MCP verification (Section 3.5).'),

    # Ch2.5: recurring theme
    ('A recurring theme in recent SLR research is the tension between model accuracy and computational efficiency, particularly for deployment scenarios where GPU access is unavailable. Several 2024-2025 works have specifically targeted this efficiency frontier.',
     'Recent research has targeted the accuracy-efficiency frontier for GPU-unavailable deployment scenarios.'),

    # Ch3: MotionEncoder intro
    ('The MotionEncoder captures temporal dynamics -- how the hand moves and changes configuration between consecutive frames. While the gesture vocabulary consists of static poses (the user holds a fixed hand shape), the transitions between poses and subtle within-pose movements (tremors, adjustments, finger pressure variations) provide discriminative information that purely spatial features miss.',
     'The MotionEncoder captures inter-frame dynamics, including subtle within-pose movements (tremors, adjustments) that provide discriminative information beyond static spatial features.'),

    # Ch3: XAI section intro
    ('The rule-based recognition layer serves a dual purpose: it is both the production recognition path (providing instantaneous, explainable classifications) and a conceptual framework for understanding what the DL model learns to do implicitly. The geometric decision tree implements the clinical standard for finger extension assessment, adapted from the goniometric measurement protocols validated in the rehabilitation literature [1][4].',
     'The rule-based layer provides instantaneous, explainable classifications using a geometric decision tree adapted from clinical goniometric protocols [1][4].'),

    # Ch4: requirements intro
    ('The requirements were derived from three sources: the official task book, practical deployment considerations for the target hospital triage scenario, and general software engineering best practices for real-time interactive systems.',
     'Requirements were derived from the task book, hospital triage deployment considerations, and real-time systems best practices.'),

    # Ch6: conclusion intro
    ('This thesis presented the design, implementation, and experimental evaluation of a lightweight multimodal real-time sign language recognition system. The system addresses a genuine accessibility need',
     'This thesis presented the design, implementation, and experimental evaluation of a lightweight multimodal real-time sign language recognition system for barrier-free deaf communication. The system addresses an accessibility need'),

    # Ch4: feature intro
    ('The system provides four core features, each accessible through the web interface with clear visual affordances.',
     'The system provides four core features accessible through the web interface.'),

    # Ch4: Feature 2 triage
    ('**Feature 2 -- AI Triage Consultation:** In triage mode, patients describe their symptoms through gestures. The symptom + severity layers map to medical concepts (e.g., Eight + Seven = dizziness, Two + Seven = worsening). DeepSeek processes the initial complaint and initiates multi-turn dialogue with yes/no follow-up questions. Patients respond using the Common Phrases layer (Six = Yes, Nine = No). After 2-3 rounds of targeted information gathering, DeepSeek recommends a medical department.',
     '**Feature 2 -- AI Triage:** Patients describe symptoms via gestures; DeepSeek conducts multi-turn yes/no follow-up (Six=Yes, Nine=No in Phrase layer); after 2-3 rounds, a department recommendation is provided.'),

    # Ch4: Feature 3 DL/Rule
    ('**Feature 3 -- DL/Rule Dual-Mode Switching:** A single button toggles between the rule-based recognition engine (production path, <1 ms latency, XAI-compliant with traceable decision provenance) and the DL model (enhanced path, approx 50 ms latency, data-driven with 82.8% accuracy). Both modes share the same downstream semantic parsing and LLM enhancement pipeline, enabling direct comparison of recognition outputs.',
     '**Feature 3 -- DL/Rule Dual Mode:** One-click toggle between rule engine (<1ms, XAI, production path) and DL model (~50ms, 82.8%, enhancement path), sharing the downstream pipeline.'),

    # Ch4: Feature 4 scene migration
    ('**Feature 4 -- Scene Configuration Migration:** The system shall support migration between application scenarios (hospital, banking, airport, smart home, government services) through replacement of JSON configuration files (semantic_chains.json, hospital_chains.json, DeepSeek system prompts) without modification to the underlying recognition code.',
     '**Feature 4 -- Scene Migration:** JSON configuration replacement (semantic_chains.json, scene-specific chains, DeepSeek prompts) enables cross-domain deployment without code modification.'),

    # Ch2.5: MSE-GCN detail
    ('MSE-GCN [17] proposed a multi-scale spatio-temporal feature aggregation enhanced efficient GCN for dynamic sign language recognition. Using separable convolution layers in a multi-scale, multi-branch setting with an early fusion scheme, the model introduced Spatial-Temporal Joint Part Attention (ST-JPA) to distinguish important body parts and joints. Key results: 85.27% on WLASL-100, 81.59% on WLASL-300, and 71.75% on WLASL-1000, all with significantly lower computational costs than competing methods.',
     'MSE-GCN [17] proposed multi-scale separable GCN with ST-JPA attention for dynamic SLR, achieving 85.27% on WLASL-100 with low computational costs.'),

    # Ch2.5: DSLNet detail
    ('DSLNet [18] achieved state-of-the-art parameter efficiency through dual-reference coordinate normalization (wrist-centric + facial-centric) combined with topology-aware GCN for shape features, a Finsler geometry encoder for trajectory features, and optimal transport fusion. Results: 93.70% on WLASL-100 and 89.97% on WLASL-300 with significantly fewer parameters than competing high-accuracy models.',
     'DSLNet [18] achieved 93.70% on WLASL-100 through dual-reference GCN with optimal transport fusion, using significantly fewer parameters than competitors.'),

    # Ch2.5: TGCN detail
    ('TGCN [19] conducted a comprehensive comparison of seven models across three modalities for Thai sign language, empirically validating TGCN as the optimal lightweight baseline across single-hand poses, multi-stroke gestures, and two-handed postures. Their finding that skeleton-only models suffice for single-hand signs while RGB augmentation is needed for two-handed cases (due to hand obstructions) directly informed the dual-modality design in this thesis.',
     'TGCN [19] empirically validated skeleton-only models for single-hand signs and RGB augmentation for two-handed cases, informing the dual-modality design in this thesis.'),

    # Ch4: Feature 1 verbosity
    ('The translation workflow proceeds through five stages: (1) WebSocket connection at 25 FPS; (2) recording initiation via both-hands transition (Closed_Fist to Open_Palm) or sustained hold (1.0s); (3) gesture accumulation with 0.3s min hold, 0.35s cooldown, 15s idle timeout, and auto-separator on mode transitions; (4) termination via sustained Closed_Fist (1.5s) or timeout; (5) token processing through longest-suffix chain matching (40 general + 65 hospital chains) and DeepSeek polishing (1.5s connect, 2.0s read timeout) with graceful degradation.',
     'The workflow: (1) WebSocket at 25 FPS; (2) start via both-hands transition or 1.0s hold; (3) accumulate with 0.3s/0.35s/15s timing and auto-separator; (4) end via 1.5s hold or timeout; (5) chain matching + DeepSeek polishing with graceful fallback.'),

    # Ch5: very long operational flow has already been cut above, skip
]

total_cuts = 0
for old, new in replacements:
    if old in content:
        cut = len(old.split()) - len(new.split())
        content = content.replace(old, new)
        total_cuts += cut
    else:
        # Try with different dash characters
        alt_old = old.replace('--', '—')
        if alt_old in content:
            cut = len(alt_old.split()) - len(new.split())
            content = content.replace(alt_old, new)
            total_cuts += cut

words = len(content.split())
print(f'Applied {total_cuts} words of cuts')
print(f'New total: {words:,} words')

with open('docs/毕业论文_完整英文版.md', 'w', encoding='utf-8') as f:
    f.write(content)
