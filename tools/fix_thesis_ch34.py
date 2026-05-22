"""Fix Ch3.6/Ch4.3 boundary and rewrite Ch4.4 UI description."""
with open('docs/毕业论文_完整英文版.md', 'r', encoding='utf-8') as f:
    content = f.read()

# === FIX 2: Trim Ch3.6.3 state machine implementation details ===
old_ss = '''**SentenceRecorder State Machine** (Figure 3-4):
- **IDLE** → RECORDING: Both Open_Palm sustained 1.0s, or transition Closed_Fist→Open_Palm
- **RECORDING**: Accumulate gesture tokens (0.3s min hold, 0.35s cooldown). Mode transitions auto-insert comma separators.
- **RECORDING** → OUTPUT: Both Closed_Fist sustained 1.5s, or 15s idle timeout
- Control: dual Victory (undo-word), dual Seven (undo-sentence), dual One (separator), all 0.5s hold

**Innovation over existing encoding schemes**: Traditional SLR maps gestures 1:1 to words. The combinatorial protocol achieves 70 slots from 15 gestures — a 4.7× encoding efficiency. Mode/content separation mirrors linguistic compositionality (function words + content words). The auto-separator eliminates the need for explicit delimiter gestures on every mode transition.'''

new_ss = '''**SentenceRecorder State Machine** (Figure 3-4): The finite state machine manages the recording lifecycle through three states — IDLE (waiting for start trigger), RECORDING (accumulating gesture tokens with minimum hold duration and inter-gesture cooldown enforcement), and OUTPUT (generating final sentence with auto-reset). Mode transitions automatically insert comma separators, eliminating explicit delimiter gestures when switching semantic layers. Symmetric dual-hand patterns (both hands showing the same mode gesture) serve as control commands: undo-word, undo-sentence, and manual separator. The detailed timing constants, transition conditions, and engineering implementation are provided in Section 4.3.1.

**Innovation over existing encoding schemes**: Traditional SLR maps gestures 1:1 to words. The combinatorial protocol achieves 70 slots from 15 gestures — a 4.7x encoding efficiency. Mode/content separation mirrors linguistic compositionality (function words + content words). The auto-separator and symmetric control gestures minimize explicit meta-communication overhead: the user focuses on WHAT to say, not HOW to operate the system.'''

if old_ss in content:
    content = content.replace(old_ss, new_ss)
    print('Fix 2: Ch3.6 state machine trimmed')
else:
    print('Fix 2: old_ss text not found')

# === FIX 3: Rewrite Ch4.4 from component list to interaction paradigm ===
old_ui = '''The web-based user interface, built with Vue.js 3.5 as a single-page application using the Composition API, communicates with the FastAPI backend exclusively through a single WebSocket connection. This design choice eliminates the complexity of multiple communication channels and ensures consistent state synchronization between frontend and backend.

**Layout Structure:** The interface follows a two-row layout. The top row contains the camera panel (left, fixed 640×480px) and a sidebar (right, flexible width). The bottom row contains the recognition history table and, in triage mode, the AI triage conversation panel in a 50/50 split arrangement.

**Camera Panel (Left):** The live video feed is displayed in a 640×480 pixel container. When the camera is active, an HTML5 Canvas overlay renders the 21-keypoint hand skeleton in real time: left hand keypoints and connections in blue (#2196F3), right hand in green (#4CAF50). An FPS (frames per second) badge in the top-left corner shows real-time performance metrics: current FPS, detection latency in milliseconds, and processing latency in milliseconds. A WebSocket status indicator in the top-right corner shows green when connected, red when disconnected. Below the camera, a horizontal button bar provides all controls.

**Sidebar Cards (Right, Top to Bottom):**
1. **Hand Gesture Recognition Card:** Displays two sub-cards, one per detected hand. Each shows the hand label (Left/Right), the recognized gesture name (large, bold), finger count, and confidence percentage. When only one hand is detected, the second card shows \"Not Detected\" in a dimmed state.
2. **Semantic Output Card:** Shows the current gesture token sequence in real time. State-dependent color coding: blue when IDLE (system waiting for input), orange when actively recording. The default idle message displays a handshake emoji with instructional text.
3. **Operation Log Card:** Chronological event log showing recording lifecycle events with distinct colors: start (blue), end/output (green), undo (red), separator (gray). New events push to the bottom with automatic scrolling.
4. **AI Triage Conversation Card (Triage Mode Only):** Displays the multi-turn conversation between the AI triage agent and the patient. AI messages appear in blue-tinted bubbles (left-aligned), patient messages (derived from gesture input) in gray bubbles (right-aligned). The department recommendation, when received, appears in a green-highlighted panel with the department name.
5. **Confidence Card:** A horizontal progress bar showing the current trajectory stability metric. Color transitions from red (low stability) through orange (medium) to green (high stability).

**Bottom Panels:**
- **Recognition History Table:** Chronological list of completed recordings showing timestamp, raw gesture tokens, and DeepSeek-polished output. Table headers are sticky during scrolling. Newest entries appear at the bottom with automatic scrolling.
- **AI Triage Panel (Triage Mode):** Shares the bottom row with the history table in a 50/50 split when triage mode is active. When triage mode is off, the history table expands to full width.'''

new_ui = '''The web-based user interface, built with Vue.js 3.5 as a single-page application, communicates with the FastAPI backend through a single WebSocket connection, eliminating multi-channel synchronization complexity. The design follows three interaction paradigms that collectively shape the information architecture.

**Paradigm 1 — Spatial Consistency:** The interface is organized as a two-row layout with fixed camera panel (640x480px, left) and scrollable information sidebar (right). This spatial partitioning reflects the natural visual flow: primary input (camera feed with real-time HTML5 Canvas skeleton overlay, blue for left hand, green for right) occupies the dominant left position, while interpreted output (gesture cards, semantic display, operation log, confidence bar) flows vertically on the right. The camera panel displays the application branding image when inactive, maintaining visual continuity. A horizontal button bar below the camera provides all controls: camera toggle, DL/Rule switch, translate/triage mode, hand swap, video upload, and force-end recording.

**Paradigm 2 — State-Visible Feedback:** Every system state transition produces an immediate, visually distinct UI change. The recording lifecycle (IDLE to RECORDING to OUTPUT) is surfaced through color-coded semantic output: blue for IDLE (instructional idle message), orange for active recording (real-time token buffer display). Gesture recognition confidence is dual-encoded as both a numerical percentage and a color-graded progress bar (red below 30%, orange 30-60%, green above 60%). WebSocket connectivity is monitored with a real-time indicator. The DL/Rule mode toggle changes button color to reflect the active recognition path. This principle ensures the user always perceives system state and confidence without conscious effort.

**Paradigm 3 — Chronological Transparency:** All user actions and system responses are logged in temporal order with automatic scroll-to-latest. The operation log records lifecycle events (start recording, separator inserted, word undone, sentence output) with type-specific color coding (blue/gray/red/green). The recognition history table stores completed sentences with timestamps, raw gesture tokens, and DeepSeek-polished output, enabling post-hoc communication review. New entries append to the bottom following natural chronological reading direction. In triage mode, an additional panel shares the bottom row in a 50/50 split, displaying the multi-turn AI consultation: patient messages derived from gesture input (gray, right-aligned), AI follow-up questions (blue, left-aligned), and department recommendations (green-highlighted).'''

if old_ui in content:
    content = content.replace(old_ui, new_ui)
    print('Fix 3: Ch4.4 rewritten as interaction paradigms')
else:
    print('Fix 3: old_ui text not found - trying partial match')
    if 'Paradigm 1' not in content:
        # Try finding the section start
        idx = content.find('The web-based user interface, built with Vue.js 3.5')
        if idx > 0:
            print(f'  Found UI section at position {idx}')
    else:
        print('  UI section already updated')

# === FIX 4: Add implementation details to Ch4.3.1 from Ch3.6 ===
old_feat1 = '''### 4.3.1 Feature 1: Sign Language Translation Mode

**Operational Flow:**
1. User clicks camera toggle to activate webcam. System establishes WebSocket connection, begins streaming frames at 25 FPS.
2. User initiates recording by transitioning both hands from Closed_Fist to Open_Palm (or holding both Open_Palm for 1.0 second). The SentenceRecorder state changes from IDLE to RECORDING. UI displays \"Recording\" status with orange color coding.
3. User performs gesture sequences following the dual-hand encoding protocol. Left hand selects mode layer; right hand selects content within that layer. Mode transitions automatically insert comma separators. The semantic output panel updates in real time to show the current token buffer.
4. User ends recording by transitioning both hands to Closed_Fist (or holding for 1.5 seconds). The SentenceRecorder transitions to OUTPUT state.
5. The token sequence is parsed through semantic chain matching. If a chain matches, the base sentence is sent to DeepSeek for polishing. If the API times out or is unavailable, the base sentence is output directly. The polished/natural sentence appears in the recognition history panel.'''

new_feat1 = '''### 4.3.1 Feature 1: Sign Language Translation Mode

**Operational Flow:**
1. User activates webcam, establishing WebSocket connection at 25 FPS frame streaming.
2. Recording initiation: both-hands Closed_Fist to Open_Palm transition (natural "preparing to speak" motion) or sustained both-Open_Palm for 1.0s (fallback). SentenceRecorder: IDLE to RECORDING.
3. During RECORDING, gesture tokens are accumulated with engineering parameters: minimum hold duration 0.3s (filters transient gestures caused by frame-to-frame fluctuation), inter-gesture cooldown 0.35s (prevents double-registration of the same gesture), idle timeout 15s (auto-ends recording when no gesture is added). Mode transitions trigger automatic comma insertion into the token buffer. Mode persistence for 1.5s after hand loss provides robustness against brief MediaPipe tracking dropouts.
4. Recording termination: both-hands Closed_Fist sustained 1.5s (intentional end signal) or idle timeout. SentenceRecorder: RECORDING to OUTPUT, auto-reset to IDLE after output generation.
5. Token sequence processing: longest-suffix semantic chain matching against 40 general-purpose and 65 hospital-specific chains to produce a base sentence. DeepSeek API polishing (1.5s connect timeout, 2.0s read timeout) for word reordering, sentence completion, and tone customization. Graceful fallback to chain-matched base sentence on API timeout (Level 2 degradation) or raw token concatenation when no chain matches (Level 3). Polished sentence displayed in recognition history with timestamp and raw tokens.

**Error Handling:** MediaPipe hand loss triggers persistent mode memory (1.5s window) preventing single-frame dropout from breaking mode+content combinations. DeepSeek API unavailability falls back to chain-matched output. Complete API and DL failure falls back to rule-engine-only operation (Level 4).'''

if old_feat1 in content:
    content = content.replace(old_feat1, new_feat1)
    print('Fix 4: Ch4.3.1 expanded with implementation details')
else:
    print('Fix 4: old_feat1 text not found')

with open('docs/毕业论文_完整英文版.md', 'w', encoding='utf-8') as f:
    f.write(content)
print('All fixes applied')
