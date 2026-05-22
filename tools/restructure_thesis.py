"""Restructure thesis: split experiments (old 3.9) into new Chapter 4.
Old: Ch1 Ch2 Ch3(3.1-3.8+3.9 experiments+3.10) Ch4 Ch5
New: Ch1 Ch2 Ch3(3.1-3.8 method+3.9 summary) Ch4(experiments) Ch5(system) Ch6(conclusion)
"""
import re

with open('docs/毕业论文_完整英文版.md', 'r', encoding='utf-8') as f:
    content = f.read()

print('=== Step 1: Split Chapter 3 at experiments section ===')

# Find the experiment section start and the chapter 4 start
exp_start = content.find('## 3.9 Experiments and Analysis')
ch4_start = content.find('\n# CHAPTER 4: SYSTEM DESIGN')
ch3_summary_start = content.find('## 3.10 Chapter Summary')

if exp_start < 0 or ch4_start < 0:
    print(f'ERROR: exp_start={exp_start}, ch4_start={ch4_start}')
    exit(1)

# Extract the three pieces
ch3_method = content[:exp_start]  # Ch3 up to but not including experiments
experiments = content[exp_start:ch4_start]  # The experiments section
rest = content[ch4_start:]  # Old Ch4 and Ch5

# Add a short chapter summary to end Ch3 (method)
ch3_closing = '''
## 3.9 Chapter Summary (Method)

This chapter presented the complete algorithmic design of the proposed sign language recognition system spanning seven processing layers: MediaPipe perception with wrist normalization, multi-modal feature extraction (SpatialGCN with auxiliary angle encoder, LightweightVisualEncoder with frozen backbone, MotionEncoder with three-path adaptive fusion, and CrossModalFusion with learnable modality embeddings), SlowFast TCN temporal modeling with a principled choice of CE temporal pooling over CTC loss, XAI-compliant rule-based recognition with PIP/MCP dual-channel verification, dual-hand combinatorial encoding achieving 4.7-fold semantic efficiency, and DeepSeek-enhanced LLM pipeline supporting both translation and triage modes. The comprehensive experimental evaluation of this method is presented in Chapter 4.

---

'''

# === Rebuild the document ===

# New Chapter 4: Experiments (was 3.9)
# Rename the heading
experiments = experiments.replace('## 3.9 Experiments and Analysis', '# CHAPTER 4: EXPERIMENTS AND ANALYSIS')

# Renumber experiment subsections: 3.9.X → 4.X
for i in range(1, 11):
    experiments = experiments.replace(f'### 3.9.{i} ', f'### 4.{i} ')
    experiments = experiments.replace(f'## 3.10 Chapter Summary', f'## 4.10 Chapter Summary')

# Renumber experiment figures: Figure 3-6 through 3-14 → Figure 4-1 through 4-9
for old, new in [('3-14','4-9'),('3-13','4-8'),('3-12','4-7'),('3-11','4-6'),
                  ('3-10','4-5'),('3-9','4-4'),('3-8','4-3'),('3-7','4-2'),('3-6','4-1')]:
    experiments = experiments.replace(f'Figure {old}:', f'Figure {new}:')
    experiments = experiments.replace(f'Figure {old} —', f'Figure {new} —')
    experiments = experiments.replace(f'Figure {old} -', f'Figure {new} -')

# New Chapter 5: System Design (was Chapter 4)
rest = rest.replace('# CHAPTER 4: SYSTEM DESIGN AND DEVELOPMENT', '# CHAPTER 5: SYSTEM DESIGN AND DEVELOPMENT')

# Renumber old Ch4 sections: 4.X → 5.X
for i in range(1, 6):
    rest = rest.replace(f'## 4.{i} ', f'## 5.{i} ')
    rest = rest.replace(f'### 4.{i}.', f'### 5.{i}.')
    rest = rest.replace(f'### 4.{i} ', f'### 5.{i} ')

# Renumber old Ch4 figures (screenshots): Figure 4-1 to 4-3 → Figure 5-1 to 5-3
for old, new in [('4-3','5-3'),('4-2','5-2'),('4-1','5-1')]:
    rest = rest.replace(f'Figure {old}:', f'Figure {new}:')
    rest = rest.replace(f'Figure {old} —', f'Figure {new} —')
    rest = rest.replace(f'Figure {old} -', f'Figure {new} -')

# New Chapter 6: Conclusion (was Chapter 5)
rest = rest.replace('# CHAPTER 5: CONCLUSION AND FUTURE WORK', '# CHAPTER 6: CONCLUSION AND FUTURE WORK')

# Renumber old Ch5 sections: 5.X → 6.X
for i in range(1, 4):
    rest = rest.replace(f'## 5.{i} ', f'## 6.{i} ')

# === Update cross-references in the body text ===
# Update references to experiments section
body = ch3_method + ch3_closing + experiments + rest

# Update "Section 3.9" → "Chapter 4" in Ch1-3
body = body.replace('Section 3.9', 'Chapter 4')
body = body.replace('(Section 3.9)', '(Chapter 4)')
body = body.replace('Section 3.9.6', 'Section 4.6')
body = body.replace('Section 3.9.7', 'Section 4.7')
body = body.replace('in Section 3.9', 'in Chapter 4')

# Update "Chapter 4" → "Chapter 5" for system design references
body = body.replace('described in Chapter 5', 'described in Chapter 5')  # already correct

# Update "Chapter 5" → "Chapter 6" for future work references
body = body.replace('(Chapter 5)', '(Chapter 6)')

# Update 1.4 Organization paragraph
old_org = '''- **Chapter 3:** Presents the core algorithm design — seven-layer pipeline architecture, feature extraction, temporal modeling, rule-based recognition, semantic parsing, LLM enhancement, and comprehensive experiments with analysis.
- **Chapter 4:** Details system design including requirements analysis, system context, functional design, and user interface.
- **Chapter 5:** Concludes the thesis and discusses future work directions.'''
new_org = '''- **Chapter 3:** Presents the core algorithm design — seven-layer pipeline architecture, feature extraction, temporal modeling, rule-based recognition, semantic parsing, and LLM enhancement.
- **Chapter 4:** Presents comprehensive experimental evaluation — dataset construction, classification performance, ablation studies, training dynamics, system benchmarks, and discussion.
- **Chapter 5:** Details system design including requirements analysis, system context, functional design, and user interface.
- **Chapter 6:** Concludes the thesis with innovation summary, limitations reflection, and future work directions.'''
body = body.replace(old_org, new_org)

# Update the "presented in Chapter 3" reference in the conclusion
body = body.replace('described in Chapter 3', 'described in Chapter 3')
body = body.replace('Section 3.9.6', 'Section 4.6')

with open('docs/毕业论文_完整英文版.md', 'w', encoding='utf-8') as f:
    f.write(body)

print('=== Restructuring complete ===')
print('New structure:')
print('  Chapter 1: Introduction')
print('  Chapter 2: Related Work')
print('  Chapter 3: Proposed Method (algorithm only)')
print('  Chapter 4: Experiments and Analysis (NEW)')
print('  Chapter 5: System Design and Development')
print('  Chapter 6: Conclusion and Future Work')
