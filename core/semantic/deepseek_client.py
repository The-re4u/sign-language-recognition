# coding:utf-8
"""DeepSeek API client for semantic enhancement of sign language recognition."""
import os
import json


class DeepSeekClient:
    """Async-capable client for DeepSeek Chat API with caching and fallback.

    API Key is read from environment variable DEEPSEEK_API_KEY.
    Fails gracefully when API is unavailable or key is not configured.
    """

    SYSTEM_PROMPT = """You are a sign language translation assistant. Your task is to:

1. Convert recognized gesture sequences into natural Chinese sentences
2. Fill in omitted subjects and function words based on context
3. Resolve ambiguity in gesture tokens
4. Add appropriate emotional tone based on the gesture category

Input format: JSON with "tokens" (list of gesture names) and optional "context" (previous sentences).
Output: Short natural Chinese sentence. No explanations."""

    POLISH_PROMPT = """You are a Chinese language polishing assistant. Your task is to refine a rough sign language translation into natural, fluent Chinese.

Rules:
1. Fix awkward phrasing and make the sentence sound natural
2. Add appropriate modal particles (呢, 吧, 啊, 哦) to convey emotion
3. Keep the original meaning — do not add or remove information
4. Output ONLY the polished sentence, no explanations, no quotation marks"""

    # v3.0: Hospital scene system prompt — carefully engineered for
    # scrambled gesture token → natural Chinese hospital sentence.
    # Design principles:
    #   1. Role anchoring: explicitly constrain the model to hospital domain
    #   2. Task decomposition: reorder → complete → tone → output
    #   3. Few-shot examples: concrete input→output pairs for each category
    #   4. Safety guardrails: anti-hallucination, medical disclaimer
    #   5. Output discipline: strict formatting, no extra content

    HOSPITAL_SYSTEM_PROMPT = """# 角色
你是一名中国医院手语翻译助手，服务于因气管插管、脑卒中、听障等原因无法说话的患者。
患者通过手势序列拼出词汇，你需要将这些零散、可能乱序的词汇重组为自然流畅的中文句子。

# 输入格式
{"tokens": ["词1","词2",...], "category": "类别", "context": ["历史句子",...]}

- tokens: 患者手势拼出的词汇（可能是乱序、省略主语、缺少虚词的零散词列表）
- category: 语义类别 (emergency/pain/medical/request/emotion/greeting/common)
- context: 最近几条已输出句子（用于对话连贯性）

# 核心任务

## 1. 词序重排
将乱序词汇按中文主谓宾语序重新排列。
例: ["药","吃","我"] → "我要吃药"
例: ["疼","头","很"] → "头很疼"
例: ["护士","叫"] → "请帮我叫护士"
例: ["7","疼","头"] → "头痛，疼痛等级7级"

## 2. 句子补全
补充省略的主语(我)、谓语、虚词，使句子语法完整。
- ["喝水"] → "我想喝水"
- ["疼","胸口"] → "我胸口疼"
- ["冷"] → "我觉得冷"
- ["翻身"] → "请帮我翻身"

## 3. 数字处理
根据医疗上下文智能解析数字 token。
- 与身体/症状搭配的1-10 → 疼痛/严重程度等级
- 连续2-3个数字 → 可能是体温(38.5)、血压(120/80)
- 孤立的数字 → 保留原样，不强行赋予医学含义
例: ["3","8"] 在发烧上下文中 → "体温38度"
例: ["1","3","5"] 在医学上下文中 → "血压135"
例: ["3"] 单独出现 → "3"（不强行解释）

## 4. 类别定制语气
根据 category 字段调整句子语调和紧急程度：

【emergency 紧急类】
语气: 急促、直接、大声呼救
语气词: 极少，直接表达
例: "救命！喘不过气了！快叫医生！"
例: "胸口疼得厉害，呼吸困难！"

【pain 疼痛类】
语气: 痛苦、求助
语气词: 啊、了、很、好
例: "我头好疼啊，能帮帮我吗？"
例: "胸口疼得受不了了"

【medical 医疗需求类】
语气: 平静、陈述事实
语气词: 适中
例: "我需要按时吃药，麻烦您帮我拿一下"
例: "昨晚一直睡不着"

【request 请求类】
语气: 礼貌、温和
语气词: 请、麻烦、一下、吧
例: "麻烦帮我倒杯水，谢谢"
例: "我想去洗手间，麻烦扶我一下"

【emotion/greeting 情感/问候类】
语气: 温暖、礼貌、感激
语气词: 啦、呢、哦、谢谢
例: "谢谢您，辛苦啦"
例: "感觉好多了呢，太感谢了"

## 5. 对话连贯性
若 context 中有历史句子，当前输出应自然地延续对话。
例: 历史=["我头疼"] + tokens=["药","吃"] → "那我能吃点止痛药吗？"
例: 历史=["我想喝水"] + tokens=["谢谢"] → "谢谢您，水很甜"

# 安全红线（必须遵守）
1. 严禁编造任何医学诊断、药物名称、剂量、治疗方案
2. 只能使用输入 tokens 中已有的信息
3. 不确定时保持保守，宁可输出短句也不自行推断
4. 不添加患者未表达的症状或需求
5. 不提供医疗建议

# 输出规则
- 只输出最终的中文句子
- 不加解释、引号、前缀或后缀
- 不加"患者说：""病人想表达："等引导语
- 一句话即可，简洁自然"""

    HOSPITAL_POLISH_PROMPT = """# 角色
你是中文医院沟通润色助手。将生硬的手语翻译结果润色为自然口语化中文。

# 规则
1. 修正生硬措辞，使句子自然流畅
2. 根据场景添加合适的语气词：
   - 疼痛/痛苦 → 啊、了（"好疼啊""受不了了"）
   - 感谢 → 了、啦、哦（"谢谢您了""好多啦"）
   - 紧急 → 保持直接，最少语气词
   - 请求 → 请、麻烦、一下、吧（"麻烦帮我一下"）
   - 欣慰/好转 → 呢、啦（"好多啦""不疼了呢"）
3. 保持原意不变，不增删症状或医疗信息
4. 对医护人员说话时用"您"，体现尊重

# 输出规则
只输出润色后的句子，不加解释和引号"""

    DEFAULT_API_KEY = ''  # Set via environment variable DEEPSEEK_API_KEY

    def __init__(self, api_key=None):
        self.api_key = (api_key
                        or os.environ.get('DEEPSEEK_API_KEY', '')
                        or self.DEFAULT_API_KEY)
        self.cache = {}
        self.enabled = bool(self.api_key)

        if self.enabled:
            try:
                import requests
                self._requests = requests
            except ImportError:
                print('[DeepSeek] requests library not installed — API calls disabled')
                self.enabled = False

    def enhance(self, tokens, context=None):
        """Enhance gesture token sequence using DeepSeek API.

        Args:
            tokens: list of gesture token strings
            context: optional list of previous recognized sentences

        Returns:
            Enhanced Chinese sentence string, or None on failure.
        """
        if not self.enabled or not tokens:
            return None

        cache_key = json.dumps({'tokens': tokens, 'context': context}, ensure_ascii=False)
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            user_prompt = f"Gesture tokens: {', '.join(tokens)}"
            if context:
                user_prompt += f"\nContext: {'; '.join(context[-3:])}"

            response = self._requests.post(
                'https://api.deepseek.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'deepseek-chat',
                    'messages': [
                        {'role': 'system', 'content': self.SYSTEM_PROMPT},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    'temperature': 0.3,
                    'max_tokens': 100,
                    'stream': False
                },
                timeout=(1.5, 2.0)
            )

            if response.status_code == 200:
                result = response.json()
                text = result['choices'][0]['message']['content'].strip()
                self.cache[cache_key] = text
                return text
            else:
                print(f'[DeepSeek] enhance failed: HTTP {response.status_code}')

        except Exception as e:
            print(f'[DeepSeek] enhance error: {e}')

        return None

    def polish(self, text):
        """Polish a rough translation into natural Chinese using DeepSeek API.

        Args:
            text: raw Chinese sentence from the semantic parser

        Returns:
            Polished Chinese sentence string, or None on failure.
        """
        if not self.enabled or not text:
            return None

        cache_key = f'polish:{text}'
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            response = self._requests.post(
                'https://api.deepseek.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'deepseek-chat',
                    'messages': [
                        {'role': 'system', 'content': self.POLISH_PROMPT},
                        {'role': 'user', 'content': text}
                    ],
                    'temperature': 0.3,
                    'max_tokens': 100,
                    'stream': False
                },
                timeout=(1.5, 2.0)
            )

            if response.status_code == 200:
                result = response.json()
                polished = result['choices'][0]['message']['content'].strip()
                self.cache[cache_key] = polished
                return polished
            else:
                print(f'[DeepSeek] polish failed: HTTP {response.status_code}')

        except Exception as e:
            print(f'[DeepSeek] polish error: {e}')

        return None

    # v3.0: Hospital word reordering and sentence assembly
    def reorder_and_assemble(self, tokens, category='common', mode='base',
                             context=None):
        """Reorder scrambled tokens and assemble into hospital-appropriate
        Chinese sentence.

        Args:
            tokens: list of gesture token strings (may be out of order)
            category: semantic category (pain/medical/emergency/request/emotion/greeting/common)
            mode: input mode (number/base/emotion)
            context: optional list of previous sentences for continuity

        Returns:
            Assembled Chinese sentence, or None on failure.
        """
        if not self.enabled or not tokens:
            return None

        cache_key = json.dumps({
            'hospital': 'v3.0',
            'tokens': tokens, 'category': category,
            'mode': mode, 'context': context
        }, ensure_ascii=False)
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            user_prompt = json.dumps({
                'tokens': tokens,
                'category': category,
                'mode': mode,
                'context': context if context else []
            }, ensure_ascii=False)

            response = self._requests.post(
                'https://api.deepseek.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'deepseek-chat',
                    'messages': [
                        {'role': 'system', 'content': self.HOSPITAL_SYSTEM_PROMPT},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    'temperature': 0.3,
                    'max_tokens': 200,
                    'stream': False
                },
                timeout=(1.5, 2.0)  # (connect, read) — fast fallback to avoid frame lag
            )

            if response.status_code == 200:
                result = response.json()
                assembled = result['choices'][0]['message']['content'].strip()
                self.cache[cache_key] = assembled
                return assembled
            else:
                print(f'[DeepSeek] reorder HTTP {response.status_code}')

        except Exception as e:
            print(f'[DeepSeek] reorder error (fallback to raw): {e}')

        return None

    # v3.0: Hospital polish — uses HOSPITAL_POLISH_PROMPT
    def hospital_polish(self, text):
        """Polish a rough hospital sentence into natural Chinese."""
        if not self.enabled or not text:
            return None

        cache_key = f'hospital_polish:{text}'
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            response = self._requests.post(
                'https://api.deepseek.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'deepseek-chat',
                    'messages': [
                        {'role': 'system', 'content': self.HOSPITAL_POLISH_PROMPT},
                        {'role': 'user', 'content': text}
                    ],
                    'temperature': 0.3,
                    'max_tokens': 150,
                    'stream': False
                },
                timeout=(1.5, 2.0)
            )

            if response.status_code == 200:
                result = response.json()
                polished = result['choices'][0]['message']['content'].strip()
                self.cache[cache_key] = polished
                return polished
        except Exception as e:
            print(f'[DeepSeek] hospital_polish error: {e}')
        return None

    # v3.2: Hospital triage / diagnosis assistant
    TRIAGE_SYSTEM_PROMPT = """# 角色
你是一名中国医院智能导诊助手。患者通过手语描述症状，你需要多轮对话逐步了解病情，最终推荐合适的科室。

# 对话规则
1. 根据患者描述的 symptoms 追问 1-2 个相关问题，每次只问一个
2. 问题应为封闭式（是/否回答），方便患者用手语回答
3. 追问方向：疼痛部位、持续时间、伴随症状、严重程度变化
4. 在给出科室推荐前，必须加问一句"您是否还有别的症状？"
5. 收集 2-3 轮症状信息 + 1轮"还有别的症状"确认后，给出科室推荐
6. 语气温暖、耐心、专业

# 科室推荐参考
- 头疼 + 头晕 → 神经内科
- 胸口疼 + 呼吸困难 → 心内科 / 急诊
- 腹痛 + 腹泻 → 消化内科
- 咳嗽 + 发烧 → 呼吸内科
- 骨伤/扭伤 → 骨科
- 皮肤问题 → 皮肤科
- 眼部不适 → 眼科
- 耳鼻喉 → 耳鼻喉科
- 失眠/情绪 → 神经内科 / 心理科
- 发热不退 → 发热门诊 / 感染科

# 输出格式
用 JSON 回复：
{"stage": "question", "message": "您是否还有XXX的症状？"}
或
{"stage": "recommend", "message": "根据您的症状，建议前往XXX科室就诊。", "department": "XXX科"}

只输出 JSON，不要其他内容。"""

    def triage_conversation(self, patient_message, history=None):
        """Multi-round triage conversation.

        Args:
            patient_message: latest patient sentence (Chinese)
            history: list of {"role": "assistant"/"patient", "content": "..."}

        Returns:
            dict with keys: stage, message, department (if recommend)
        """
        if not self.enabled:
            return None

        cache_key = json.dumps({'triage': 'v3.2', 'msg': patient_message,
                                'hist': history[-6:] if history else []},
                               ensure_ascii=False)
        if cache_key in self.cache:
            return self.cache[cache_key]

        messages = [{'role': 'system', 'content': self.TRIAGE_SYSTEM_PROMPT}]
        if history:
            for h in history[-6:]:  # last 6 messages for context
                role = 'assistant' if h['role'] == 'assistant' else 'user'
                messages.append({'role': role, 'content': h['content']})
        messages.append({'role': 'user', 'content': patient_message})

        try:
            response = self._requests.post(
                'https://api.deepseek.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'deepseek-chat',
                    'messages': messages,
                    'temperature': 0.3,
                    'max_tokens': 200,
                    'stream': False
                },
                timeout=(1.5, 2.0)
            )

            if response.status_code == 200:
                result = response.json()
                text = result['choices'][0]['message']['content'].strip()
                # Parse JSON from response
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    # Fallback: treat as plain text question
                    parsed = {'stage': 'question', 'message': text}
                self.cache[cache_key] = parsed
                return parsed
        except Exception as e:
            print(f'[DeepSeek] triage error: {e}')
        return None

    def is_available(self):
        return self.enabled
