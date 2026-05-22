# coding:utf-8
"""Temporal semantic parser: converts gesture sequences into semantic chains."""
from collections import deque
import json
import os
import time


class TemporalSemanticParser:
    """Parses sequences of gesture tokens into semantic chains.

    Features:
    - Sliding buffer (last 5 tokens)
    - 3-second timeout (auto-flush)
    - De-duplication (consecutive duplicates merged)
    - Longest-match strategy for chain lookup
    - Suffix matching (allow prefix noise)
    - Priority-based activation
    """

    def __init__(self, chains_path='config/hospital_chains.json'):
        self.buffer = deque(maxlen=5)
        self.chains = {}
        self.synonyms = {}
        self.categories = {}
        self.last_token_time = 0
        self.timeout = 3.0  # seconds
        self._load_chains(chains_path)

    def _load_chains(self, path):
        """Load semantic chains from JSON config."""
        default_paths = [
            path,
            os.path.join(os.path.dirname(__file__), '..', '..', path),
            os.path.join('E:', 'gestureRecognition', path)
        ]
        for p in default_paths:
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.chains = data.get('chains', {})
                self.synonyms = data.get('synonyms', {})
                self.categories = data.get('categories', {})
                return
            except (FileNotFoundError, json.JSONDecodeError):
                continue

        # Default: initialize with empty chains (semantic parser works but returns raw tokens)
        print('[TemporalParser] No chain config found, using raw token mode')

    def _normalize_token(self, token):
        """Normalize a gesture token using synonym table."""
        for canonical, syns in self.synonyms.items():
            if token in syns or token == canonical:
                return canonical
        return token

    def add_token(self, token, confidence=0.5):
        """Add a recognized gesture token to the buffer with deduplication."""
        # v3.0: SEP token breaks dedup chain but doesn't enter buffer
        if token == '__SEP__':
            self.last_token_time = time.time()
            return None

        normalized = self._normalize_token(token)
        now = time.time()

        # Timeout check
        if self.last_token_time > 0 and (now - self.last_token_time) > self.timeout:
            old_buffer = list(self.buffer)
            self.buffer.clear()
            # Try to match what we had before clearing
            matched = self._match_chain(old_buffer)
            if matched:
                self.buffer.append(normalized)
                self.last_token_time = now
                return matched

        # Deduplicate consecutive tokens
        if len(self.buffer) > 0 and self.buffer[-1] == normalized:
            self.last_token_time = now
            return None

        self.buffer.append(normalized)
        self.last_token_time = now

        # Attempt chain matching
        return self._match_chain(self.buffer)

    def _match_chain(self, tokens, force=False):
        """Match buffer against chains.

        Rule: match the LONGEST chain whose sequence is a suffix of the buffer.
        If force=False and buffer is a prefix of a longer chain, defer (wait for more tokens).
        If force=True, return the best match immediately (used for timeout).
        """
        if not tokens or len(tokens) < 1:
            return None

        token_list = list(tokens)

        # 1) Find all suffix matches
        all_matches = []  # (chain_name, chain_data, pattern_length)

        for chain_name, chain_data in self.chains.items():
            all_patterns = [chain_data['sequence']] + chain_data.get('variants', [])
            for pattern in all_patterns:
                if len(pattern) <= len(token_list) and token_list[-len(pattern):] == pattern:
                    all_matches.append((chain_name, chain_data, len(pattern)))
                    break

        if not all_matches:
            if force:
                return None
            # Check if buffer is building toward a longer chain
            for chain_name, chain_data in self.chains.items():
                all_patterns = [chain_data['sequence']] + chain_data.get('variants', [])
                for pattern in all_patterns:
                    if len(pattern) > len(token_list) and pattern[:len(token_list)] == token_list:
                        return None
            return None

        # 2) Get the longest match
        best = max(all_matches, key=lambda x: x[2])

        # 3) Defer if a longer chain could still complete (only when not forced)
        if not force:
            for chain_name, chain_data in self.chains.items():
                all_patterns = [chain_data['sequence']] + chain_data.get('variants', [])
                for pattern in all_patterns:
                    if len(pattern) > len(token_list) and pattern[:len(token_list)] == token_list:
                        return None

        return self._build_result(best[0], best[1])

    def _is_suffix_match(self, tokens, pattern):
        """Check if tokens ends with pattern (suffix matching)."""
        if len(pattern) > len(tokens):
            return False
        for i in range(1, len(pattern) + 1):
            if tokens[-i] != pattern[-i]:
                return False
        return True

    def _build_result(self, chain_name, chain_data):
        """Build a structured match result."""
        category = chain_data.get('category', 'common')
        priority = self.categories.get(category, {}).get('priority', 99)
        return {
            'chain': chain_name,
            'category': category,
            'priority': priority,
            'matched_at': time.time()
        }

    def check_timeout(self):
        """Check if buffer has timed out — if so, flush and return best match."""
        if len(self.buffer) == 0:
            return None
        if self.last_token_time > 0 and (time.time() - self.last_token_time) > self.timeout:
            tokens = list(self.buffer)
            # Force-match (ignore potential longer chains since we timed out)
            matched = self._match_chain(tokens, force=True)
            self.buffer.clear()
            if matched:
                return matched
            # No match — return raw concatenation
            return {
                'chain': '-'.join(tokens),
                'category': 'raw',
                'priority': 99,
                'matched_at': time.time()
            }
        return None

    def flush(self):
        """Force output current buffer as raw concatenated tokens."""
        tokens = list(self.buffer)
        self.buffer.clear()
        if not tokens:
            return None
        return {
            'chain': '-'.join(tokens),
            'category': 'raw',
            'priority': 99,
            'matched_at': time.time()
        }

    def get_buffer_state(self):
        """Return current buffer contents for UI display."""
        return list(self.buffer)
