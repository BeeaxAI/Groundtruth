"""
Phase 7: Security layer — prompt injection defense & input sanitization.
Enterprise-grade security for AI applications.
"""

import re
import html
import logging
import time
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

# Known prompt injection patterns
INJECTION_PATTERNS = [
    re.compile(r'ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)', re.I),
    re.compile(r'(system\s+prompt|system\s+message|system\s+instruction)', re.I),
    re.compile(r'you\s+are\s+now\s+(a|an)\s+', re.I),
    re.compile(r'(forget|disregard|override)\s+(everything|all|your)', re.I),
    re.compile(r'new\s+(instructions?|role|persona|identity)', re.I),
    re.compile(r'(pretend|act\s+as|roleplay|assume\s+the\s+role)', re.I),
    re.compile(r'(do\s+not|don\'t)\s+(follow|obey)\s+(the|your)', re.I),
    re.compile(r'(reveal|show|display|print|output)\s+(your\s+)?(system|hidden|secret|internal)', re.I),
    re.compile(r'<\s*(system|prompt|instruction|admin)', re.I),
    re.compile(r'\[\s*(SYSTEM|ADMIN|OVERRIDE|INJECT)', re.I),
]

# Characters that could break formatting
CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


class InputSanitizer:
    """Sanitizes user input and document content for security."""

    def sanitize_query(self, text: str, max_length: int = 5000) -> tuple[str, list[str]]:
        warnings = []
        if not text or not text.strip():
            return "", ["Empty input"]

        text = text[:max_length]
        text = CONTROL_CHARS.sub('', text)
        text = text.strip()

        injection_found = self._check_injection(text)
        if injection_found:
            warnings.append(f"Potential prompt injection detected: {injection_found}")
            logger.warning(f"Injection attempt in query: {injection_found}")

        return text, warnings

    def sanitize_document_content(self, text: str) -> tuple[str, list[str]]:
        warnings = []
        text = CONTROL_CHARS.sub('', text)

        injection_matches = []
        for pattern in INJECTION_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                injection_matches.extend(str(m) for m in matches)

        if injection_matches:
            warnings.append(f"Document contains {len(injection_matches)} potential injection patterns (treated as data)")
            logger.warning(f"Injection patterns in document: {injection_matches[:3]}")

        return text, warnings

    def _check_injection(self, text: str) -> Optional[str]:
        for pattern in INJECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(0)
        return None


class RateLimiter:
    """Simple in-memory rate limiter per client."""

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def check(self, client_id: str) -> tuple[bool, int]:
        now = time.time()
        cutoff = now - self.window

        self._requests[client_id] = [t for t in self._requests[client_id] if t > cutoff]

        remaining = self.max_requests - len(self._requests[client_id])
        if remaining <= 0:
            return False, 0

        self._requests[client_id].append(now)
        return True, remaining - 1

    def cleanup(self):
        now = time.time()
        cutoff = now - self.window
        empty_keys = [k for k, v in self._requests.items() if all(t <= cutoff for t in v)]
        for k in empty_keys:
            del self._requests[k]
