from __future__ import annotations

import re
from typing import List

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def simple_sentence_split(text: str) -> List[str]:
    """Regex-based fallback sentence splitter."""
    text = text.strip()
    if not text:
        return []
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    sents = re.split(SENT_SPLIT_RE, text)
    return [s for s in sents if s]
