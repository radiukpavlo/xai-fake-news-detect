from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from ..data.preprocess import simple_sentence_split


@dataclass
class Evidence:
    spans: List[Tuple[int, int]]  # (start_char, end_char)
    notes: str
    links: List[str]


@dataclass
class FeatureResult:
    name: str
    value: float  # scaled to [0, 1]
    evidence: Evidence
    meta: Dict[str, str]


# --- Embedding backend (auto / sentence-transformers / hash) ---
class EmbeddingBackend:
    def __init__(self, backend: str = "auto", model_name: str = "sentence-transformers/all-mpnet-base-v2") -> None:
        self.backend = backend
        self.model_name = model_name
        self._model = None  # lazy

    def _ensure_model(self) -> None:
        if self.backend in ("auto", "st"):
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer(self.model_name)
                self.backend = "st"
                return
            except Exception:
                if self.backend == "st":
                    raise
                # fall through to hash
        self.backend = "hash"

    def encode(self, texts: List[str]) -> np.ndarray:
        self._ensure_model()
        if self.backend == "st" and self._model is not None:
            embs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(embs, dtype=np.float32)
        # Hash-based fallback: feature hashing into 512-dim space, l2-normalized
        dim = 512
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            vec = np.zeros(dim, dtype=np.float32)
            for tok in t.lower().split():
                h = hash(tok) % dim
                vec[h] += 1.0
            n = np.linalg.norm(vec) + 1e-12
            out[i] = vec / n
        return out


class FeatureComputer:
    """Computes interpretable features and returns numeric values with evidence.

    All raw feature values are later min-max scaled on training data.
    """

    def __init__(
        self,
        emb_backend: EmbeddingBackend,
        pos_lex: Iterable[str] | None = None,
        neg_lex: Iterable[str] | None = None,
        ul_lexicons: List[Path] | None = None,
    ) -> None:
        self.emb = emb_backend
        self.pos_words = set([w.strip().lower() for w in (pos_lex or []) if w.strip()])
        self.neg_words = set([w.strip().lower() for w in (neg_lex or []) if w.strip()])
        self.ul_words: set[str] = set()
        for p in ul_lexicons or []:
            try:
                for line in Path(p).read_text(encoding="utf-8").splitlines():
                    w = line.strip().lower()
                    if w:
                        self.ul_words.add(w)
            except Exception:
                continue

    # Utility
    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, x))

    # --- Features ---
    def paraphrasing_ratio(self, text: str) -> FeatureResult:
        sents = simple_sentence_split(text)
        if len(sents) < 2:
            return FeatureResult("PR", 0.0, Evidence([], "single sentence", []), {"pairs": "0"})
        embs = self.emb.encode(sents)
        sims = []
        for i in range(len(sents)):
            for j in range(i + 1, len(sents)):
                a = embs[i]
                b = embs[j]
                sim = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                sims.append(sim)
        val = float(np.mean(sims)) if sims else 0.0
        return FeatureResult(
            "PR", self._clip01(val), Evidence([], "pairwise cosine mean", []), {"pairs": str(len(sims))}
        )

    def subjectivity_ratio(self, text: str, threshold: float = 0.5) -> FeatureResult:
        toks = [t.strip(""".,!?;:'"()[]{}""").lower() for t in text.split()]
        # heuristic: adjectives/adverbs approximated by suffix patterns; plus sentiment words
        subj = 0
        for t in toks:
            if t in self.pos_words or t in self.neg_words:
                subj += 1
            elif t.endswith("ly") or t.endswith("ive") or t.endswith("ous"):
                subj += 1
        value = float(subj / max(1, len(toks)))
        spans: List[Tuple[int, int]] = []
        # naive evidence: positions of first 30 subjective tokens
        count = 0
        start = 0
        for tok in text.split():
            tok_clean = tok.strip(""".,!?;:'"()[]{}""").lower()
            end = start + len(tok)
            if tok_clean in self.pos_words or tok_clean in self.neg_words or tok_clean.endswith(("ly", "ive", "ous")):
                spans.append((start, end))
                count += 1
                if count >= 30:
                    break
            start = end + 1  # account for space
        return FeatureResult(
            "SR", self._clip01(value), Evidence(spans, "lexicon+suffix heuristic", []), {"threshold": str(threshold)}
        )

    def headline_lead_coherence(self, title: str, lead: str) -> FeatureResult:
        embs = self.emb.encode([title, lead])
        a, b = embs[0], embs[1]
        sim = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        return FeatureResult("HS", self._clip01(sim), Evidence([], "cosine(title, lead)", []), {})

    def unusual_language_share(self, text: str) -> FeatureResult:
        toks = [t.strip(""".,!?;:'"()[]{}""").lower() for t in text.split() if t.strip()]
        cnt = sum(1 for t in toks if t in self.ul_words)
        value = float(cnt / max(1, len(toks)))
        spans: List[Tuple[int, int]] = []
        start = 0
        for tok in text.split():
            clean = tok.strip(""".,!?;:'"()[]{}""").lower()
            end = start + len(tok)
            if clean in self.ul_words:
                spans.append((start, end))
            start = end + 1
        return FeatureResult(
            "UL",
            self._clip01(value),
            Evidence(spans[:30], "lexicon match", []),
            {"lexicon_size": str(len(self.ul_words))},
        )

    def sentiment_and_consistency(self, text: str) -> Tuple[FeatureResult, FeatureResult]:
        sents = simple_sentence_split(text)
        if not sents:
            return (
                FeatureResult("SP", 0.0, Evidence([], "empty text", []), {}),
                FeatureResult("NC", 1.0, Evidence([], "empty text", []), {}),
            )
        scores: List[float] = []
        for s in sents:
            toks = [t.strip(""".,!?;:'"()[]{}""").lower() for t in s.split() if t.strip()]
            pos = sum(1 for t in toks if t in self.pos_words)
            neg = sum(1 for t in toks if t in self.neg_words)
            score = (pos - neg) / max(1, (pos + neg))
            # map to [0,1]
            score01 = 0.5 * (score + 1.0)
            scores.append(score01)
        sp = float(np.mean(scores))
        var = float(np.var(scores))
        nc = float(1.0 - var)  # higher variance -> lower consistency
        return (
            FeatureResult(
                "SP", self._clip01(sp), Evidence([], "lexicon polarity mean", []), {"n_sents": str(len(sents))}
            ),
            FeatureResult("NC", self._clip01(nc), Evidence([], "1 - var(sentiment)", []), {}),
        )

    def selective_quoting(self, text: str) -> FeatureResult:
        # count quoted spans
        n_quotes = text.count('"') // 2 + text.count("“") // 2 + text.count("”") // 2
        toks = [t for t in text.split() if t.strip()]
        quote_tokens = sum(1 for t in toks if t.startswith('"') or t.endswith('"') or "“" in t or "”" in t)
        imbalance = quote_tokens / max(1, len(toks))
        # combine with weights (alpha=0.7, beta=0.3) and normalize by simple Z=1.0
        value = 0.7 * imbalance + 0.3 * (1.0 - math.exp(-n_quotes))
        value = self._clip01(value)
        return FeatureResult("SQ", value, Evidence([], "quote share + count", []), {"n_quotes": str(n_quotes)})

    def fact_confirmation(self, text: str) -> FeatureResult:
        """Placeholder for fact confirmation.

        Checks for keywords suggesting external sourcing.
        A real implementation would use a fact-checking API.
        """
        keywords = ["source:", "according to", "reuters", "associated press", "ap", "fact check"]
        lower_text = text.lower()
        count = sum(1 for keyword in keywords if keyword in lower_text)

        # Normalize based on presence of any keyword
        value = 1.0 if count > 0 else 0.0

        spans: List[Tuple[int, int]] = []
        for keyword in keywords:
            idx = lower_text.find(keyword)
            if idx != -1:
                spans.append((idx, idx + len(keyword)))

        return FeatureResult(
            "FC",
            value,
            Evidence(spans[:10], "keyword match", []),
            {"keywords_checked": str(len(keywords)), "hits": str(count)},
        )
