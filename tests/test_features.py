from __future__ import annotations

from mmx_news.features.mechanisms import EmbeddingBackend, FeatureComputer


def test_feature_bounds() -> None:
    emb = EmbeddingBackend(backend="hash")
    fc = FeatureComputer(emb_backend=emb, pos_lex=["good"], neg_lex=["bad"])
    text = "Good news! This is really effective. Bad actors said bad things."
    pr = fc.paraphrasing_ratio(text)
    sr = fc.subjectivity_ratio(text)
    hs = fc.headline_lead_coherence("Good news", "This is really effective.")
    ul = fc.unusual_language_share(text)
    sp, nc = fc.sentiment_and_consistency(text)
    sq = fc.selective_quoting('"Good news!" said someone.')
    for fr in [pr, sr, hs, ul, sp, nc, sq]:
        assert 0.0 <= fr.value <= 1.0
