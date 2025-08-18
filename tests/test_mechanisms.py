from __future__ import annotations

from pathlib import Path
import pytest
from mmx_news.features.mechanisms import EmbeddingBackend, FeatureComputer


@pytest.fixture
def feature_computer() -> FeatureComputer:
    """Returns a FeatureComputer with a hash-based embedding backend."""
    emb_backend = EmbeddingBackend(backend="hash")
    return FeatureComputer(emb_backend=emb_backend, pos_lex=["good"], neg_lex=["bad"])


def test_paraphrasing_ratio_no_paraphrasing(feature_computer: FeatureComputer):
    """Tests that the paraphrasing ratio is low for dissimilar sentences."""
    text_dissimilar = "This is the first sentence. This is the second sentence."
    result_dissimilar = feature_computer.paraphrasing_ratio(text_dissimilar)
    assert result_dissimilar.name == "PR"
    assert 0.0 <= result_dissimilar.value <= 1.0

    text_similar = "This sentence is similar. This sentence is similar."
    result_similar = feature_computer.paraphrasing_ratio(text_similar)
    assert result_similar.name == "PR"
    assert 0.0 <= result_similar.value <= 1.0

    assert result_dissimilar.value < result_similar.value


def test_paraphrasing_ratio_high_paraphrasing(feature_computer: FeatureComputer):
    """Tests that the paraphrasing ratio is high for similar sentences."""
    text = "This is a sentence. This is a sentence."
    result = feature_computer.paraphrasing_ratio(text)
    assert result.name == "PR"
    assert 0.0 <= result.value <= 1.0
    # With hash embeddings and identical sentences, similarity should be 1.0
    assert result.value > 0.99


def test_paraphrasing_ratio_single_sentence(feature_computer: FeatureComputer):
    """Tests that the paraphrasing ratio is 0 for a single sentence."""
    text = "This is a single sentence."
    result = feature_computer.paraphrasing_ratio(text)
    assert result.name == "PR"
    assert result.value == 0.0


def test_subjectivity_ratio(feature_computer: FeatureComputer):
    """Tests the subjectivity ratio feature."""
    text = "This is a good and objective statement. This is a bad and subjective statement."
    result = feature_computer.subjectivity_ratio(text)
    assert result.name == "SR"
    assert 0.0 <= result.value <= 1.0
    # "good" and "bad" are subjective words
    assert result.value > 0.1


def test_headline_lead_coherence(feature_computer: FeatureComputer):
    """Tests the headline lead coherence feature."""
    title = "This is a title."
    lead = "This is a lead."
    result = feature_computer.headline_lead_coherence(title, lead)
    assert result.name == "HS"
    assert 0.0 <= result.value <= 1.0


def test_unusual_language_share_with_fake_news_words(feature_computer: FeatureComputer):
    """Tests the unusual language share feature with fake news words."""
    # Need to set up the lexicon first
    fc = FeatureComputer(emb_backend=feature_computer.emb, ul_lexicons=[Path("lexicons/profanity.txt")])
    text = "This is a normal sentence."
    result = fc.unusual_language_share(text)
    assert result.name == "UL"
    assert result.value == 0.0

    text_with_unusual_word = "This is a hoax."
    result_with_unusual_word = fc.unusual_language_share(text_with_unusual_word)
    assert result_with_unusual_word.name == "UL"
    assert result_with_unusual_word.value > 0.0


def test_sentiment_and_consistency(feature_computer: FeatureComputer):
    """Tests the sentiment and consistency features."""
    text = "This is a good sentence. This is another good sentence."
    sp, nc = feature_computer.sentiment_and_consistency(text)
    assert sp.name == "SP"
    assert nc.name == "NC"
    assert 0.0 <= sp.value <= 1.0
    assert 0.0 <= nc.value <= 1.0
    assert sp.value > 0.5  # Positive sentiment
    assert nc.value > 0.9  # High consistency

    text_mixed_sentiment = "This is a good sentence. This is a bad sentence."
    sp_mixed, nc_mixed = feature_computer.sentiment_and_consistency(text_mixed_sentiment)
    assert sp_mixed.value < sp.value
    assert nc_mixed.value < nc.value


def test_selective_quoting(feature_computer: FeatureComputer):
    """Tests the selective quoting feature."""
    text = 'He said "This is a quote."'
    result = feature_computer.selective_quoting(text)
    assert result.name == "SQ"
    assert 0.0 <= result.value <= 1.0
    assert result.value > 0.0
