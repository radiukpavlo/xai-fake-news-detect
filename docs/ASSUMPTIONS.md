# Assumptions

1. **Sentiment and Subjectivity**: A small, bundled lexicon is used for sentiment (lists in `lexicons/`). This is a deterministic fallback to avoid external downloads. Config keys:
   - `features.sentiment_lexicons.pos`
   - `features.sentiment_lexicons.neg`
   Impact: affects `SP` and `NC` calculation; replace with a richer lexicon for higher fidelity.

2. **Sentence Segmentation**: If spaCy (`en_core_web_sm`) is unavailable, a regex-based sentence splitter is used.
   - Config: automatic fallback; log warns.
   Impact: minor differences in `PR`, `HS`, and body pooling.

3. **Embeddings Backend**: Default uses sentence-transformers (`all-mpnet-base-v2`). If not available, a hash-based embedding is used.
   - Config: `embeddings.backend` ∈ {auto, st, hash}
   Impact: projection fidelity (`AT ≈ B`) may degrade in fallback; experiments remain reproducible.

4. **Deduplication**: SimHash over token features with Hamming threshold 3 groups near-duplicates. Set `data.dedupe.enabled: false` to disable.
   - Config: `data.dedupe.*`
   Impact: prevents leakage; may merge aggressively on very short texts.

5. **Fact Confirmation (`FC`)**: Off by default to preserve offline determinism. When enabled, user must provide API keys; evidence links will be recorded.
   - Config: `features.factcheck.enabled`, `features.factcheck.provider`, `features.factcheck.api_key_env`
   Impact: improves auditability; introduces external dependency and latency.

6. **Selective Quoting (`SQ`)**: Implemented as normalized combination of quote share and number of distinct quote spans; social reply signals are set to 0 due to unavailable external context.
   - Config: no extra; parameters documented in code docstrings.
   Impact: partial proxy of selective quoting behavior.

7. **License**: MIT license for this reproduction code. Dataset licenses remain with their owners.
