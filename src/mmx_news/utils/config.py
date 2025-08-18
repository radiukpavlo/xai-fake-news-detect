from __future__ import annotations

from typing import List, Tuple

from pydantic import BaseModel


class DataDedupeConfig(BaseModel):
    enabled: bool
    backend: str
    threshold_bits: int


class DataSmokeConfig(BaseModel):
    enabled: bool
    path: str


class DataConfig(BaseModel):
    root: str
    language: str
    splits: int
    split_ratio: Tuple[float, float, float]
    seed_list: List[int]
    dedupe: DataDedupeConfig
    smoke: DataSmokeConfig


class EmbeddingsConfig(BaseModel):
    backend: str
    model_name: str
    max_body_sentences: int
    weights: Tuple[float, float, float]


class SentimentLexiconsConfig(BaseModel):
    pos: str
    neg: str


class FactCheckConfig(BaseModel):
    enabled: bool
    provider: str
    api_key_env: str


class FeaturesConfig(BaseModel):
    subjectivity_threshold: float
    ul_lexicons: List[str]
    sentiment_lexicons: SentimentLexiconsConfig
    factcheck: FactCheckConfig
    calibration: str


class TransitionConfig(BaseModel):
    svd_tolerance: float


class ClassifierConfig(BaseModel):
    type: str
    C: float
    gamma: float


class EvaluationConfig(BaseModel):
    metrics: List[str]
    plots: List[str]
    n_jobs: int


class ReproConfig(BaseModel):
    global_seed: int
    deterministic: bool


class Config(BaseModel):
    run_name: str
    data: DataConfig
    embeddings: EmbeddingsConfig
    features: FeaturesConfig
    transition: TransitionConfig
    classifier: ClassifierConfig
    evaluation: EvaluationConfig
    repro: ReproConfig
