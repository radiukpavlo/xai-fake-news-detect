"""
Data module for fake news detection pipeline.
"""

from .loaders import Article, prepare_splits, stratified_splits
from .dataset_pipeline import (
    DatasetConfig,
    DatasetDownloader,
    DatasetProcessor,
    DatasetPipeline,
    DataSource,
)

__all__ = [
    'Article',
    'prepare_splits',
    'stratified_splits',
    'DatasetConfig',
    'DatasetDownloader',
    'DatasetProcessor',
    'DatasetPipeline',
    'DataSource',
]