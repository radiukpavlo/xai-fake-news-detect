"""
Comprehensive Dataset Pipeline for Fake News Detection
=======================================================

This module provides a unified interface for downloading and processing
fake news datasets from various sources including Kaggle, Mendeley Data,
Hugging Face, and other repositories.

Supported Data Sources:
- Kaggle API
- Mendeley Data
- Hugging Face Datasets
- Direct URL downloads
- GitHub repositories
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import pandas as pd
import requests
from tqdm import tqdm


class DataSource(Enum):
    """Supported data source platforms."""
    KAGGLE = "kaggle"
    MENDELEY = "mendeley"
    HUGGINGFACE = "huggingface"
    DIRECT_URL = "direct_url"
    GITHUB = "github"


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    source: DataSource
    identifier: str  # e.g., kaggle dataset ID, URL, etc.
    expected_files: List[str]
    format: str  # csv, tsv, json, etc.
    label_column: str
    text_columns: List[str]
    title_column: Optional[str] = None
    preprocessing: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class DatasetDownloader:
    """Handle downloading datasets from various sources."""
    
    def __init__(self, cache_dir: Path = Path("data/cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download(self, config: DatasetConfig, api_key: Optional[str] = None) -> Path:
        """Download dataset based on source type."""
        if config.source == DataSource.KAGGLE:
            return self._download_kaggle(config, api_key)
        elif config.source == DataSource.MENDELEY:
            return self._download_mendeley(config, api_key)
        elif config.source == DataSource.HUGGINGFACE:
            return self._download_huggingface(config, api_key)
        elif config.source == DataSource.DIRECT_URL:
            return self._download_direct(config)
        elif config.source == DataSource.GITHUB:
            return self._download_github(config)
        else:
            raise ValueError(f"Unsupported data source: {config.source}")
    
    def _download_kaggle(self, config: DatasetConfig, api_key: Optional[str] = None) -> Path:
        """Download dataset from Kaggle."""
        dataset_dir = self.cache_dir / "kaggle" / config.name
        
        # Check if already downloaded
        if all((dataset_dir / f).exists() for f in config.expected_files):
            print(f"Dataset {config.name} already cached.")
            return dataset_dir
            
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up Kaggle API credentials if provided
        if api_key:
            self._setup_kaggle_credentials(api_key)
        
        try:
            # Use Kaggle API to download
            import kaggle
            kaggle.api.dataset_download_files(
                config.identifier,
                path=str(dataset_dir),
                unzip=True
            )
            print(f"Successfully downloaded {config.name} from Kaggle.")
        except Exception as e:
            print(f"Error downloading from Kaggle: {e}")
            # Try alternative download method
            return self._download_kaggle_alternative(config, dataset_dir)
            
        return dataset_dir
    
    def _setup_kaggle_credentials(self, api_key: str):
        """Set up Kaggle API credentials."""
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        
        # Parse API key (format: username:key)
        if ":" in api_key:
            username, key = api_key.split(":", 1)
            credentials = {"username": username, "key": key}
            
            kaggle_json = kaggle_dir / "kaggle.json"
            with open(kaggle_json, "w") as f:
                json.dump(credentials, f)
            
            # Set permissions
            os.chmod(kaggle_json, 0o600)
    
    def _download_kaggle_alternative(self, config: DatasetConfig, dataset_dir: Path) -> Path:
        """Alternative Kaggle download using direct API calls."""
        print("Attempting alternative Kaggle download method...")
        
        # Use os.system as fallback
        exit_code = os.system(
            f"kaggle datasets download -d {config.identifier} "
            f"-p '{dataset_dir}' --unzip"
        )
        
        if exit_code != 0:
            raise RuntimeError(f"Failed to download {config.name} from Kaggle.")
            
        return dataset_dir
    
    def _download_mendeley(self, config: DatasetConfig, api_key: Optional[str] = None) -> Path:
        """Download dataset from Mendeley Data."""
        dataset_dir = self.cache_dir / "mendeley" / config.name
        
        if dataset_dir.exists() and any(dataset_dir.iterdir()):
            print(f"Dataset {config.name} already cached.")
            return dataset_dir
            
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Mendeley Data typically provides DOI-based URLs
        base_url = "https://data.mendeley.com/datasets"
        
        try:
            # Parse Mendeley identifier (DOI or dataset ID)
            if config.identifier.startswith("10."):  # DOI format
                doi = config.identifier
                response = requests.get(f"https://api.mendeley.com/datasets/doi/{doi}")
                if response.status_code == 200:
                    data = response.json()
                    download_url = data.get("download_url")
                    if download_url:
                        self._download_file(download_url, dataset_dir / "data.zip")
                        self._extract_archive(dataset_dir / "data.zip", dataset_dir)
            else:
                # Direct Mendeley dataset URL
                self._download_file(config.identifier, dataset_dir / "data.zip")
                self._extract_archive(dataset_dir / "data.zip", dataset_dir)
                
            print(f"Successfully downloaded {config.name} from Mendeley.")
        except Exception as e:
            print(f"Error downloading from Mendeley: {e}")
            raise
            
        return dataset_dir
    
    def _download_huggingface(self, config: DatasetConfig, api_key: Optional[str] = None) -> Path:
        """Download dataset from Hugging Face Hub."""
        dataset_dir = self.cache_dir / "huggingface" / config.name
        
        if dataset_dir.exists() and any(dataset_dir.iterdir()):
            print(f"Dataset {config.name} already cached.")
            return dataset_dir
            
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from datasets import load_dataset
            
            # Load dataset from Hugging Face
            dataset = load_dataset(config.identifier)
            
            # Save to disk in a standard format
            for split_name, split_data in dataset.items():
                output_file = dataset_dir / f"{split_name}.csv"
                split_data.to_pandas().to_csv(output_file, index=False)
                
            print(f"Successfully downloaded {config.name} from Hugging Face.")
        except ImportError:
            print("Hugging Face datasets library not installed. Installing...")
            os.system("pip install datasets")
            return self._download_huggingface(config, api_key)
        except Exception as e:
            print(f"Error downloading from Hugging Face: {e}")
            raise
            
        return dataset_dir
    
    def _download_direct(self, config: DatasetConfig) -> Path:
        """Download dataset from direct URL."""
        dataset_dir = self.cache_dir / "direct" / config.name
        
        if dataset_dir.exists() and any(dataset_dir.iterdir()):
            print(f"Dataset {config.name} already cached.")
            return dataset_dir
            
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse URL to get filename
        url = config.identifier
        parsed = urlparse(url)
        filename = Path(parsed.path).name or "data.zip"
        
        output_path = dataset_dir / filename
        self._download_file(url, output_path)
        
        # Extract if archive
        if filename.endswith(('.zip', '.tar', '.gz')):
            self._extract_archive(output_path, dataset_dir)
            
        print(f"Successfully downloaded {config.name} from URL.")
        return dataset_dir
    
    def _download_github(self, config: DatasetConfig) -> Path:
        """Download dataset from GitHub repository."""
        dataset_dir = self.cache_dir / "github" / config.name
        
        if dataset_dir.exists() and any(dataset_dir.iterdir()):
            print(f"Dataset {config.name} already cached.")
            return dataset_dir
            
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Clone or download from GitHub
        if config.identifier.endswith('.git'):
            # Clone repository
            os.system(f"git clone {config.identifier} {dataset_dir}")
        else:
            # Download as archive
            if "github.com" in config.identifier:
                # Convert to archive URL
                archive_url = config.identifier.replace("github.com", "codeload.github.com")
                archive_url += "/zip/refs/heads/main"  # or master
                
                output_path = dataset_dir / "repo.zip"
                self._download_file(archive_url, output_path)
                self._extract_archive(output_path, dataset_dir)
                
        print(f"Successfully downloaded {config.name} from GitHub.")
        return dataset_dir
    
    def _download_file(self, url: str, output_path: Path):
        """Download file with progress bar."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    def _extract_archive(self, archive_path: Path, output_dir: Path):
        """Extract archive file."""
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
        elif archive_path.suffix in ['.tar', '.gz', '.bz2']:
            import tarfile
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(output_dir)
        
        # Remove archive after extraction
        archive_path.unlink()


class DatasetProcessor:
    """Process and standardize downloaded datasets."""
    
    def __init__(self):
        self.processed_dir = Path("data/processed/datasets")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def process(self, config: DatasetConfig, raw_data_dir: Path) -> pd.DataFrame:
        """Process raw dataset into standardized format."""
        print(f"Processing {config.name}...")
        
        # Load data based on format
        if config.format == "csv":
            df = self._load_csv(config, raw_data_dir)
        elif config.format == "tsv":
            df = self._load_tsv(config, raw_data_dir)
        elif config.format == "json":
            df = self._load_json(config, raw_data_dir)
        else:
            raise ValueError(f"Unsupported format: {config.format}")
        
        # Standardize columns
        df = self._standardize_columns(df, config)
        
        # Apply preprocessing if specified
        if config.preprocessing:
            df = self._apply_preprocessing(df, config.preprocessing)
        
        # Save processed data
        output_path = self.processed_dir / f"{config.name}_processed.csv"
        df.to_csv(output_path, index=False)
        
        print(f"Processed {len(df)} samples from {config.name}")
        print(f"Saved to {output_path}")
        
        return df
    
    def _load_csv(self, config: DatasetConfig, data_dir: Path) -> pd.DataFrame:
        """Load CSV files."""
        dfs = []
        for file_pattern in config.expected_files:
            for file_path in data_dir.glob(file_pattern):
                if file_path.suffix == '.csv':
                    df = pd.read_csv(file_path)
                    dfs.append(df)
        
        if not dfs:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
            
        return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    
    def _load_tsv(self, config: DatasetConfig, data_dir: Path) -> pd.DataFrame:
        """Load TSV files."""
        dfs = []
        for file_pattern in config.expected_files:
            for file_path in data_dir.glob(file_pattern):
                if file_path.suffix == '.tsv':
                    df = pd.read_csv(file_path, sep='\t')
                    dfs.append(df)
        
        if not dfs:
            raise FileNotFoundError(f"No TSV files found in {data_dir}")
            
        return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    
    def _load_json(self, config: DatasetConfig, data_dir: Path) -> pd.DataFrame:
        """Load JSON files."""
        dfs = []
        for file_pattern in config.expected_files:
            for file_path in data_dir.glob(file_pattern):
                if file_path.suffix == '.json':
                    df = pd.read_json(file_path)
                    dfs.append(df)
        
        if not dfs:
            raise FileNotFoundError(f"No JSON files found in {data_dir}")
            
        return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    
    def _standardize_columns(self, df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
        """Standardize column names and structure."""
        # Create standardized dataframe
        standardized = pd.DataFrame()
        
        # Map text columns
        if len(config.text_columns) == 1:
            standardized['text'] = df[config.text_columns[0]].astype(str)
        else:
            # Concatenate multiple text columns
            standardized['text'] = df[config.text_columns].apply(
                lambda x: ' '.join(x.astype(str)), axis=1
            )
        
        # Map title column if exists
        if config.title_column and config.title_column in df.columns:
            standardized['title'] = df[config.title_column].astype(str)
        else:
            standardized['title'] = ""
        
        # Map label column
        if config.label_column in df.columns:
            standardized['label'] = df[config.label_column]
            # Standardize label values
            standardized['label'] = standardized['label'].apply(self._standardize_label)
        else:
            raise KeyError(f"Label column '{config.label_column}' not found in dataset")
        
        # Add source information
        standardized['source'] = config.name
        
        # Add ID column
        standardized['id'] = [f"{config.name}_{i}" for i in range(len(standardized))]
        
        return standardized
    
    def _standardize_label(self, label: Any) -> str:
        """Standardize label values to 'fake' or 'real'."""
        label = str(label).lower().strip()
        
        # Common mappings
        fake_labels = ['fake', 'false', '0', 'f', 'hoax', 'rumor', 'fabricated']
        real_labels = ['real', 'true', '1', 't', 'genuine', 'authentic', 'verified']
        
        if label in fake_labels:
            return 'fake'
        elif label in real_labels:
            return 'real'
        else:
            # Try to infer from label content
            if 'fake' in label or 'false' in label:
                return 'fake'
            elif 'real' in label or 'true' in label:
                return 'real'
            else:
                # Default to original label
                return label
    
    def _apply_preprocessing(self, df: pd.DataFrame, preprocessing: Dict[str, Any]) -> pd.DataFrame:
        """Apply custom preprocessing steps."""
        # Remove duplicates if specified
        if preprocessing.get('remove_duplicates', False):
            df = df.drop_duplicates(subset=['text'])
        
        # Remove short texts
        min_length = preprocessing.get('min_text_length', 10)
        df = df[df['text'].str.len() >= min_length]
        
        # Remove missing values
        if preprocessing.get('remove_missing', True):
            df = df.dropna(subset=['text', 'label'])
        
        # Custom filters
        if 'filters' in preprocessing:
            for filter_func in preprocessing['filters']:
                df = filter_func(df)
        
        return df


class DatasetPipeline:
    """Main pipeline for dataset management."""
    
    def __init__(self):
        self.downloader = DatasetDownloader()
        self.processor = DatasetProcessor()
        self.configs = self._load_dataset_configs()
    
    def _load_dataset_configs(self) -> Dict[str, DatasetConfig]:
        """Load predefined dataset configurations."""
        configs = {
            # Existing ISOT dataset
            "isot": DatasetConfig(
                name="isot",
                source=DataSource.KAGGLE,
                identifier="csmalarkodi/isot-fake-news-dataset",
                expected_files=["Fake.csv", "True.csv"],
                format="csv",
                label_column="label",
                text_columns=["text"],
                title_column="title"
            ),
            
            # LIAR dataset
            "liar": DatasetConfig(
                name="liar",
                source=DataSource.KAGGLE,
                identifier="doanquanvietnamca/liar-dataset",
                expected_files=["train.tsv", "test.tsv", "valid.tsv"],
                format="tsv",
                label_column="label",
                text_columns=["statement"],
                title_column=None
            ),
            
            # FakeNewsNet dataset
            "fakenewsnet": DatasetConfig(
                name="fakenewsnet",
                source=DataSource.GITHUB,
                identifier="https://github.com/KaiDMML/FakeNewsNet",
                expected_files=["*.csv"],
                format="csv",
                label_column="label",
                text_columns=["text"],
                title_column="title"
            ),
            
            # FEVER dataset
            "fever": DatasetConfig(
                name="fever",
                source=DataSource.HUGGINGFACE,
                identifier="fever",
                expected_files=["*.csv"],
                format="csv",
                label_column="label",
                text_columns=["claim"],
                title_column=None
            ),
            
            # Multi-source COVID-19 fake news
            "covid19_fake": DatasetConfig(
                name="covid19_fake",
                source=DataSource.KAGGLE,
                identifier="arashnic/covid19-fake-news",
                expected_files=["*.csv"],
                format="csv",
                label_column="label",
                text_columns=["text", "title"],
                title_column="title"
            ),
            
            # Politifact dataset
            "politifact": DatasetConfig(
                name="politifact",
                source=DataSource.DIRECT_URL,
                identifier="https://www.politifact.com/api/statements/",
                expected_files=["*.json"],
                format="json",
                label_column="ruling",
                text_columns=["statement"],
                title_column=None
            )
        }
        
        return configs
    
    def add_custom_dataset(self, config: DatasetConfig):
        """Add a custom dataset configuration."""
        self.configs[config.name] = config
    
    def list_available_datasets(self) -> List[str]:
        """List all available dataset configurations."""
        return list(self.configs.keys())
    
    def download_and_process(
        self,
        dataset_name: str,
        api_key: Optional[str] = None,
        force_redownload: bool = False
    ) -> pd.DataFrame:
        """Download and process a dataset."""
        if dataset_name not in self.configs:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Available: {self.list_available_datasets()}")
        
        config = self.configs[dataset_name]
        
        # Download dataset
        if force_redownload:
            # Clear cache for this dataset
            cache_path = self.downloader.cache_dir / config.source.value / config.name
            if cache_path.exists():
                shutil.rmtree(cache_path)
        
        raw_data_dir = self.downloader.download(config, api_key)
        
        # Process dataset
        processed_df = self.processor.process(config, raw_data_dir)
        
        return processed_df
    
    def download_multiple(
        self,
        dataset_names: List[str],
        api_keys: Optional[Dict[str, str]] = None,
        force_redownload: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Download and process multiple datasets."""
        results = {}
        api_keys = api_keys or {}
        
        for name in dataset_names:
            print(f"\n{'='*50}")
            print(f"Processing dataset: {name}")
            print('='*50)
            
            api_key = api_keys.get(name)
            df = self.download_and_process(name, api_key, force_redownload)
            results[name] = df
        
        return results
    
    def combine_datasets(
        self,
        datasets: Dict[str, pd.DataFrame],
        balance: bool = True
    ) -> pd.DataFrame:
        """Combine multiple processed datasets."""
        combined = pd.concat(datasets.values(), ignore_index=True)
        
        if balance:
            # Balance fake and real samples
            fake_samples = combined[combined['label'] == 'fake']
            real_samples = combined[combined['label'] == 'real']
            
            min_samples = min(len(fake_samples), len(real_samples))
            
            balanced = pd.concat([
                fake_samples.sample(n=min_samples, random_state=42),
                real_samples.sample(n=min_samples, random_state=42)
            ])
            
            return balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return combined
    
    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about a processed dataset."""
        stats = {
            'total_samples': len(df),
            'label_distribution': df['label'].value_counts().to_dict(),
            'source_distribution': df['source'].value_counts().to_dict() if 'source' in df else {},
            'avg_text_length': df['text'].str.len().mean(),
            'avg_word_count': df['text'].str.split().str.len().mean(),
            'has_title': (df['title'].str.len() > 0).sum() if 'title' in df else 0
        }
        
        return stats