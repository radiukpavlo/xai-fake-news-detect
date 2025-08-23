# Dataset Pipeline Documentation

## Overview

The enhanced dataset pipeline provides a unified interface for downloading, processing, and integrating fake news datasets from multiple sources including Kaggle, Mendeley Data, Hugging Face, GitHub, and direct URLs.

## Features

- **Multi-source Support**: Download datasets from Kaggle, Mendeley, Hugging Face, GitHub, and direct URLs
- **Automatic Processing**: Standardize datasets into a common format
- **Caching**: Intelligent caching to avoid re-downloading
- **Batch Operations**: Download and process multiple datasets at once
- **Custom Datasets**: Support for custom dataset integration
- **API Key Management**: Secure handling of API credentials

## Quick Start

### 1. Install Dependencies

```bash
pip install -e . -r requirements.txt
```

### 2. Set Up API Keys

#### Kaggle
1. Go to https://www.kaggle.com/account
2. Click "Create New API Token"
3. Save the downloaded `kaggle.json` to `~/.kaggle/`

Or provide directly via CLI:
```bash
python -m mmx_news.cli_dataset download isot --kaggle-key "username:api_key"
```

#### Mendeley Data
1. Register at https://data.mendeley.com
2. Get API key from account settings
3. Provide via CLI:
```bash
python -m mmx_news.cli_dataset download mendeley_fake_news --mendeley-key "your_api_key"
```

#### Hugging Face (optional for public datasets)
1. Get token from https://huggingface.co/settings/tokens
2. Provide via CLI:
```bash
python -m mmx_news.cli_dataset download fever --hf-token "hf_xxxxx"
```

## Usage Examples

### List Available Datasets

```bash
python -m mmx_news.cli_dataset list-datasets
```

Output:
```
Available Fake News Datasets
======================================================================

📊 isot
   Name: ISOT Fake News Dataset
   Source: kaggle
   Description: Primary dataset with 44,898 articles
   Format: csv

📊 liar
   Name: LIAR Dataset
   Source: kaggle
   Description: 12.8K labeled short statements from PolitiFact
   Format: tsv

📊 covid19_fake
   Name: COVID-19 Fake News
   Source: kaggle
   Description: COVID-19 related misinformation dataset
   Format: csv

[... more datasets ...]
```

### Download Single Dataset

```bash
# Download ISOT dataset
python -m mmx_news.cli_dataset download isot --kaggle-key "username:key"

# Download with force re-download
python -m mmx_news.cli_dataset download liar --kaggle-key "username:key" --force

# Save to custom location
python -m mmx_news.cli_dataset download covid19_fake \
    --kaggle-key "username:key" \
    --output data/custom/covid19.csv
```

### Download Multiple Datasets

```bash
# Download specific datasets
python -m mmx_news.cli_dataset download-multiple \
    --datasets "isot,liar,covid19_fake" \
    --kaggle-key "username:key"

# Download all available datasets
python -m mmx_news.cli_dataset download-multiple \
    --datasets all \
    --kaggle-key "username:key" \
    --mendeley-key "mendeley_key" \
    --hf-token "hf_token"

# Download and combine with balancing
python -m mmx_news.cli_dataset download-multiple \
    --datasets "isot,liar" \
    --kaggle-key "username:key" \
    --combine \
    --balance
```

### Process Custom Dataset

```bash
# From URL
python -m mmx_news.cli_dataset process-custom \
    --url "https://example.com/dataset.csv" \
    --name "my_dataset" \
    --format csv \
    --label-col "label" \
    --text-cols "content,summary" \
    --title-col "headline"

# From local file
python -m mmx_news.cli_dataset process-custom \
    --path "/path/to/dataset.csv" \
    --name "local_dataset" \
    --format csv \
    --label-col "is_fake" \
    --text-cols "article_text"
```

### Integrate into Main Pipeline

```bash
# Integrate all downloaded datasets
python -m mmx_news.cli_dataset integrate --prepare

# This will:
# 1. Combine all processed datasets
# 2. Convert to ISOT format for compatibility
# 3. Save to data/isot/
# 4. Run data preparation for train/val/test splits
```

## Complete Workflow Example

Here's a complete example workflow for setting up multiple datasets:

```bash
# Step 1: List available datasets
python -m mmx_news.cli_dataset list-datasets

# Step 2: Download multiple datasets with API keys
python -m mmx_news.cli_dataset download-multiple \
    --datasets "isot,liar,covid19_fake" \
    --kaggle-key "your_username:your_api_key" \
    --combine \
    --balance

# Step 3: Integrate into main pipeline
python -m mmx_news.cli_dataset integrate --prepare

# Step 4: Train model with integrated datasets
python -m mmx_news.cli train --config configs/default.yaml --seed 13

# Step 5: Evaluate model
python -m mmx_news.cli evaluate --checkpoint runs/exp1/best.joblib --split test
```

## Python API Usage

You can also use the dataset pipeline programmatically:

```python
from mmx_news.data.dataset_pipeline import DatasetPipeline

# Initialize pipeline
pipeline = DatasetPipeline()

# List available datasets
datasets = pipeline.list_available_datasets()
print(f"Available datasets: {datasets}")

# Download and process single dataset
df_isot = pipeline.download_and_process(
    "isot",
    api_key="username:key"
)

# Download multiple datasets
results = pipeline.download_multiple(
    ["isot", "liar", "covid19_fake"],
    api_keys={"kaggle": "username:key"}
)

# Combine datasets with balancing
combined = pipeline.combine_datasets(results, balance=True)

# Get statistics
stats = pipeline.get_statistics(combined)
print(f"Total samples: {stats['total_samples']}")
print(f"Label distribution: {stats['label_distribution']}")
```

## Adding Custom Datasets

To add a new dataset to the pipeline, edit `configs/datasets.yaml`:

```yaml
datasets:
  my_custom_dataset:
    name: "My Custom Dataset"
    source: direct_url  # or kaggle, mendeley, huggingface, github
    identifier: "https://example.com/dataset.zip"
    expected_files: ["data.csv"]
    format: csv
    label_column: truth_value
    text_columns: [article_text]
    title_column: headline
    description: "Custom fake news dataset"
```

Then use it:
```bash
python -m mmx_news.cli_dataset download my_custom_dataset
```

## Dataset Formats

The pipeline standardizes all datasets to the following format:

| Column | Type | Description |
|--------|------|-------------|
| id | string | Unique identifier |
| text | string | Main text content |
| title | string | Article title (if available) |
| label | string | "fake" or "real" |
| source | string | Dataset source name |

## Caching

Downloaded datasets are cached in `data/cache/` to avoid re-downloading:

```
data/cache/
├── kaggle/
│   ├── isot/
│   └── liar/
├── mendeley/
├── huggingface/
├── github/
└── direct/
```

Use `--force` flag to force re-download.

## Troubleshooting

### Kaggle API Issues
- Ensure `kaggle.json` is in `~/.kaggle/` with correct permissions (600)
- Check API key format: `username:key`
- Verify dataset identifier matches Kaggle URL

### Mendeley Data Issues
- Some Mendeley datasets require authentication
- Check DOI format if using DOI identifier

### Memory Issues with Large Datasets
- Process datasets in batches
- Use `--balance` flag to limit dataset size
- Increase system swap space if needed

### Network Issues
- Check internet connection
- Try using a VPN if datasets are region-restricted
- Use `--force` flag if partial download occurred

## Supported Datasets

Currently configured datasets:

1. **ISOT** - 44,898 articles from various news sources
2. **LIAR** - 12.8K short statements from PolitiFact
3. **FakeNewsNet** - Multi-domain with social context
4. **COVID-19 Fake News** - COVID-related misinformation
5. **Fake News Corpus** - Large corpus from various sources
6. **BuzzFeed News** - Facebook news analysis
7. **FEVER** - Fact extraction and verification
8. **Multi-FC** - Multi-domain fact-checking

## Performance Considerations

- **Batch Processing**: Use `batch_size` in config for large datasets
- **Parallel Processing**: Configure `n_workers` for parallel processing
- **Memory Management**: Process large datasets in chunks
- **Disk Space**: Ensure sufficient space for cache and processed data

## API Rate Limits

Be aware of API rate limits:
- **Kaggle**: 1000 requests per day
- **Hugging Face**: Varies by account type
- **GitHub**: 60 requests per hour (unauthenticated)

## Contributing

To add support for new data sources:

1. Add source type to `DataSource` enum
2. Implement download method in `DatasetDownloader`
3. Add configuration to `configs/datasets.yaml`
4. Submit pull request with documentation

## License

This dataset pipeline is part of the XAI Fake News Detection project and is licensed under MIT License.