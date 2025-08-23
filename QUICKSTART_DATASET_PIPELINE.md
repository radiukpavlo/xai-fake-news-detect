# Quick Start Guide: Dataset Pipeline

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies
```bash
pip install -e . -r requirements.txt
```

### Step 2: Get Your API Keys

#### For Kaggle (Required for most datasets):
1. Go to https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. Note your username and key from the downloaded `kaggle.json`

#### For Mendeley Data (Optional):
1. Register at https://data.mendeley.com
2. Get API key from account settings

#### For Hugging Face (Optional):
1. Get token from https://huggingface.co/settings/tokens

### Step 3: Download Your First Dataset
```bash
# Replace with your actual Kaggle credentials
python -m mmx_news.cli dataset download isot --kaggle-key "your_username:your_api_key"
```

### Step 4: Download Multiple Datasets
```bash
# Download and combine popular datasets
python -m mmx_news.cli dataset download-multiple \
    --datasets "isot,liar,covid19_fake" \
    --kaggle-key "your_username:your_api_key" \
    --combine \
    --balance
```

### Step 5: Integrate and Train
```bash
# Integrate datasets into the main pipeline
python -m mmx_news.cli dataset integrate --prepare

# Train your model
python -m mmx_news.cli train --config configs/default.yaml --seed 13

# Evaluate
python -m mmx_news.cli evaluate --checkpoint runs/exp1/best.joblib --split test
```

## 📊 Available Commands

### List All Available Datasets
```bash
python -m mmx_news.cli dataset list-datasets
```

### Download Specific Dataset
```bash
python -m mmx_news.cli dataset download <dataset_name> --kaggle-key "username:key"
```

Available datasets:
- `isot` - 44,898 news articles
- `liar` - 12.8K PolitiFact statements  
- `covid19_fake` - COVID-19 misinformation
- `fakenewsnet` - Multi-domain dataset
- `fever` - Fact verification (Hugging Face)
- `fake_news_corpus` - Large news corpus
- `buzzfeed_news` - Facebook news analysis

### Process Your Own Dataset
```bash
# From URL
python -m mmx_news.cli dataset process-custom \
    --url "https://your-dataset-url.com/data.csv" \
    --label-col "truth_label" \
    --text-cols "article_text,summary" \
    --title-col "headline"

# From local file
python -m mmx_news.cli dataset process-custom \
    --path "/path/to/your/dataset.csv" \
    --label-col "is_fake" \
    --text-cols "content"
```

## 🐍 Python API Usage

```python
from mmx_news.data.dataset_pipeline import DatasetPipeline

# Initialize
pipeline = DatasetPipeline()

# Download single dataset
df = pipeline.download_and_process("isot", api_key="username:key")

# Download multiple and combine
results = pipeline.download_multiple(
    ["isot", "liar"],
    api_keys={"kaggle": "username:key"}
)
combined = pipeline.combine_datasets(results, balance=True)

# Get statistics
stats = pipeline.get_statistics(combined)
print(f"Total: {stats['total_samples']} samples")
print(f"Labels: {stats['label_distribution']}")
```

## 🎯 Common Use Cases

### 1. Quick Test with Small Dataset
```bash
# Use the built-in smoke dataset for testing
python -m mmx_news.cli train --config configs/default.yaml --mode smoke
```

### 2. Combine Multiple Sources for Robust Training
```bash
# Download all available Kaggle datasets
python -m mmx_news.cli dataset download-multiple \
    --datasets "isot,liar,covid19_fake,fake_news_corpus" \
    --kaggle-key "username:key" \
    --combine --balance

# Integrate and train
python -m mmx_news.cli dataset integrate --prepare
python -m mmx_news.cli train --config configs/default.yaml
```

### 3. Add Your Research Dataset
1. Edit `configs/datasets.yaml` to add your dataset:
```yaml
datasets:
  my_dataset:
    name: "My Research Dataset"
    source: direct_url
    identifier: "https://my-university.edu/dataset.zip"
    expected_files: ["data.csv"]
    format: csv
    label_column: verification_status
    text_columns: [article_body]
    title_column: headline
```

2. Download and use:
```bash
python -m mmx_news.cli dataset download my_dataset
```

## 📁 Where Are My Datasets?

- **Downloaded (cached)**: `data/cache/`
- **Processed**: `data/processed/datasets/`
- **Integrated (ISOT format)**: `data/isot/`
- **Train/Val/Test splits**: `data/processed/splits/`

## 🔧 Troubleshooting

### "Kaggle API key not found"
- Make sure you provide `--kaggle-key "username:key"` 
- Or place `kaggle.json` in `~/.kaggle/`

### "Dataset not found"
- Run `python -m mmx_news.cli dataset list-datasets` to see available datasets
- Check spelling of dataset name

### "Out of memory"
- Use `--balance` flag to limit dataset size
- Process datasets one at a time

### "Network error"
- Check internet connection
- Some datasets may be region-restricted
- Use `--force` to retry failed downloads

## 📚 More Information

- Full documentation: [docs/DATASET_PIPELINE.md](docs/DATASET_PIPELINE.md)
- Example script: [examples/dataset_pipeline_demo.py](examples/dataset_pipeline_demo.py)
- Dataset configs: [configs/datasets.yaml](configs/datasets.yaml)

## 💡 Tips

1. **Start Small**: Test with one dataset first
2. **Use Caching**: Downloaded datasets are cached - no need to re-download
3. **Balance Data**: Use `--balance` to ensure equal fake/real samples
4. **Check Stats**: Always review dataset statistics before training
5. **Custom Datasets**: The pipeline supports any CSV/JSON with text and labels

## 🎉 You're Ready!

You now have a powerful dataset pipeline that can handle fake news datasets from multiple sources. Happy training!