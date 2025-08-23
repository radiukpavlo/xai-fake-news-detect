#!/usr/bin/env python3
"""
Dataset Pipeline Demo
=====================

This script demonstrates how to use the dataset pipeline to:
1. Download multiple fake news datasets
2. Process and standardize them
3. Integrate them into the main training pipeline
4. Train and evaluate models

Usage:
    python examples/dataset_pipeline_demo.py --kaggle-key "username:api_key"
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mmx_news.data.dataset_pipeline import DatasetPipeline, DatasetConfig, DataSource
from mmx_news.data.loaders import prepare_splits
from mmx_news.training.utils import load_config


def setup_environment():
    """Set up the environment for the demo."""
    # Create necessary directories
    dirs = [
        "data/cache",
        "data/processed/datasets",
        "data/isot",
        "runs",
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Environment setup complete")


def demo_single_dataset(api_key: str = None):
    """Demonstrate downloading and processing a single dataset."""
    print("\n" + "="*70)
    print("DEMO: Single Dataset Download")
    print("="*70)
    
    pipeline = DatasetPipeline()
    
    # Download ISOT dataset
    print("\n📊 Downloading ISOT dataset...")
    df_isot = pipeline.download_and_process("isot", api_key=api_key)
    
    # Get statistics
    stats = pipeline.get_statistics(df_isot)
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Label distribution: {stats['label_distribution']}")
    print(f"   Avg text length: {stats['avg_text_length']:.0f} chars")
    print(f"   Avg word count: {stats['avg_word_count']:.0f} words")
    
    # Show sample
    print(f"\n📝 Sample article:")
    sample = df_isot.iloc[0]
    print(f"   ID: {sample['id']}")
    print(f"   Label: {sample['label']}")
    print(f"   Title: {sample['title'][:80]}...")
    print(f"   Text: {sample['text'][:200]}...")
    
    return df_isot


def demo_multiple_datasets(api_key: str = None):
    """Demonstrate downloading and combining multiple datasets."""
    print("\n" + "="*70)
    print("DEMO: Multiple Dataset Download and Combination")
    print("="*70)
    
    pipeline = DatasetPipeline()
    
    # Download multiple datasets
    datasets_to_download = ["isot", "liar"]
    
    print(f"\n📊 Downloading {len(datasets_to_download)} datasets...")
    results = {}
    
    for dataset_name in datasets_to_download:
        try:
            print(f"\n   Processing {dataset_name}...")
            df = pipeline.download_and_process(dataset_name, api_key=api_key)
            results[dataset_name] = df
            print(f"   ✅ {dataset_name}: {len(df)} samples")
        except Exception as e:
            print(f"   ❌ {dataset_name}: Failed - {e}")
    
    if not results:
        print("No datasets downloaded successfully.")
        return None
    
    # Combine datasets
    print("\n🔗 Combining datasets...")
    combined = pipeline.combine_datasets(results, balance=True)
    
    print(f"\n📈 Combined Dataset Statistics:")
    stats = pipeline.get_statistics(combined)
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Sources: {stats['source_distribution']}")
    print(f"   Label distribution: {stats['label_distribution']}")
    
    # Save combined dataset
    output_path = Path("data/processed/combined_demo.csv")
    combined.to_csv(output_path, index=False)
    print(f"\n💾 Saved combined dataset to: {output_path}")
    
    return combined


def demo_custom_dataset():
    """Demonstrate processing a custom dataset."""
    print("\n" + "="*70)
    print("DEMO: Custom Dataset Processing")
    print("="*70)
    
    pipeline = DatasetPipeline()
    
    # Create a sample custom dataset
    import pandas as pd
    
    custom_data = pd.DataFrame({
        'headline': [
            'Breaking: Major Scientific Discovery',
            'Celebrity Scandal Rocks Hollywood',
            'New Study Shows Surprising Results',
            'Shocking Truth Revealed',
        ],
        'content': [
            'Scientists have discovered a new method for detecting fake news using AI...',
            'A fabricated story about a celebrity that never happened...',
            'Researchers found that machine learning can accurately identify misinformation...',
            'This sensational headline is completely made up for clicks...',
        ],
        'veracity': ['real', 'fake', 'real', 'fake']
    })
    
    # Save to temporary file
    temp_path = Path("data/cache/custom_demo.csv")
    custom_data.to_csv(temp_path, index=False)
    
    # Create custom configuration
    custom_config = DatasetConfig(
        name="custom_demo",
        source=DataSource.DIRECT_URL,
        identifier=str(temp_path.absolute()),
        expected_files=["custom_demo.csv"],
        format="csv",
        label_column="veracity",
        text_columns=["content"],
        title_column="headline"
    )
    
    # Add to pipeline and process
    pipeline.add_custom_dataset(custom_config)
    
    print("\n📊 Processing custom dataset...")
    df_custom = pipeline.download_and_process("custom_demo")
    
    print(f"\n✅ Custom dataset processed:")
    print(f"   Samples: {len(df_custom)}")
    print(f"   Columns: {df_custom.columns.tolist()}")
    
    return df_custom


def demo_integration(combined_df):
    """Demonstrate integration with main pipeline."""
    print("\n" + "="*70)
    print("DEMO: Integration with Main Pipeline")
    print("="*70)
    
    if combined_df is None:
        print("No data to integrate.")
        return
    
    # Convert to ISOT format
    print("\n🔄 Converting to ISOT format...")
    
    fake_samples = combined_df[combined_df['label'] == 'fake'][['title', 'text']]
    real_samples = combined_df[combined_df['label'] == 'real'][['title', 'text']]
    
    # Add label column for ISOT format
    fake_samples = fake_samples.copy()
    real_samples = real_samples.copy()
    fake_samples['label'] = 'fake'
    real_samples['label'] = 'real'
    
    # Save in ISOT format
    isot_dir = Path("data/isot")
    fake_samples.to_csv(isot_dir / "Fake.csv", index=False)
    real_samples.to_csv(isot_dir / "True.csv", index=False)
    
    print(f"✅ Saved to ISOT format:")
    print(f"   Fake samples: {len(fake_samples)}")
    print(f"   Real samples: {len(real_samples)}")
    print(f"   Location: {isot_dir}")
    
    # Prepare splits
    print("\n🔄 Preparing train/val/test splits...")
    
    try:
        config = load_config("configs/default.yaml")
        prepare_splits(config, mode="full")
        print("✅ Data splits prepared successfully")
    except Exception as e:
        print(f"⚠️  Could not prepare splits: {e}")
    
    print("\n🎯 Ready for training!")
    print("   Run: python -m mmx_news.cli train --config configs/default.yaml --seed 13")


def demo_api_usage():
    """Demonstrate programmatic API usage."""
    print("\n" + "="*70)
    print("DEMO: Programmatic API Usage")
    print("="*70)
    
    print("""
    Example Python code for using the dataset pipeline:
    
    ```python
    from mmx_news.data.dataset_pipeline import DatasetPipeline
    
    # Initialize pipeline
    pipeline = DatasetPipeline()
    
    # List available datasets
    datasets = pipeline.list_available_datasets()
    print(f"Available: {datasets}")
    
    # Download and process
    df = pipeline.download_and_process("isot", api_key="username:key")
    
    # Get statistics
    stats = pipeline.get_statistics(df)
    print(f"Total samples: {stats['total_samples']}")
    
    # Combine multiple datasets
    results = pipeline.download_multiple(
        ["isot", "liar"],
        api_keys={"kaggle": "username:key"}
    )
    combined = pipeline.combine_datasets(results, balance=True)
    ```
    """)


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="Dataset Pipeline Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with Kaggle API key
  python examples/dataset_pipeline_demo.py --kaggle-key "username:api_key"
  
  # Run specific demos
  python examples/dataset_pipeline_demo.py --demo single --kaggle-key "username:api_key"
  python examples/dataset_pipeline_demo.py --demo custom
  
  # Skip certain demos
  python examples/dataset_pipeline_demo.py --skip-multiple --kaggle-key "username:api_key"
        """
    )
    
    parser.add_argument(
        "--kaggle-key",
        help="Kaggle API key (format: username:key)"
    )
    parser.add_argument(
        "--demo",
        choices=["all", "single", "multiple", "custom", "api"],
        default="all",
        help="Which demo to run"
    )
    parser.add_argument(
        "--skip-multiple",
        action="store_true",
        help="Skip multiple dataset demo (faster)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("FAKE NEWS DATASET PIPELINE DEMO")
    print("="*70)
    
    # Setup environment
    setup_environment()
    
    # Set up Kaggle credentials if provided
    if args.kaggle_key:
        os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = args.kaggle_key.split(":", 1)
        print(f"✅ Kaggle credentials configured")
    
    combined_df = None
    
    # Run demos based on selection
    if args.demo in ["all", "single"]:
        df_single = demo_single_dataset(args.kaggle_key)
    
    if args.demo in ["all", "multiple"] and not args.skip_multiple:
        combined_df = demo_multiple_datasets(args.kaggle_key)
    
    if args.demo in ["all", "custom"]:
        df_custom = demo_custom_dataset()
    
    if args.demo in ["all", "api"]:
        demo_api_usage()
    
    # Integration demo (only if we have data)
    if combined_df is not None and args.demo == "all":
        demo_integration(combined_df)
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Use the CLI to download more datasets:")
    print("   python -m mmx_news.cli dataset list-datasets")
    print("   python -m mmx_news.cli dataset download <dataset_name> --kaggle-key <key>")
    print("\n2. Train a model with the integrated datasets:")
    print("   python -m mmx_news.cli train --config configs/default.yaml --seed 13")
    print("\n3. See docs/DATASET_PIPELINE.md for more information")


if __name__ == "__main__":
    main()