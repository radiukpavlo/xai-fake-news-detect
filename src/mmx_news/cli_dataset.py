"""
Enhanced CLI for Dataset Pipeline Management
============================================

This module provides CLI commands for managing fake news datasets,
including downloading, processing, and integrating them into the
main pipeline.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from .data.dataset_pipeline import (
    DatasetConfig,
    DatasetPipeline,
    DataSource,
)
from .utils.config import Config
from .training.utils import load_config


def load_dataset_config(config_path: str = "configs/datasets.yaml") -> Dict:
    """Load dataset configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_api_credentials(api_keys: Dict[str, str]):
    """Set up API credentials from provided keys."""
    for service, key in api_keys.items():
        if service == "kaggle":
            # Set up Kaggle credentials
            if ":" in key:
                os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = key.split(":", 1)
            # Also create kaggle.json if needed
            kaggle_dir = Path.home() / ".kaggle"
            kaggle_dir.mkdir(exist_ok=True)
            kaggle_json = kaggle_dir / "kaggle.json"
            if ":" in key:
                username, api_key = key.split(":", 1)
                with open(kaggle_json, 'w') as f:
                    json.dump({"username": username, "key": api_key}, f)
                os.chmod(kaggle_json, 0o600)
        elif service == "mendeley":
            os.environ["MENDELEY_API_KEY"] = key
        elif service == "huggingface":
            os.environ["HF_TOKEN"] = key


def cmd_list_datasets(args):
    """List all available datasets."""
    config = load_dataset_config(args.config)
    
    print("\n" + "="*70)
    print("Available Fake News Datasets")
    print("="*70)
    
    for dataset_id, dataset_info in config['datasets'].items():
        print(f"\n📊 {dataset_id}")
        print(f"   Name: {dataset_info.get('name', 'N/A')}")
        print(f"   Source: {dataset_info.get('source', 'N/A')}")
        print(f"   Description: {dataset_info.get('description', 'N/A')}")
        print(f"   Format: {dataset_info.get('format', 'N/A')}")
    
    print("\n" + "="*70)
    print(f"Total datasets available: {len(config['datasets'])}")
    print("="*70)


def cmd_download_dataset(args):
    """Download a specific dataset."""
    config = load_dataset_config(args.config)
    pipeline = DatasetPipeline()
    
    # Parse API keys if provided
    api_keys = {}
    if args.kaggle_key:
        api_keys['kaggle'] = args.kaggle_key
        setup_api_credentials({'kaggle': args.kaggle_key})
    if args.mendeley_key:
        api_keys['mendeley'] = args.mendeley_key
        setup_api_credentials({'mendeley': args.mendeley_key})
    if args.hf_token:
        api_keys['huggingface'] = args.hf_token
        setup_api_credentials({'huggingface': args.hf_token})
    
    # Check if dataset exists in config
    if args.dataset not in config['datasets']:
        print(f"Error: Dataset '{args.dataset}' not found in configuration.")
        print("Use 'list-datasets' to see available datasets.")
        return
    
    dataset_config = config['datasets'][args.dataset]
    
    # Create DatasetConfig object
    ds_config = DatasetConfig(
        name=args.dataset,
        source=DataSource(dataset_config['source']),
        identifier=dataset_config['identifier'],
        expected_files=dataset_config['expected_files'],
        format=dataset_config['format'],
        label_column=dataset_config['label_column'],
        text_columns=dataset_config['text_columns'],
        title_column=dataset_config.get('title_column'),
    )
    
    # Add to pipeline
    pipeline.add_custom_dataset(ds_config)
    
    print(f"\n🔄 Downloading dataset: {args.dataset}")
    print("="*50)
    
    try:
        # Download and process
        df = pipeline.download_and_process(
            args.dataset,
            api_key=api_keys.get(dataset_config['source']),
            force_redownload=args.force
        )
        
        # Show statistics
        stats = pipeline.get_statistics(df)
        print(f"\n✅ Successfully processed {args.dataset}")
        print(f"   Total samples: {stats['total_samples']}")
        print(f"   Label distribution: {stats['label_distribution']}")
        print(f"   Avg text length: {stats['avg_text_length']:.0f} chars")
        print(f"   Avg word count: {stats['avg_word_count']:.0f} words")
        
        if args.output:
            # Save to custom location
            output_path = Path(args.output)
            df.to_csv(output_path, index=False)
            print(f"   Saved to: {output_path}")
            
    except Exception as e:
        print(f"\n❌ Error downloading {args.dataset}: {e}")
        return 1
    
    return 0


def cmd_download_multiple(args):
    """Download multiple datasets."""
    config = load_dataset_config(args.config)
    pipeline = DatasetPipeline()
    
    # Parse dataset list
    if args.datasets == "all":
        dataset_names = list(config['datasets'].keys())
    else:
        dataset_names = args.datasets.split(",")
    
    # Setup API credentials
    api_keys = {}
    if args.kaggle_key:
        setup_api_credentials({'kaggle': args.kaggle_key})
    if args.mendeley_key:
        setup_api_credentials({'mendeley': args.mendeley_key})
    if args.hf_token:
        setup_api_credentials({'huggingface': args.hf_token})
    
    print(f"\n🔄 Downloading {len(dataset_names)} datasets...")
    print("="*70)
    
    # Configure all datasets
    for name in dataset_names:
        if name not in config['datasets']:
            print(f"⚠️  Warning: Dataset '{name}' not found, skipping...")
            continue
            
        dataset_config = config['datasets'][name]
        ds_config = DatasetConfig(
            name=name,
            source=DataSource(dataset_config['source']),
            identifier=dataset_config['identifier'],
            expected_files=dataset_config['expected_files'],
            format=dataset_config['format'],
            label_column=dataset_config['label_column'],
            text_columns=dataset_config['text_columns'],
            title_column=dataset_config.get('title_column'),
        )
        pipeline.add_custom_dataset(ds_config)
    
    # Download all
    results = {}
    for name in dataset_names:
        if name not in config['datasets']:
            continue
            
        try:
            dataset_source = config['datasets'][name]['source']
            api_key = None
            
            if dataset_source == 'kaggle' and args.kaggle_key:
                api_key = args.kaggle_key
            elif dataset_source == 'mendeley' and args.mendeley_key:
                api_key = args.mendeley_key
            elif dataset_source == 'huggingface' and args.hf_token:
                api_key = args.hf_token
            
            df = pipeline.download_and_process(name, api_key, args.force)
            results[name] = df
            print(f"✅ {name}: {len(df)} samples")
        except Exception as e:
            print(f"❌ {name}: Failed - {e}")
    
    # Combine if requested
    if args.combine and results:
        print("\n🔗 Combining datasets...")
        combined = pipeline.combine_datasets(results, balance=args.balance)
        
        output_path = Path("data/processed/combined_dataset.csv")
        combined.to_csv(output_path, index=False)
        
        print(f"✅ Combined dataset saved to: {output_path}")
        print(f"   Total samples: {len(combined)}")
        print(f"   Sources: {combined['source'].value_counts().to_dict()}")
    
    return 0


def cmd_process_custom(args):
    """Process a custom dataset."""
    pipeline = DatasetPipeline()
    
    # Create custom config
    custom_config = DatasetConfig(
        name=args.name or "custom",
        source=DataSource.DIRECT_URL if args.url else DataSource.DIRECT_URL,
        identifier=args.url or args.path,
        expected_files=args.files.split(",") if args.files else ["*.csv"],
        format=args.format or "csv",
        label_column=args.label_col,
        text_columns=args.text_cols.split(","),
        title_column=args.title_col,
    )
    
    pipeline.add_custom_dataset(custom_config)
    
    print(f"\n🔄 Processing custom dataset: {custom_config.name}")
    
    try:
        df = pipeline.download_and_process(custom_config.name)
        
        stats = pipeline.get_statistics(df)
        print(f"\n✅ Successfully processed custom dataset")
        print(f"   Total samples: {stats['total_samples']}")
        print(f"   Label distribution: {stats['label_distribution']}")
        
        if args.output:
            output_path = Path(args.output)
            df.to_csv(output_path, index=False)
            print(f"   Saved to: {output_path}")
            
    except Exception as e:
        print(f"\n❌ Error processing custom dataset: {e}")
        return 1
    
    return 0


def cmd_integrate(args):
    """Integrate downloaded datasets into the main pipeline."""
    config = load_config(args.config)
    pipeline = DatasetPipeline()
    
    # Find all processed datasets
    processed_dir = Path("data/processed/datasets")
    if not processed_dir.exists():
        print("No processed datasets found. Please download datasets first.")
        return 1
    
    processed_files = list(processed_dir.glob("*_processed.csv"))
    
    if not processed_files:
        print("No processed datasets found.")
        return 1
    
    print(f"\n📦 Found {len(processed_files)} processed datasets")
    
    # Load and combine datasets
    dfs = []
    for file in processed_files:
        df = pd.read_csv(file)
        dfs.append(df)
        print(f"   - {file.stem}: {len(df)} samples")
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Convert to ISOT format for compatibility
    isot_fake = combined[combined['label'] == 'fake'][['title', 'text']]
    isot_real = combined[combined['label'] == 'real'][['title', 'text']]
    
    # Save in ISOT format
    isot_dir = Path(config.data.root)
    isot_dir.mkdir(parents=True, exist_ok=True)
    
    isot_fake.to_csv(isot_dir / "Fake.csv", index=False)
    isot_real.to_csv(isot_dir / "True.csv", index=False)
    
    print(f"\n✅ Integrated datasets into main pipeline")
    print(f"   Fake samples: {len(isot_fake)}")
    print(f"   Real samples: {len(isot_real)}")
    print(f"   Saved to: {isot_dir}")
    
    # Run data preparation
    if args.prepare:
        print("\n🔄 Running data preparation...")
        from .data.loaders import prepare_splits
        prepare_splits(config, mode="full")
        print("✅ Data preparation completed")
    
    return 0


def build_parser():
    """Build argument parser for dataset CLI."""
    parser = argparse.ArgumentParser(
        prog="mmx_news.dataset",
        description="Dataset Pipeline Management for Fake News Detection"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List datasets command
    list_parser = subparsers.add_parser(
        "list-datasets",
        help="List all available datasets"
    )
    list_parser.add_argument(
        "--config",
        default="configs/datasets.yaml",
        help="Path to datasets configuration file"
    )
    list_parser.set_defaults(func=cmd_list_datasets)
    
    # Download single dataset
    download_parser = subparsers.add_parser(
        "download",
        help="Download a specific dataset"
    )
    download_parser.add_argument(
        "dataset",
        help="Dataset ID to download (e.g., isot, liar, covid19_fake)"
    )
    download_parser.add_argument(
        "--config",
        default="configs/datasets.yaml",
        help="Path to datasets configuration file"
    )
    download_parser.add_argument(
        "--kaggle-key",
        help="Kaggle API key (format: username:key)"
    )
    download_parser.add_argument(
        "--mendeley-key",
        help="Mendeley API key"
    )
    download_parser.add_argument(
        "--hf-token",
        help="Hugging Face API token"
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached"
    )
    download_parser.add_argument(
        "--output",
        help="Custom output path for processed dataset"
    )
    download_parser.set_defaults(func=cmd_download_dataset)
    
    # Download multiple datasets
    multi_parser = subparsers.add_parser(
        "download-multiple",
        help="Download multiple datasets"
    )
    multi_parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated list of dataset IDs or 'all'"
    )
    multi_parser.add_argument(
        "--config",
        default="configs/datasets.yaml",
        help="Path to datasets configuration file"
    )
    multi_parser.add_argument(
        "--kaggle-key",
        help="Kaggle API key (format: username:key)"
    )
    multi_parser.add_argument(
        "--mendeley-key",
        help="Mendeley API key"
    )
    multi_parser.add_argument(
        "--hf-token",
        help="Hugging Face API token"
    )
    multi_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached"
    )
    multi_parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine all datasets into one"
    )
    multi_parser.add_argument(
        "--balance",
        action="store_true",
        help="Balance fake/real samples when combining"
    )
    multi_parser.set_defaults(func=cmd_download_multiple)
    
    # Process custom dataset
    custom_parser = subparsers.add_parser(
        "process-custom",
        help="Process a custom dataset"
    )
    custom_parser.add_argument(
        "--url",
        help="URL to download dataset from"
    )
    custom_parser.add_argument(
        "--path",
        help="Local path to dataset"
    )
    custom_parser.add_argument(
        "--name",
        help="Name for the custom dataset"
    )
    custom_parser.add_argument(
        "--format",
        choices=["csv", "tsv", "json"],
        help="Dataset format"
    )
    custom_parser.add_argument(
        "--files",
        help="Comma-separated list of expected files"
    )
    custom_parser.add_argument(
        "--label-col",
        required=True,
        help="Name of the label column"
    )
    custom_parser.add_argument(
        "--text-cols",
        required=True,
        help="Comma-separated list of text columns"
    )
    custom_parser.add_argument(
        "--title-col",
        help="Name of the title column (optional)"
    )
    custom_parser.add_argument(
        "--output",
        help="Output path for processed dataset"
    )
    custom_parser.set_defaults(func=cmd_process_custom)
    
    # Integrate datasets into main pipeline
    integrate_parser = subparsers.add_parser(
        "integrate",
        help="Integrate downloaded datasets into main pipeline"
    )
    integrate_parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to main configuration file"
    )
    integrate_parser.add_argument(
        "--prepare",
        action="store_true",
        help="Run data preparation after integration"
    )
    integrate_parser.set_defaults(func=cmd_integrate)
    
    return parser


def main():
    """Main entry point for dataset CLI."""
    parser = build_parser()
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        return args.func(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())