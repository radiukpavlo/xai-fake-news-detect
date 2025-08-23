"""
Tests for Dataset Pipeline
===========================

Unit tests for the dataset pipeline functionality.
"""

import unittest
from pathlib import Path
import tempfile
import pandas as pd
import json
from unittest.mock import patch, MagicMock

from mmx_news.data.dataset_pipeline import (
    DatasetConfig,
    DatasetDownloader,
    DatasetProcessor,
    DatasetPipeline,
    DataSource
)


class TestDatasetConfig(unittest.TestCase):
    """Test DatasetConfig dataclass."""
    
    def test_create_config(self):
        """Test creating a dataset configuration."""
        config = DatasetConfig(
            name="test_dataset",
            source=DataSource.KAGGLE,
            identifier="test/dataset",
            expected_files=["test.csv"],
            format="csv",
            label_column="label",
            text_columns=["text"],
            title_column="title"
        )
        
        self.assertEqual(config.name, "test_dataset")
        self.assertEqual(config.source, DataSource.KAGGLE)
        self.assertEqual(config.identifier, "test/dataset")
        self.assertEqual(config.expected_files, ["test.csv"])
        self.assertEqual(config.format, "csv")
        self.assertEqual(config.label_column, "label")
        self.assertEqual(config.text_columns, ["text"])
        self.assertEqual(config.title_column, "title")


class TestDatasetDownloader(unittest.TestCase):
    """Test DatasetDownloader class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.downloader = DatasetDownloader(cache_dir=Path(self.temp_dir))
    
    def test_cache_directory_creation(self):
        """Test that cache directory is created."""
        self.assertTrue(Path(self.temp_dir).exists())
    
    @patch('mmx_news.data.dataset_pipeline.requests.get')
    def test_download_direct_url(self, mock_get):
        """Test downloading from direct URL."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'content-length': '1000'}
        mock_response.iter_content = lambda chunk_size: [b'test data']
        mock_get.return_value = mock_response
        
        config = DatasetConfig(
            name="test_url",
            source=DataSource.DIRECT_URL,
            identifier="https://example.com/data.csv",
            expected_files=["data.csv"],
            format="csv",
            label_column="label",
            text_columns=["text"]
        )
        
        result_dir = self.downloader.download(config)
        self.assertTrue(result_dir.exists())
    
    def test_setup_kaggle_credentials(self):
        """Test setting up Kaggle credentials."""
        with tempfile.TemporaryDirectory() as temp_home:
            kaggle_dir = Path(temp_home) / ".kaggle"
            with patch('pathlib.Path.home', return_value=Path(temp_home)):
                self.downloader._setup_kaggle_credentials("testuser:testkey")
                
                kaggle_json = kaggle_dir / "kaggle.json"
                self.assertTrue(kaggle_json.exists())
                
                with open(kaggle_json) as f:
                    creds = json.load(f)
                    self.assertEqual(creds["username"], "testuser")
                    self.assertEqual(creds["key"], "testkey")


class TestDatasetProcessor(unittest.TestCase):
    """Test DatasetProcessor class."""
    
    def setUp(self):
        """Set up test environment."""
        self.processor = DatasetProcessor()
        self.temp_dir = tempfile.mkdtemp()
    
    def test_standardize_label(self):
        """Test label standardization."""
        test_cases = [
            ("fake", "fake"),
            ("FAKE", "fake"),
            ("false", "fake"),
            ("0", "fake"),
            ("real", "real"),
            ("REAL", "real"),
            ("true", "real"),
            ("1", "real"),
            ("hoax", "fake"),
            ("genuine", "real"),
        ]
        
        for input_label, expected in test_cases:
            result = self.processor._standardize_label(input_label)
            self.assertEqual(result, expected, 
                           f"Failed for input '{input_label}'")
    
    def test_standardize_columns(self):
        """Test column standardization."""
        # Create test dataframe
        df = pd.DataFrame({
            'article_text': ['Text 1', 'Text 2'],
            'headline': ['Title 1', 'Title 2'],
            'is_fake': ['fake', 'real']
        })
        
        config = DatasetConfig(
            name="test",
            source=DataSource.DIRECT_URL,
            identifier="test.csv",
            expected_files=["test.csv"],
            format="csv",
            label_column="is_fake",
            text_columns=["article_text"],
            title_column="headline"
        )
        
        standardized = self.processor._standardize_columns(df, config)
        
        # Check columns exist
        self.assertIn('id', standardized.columns)
        self.assertIn('text', standardized.columns)
        self.assertIn('title', standardized.columns)
        self.assertIn('label', standardized.columns)
        self.assertIn('source', standardized.columns)
        
        # Check values
        self.assertEqual(standardized['text'].iloc[0], 'Text 1')
        self.assertEqual(standardized['title'].iloc[0], 'Title 1')
        self.assertEqual(standardized['label'].iloc[0], 'fake')
        self.assertEqual(standardized['source'].iloc[0], 'test')
    
    def test_apply_preprocessing(self):
        """Test preprocessing application."""
        df = pd.DataFrame({
            'text': ['Short', 'This is a longer text that meets minimum length', 
                    'Another long text sample for testing'],
            'label': ['fake', 'real', 'fake']
        })
        
        preprocessing = {
            'remove_duplicates': True,
            'min_text_length': 20,
            'remove_missing': True
        }
        
        result = self.processor._apply_preprocessing(df, preprocessing)
        
        # Only texts with length >= 20 should remain
        self.assertEqual(len(result), 2)
        self.assertTrue(all(result['text'].str.len() >= 20))


class TestDatasetPipeline(unittest.TestCase):
    """Test DatasetPipeline class."""
    
    def setUp(self):
        """Set up test environment."""
        self.pipeline = DatasetPipeline()
    
    def test_list_available_datasets(self):
        """Test listing available datasets."""
        datasets = self.pipeline.list_available_datasets()
        
        # Check that predefined datasets are available
        self.assertIn('isot', datasets)
        self.assertIn('liar', datasets)
        self.assertIsInstance(datasets, list)
    
    def test_add_custom_dataset(self):
        """Test adding custom dataset configuration."""
        custom_config = DatasetConfig(
            name="custom_test",
            source=DataSource.DIRECT_URL,
            identifier="https://example.com/test.csv",
            expected_files=["test.csv"],
            format="csv",
            label_column="label",
            text_columns=["text"]
        )
        
        self.pipeline.add_custom_dataset(custom_config)
        
        self.assertIn('custom_test', self.pipeline.configs)
        self.assertEqual(self.pipeline.configs['custom_test'], custom_config)
    
    def test_combine_datasets(self):
        """Test combining multiple datasets."""
        # Create test datasets
        df1 = pd.DataFrame({
            'text': ['Text 1', 'Text 2', 'Text 3', 'Text 4'],
            'label': ['fake', 'fake', 'real', 'real'],
            'source': ['dataset1'] * 4
        })
        
        df2 = pd.DataFrame({
            'text': ['Text 5', 'Text 6', 'Text 7', 'Text 8'],
            'label': ['fake', 'real', 'fake', 'real'],
            'source': ['dataset2'] * 4
        })
        
        datasets = {'dataset1': df1, 'dataset2': df2}
        
        # Test without balancing
        combined = self.pipeline.combine_datasets(datasets, balance=False)
        self.assertEqual(len(combined), 8)
        
        # Test with balancing
        combined_balanced = self.pipeline.combine_datasets(datasets, balance=True)
        fake_count = (combined_balanced['label'] == 'fake').sum()
        real_count = (combined_balanced['label'] == 'real').sum()
        self.assertEqual(fake_count, real_count)
    
    def test_get_statistics(self):
        """Test getting dataset statistics."""
        df = pd.DataFrame({
            'text': ['Hello world', 'This is a test', 'Another sample'],
            'label': ['fake', 'real', 'fake'],
            'source': ['test'] * 3,
            'title': ['Title 1', '', 'Title 3']
        })
        
        stats = self.pipeline.get_statistics(df)
        
        self.assertEqual(stats['total_samples'], 3)
        self.assertEqual(stats['label_distribution']['fake'], 2)
        self.assertEqual(stats['label_distribution']['real'], 1)
        self.assertEqual(stats['source_distribution']['test'], 3)
        self.assertEqual(stats['has_title'], 2)
        self.assertIn('avg_text_length', stats)
        self.assertIn('avg_word_count', stats)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_custom_dataset(self):
        """Test end-to-end processing of a custom dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test CSV file
            test_data = pd.DataFrame({
                'content': [
                    'This is a fake news article about something that never happened.',
                    'This is a real news article reporting actual events.',
                    'Another fake story designed to mislead readers.',
                    'A genuine report from a credible news source.'
                ],
                'headline': [
                    'Shocking Discovery!',
                    'Scientific Breakthrough',
                    'Unbelievable Claims',
                    'Research Update'
                ],
                'veracity': ['fake', 'real', 'fake', 'real']
            })
            
            test_file = Path(temp_dir) / "test_data.csv"
            test_data.to_csv(test_file, index=False)
            
            # Create pipeline with custom cache directory
            pipeline = DatasetPipeline()
            pipeline.downloader.cache_dir = Path(temp_dir) / "cache"
            pipeline.processor.processed_dir = Path(temp_dir) / "processed"
            
            # Add custom dataset
            config = DatasetConfig(
                name="integration_test",
                source=DataSource.DIRECT_URL,
                identifier=str(test_file),
                expected_files=["test_data.csv"],
                format="csv",
                label_column="veracity",
                text_columns=["content"],
                title_column="headline"
            )
            
            pipeline.add_custom_dataset(config)
            
            # Process dataset
            processed = pipeline.download_and_process("integration_test")
            
            # Verify results
            self.assertEqual(len(processed), 4)
            self.assertIn('text', processed.columns)
            self.assertIn('title', processed.columns)
            self.assertIn('label', processed.columns)
            self.assertIn('source', processed.columns)
            self.assertIn('id', processed.columns)
            
            # Check label standardization
            self.assertEqual(set(processed['label'].unique()), {'fake', 'real'})
            
            # Check statistics
            stats = pipeline.get_statistics(processed)
            self.assertEqual(stats['total_samples'], 4)
            self.assertEqual(stats['label_distribution']['fake'], 2)
            self.assertEqual(stats['label_distribution']['real'], 2)


if __name__ == '__main__':
    unittest.main()