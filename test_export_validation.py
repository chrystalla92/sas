"""
Test Suite for Data Export and Validation Utilities
==================================================

Comprehensive tests for the data export and validation functionality that
replicates SAS PROC SURVEYSELECT stratified sampling using scikit-learn.

Test Coverage:
- Stratified sampling accuracy and distribution preservation
- CSV export format validation and SAS compatibility
- Data quality validation framework functionality
- Statistical distribution comparison between train/test sets
- Range validation and consistency checks
- Missing value analysis and reporting

Author: Risk Analytics Team
Date: 2025
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import json
from unittest.mock import patch, MagicMock

# Import the module to test
from data_export_utils import (
    DataExportValidator, 
    create_stratified_splits, 
    export_with_validation
)


class TestDataExportValidator:
    """Test class for DataExportValidator functionality."""
    
    @pytest.fixture
    def sample_credit_data(self):
        """Create sample credit data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'customer_id': [f'CUST{i:06d}' for i in range(1, n_samples + 1)],
            'age': np.random.randint(18, 75, n_samples),
            'employment_years': np.random.randint(0, 20, n_samples),
            'monthly_income': np.random.randint(2000, 15000, n_samples),
            'annual_income': [],
            'loan_amount': np.random.randint(5000, 200000, n_samples),
            'credit_score': np.random.randint(300, 850, n_samples),
            'credit_utilization': np.random.uniform(0, 100, n_samples),
            'debt_to_income_ratio': np.random.uniform(10, 80, n_samples),
            'num_dependents': np.random.randint(0, 5, n_samples),
            'credit_history_years': [],
            'num_credit_accounts': np.random.randint(1, 10, n_samples),
            'num_late_payments': np.random.randint(0, 10, n_samples),
            'previous_defaults': np.random.randint(0, 3, n_samples),
            'loan_term_months': np.random.choice([12, 24, 36, 48], n_samples),
            'interest_rate': np.random.uniform(0.03, 0.25, n_samples),
            'default_flag': np.random.binomial(1, 0.15, n_samples),  # 15% default rate
            'employment_status': np.random.choice(
                ['Full-time', 'Part-time', 'Self-employed', 'Retired'], n_samples
            ),
            'education': np.random.choice(
                ['High School', 'Bachelors', 'Masters', 'Doctorate'], n_samples
            ),
            'home_ownership': np.random.choice(['Rent', 'Own', 'Mortgage'], n_samples),
            'risk_rating': np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'], n_samples)
        }
        
        # Calculate derived fields
        for i in range(n_samples):
            data['annual_income'].append(data['monthly_income'][i] * 12)
            # Credit history should be reasonable relative to age
            max_credit_years = max(0, data['age'][i] - 18)
            data['credit_history_years'].append(
                np.random.randint(0, min(max_credit_years + 1, 30))
            )
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def validator(self):
        """Create a DataExportValidator instance for testing."""
        return DataExportValidator(random_state=42, validation_tolerance=0.01)
    
    def test_stratified_split_basic_functionality(self, validator, sample_credit_data):
        """Test basic stratified split functionality."""
        train, test = validator.stratified_train_test_split(
            sample_credit_data, 
            target_col='default_flag',
            test_size=0.3
        )
        
        # Check that we got the right sizes
        expected_train_size = int(len(sample_credit_data) * 0.7)
        expected_test_size = len(sample_credit_data) - expected_train_size
        
        assert len(train) == expected_train_size
        assert len(test) == expected_test_size
        
        # Check that all original columns are preserved
        assert set(train.columns) == set(sample_credit_data.columns)
        assert set(test.columns) == set(sample_credit_data.columns)
        
        # Check that the splits don't overlap
        train_ids = set(train['customer_id'])
        test_ids = set(test['customer_id'])
        assert len(train_ids.intersection(test_ids)) == 0
    
    def test_stratification_accuracy(self, validator, sample_credit_data):
        """Test that stratification maintains target distribution within tolerance."""
        original_default_rate = sample_credit_data['default_flag'].mean()
        
        train, test = validator.stratified_train_test_split(
            sample_credit_data, 
            target_col='default_flag',
            test_size=0.3
        )
        
        train_default_rate = train['default_flag'].mean()
        test_default_rate = test['default_flag'].mean()
        
        # Check that rates are within 1% tolerance
        assert abs(train_default_rate - original_default_rate) <= 0.01
        assert abs(test_default_rate - original_default_rate) <= 0.01
        
        # Check validation results
        assert 'stratification' in validator.validation_results
        assert validator.validation_results['stratification']['within_tolerance']
    
    def test_different_split_ratios(self, validator, sample_credit_data):
        """Test different split ratios (70/30, 80/20)."""
        # Test 70/30 split
        train_70, test_30 = validator.stratified_train_test_split(
            sample_credit_data, test_size=0.3
        )
        assert len(train_70) / len(sample_credit_data) == pytest.approx(0.7, abs=0.01)
        assert len(test_30) / len(sample_credit_data) == pytest.approx(0.3, abs=0.01)
        
        # Test 80/20 split
        train_80, test_20 = validator.stratified_train_test_split(
            sample_credit_data, test_size=0.2
        )
        assert len(train_80) / len(sample_credit_data) == pytest.approx(0.8, abs=0.01)
        assert len(test_20) / len(sample_credit_data) == pytest.approx(0.2, abs=0.01)
    
    def test_random_state_reproducibility(self, sample_credit_data):
        """Test that random state ensures reproducible splits."""
        validator1 = DataExportValidator(random_state=42)
        validator2 = DataExportValidator(random_state=42)
        validator3 = DataExportValidator(random_state=99)
        
        train1, test1 = validator1.stratified_train_test_split(sample_credit_data)
        train2, test2 = validator2.stratified_train_test_split(sample_credit_data)
        train3, test3 = validator3.stratified_train_test_split(sample_credit_data)
        
        # Same random state should produce identical results
        pd.testing.assert_frame_equal(train1.sort_values('customer_id').reset_index(drop=True), 
                                    train2.sort_values('customer_id').reset_index(drop=True))
        pd.testing.assert_frame_equal(test1.sort_values('customer_id').reset_index(drop=True), 
                                    test2.sort_values('customer_id').reset_index(drop=True))
        
        # Different random state should produce different results
        assert not train1.equals(train3)
    
    def test_csv_export_basic_functionality(self, validator, sample_credit_data):
        """Test basic CSV export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / 'test_export.csv'
            
            validator.export_to_csv(sample_credit_data, filepath)
            
            # Check that file was created
            assert filepath.exists()
            
            # Read back and verify content
            exported_data = pd.read_csv(filepath)
            assert len(exported_data) == len(sample_credit_data)
            assert set(exported_data.columns) == set(sample_credit_data.columns)
    
    def test_sas_formatting(self, validator, sample_credit_data):
        """Test SAS-compatible formatting in CSV export."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / 'test_sas_format.csv'
            
            validator.export_to_csv(sample_credit_data, filepath, sas_format=True)
            
            # Read the exported data
            exported_data = pd.read_csv(filepath)
            
            # Check that monetary columns are formatted with dollar signs
            if 'monthly_income' in exported_data.columns:
                # Should have dollar signs and commas
                sample_income = str(exported_data['monthly_income'].iloc[0])
                assert sample_income.startswith('$')
                assert ',' in sample_income or float(sample_income.replace('$', '')) < 1000
    
    def test_column_ordering(self, validator, sample_credit_data):
        """Test that CSV export maintains expected column ordering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / 'test_column_order.csv'
            
            validator.export_to_csv(sample_credit_data, filepath, sas_format=True)
            
            exported_data = pd.read_csv(filepath)
            columns = list(exported_data.columns)
            
            # Check that customer_id is first
            assert columns[0] == 'customer_id'
            
            # Check that key columns are in expected positions
            expected_early_columns = ['customer_id', 'employment_status', 'education']
            for i, expected_col in enumerate(expected_early_columns):
                if expected_col in columns:
                    assert columns.index(expected_col) == i
    
    def test_data_quality_validation_basic(self, validator, sample_credit_data):
        """Test basic data quality validation functionality."""
        results = validator.validate_data_quality(sample_credit_data)
        
        # Check that all expected sections are present
        expected_sections = [
            'timestamp', 'dataset_info', 'range_validation', 
            'consistency_checks', 'missing_value_analysis', 
            'statistical_summary', 'data_type_validation', 'quality_score'
        ]
        
        for section in expected_sections:
            assert section in results
        
        # Check dataset info
        assert results['dataset_info']['rows'] == len(sample_credit_data)
        assert results['dataset_info']['columns'] == len(sample_credit_data.columns)
        
        # Check quality score is reasonable
        assert 0 <= results['quality_score'] <= 100
    
    def test_range_validation(self, validator):
        """Test range validation functionality."""
        # Create data with some out-of-range values
        bad_data = pd.DataFrame({
            'age': [15, 25, 80, 35],  # 15 and 80 are out of range
            'credit_score': [250, 650, 900, 750],  # 250 and 900 are out of range
            'default_flag': [0, 1, 0, 2]  # 2 is out of range
        })
        
        results = validator.validate_data_quality(bad_data)
        range_results = results['range_validation']
        
        # Check that out-of-range values are detected
        assert range_results['age']['out_of_range_count'] == 2
        assert range_results['credit_score']['out_of_range_count'] == 2
        assert range_results['default_flag']['out_of_range_count'] == 1
        
        # Check validity flags
        assert not range_results['age']['valid']
        assert not range_results['credit_score']['valid']
        assert not range_results['default_flag']['valid']
    
    def test_consistency_validation(self, validator):
        """Test consistency validation between related variables."""
        # Create data with consistency issues
        inconsistent_data = pd.DataFrame({
            'age': [25, 30, 40, 50],
            'employment_years': [15, 5, 10, 20],  # First value is impossible (25 - 16 = 9 max)
            'credit_history_years': [10, 8, 15, 25],  # First value is impossible (25 - 18 = 7 max)
            'monthly_income': [3000, 4000, 5000, 6000],
            'annual_income': [36000, 40000, 60000, 72000],  # Third value is inconsistent
            'default_flag': [0, 1, 0, 1]
        })
        
        results = validator.validate_data_quality(inconsistent_data)
        consistency_results = results['consistency_checks']
        
        # Check that consistency issues are detected
        assert 'employment_age' in consistency_results
        assert 'credit_history_age' in consistency_results
        assert 'annual_monthly_income' in consistency_results
        
        # Check that invalid cases are detected
        assert consistency_results['employment_age']['inconsistent_count'] == 1
        assert consistency_results['credit_history_age']['inconsistent_count'] == 1
        assert consistency_results['annual_monthly_income']['inconsistent_count'] == 1
    
    def test_missing_value_analysis(self, validator):
        """Test missing value analysis functionality."""
        # Create data with missing values
        data_with_missing = pd.DataFrame({
            'col1': [1, 2, None, 4, 5],
            'col2': [1, None, None, 4, 5],
            'col3': [1, 2, 3, 4, 5]  # No missing values
        })
        
        results = validator.validate_data_quality(data_with_missing)
        missing_results = results['missing_value_analysis']
        
        # Check individual column results
        assert missing_results['col1']['missing_count'] == 1
        assert missing_results['col1']['missing_percentage'] == 20.0
        assert missing_results['col2']['missing_count'] == 2
        assert missing_results['col2']['missing_percentage'] == 40.0
        assert missing_results['col3']['missing_count'] == 0
        assert missing_results['col3']['missing_percentage'] == 0.0
        
        # Check summary
        assert missing_results['_summary']['total_missing_values'] == 3
        assert missing_results['_summary']['columns_with_missing'] == 2
    
    def test_statistical_summary(self, validator, sample_credit_data):
        """Test statistical summary generation."""
        results = validator.validate_data_quality(sample_credit_data)
        stats = results['statistical_summary']
        
        # Check that numeric columns have statistical summaries
        numeric_columns = sample_credit_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in stats:
                col_stats = stats[col]
                required_stats = ['count', 'mean', 'std', 'min', 'max', 'median']
                for stat in required_stats:
                    assert stat in col_stats
                    assert isinstance(col_stats[stat], (int, float))
        
        # Check categorical summary
        assert '_categorical' in stats
        categorical_stats = stats['_categorical']
        
        for col in ['employment_status', 'education', 'home_ownership']:
            if col in categorical_stats:
                assert 'unique_values' in categorical_stats[col]
                assert 'most_common' in categorical_stats[col]
    
    def test_distribution_comparison(self, validator, sample_credit_data):
        """Test statistical distribution comparison between datasets."""
        train, test = validator.stratified_train_test_split(sample_credit_data)
        
        comparison = validator.compare_distributions(train, test)
        
        # Check that comparison includes all numeric columns
        numeric_columns = sample_credit_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in comparison:
                col_comparison = comparison[col]
                
                # Check that required metrics are present
                required_metrics = [
                    'train_mean', 'test_mean', 'mean_difference',
                    'train_std', 'test_std', 'std_difference',
                    'distributions_similar'
                ]
                
                for metric in required_metrics:
                    assert metric in col_comparison
                
                # Check that distributions are similar (due to stratification)
                # Most columns should have similar distributions
                assert isinstance(col_comparison['distributions_similar'], bool)
    
    def test_validation_report_generation(self, validator, sample_credit_data):
        """Test validation report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / 'validation_report.json'
            
            # First run validation to populate results
            validator.validate_data_quality(sample_credit_data)
            
            # Generate report
            validator.generate_validation_report(report_path)
            
            # Check that report file was created
            assert report_path.exists()
            
            # Read and validate report content
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            
            # Check that report contains expected sections
            assert 'data_quality' in report_data
            assert 'timestamp' in report_data['data_quality']
            assert 'quality_score' in report_data['data_quality']
    
    def test_quality_score_calculation(self, validator):
        """Test data quality score calculation."""
        # Create perfect data
        perfect_data = pd.DataFrame({
            'age': [25, 30, 35, 40],
            'credit_score': [650, 700, 750, 800],
            'employment_years': [5, 10, 15, 20],
            'credit_history_years': [5, 8, 12, 18],
            'monthly_income': [3000, 4000, 5000, 6000],
            'annual_income': [36000, 48000, 60000, 72000],
            'default_flag': [0, 0, 1, 1]
        })
        
        results = validator.validate_data_quality(perfect_data)
        
        # Quality score should be high for perfect data
        assert results['quality_score'] >= 90
        
        # Create problematic data
        bad_data = pd.DataFrame({
            'age': [15, 85, None, 40],  # Out of range and missing
            'credit_score': [200, 900, 650, None],  # Out of range and missing
            'employment_years': [25, 30, 15, 20],  # Inconsistent with age
            'default_flag': [0, 2, 1, 0]  # Out of range
        })
        
        bad_results = validator.validate_data_quality(bad_data)
        
        # Quality score should be low for problematic data
        assert bad_results['quality_score'] <= 50


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create minimal sample data for testing."""
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'default_flag': [0, 0, 1, 0, 1, 0, 1, 1, 0, 1]
        })
    
    def test_create_stratified_splits_function(self, sample_data):
        """Test the create_stratified_splits convenience function."""
        train, test = create_stratified_splits(sample_data, test_size=0.3, random_state=42)
        
        assert len(train) == 7  # 70% of 10
        assert len(test) == 3   # 30% of 10
        
        # Check stratification
        original_rate = sample_data['default_flag'].mean()
        train_rate = train['default_flag'].mean()
        test_rate = test['default_flag'].mean()
        
        # Rates should be reasonably close (allowing for small sample size)
        assert abs(train_rate - original_rate) <= 0.2
        assert abs(test_rate - original_rate) <= 0.2
    
    def test_export_with_validation_function(self, sample_data):
        """Test the export_with_validation convenience function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / 'test_export.csv'
            
            # Test with validation
            validation_results = export_with_validation(
                sample_data, filepath, run_validation=True
            )
            
            # Check that file was created
            assert filepath.exists()
            
            # Check that validation results were returned
            assert validation_results is not None
            assert 'quality_score' in validation_results
            
            # Test without validation
            filepath2 = Path(temp_dir) / 'test_export2.csv'
            validation_results2 = export_with_validation(
                sample_data, filepath2, run_validation=False
            )
            
            assert filepath2.exists()
            assert validation_results2 is None


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_missing_target_column(self):
        """Test behavior when target column is missing."""
        validator = DataExportValidator()
        data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        
        with pytest.raises(ValueError, match="Target column 'default_flag' not found"):
            validator.stratified_train_test_split(data)
    
    def test_target_column_with_missing_values(self):
        """Test behavior when target column has missing values."""
        validator = DataExportValidator()
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'default_flag': [0, 1, None, 0, 1]
        })
        
        with pytest.warns(UserWarning, match="Missing values found in target column"):
            train, test = validator.stratified_train_test_split(data)
        
        # Should exclude rows with missing target values
        assert len(train) + len(test) == 4  # One row excluded
    
    def test_empty_dataframe(self):
        """Test behavior with empty dataframe."""
        validator = DataExportValidator()
        empty_data = pd.DataFrame()
        
        with pytest.raises(Exception):  # Should raise some kind of error
            validator.validate_data_quality(empty_data)
    
    def test_single_class_target(self):
        """Test behavior when target has only one class."""
        validator = DataExportValidator()
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'default_flag': [0, 0, 0, 0, 0]  # All zeros
        })
        
        # Should raise an error due to inability to stratify
        with pytest.raises(ValueError):
            validator.stratified_train_test_split(data)


if __name__ == '__main__':
    # Run the tests
    pytest.main([__file__, '-v'])
