"""
Data Export and Validation Utilities
====================================

This module provides functionality to replicate SAS PROC SURVEYSELECT 
stratified sampling using scikit-learn, along with comprehensive data 
quality validation and CSV export capabilities.

Key Features:
- Stratified sampling by default status with configurable split ratios
- CSV export matching SAS format specifications
- Comprehensive data quality validation framework
- Statistical summary generation and comparison
- Validation report generation

Author: Risk Analytics Team
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report
from typing import Tuple, Dict, List, Optional, Union
import warnings
import logging
from pathlib import Path
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataExportValidator:
    """
    Main class for data export and validation operations.
    Replicates SAS PROC SURVEYSELECT functionality with enhanced validation.
    """
    
    def __init__(self, random_state: int = 42, validation_tolerance: float = 0.01):
        """
        Initialize the DataExportValidator.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random seed for reproducible splits
        validation_tolerance : float, default=0.01
            Tolerance for stratification validation (1% by default)
        """
        self.random_state = random_state
        self.validation_tolerance = validation_tolerance
        self.validation_results = {}
        
    def stratified_train_test_split(self, 
                                  data: pd.DataFrame, 
                                  target_col: str = 'default_flag',
                                  test_size: float = 0.3,
                                  train_size: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform stratified train/test split maintaining target distribution.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        target_col : str, default='default_flag'
            Column name for stratification
        test_size : float, default=0.3
            Proportion of dataset for test set (0.3 = 30%)
        train_size : float, optional
            Proportion of dataset for training set. If None, complement of test_size
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Training and validation datasets
        """
        logger.info(f"Performing stratified split with test_size={test_size}, random_state={self.random_state}")
        
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        # Check for missing values in target column
        if data[target_col].isna().any():
            warnings.warn(f"Missing values found in target column '{target_col}'. These will be excluded from split.")
            data_clean = data.dropna(subset=[target_col])
        else:
            data_clean = data.copy()
            
        # Perform stratified split
        try:
            X = data_clean.drop(columns=[target_col])
            y = data_clean[target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size,
                train_size=train_size,
                random_state=self.random_state,
                stratify=y
            )
            
            # Reconstruct full dataframes
            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)
            
            # Validate stratification
            self._validate_stratification(data_clean, train_data, test_data, target_col)
            
            logger.info(f"Split successful: Train={len(train_data)}, Test={len(test_data)}")
            return train_data, test_data
            
        except Exception as e:
            logger.error(f"Error in stratified split: {str(e)}")
            raise
    
    def _validate_stratification(self, original: pd.DataFrame, 
                               train: pd.DataFrame, 
                               test: pd.DataFrame, 
                               target_col: str) -> None:
        """
        Validate that stratification maintains target distribution within tolerance.
        """
        orig_rate = original[target_col].mean()
        train_rate = train[target_col].mean()
        test_rate = test[target_col].mean()
        
        train_diff = abs(train_rate - orig_rate)
        test_diff = abs(test_rate - orig_rate)
        
        self.validation_results['stratification'] = {
            'original_default_rate': orig_rate,
            'train_default_rate': train_rate,
            'test_default_rate': test_rate,
            'train_difference': train_diff,
            'test_difference': test_diff,
            'within_tolerance': train_diff <= self.validation_tolerance and test_diff <= self.validation_tolerance
        }
        
        if not self.validation_results['stratification']['within_tolerance']:
            warnings.warn(f"Stratification outside tolerance: Train diff={train_diff:.4f}, Test diff={test_diff:.4f}")
        else:
            logger.info(f"Stratification validated: Original={orig_rate:.4f}, Train={train_rate:.4f}, Test={test_rate:.4f}")
    
    def export_to_csv(self, data: pd.DataFrame, 
                     filepath: Union[str, Path], 
                     sas_format: bool = True) -> None:
        """
        Export data to CSV with SAS-compatible formatting.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to export
        filepath : str or Path
            Output file path
        sas_format : bool, default=True
            Whether to apply SAS-compatible formatting
        """
        logger.info(f"Exporting data to {filepath} (SAS format: {sas_format})")
        
        export_data = data.copy()
        
        if sas_format:
            export_data = self._apply_sas_formatting(export_data)
        
        # Ensure output directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Export to CSV
        export_data.to_csv(filepath, index=False)
        logger.info(f"Successfully exported {len(export_data)} records to {filepath}")
    
    def _apply_sas_formatting(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply SAS-compatible formatting to the dataset.
        """
        formatted_data = data.copy()
        
        # Format monetary columns with dollar signs
        monetary_columns = ['monthly_income', 'loan_amount', 'monthly_payment', 
                          'existing_monthly_debt', 'total_monthly_debt']
        
        for col in monetary_columns:
            if col in formatted_data.columns:
                formatted_data[col] = formatted_data[col].apply(
                    lambda x: f'${x:,.2f}' if pd.notnull(x) else ''
                )
        
        # Format application_date
        if 'application_date' in formatted_data.columns:
            formatted_data['application_date'] = pd.to_datetime(
                formatted_data['application_date']
            ).dt.strftime('%d%b%Y').str.upper()
        
        # Ensure proper column ordering (match SAS output)
        expected_columns = [
            'customer_id', 'employment_status', 'education', 'home_ownership',
            'application_date', 'monthly_income', 'loan_amount', 'age',
            'employment_years', 'annual_income', 'num_dependents',
            'credit_history_years', 'num_credit_accounts', 'num_late_payments',
            'credit_utilization', 'previous_defaults', 'loan_term_months',
            'loan_purpose', 'interest_rate', 'monthly_payment',
            'existing_monthly_debt', 'total_monthly_debt', 'debt_to_income_ratio',
            'credit_score', 'default_flag', 'risk_rating'
        ]
        
        # Reorder columns to match expected order
        available_columns = [col for col in expected_columns if col in formatted_data.columns]
        remaining_columns = [col for col in formatted_data.columns if col not in available_columns]
        
        formatted_data = formatted_data[available_columns + remaining_columns]
        
        return formatted_data
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict:
        """
        Comprehensive data quality validation framework.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dataset to validate
            
        Returns:
        --------
        Dict
            Validation results dictionary
        """
        logger.info("Starting comprehensive data quality validation")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'rows': len(data),
                'columns': len(data.columns),
                'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024**2)
            },
            'range_validation': self._validate_ranges(data),
            'consistency_checks': self._validate_consistency(data),
            'missing_value_analysis': self._analyze_missing_values(data),
            'statistical_summary': self._generate_statistical_summary(data),
            'data_type_validation': self._validate_data_types(data)
        }
        
        # Calculate overall data quality score
        validation_results['quality_score'] = self._calculate_quality_score(validation_results)
        
        self.validation_results['data_quality'] = validation_results
        logger.info(f"Data quality validation complete. Quality score: {validation_results['quality_score']:.2f}")
        
        return validation_results
    
    def _validate_ranges(self, data: pd.DataFrame) -> Dict:
        """
        Validate numeric variables are within expected ranges.
        """
        range_rules = {
            'age': (18, 75),
            'employment_years': (0, 50),
            'monthly_income': (0, 50000),
            'annual_income': (0, 600000),
            'loan_amount': (1000, 500000),
            'credit_score': (300, 850),
            'credit_utilization': (0, 100),
            'debt_to_income_ratio': (0, 300),
            'num_dependents': (0, 10),
            'credit_history_years': (0, 50),
            'num_credit_accounts': (0, 20),
            'num_late_payments': (0, 50),
            'previous_defaults': (0, 10),
            'loan_term_months': (6, 120),
            'interest_rate': (0, 1.0),
            'default_flag': (0, 1)
        }
        
        validation_results = {}
        
        for column, (min_val, max_val) in range_rules.items():
            if column in data.columns:
                # Convert formatted monetary strings to numeric for validation
                if column in ['monthly_income', 'loan_amount']:
                    series = pd.to_numeric(data[column].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
                else:
                    series = data[column].dropna()
                
                out_of_range = ((series < min_val) | (series > max_val)).sum()
                
                validation_results[column] = {
                    'min_allowed': min_val,
                    'max_allowed': max_val,
                    'actual_min': float(series.min()),
                    'actual_max': float(series.max()),
                    'out_of_range_count': int(out_of_range),
                    'out_of_range_percentage': float(out_of_range / len(series) * 100),
                    'valid': out_of_range == 0
                }
        
        return validation_results
    
    def _validate_consistency(self, data: pd.DataFrame) -> Dict:
        """
        Validate consistency between related variables.
        """
        consistency_checks = {}
        
        # Annual income should equal monthly income * 12
        if 'annual_income' in data.columns and 'monthly_income' in data.columns:
            # Convert monetary strings to numeric if needed
            monthly_income = pd.to_numeric(data['monthly_income'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
            annual_income = data['annual_income']
            
            expected_annual = monthly_income * 12
            inconsistent = (abs(annual_income - expected_annual) > 100).sum()
            
            consistency_checks['annual_monthly_income'] = {
                'description': 'Annual income should equal monthly income * 12',
                'inconsistent_count': int(inconsistent),
                'inconsistent_percentage': float(inconsistent / len(data) * 100),
                'valid': inconsistent == 0
            }
        
        # Employment years should not exceed (age - 16)
        if 'employment_years' in data.columns and 'age' in data.columns:
            max_possible_employment = data['age'] - 16
            invalid_employment = (data['employment_years'] > max_possible_employment).sum()
            
            consistency_checks['employment_age'] = {
                'description': 'Employment years should not exceed (age - 16)',
                'inconsistent_count': int(invalid_employment),
                'inconsistent_percentage': float(invalid_employment / len(data) * 100),
                'valid': invalid_employment == 0
            }
        
        # Credit history years should not exceed (age - 18)
        if 'credit_history_years' in data.columns and 'age' in data.columns:
            max_possible_credit_history = data['age'] - 18
            invalid_credit_history = (data['credit_history_years'] > max_possible_credit_history).sum()
            
            consistency_checks['credit_history_age'] = {
                'description': 'Credit history years should not exceed (age - 18)',
                'inconsistent_count': int(invalid_credit_history),
                'inconsistent_percentage': float(invalid_credit_history / len(data) * 100),
                'valid': invalid_credit_history == 0
            }
        
        return consistency_checks
    
    def _analyze_missing_values(self, data: pd.DataFrame) -> Dict:
        """
        Analyze missing value patterns in the dataset.
        """
        missing_analysis = {}
        
        for column in data.columns:
            missing_count = data[column].isna().sum()
            missing_percentage = (missing_count / len(data)) * 100
            
            missing_analysis[column] = {
                'missing_count': int(missing_count),
                'missing_percentage': float(missing_percentage),
                'has_missing': missing_count > 0
            }
        
        # Overall missing value statistics
        total_missing = sum(info['missing_count'] for info in missing_analysis.values())
        total_cells = len(data) * len(data.columns)
        
        missing_analysis['_summary'] = {
            'total_missing_values': int(total_missing),
            'total_cells': int(total_cells),
            'overall_missing_percentage': float((total_missing / total_cells) * 100),
            'columns_with_missing': int(sum(1 for info in missing_analysis.values() if isinstance(info, dict) and info.get('has_missing', False)))
        }
        
        return missing_analysis
    
    def _generate_statistical_summary(self, data: pd.DataFrame) -> Dict:
        """
        Generate comprehensive statistical summary.
        """
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        statistical_summary = {}
        
        for column in numeric_columns:
            series = data[column].dropna()
            
            if len(series) > 0:
                statistical_summary[column] = {
                    'count': int(len(series)),
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'q25': float(series.quantile(0.25)),
                    'median': float(series.median()),
                    'q75': float(series.quantile(0.75)),
                    'max': float(series.max()),
                    'skewness': float(series.skew()),
                    'kurtosis': float(series.kurtosis())
                }
        
        # Categorical variable summaries
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        
        categorical_summary = {}
        for column in categorical_columns:
            if column not in ['customer_id']:  # Skip ID columns
                value_counts = data[column].value_counts()
                categorical_summary[column] = {
                    'unique_values': int(data[column].nunique()),
                    'most_common': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'most_common_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'value_distribution': {str(k): int(v) for k, v in value_counts.head(10).items()}
                }
        
        statistical_summary['_categorical'] = categorical_summary
        
        return statistical_summary
    
    def _validate_data_types(self, data: pd.DataFrame) -> Dict:
        """
        Validate data types match expectations.
        """
        expected_types = {
            'customer_id': 'object',
            'age': 'int64',
            'employment_years': 'int64', 
            'annual_income': 'int64',
            'credit_score': 'int64',
            'default_flag': 'int64',
            'num_dependents': 'int64',
            'credit_history_years': 'int64',
            'num_credit_accounts': 'int64',
            'num_late_payments': 'int64',
            'previous_defaults': 'int64',
            'loan_term_months': 'int64'
        }
        
        type_validation = {}
        
        for column, expected_type in expected_types.items():
            if column in data.columns:
                actual_type = str(data[column].dtype)
                type_validation[column] = {
                    'expected_type': expected_type,
                    'actual_type': actual_type,
                    'valid': actual_type == expected_type or (expected_type == 'object' and actual_type in ['object', 'string'])
                }
        
        return type_validation
    
    def _calculate_quality_score(self, validation_results: Dict) -> float:
        """
        Calculate overall data quality score (0-100).
        """
        scores = []
        
        # Range validation score
        range_results = validation_results.get('range_validation', {})
        if range_results:
            valid_ranges = sum(1 for result in range_results.values() if result.get('valid', False))
            range_score = (valid_ranges / len(range_results)) * 100
            scores.append(range_score)
        
        # Consistency check score
        consistency_results = validation_results.get('consistency_checks', {})
        if consistency_results:
            valid_consistency = sum(1 for result in consistency_results.values() if result.get('valid', False))
            consistency_score = (valid_consistency / len(consistency_results)) * 100
            scores.append(consistency_score)
        
        # Missing value score (penalize high missing percentages)
        missing_results = validation_results.get('missing_value_analysis', {})
        if missing_results and '_summary' in missing_results:
            overall_missing_pct = missing_results['_summary']['overall_missing_percentage']
            missing_score = max(0, 100 - overall_missing_pct * 2)  # 2 points penalty per 1% missing
            scores.append(missing_score)
        
        # Data type validation score
        type_results = validation_results.get('data_type_validation', {})
        if type_results:
            valid_types = sum(1 for result in type_results.values() if result.get('valid', False))
            type_score = (valid_types / len(type_results)) * 100
            scores.append(type_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def compare_distributions(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
        """
        Compare statistical distributions between train and test sets.
        """
        logger.info("Comparing distributions between train and test sets")
        
        comparison_results = {}
        
        # Compare numeric variables
        numeric_columns = train_data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in test_data.columns:
                train_series = train_data[column].dropna()
                test_series = test_data[column].dropna()
                
                comparison_results[column] = {
                    'train_mean': float(train_series.mean()),
                    'test_mean': float(test_series.mean()),
                    'mean_difference': float(abs(train_series.mean() - test_series.mean())),
                    'train_std': float(train_series.std()),
                    'test_std': float(test_series.std()),
                    'std_difference': float(abs(train_series.std() - test_series.std())),
                    'distributions_similar': abs(train_series.mean() - test_series.mean()) / train_series.mean() < 0.05
                }
        
        return comparison_results
    
    def generate_validation_report(self, filepath: Union[str, Path]) -> None:
        """
        Generate comprehensive validation report.
        """
        if not self.validation_results:
            logger.warning("No validation results available. Run validation first.")
            return
        
        logger.info(f"Generating validation report: {filepath}")
        
        # Ensure output directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save validation results as JSON
        with open(filepath, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {filepath}")


# Convenience functions for common operations
def create_stratified_splits(data: pd.DataFrame, 
                           test_size: float = 0.3,
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function for creating stratified train/test splits.
    """
    validator = DataExportValidator(random_state=random_state)
    return validator.stratified_train_test_split(data, test_size=test_size)


def export_with_validation(data: pd.DataFrame,
                         filepath: Union[str, Path],
                         run_validation: bool = True) -> Optional[Dict]:
    """
    Convenience function for exporting data with optional validation.
    """
    validator = DataExportValidator()
    
    # Export data
    validator.export_to_csv(data, filepath)
    
    # Run validation if requested
    if run_validation:
        return validator.validate_data_quality(data)
    
    return None
