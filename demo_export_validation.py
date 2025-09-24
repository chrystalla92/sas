#!/usr/bin/env python3
"""
Demonstration Script for Data Export and Validation
==================================================

This script demonstrates how to use the new Python data export and validation
utilities that replicate SAS PROC SURVEYSELECT functionality.

The script:
1. Loads the existing credit data from CSV
2. Performs stratified sampling to create train/validation splits
3. Exports datasets with SAS-compatible formatting
4. Runs comprehensive data quality validation
5. Generates validation reports

Author: Risk Analytics Team
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from data_export_utils import DataExportValidator, create_stratified_splits, export_with_validation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_credit_data(filepath: str = "output/credit_data_sample.csv") -> pd.DataFrame:
    """Load the existing credit data."""
    logger.info(f"Loading credit data from {filepath}")
    
    try:
        data = pd.read_csv(filepath)
        logger.info(f"Successfully loaded {len(data)} records with {len(data.columns)} columns")
        return data
    except FileNotFoundError:
        logger.error(f"Credit data file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading credit data: {str(e)}")
        raise


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data for validation (convert monetary strings to numeric)."""
    logger.info("Preprocessing data for validation")
    
    processed_data = data.copy()
    
    # Convert monetary columns from string format to numeric for validation
    monetary_columns = ['monthly_income', 'loan_amount', 'monthly_payment', 
                       'existing_monthly_debt', 'total_monthly_debt']
    
    for col in monetary_columns:
        if col in processed_data.columns:
            # Convert from "$X,XXX.XX" format to numeric
            processed_data[col] = pd.to_numeric(
                processed_data[col].astype(str).str.replace('$', '').str.replace(',', ''), 
                errors='coerce'
            )
    
    # Convert application_date to datetime if needed
    if 'application_date' in processed_data.columns:
        processed_data['application_date'] = pd.to_datetime(
            processed_data['application_date'], 
            format='%d%b%Y', 
            errors='coerce'
        )
    
    logger.info("Data preprocessing complete")
    return processed_data


def demonstrate_stratified_sampling():
    """Demonstrate stratified sampling functionality."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATION: Stratified Sampling")
    logger.info("=" * 60)
    
    # Load and preprocess data
    raw_data = load_credit_data()
    data = preprocess_data(raw_data)
    
    # Initialize validator
    validator = DataExportValidator(random_state=42, validation_tolerance=0.01)
    
    # Demonstrate different split ratios
    split_ratios = [0.3, 0.2]  # 70/30 and 80/20
    
    for test_size in split_ratios:
        logger.info(f"\n--- Testing {int((1-test_size)*100)}/{int(test_size*100)} split ---")
        
        train_data, validation_data = validator.stratified_train_test_split(
            data, 
            target_col='default_flag',
            test_size=test_size
        )
        
        # Calculate and display statistics
        original_default_rate = data['default_flag'].mean()
        train_default_rate = train_data['default_flag'].mean()
        validation_default_rate = validation_data['default_flag'].mean()
        
        logger.info(f"Original default rate: {original_default_rate:.4f}")
        logger.info(f"Train default rate: {train_default_rate:.4f}")
        logger.info(f"Validation default rate: {validation_default_rate:.4f}")
        logger.info(f"Train difference: {abs(train_default_rate - original_default_rate):.4f}")
        logger.info(f"Validation difference: {abs(validation_default_rate - original_default_rate):.4f}")
        
        # Check stratification results
        stratification = validator.validation_results.get('stratification', {})
        if stratification.get('within_tolerance', False):
            logger.info("✓ Stratification within tolerance")
        else:
            logger.warning("⚠ Stratification outside tolerance")


def demonstrate_csv_export():
    """Demonstrate CSV export with SAS formatting."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATION: CSV Export with SAS Formatting")
    logger.info("=" * 60)
    
    # Load and preprocess data
    raw_data = load_credit_data()
    data = preprocess_data(raw_data)
    
    # Create validator
    validator = DataExportValidator(random_state=42)
    
    # Create train/validation split
    train_data, validation_data = validator.stratified_train_test_split(data, test_size=0.3)
    
    # Create output directory
    output_dir = Path("output/python_exports")
    output_dir.mkdir(exist_ok=True)
    
    # Export datasets with SAS formatting
    train_file = output_dir / "credit_train_python.csv"
    validation_file = output_dir / "credit_validation_python.csv"
    
    logger.info(f"Exporting training data to {train_file}")
    validator.export_to_csv(train_data, train_file, sas_format=True)
    
    logger.info(f"Exporting validation data to {validation_file}")
    validator.export_to_csv(validation_data, validation_file, sas_format=True)
    
    # Verify exports
    train_exported = pd.read_csv(train_file)
    validation_exported = pd.read_csv(validation_file)
    
    logger.info(f"✓ Training export: {len(train_exported)} records")
    logger.info(f"✓ Validation export: {len(validation_exported)} records")
    
    # Show sample of formatted data
    logger.info("\nSample of SAS-formatted training data:")
    logger.info(train_exported.head(3).to_string(max_cols=6))


def demonstrate_data_quality_validation():
    """Demonstrate comprehensive data quality validation."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATION: Data Quality Validation")
    logger.info("=" * 60)
    
    # Load and preprocess data
    raw_data = load_credit_data()
    data = preprocess_data(raw_data)
    
    # Create validator
    validator = DataExportValidator(random_state=42)
    
    # Run comprehensive validation
    logger.info("Running comprehensive data quality validation...")
    validation_results = validator.validate_data_quality(data)
    
    # Display key results
    logger.info(f"\nData Quality Score: {validation_results['quality_score']:.2f}/100")
    
    # Dataset info
    dataset_info = validation_results['dataset_info']
    logger.info(f"Dataset: {dataset_info['rows']} rows, {dataset_info['columns']} columns")
    logger.info(f"Memory usage: {dataset_info['memory_usage_mb']:.2f} MB")
    
    # Range validation summary
    range_results = validation_results['range_validation']
    range_issues = sum(1 for r in range_results.values() if not r.get('valid', True))
    logger.info(f"Range validation: {range_issues} columns with out-of-range values")
    
    # Consistency checks summary  
    consistency_results = validation_results['consistency_checks']
    consistency_issues = sum(1 for r in consistency_results.values() if not r.get('valid', True))
    logger.info(f"Consistency checks: {consistency_issues} failed consistency checks")
    
    # Missing value summary
    missing_summary = validation_results['missing_value_analysis']['_summary']
    logger.info(f"Missing values: {missing_summary['overall_missing_percentage']:.2f}% overall")
    
    # Show detailed results for problematic columns
    logger.info("\nDetailed validation results:")
    
    # Range validation details
    logger.info("\n--- Range Validation Issues ---")
    for column, result in range_results.items():
        if not result.get('valid', True):
            logger.info(f"{column}: {result['out_of_range_count']} values out of range "
                       f"({result['out_of_range_percentage']:.1f}%)")
    
    # Consistency check details
    logger.info("\n--- Consistency Check Issues ---")
    for check_name, result in consistency_results.items():
        if not result.get('valid', True):
            logger.info(f"{check_name}: {result['inconsistent_count']} inconsistent values "
                       f"({result['inconsistent_percentage']:.1f}%)")
    
    return validation_results


def demonstrate_distribution_comparison():
    """Demonstrate statistical distribution comparison."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATION: Distribution Comparison")
    logger.info("=" * 60)
    
    # Load and preprocess data
    raw_data = load_credit_data()
    data = preprocess_data(raw_data)
    
    # Create validator and split data
    validator = DataExportValidator(random_state=42)
    train_data, validation_data = validator.stratified_train_test_split(data, test_size=0.3)
    
    # Compare distributions
    logger.info("Comparing statistical distributions between train and validation sets...")
    comparison_results = validator.compare_distributions(train_data, validation_data)
    
    # Display results for key columns
    key_columns = ['age', 'credit_score', 'monthly_income', 'debt_to_income_ratio']
    
    logger.info("\nDistribution Comparison Results:")
    logger.info(f"{'Column':<20} {'Train Mean':<12} {'Val Mean':<12} {'Difference':<12} {'Similar?':<8}")
    logger.info("-" * 70)
    
    for column in key_columns:
        if column in comparison_results:
            result = comparison_results[column]
            similar = "Yes" if result['distributions_similar'] else "No"
            logger.info(f"{column:<20} {result['train_mean']:<12.2f} "
                       f"{result['test_mean']:<12.2f} {result['mean_difference']:<12.2f} {similar:<8}")


def demonstrate_validation_report():
    """Demonstrate validation report generation."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATION: Validation Report Generation")
    logger.info("=" * 60)
    
    # Load and preprocess data
    raw_data = load_credit_data()
    data = preprocess_data(raw_data)
    
    # Create validator and run validation
    validator = DataExportValidator(random_state=42)
    validation_results = validator.validate_data_quality(data)
    
    # Create train/validation split and compare distributions
    train_data, validation_data = validator.stratified_train_test_split(data, test_size=0.3)
    comparison_results = validator.compare_distributions(train_data, validation_data)
    
    # Store comparison results
    validator.validation_results['distribution_comparison'] = comparison_results
    
    # Generate comprehensive report
    output_dir = Path("output/validation_reports")
    output_dir.mkdir(exist_ok=True)
    
    report_file = output_dir / "comprehensive_validation_report.json"
    
    logger.info(f"Generating validation report: {report_file}")
    validator.generate_validation_report(report_file)
    
    # Create a summary report in text format
    summary_file = output_dir / "validation_summary.txt"
    
    logger.info(f"Creating summary report: {summary_file}")
    with open(summary_file, 'w') as f:
        f.write("Credit Data Validation Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Dataset info
        dataset_info = validation_results['dataset_info']
        f.write(f"Dataset Size: {dataset_info['rows']} rows, {dataset_info['columns']} columns\n")
        f.write(f"Quality Score: {validation_results['quality_score']:.2f}/100\n\n")
        
        # Range validation summary
        range_results = validation_results['range_validation']
        range_issues = sum(1 for r in range_results.values() if not r.get('valid', True))
        f.write(f"Range Validation: {len(range_results) - range_issues}/{len(range_results)} columns valid\n")
        
        # Consistency summary
        consistency_results = validation_results['consistency_checks']
        consistency_issues = sum(1 for r in consistency_results.values() if not r.get('valid', True))
        f.write(f"Consistency Checks: {len(consistency_results) - consistency_issues}/{len(consistency_results)} checks passed\n")
        
        # Missing values
        missing_summary = validation_results['missing_value_analysis']['_summary']
        f.write(f"Missing Values: {missing_summary['overall_missing_percentage']:.2f}% overall\n\n")
        
        # Stratification summary
        if 'stratification' in validator.validation_results:
            strat_info = validator.validation_results['stratification']
            f.write("Stratified Sampling Results:\n")
            f.write(f"  Original default rate: {strat_info['original_default_rate']:.4f}\n")
            f.write(f"  Training default rate: {strat_info['train_default_rate']:.4f}\n")
            f.write(f"  Validation default rate: {strat_info['test_default_rate']:.4f}\n")
            f.write(f"  Within tolerance: {'Yes' if strat_info['within_tolerance'] else 'No'}\n")
    
    logger.info("✓ Validation reports generated successfully")


def main():
    """Main demonstration function."""
    logger.info("Starting Data Export and Validation Demonstration")
    logger.info("Python implementation of SAS PROC SURVEYSELECT functionality")
    
    try:
        # Run all demonstrations
        demonstrate_stratified_sampling()
        demonstrate_csv_export()
        validation_results = demonstrate_data_quality_validation()
        demonstrate_distribution_comparison()
        demonstrate_validation_report()
        
        logger.info("=" * 60)
        logger.info("DEMONSTRATION COMPLETE")
        logger.info("=" * 60)
        logger.info("All functionality has been successfully demonstrated.")
        logger.info("Check the output/ directory for exported files and reports.")
        
    except FileNotFoundError:
        logger.error("Could not find credit data file. Please ensure 'output/credit_data_sample.csv' exists.")
        logger.error("You may need to run the SAS data generation script first.")
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
