"""
Feature Engineering Script for Credit Risk Model

This script performs comprehensive feature engineering including:
- Ratio feature creation
- Composite score generation
- WOE (Weight of Evidence) transformation
- Information Value calculation
- Categorical encoding
- Feature standardization

Author: Risk Analytics Team
Date: 2025
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import sys

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent))
from logging_config import setup_logging


def load_data(train_path, validation_path, logger):
    """
    Load training and validation datasets from CSV files.
    
    Parameters
    ----------
    train_path : str or Path
        Path to training CSV file
    validation_path : str or Path
        Path to validation CSV file
    logger : logging.Logger
        Logger instance for logging
        
    Returns
    -------
    tuple
        (train_df, validation_df) DataFrames
    """
    logger.info(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    logger.info(f"Training data loaded: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    
    logger.info(f"Loading validation data from: {validation_path}")
    val_df = pd.read_csv(validation_path)
    logger.info(f"Validation data loaded: {val_df.shape[0]} rows, {val_df.shape[1]} columns")
    
    # Check for missing values
    train_missing = train_df.isnull().sum().sum()
    val_missing = val_df.isnull().sum().sum()
    
    if train_missing > 0:
        logger.warning(f"Training data contains {train_missing} missing values")
    if val_missing > 0:
        logger.warning(f"Validation data contains {val_missing} missing values")
    
    return train_df, val_df


def create_ratio_features(df, logger):
    """
    Create ratio features for financial analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added ratio features
    """
    logger.info("Creating ratio features")
    
    # Note: debt_to_income_ratio already exists in input data
    # But we'll keep it for consistency
    
    # payment_to_income_ratio = monthly_payment / monthly_income
    df['payment_to_income_ratio'] = df['monthly_payment'] / df['monthly_income']
    
    # loan_to_income_ratio = loan_amount / annual_income
    df['loan_to_income_ratio'] = df['loan_amount'] / df['annual_income']
    
    logger.info("Ratio features created: payment_to_income_ratio, loan_to_income_ratio")
    
    return df


def create_composite_scores(df, logger):
    """
    Create composite score features based on multiple variables.
    
    Formulas:
    - employment_score = emp_stability * employment_years
      where emp_stability: Full-time=5, Self-employed=4, Retired=3, Part-time=2, Unemployed=1
    - credit_quality_score = credit_score - (num_late_payments * 50) - (previous_defaults * 150)
    - affordability_score = (monthly_income - total_monthly_debt) / monthly_payment
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added composite scores
    """
    logger.info("Creating composite score features")
    
    # Employment score based on employment status and years
    emp_stability_map = {
        'Full-time': 5,
        'Self-employed': 4,
        'Retired': 3,
        'Part-time': 2,
        'Unemployed': 1
    }
    df['emp_stability'] = df['employment_status'].map(emp_stability_map)
    df['employment_score'] = df['emp_stability'] * df['employment_years']
    
    # Credit quality score
    df['credit_quality_score'] = (
        df['credit_score'] 
        - (df['num_late_payments'] * 50) 
        - (df['previous_defaults'] * 150)
    )
    
    # Affordability score
    df['affordability_score'] = (
        (df['monthly_income'] - df['total_monthly_debt']) / df['monthly_payment']
    )
    
    # Drop intermediate emp_stability column
    df = df.drop('emp_stability', axis=1)
    
    logger.info("Composite scores created: employment_score, credit_quality_score, affordability_score")
    
    return df


def create_binary_flags(df, logger):
    """
    Create binary flag features for risk indicators.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added binary flags
    """
    logger.info("Creating binary flag features")
    
    # Flag for high debt-to-income ratio (> 43%)
    df['flag_high_dti'] = (df['debt_to_income_ratio'] > 43).astype(int)
    
    # Flag for low credit score (< 650)
    df['flag_low_credit'] = (df['credit_score'] < 650).astype(int)
    
    # Additional flag: has delinquency (for IV calculation)
    df['has_delinquency'] = (df['num_late_payments'] > 0).astype(int)
    
    logger.info("Binary flags created: flag_high_dti, flag_low_credit, has_delinquency")
    
    return df


def calculate_woe_mapping(df, target_col='default_flag', logger=None):
    """
    Calculate Weight of Evidence (WOE) for credit_score bins on training data.
    
    Bins: <580, 580-649, 650-699, 700-749, 750+
    
    WOE Formula:
    - Good rate = count(default_flag=0 in bin) / total_count(default_flag=0)
    - Bad rate = count(default_flag=1 in bin) / total_count(default_flag=1)
    - WOE = log(good_rate / bad_rate)
    
    Parameters
    ----------
    df : pd.DataFrame
        Training dataframe
    target_col : str
        Name of target variable column
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Dictionary mapping bin labels to WOE values
    """
    if logger:
        logger.info("Calculating WOE mapping for credit_score bins")
    
    # Define bins
    bins = [0, 580, 650, 700, 750, np.inf]
    labels = ['<580', '580-649', '650-699', '700-749', '750+']
    
    # Create binned credit score
    df['credit_bin'] = pd.cut(df['credit_score'], bins=bins, labels=labels, right=False)
    
    # Calculate total goods and bads
    total_good = (df[target_col] == 0).sum()
    total_bad = (df[target_col] == 1).sum()
    
    if logger:
        logger.info(f"Total good: {total_good}, Total bad: {total_bad}")
    
    # Calculate WOE for each bin
    woe_mapping = {}
    
    for label in labels:
        bin_data = df[df['credit_bin'] == label]
        
        good_count = (bin_data[target_col] == 0).sum()
        bad_count = (bin_data[target_col] == 1).sum()
        
        # Calculate rates with small constant to avoid division by zero
        good_rate = (good_count + 0.5) / (total_good + 1)
        bad_rate = (bad_count + 0.5) / (total_bad + 1)
        
        # Calculate WOE
        woe = np.log(good_rate / bad_rate)
        woe_mapping[label] = woe
        
        if logger:
            logger.info(f"Bin {label}: Good={good_count}, Bad={bad_count}, WOE={woe:.4f}")
    
    # Remove temporary column
    df.drop('credit_bin', axis=1, inplace=True)
    
    return woe_mapping


def apply_woe_transformation(df, woe_mapping, logger):
    """
    Apply WOE transformation to credit_score using pre-calculated mapping.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    woe_mapping : dict
        Dictionary mapping bin labels to WOE values
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added credit_score_woe column
    """
    logger.info("Applying WOE transformation to credit_score")
    
    # Define bins matching the WOE calculation
    bins = [0, 580, 650, 700, 750, np.inf]
    labels = ['<580', '580-649', '650-699', '700-749', '750+']
    
    # Create binned credit score
    df['credit_bin'] = pd.cut(df['credit_score'], bins=bins, labels=labels, right=False)
    
    # Map WOE values
    df['credit_score_woe'] = df['credit_bin'].map(woe_mapping)
    
    # Remove temporary column
    df.drop('credit_bin', axis=1, inplace=True)
    
    logger.info("WOE transformation applied: credit_score_woe created")
    
    return df


def calculate_information_value(df, feature_col, target_col='default_flag', logger=None):
    """
    Calculate Information Value (IV) for a binary feature.
    
    IV Formula:
    IV = sum((good_rate - bad_rate) * WOE) for each category
    
    Interpretation:
    - IV < 0.02: Not useful for prediction
    - 0.02 <= IV < 0.1: Weak predictive power
    - 0.1 <= IV < 0.3: Medium predictive power
    - 0.3 <= IV < 0.5: Strong predictive power
    - IV >= 0.5: Suspicious (too good to be true)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_col : str
        Name of feature column
    target_col : str
        Name of target variable column
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    float
        Information Value
    """
    # Calculate total goods and bads
    total_good = (df[target_col] == 0).sum()
    total_bad = (df[target_col] == 1).sum()
    
    # Group by feature values
    grouped = df.groupby(feature_col)[target_col].agg(['count', 'sum'])
    grouped.columns = ['total', 'bad_count']
    grouped['good_count'] = grouped['total'] - grouped['bad_count']
    
    # Calculate rates with small constant to avoid division by zero
    grouped['good_rate'] = (grouped['good_count'] + 0.5) / (total_good + 1)
    grouped['bad_rate'] = (grouped['bad_count'] + 0.5) / (total_bad + 1)
    
    # Calculate WOE
    grouped['woe'] = np.log(grouped['good_rate'] / grouped['bad_rate'])
    
    # Calculate IV
    grouped['iv_component'] = (grouped['good_rate'] - grouped['bad_rate']) * grouped['woe']
    iv = grouped['iv_component'].sum()
    
    if logger:
        logger.info(f"Information Value for {feature_col}: {iv:.4f}")
    
    return iv


def create_dummy_variables(df, logger):
    """
    Create dummy variables for categorical features.
    
    Creates dummy variables for:
    - employment_status
    - education
    
    Uses training set to determine categories.
    Drops reference categories to avoid multicollinearity.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        DataFrame with dummy variables added
    """
    logger.info("Creating dummy variables for categorical features")
    
    # Employment status dummies (drop one for reference category)
    emp_dummies = pd.get_dummies(df['employment_status'], prefix='emp', dtype=int)
    # Keep all but drop the first alphabetically or explicitly drop 'Part-time' to match SAS
    # Based on SAS, they keep: emp_fulltime, emp_selfemployed, emp_unemployed
    # Drop: Part-time, Retired
    df['emp_fulltime'] = (df['employment_status'] == 'Full-time').astype(int)
    df['emp_selfemployed'] = (df['employment_status'] == 'Self-employed').astype(int)
    df['emp_unemployed'] = (df['employment_status'] == 'Unemployed').astype(int)
    
    # Education dummies (drop reference category)
    # Based on SAS, they keep: edu_bachelors, edu_masters, edu_doctorate
    # Drop: High School, Other
    df['edu_bachelors'] = (df['education'] == 'Bachelors').astype(int)
    df['edu_masters'] = (df['education'] == 'Masters').astype(int)
    df['edu_doctorate'] = (df['education'] == 'Doctorate').astype(int)
    
    logger.info("Dummy variables created for employment_status and education")
    
    return df


def standardize_features(train_df, val_df, continuous_vars, logger):
    """
    Standardize continuous features using StandardScaler.
    
    Fit scaler on training set ONLY, then transform both train and validation sets.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe
    val_df : pd.DataFrame
        Validation dataframe
    continuous_vars : list
        List of continuous variable names to standardize
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    tuple
        (train_df, val_df, scaler) - transformed dataframes and fitted scaler
    """
    logger.info("Standardizing continuous features")
    logger.info(f"Features to standardize: {continuous_vars}")
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit on training data only
    scaler.fit(train_df[continuous_vars])
    
    # Transform both datasets
    train_df[continuous_vars] = scaler.transform(train_df[continuous_vars])
    val_df[continuous_vars] = scaler.transform(val_df[continuous_vars])
    
    logger.info(f"Standardization complete. Scaler fitted on {len(continuous_vars)} features")
    logger.info(f"Training set - Mean: ~0, Std: ~1 (expected after standardization)")
    
    return train_df, val_df, scaler


def save_outputs(train_df, val_df, woe_mapping, scaler, output_dir, logger):
    """
    Save all outputs: CSVs and pickle files.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Processed training dataframe
    val_df : pd.DataFrame
        Processed validation dataframe
    woe_mapping : dict
        WOE mapping dictionary
    scaler : StandardScaler
        Fitted scaler object
    output_dir : Path
        Output directory path
    logger : logging.Logger
        Logger instance
    """
    logger.info("Saving output files")
    
    # Ensure output directories exist
    data_dir = output_dir / 'data'
    models_dir = output_dir / 'models'
    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    
    # Save CSVs
    train_output_path = data_dir / 'model_features_train.csv'
    val_output_path = data_dir / 'model_features_validation.csv'
    
    train_df.to_csv(train_output_path, index=False)
    logger.info(f"Training features saved to: {train_output_path} ({train_df.shape[0]} rows, {train_df.shape[1]} columns)")
    
    val_df.to_csv(val_output_path, index=False)
    logger.info(f"Validation features saved to: {val_output_path} ({val_df.shape[0]} rows, {val_df.shape[1]} columns)")
    
    # Save WOE mapping
    woe_path = models_dir / 'woe_mapping.pkl'
    with open(woe_path, 'wb') as f:
        pickle.dump(woe_mapping, f)
    logger.info(f"WOE mapping saved to: {woe_path}")
    
    # Save scaler
    scaler_path = models_dir / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to: {scaler_path}")
    logger.info(f"Scaler n_features_in_: {scaler.n_features_in_}")


def main():
    """
    Main execution function for feature engineering pipeline.
    """
    # Setup logging
    logger = setup_logging('credit_risk_model.log')
    logger.info("="*80)
    logger.info("Starting Feature Engineering Pipeline")
    logger.info("="*80)
    
    try:
        # Define paths
        project_root = Path(__file__).parent.parent
        train_path = project_root / 'credit_train.csv'
        val_path = project_root / 'credit_validation.csv'
        
        # 1. Load data
        train_df, val_df = load_data(train_path, val_path, logger)
        
        # 2. Create ratio features
        train_df = create_ratio_features(train_df, logger)
        val_df = create_ratio_features(val_df, logger)
        
        # 3. Create composite scores
        train_df = create_composite_scores(train_df, logger)
        val_df = create_composite_scores(val_df, logger)
        
        # 4. Create binary flags
        train_df = create_binary_flags(train_df, logger)
        val_df = create_binary_flags(val_df, logger)
        
        # 5. Calculate WOE mapping on training data
        woe_mapping = calculate_woe_mapping(train_df, target_col='default_flag', logger=logger)
        
        # 6. Apply WOE transformation to both datasets
        train_df = apply_woe_transformation(train_df, woe_mapping, logger)
        val_df = apply_woe_transformation(val_df, woe_mapping, logger)
        
        # 7. Calculate Information Value for binary features
        logger.info("Calculating Information Values for binary features")
        iv_high_dti = calculate_information_value(train_df, 'flag_high_dti', logger=logger)
        iv_low_credit = calculate_information_value(train_df, 'flag_low_credit', logger=logger)
        iv_delinquency = calculate_information_value(train_df, 'has_delinquency', logger=logger)
        
        # 8. Create dummy variables
        train_df = create_dummy_variables(train_df, logger)
        val_df = create_dummy_variables(val_df, logger)
        
        # 9. Standardize continuous features
        # Define the 13 continuous variables to standardize
        continuous_vars = [
            'age', 'employment_years', 'monthly_income', 'annual_income', 'loan_amount',
            'credit_utilization', 'debt_to_income_ratio', 'num_late_payments',
            'payment_to_income_ratio', 'loan_to_income_ratio',
            'employment_score', 'credit_quality_score', 'affordability_score'
        ]
        
        train_df, val_df, scaler = standardize_features(train_df, val_df, continuous_vars, logger)
        
        # 10. Select and order final columns
        # Based on specs and SAS file, keep these columns in consistent order
        final_columns = [
            'customer_id', 'default_flag',
            # Standardized continuous features
            'age', 'employment_years', 'monthly_income', 'annual_income', 'loan_amount',
            'credit_utilization', 'debt_to_income_ratio', 'num_late_payments',
            'payment_to_income_ratio', 'loan_to_income_ratio',
            'employment_score', 'credit_quality_score', 'affordability_score',
            # Original continuous features (not standardized)
            'num_credit_accounts', 'credit_history_years', 'previous_defaults',
            'loan_term_months', 'credit_score',
            # WOE feature
            'credit_score_woe',
            # Binary flags
            'flag_high_dti', 'flag_low_credit', 'has_delinquency',
            # Dummy variables
            'emp_fulltime', 'emp_selfemployed', 'emp_unemployed',
            'edu_bachelors', 'edu_masters', 'edu_doctorate'
        ]
        
        train_df = train_df[final_columns]
        val_df = val_df[final_columns]
        
        logger.info(f"Final feature columns: {len(final_columns)}")
        
        # 11. Save outputs
        save_outputs(train_df, val_df, woe_mapping, scaler, project_root, logger)
        
        logger.info("="*80)
        logger.info("Feature Engineering Pipeline Completed Successfully")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
