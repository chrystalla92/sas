#!/usr/bin/env python3
"""
Bank Credit Risk Scoring Model - Step 3: Feature Engineering

Purpose: Create derived features and transform variables for modeling
Author: Risk Analytics Team
Date: 2025

This script performs feature engineering including:
- WOE transformations for categorical variables
- Creating risk indicators and ratios
- Binning continuous variables
- Handling categorical variables with dummy encoding
- Feature scaling and transformation
- Train/validation split with stratification

MIGRATION FROM SAS:
This Python implementation replicates the exact logic from 03_feature_engineering.sas,
maintaining the same transformations, business rules, and feature engineering steps.

USAGE:
    python 03_feature_engineering.py

INPUT:
    - output/credit_data_sample.csv (from Script 1)

OUTPUT:
    - output/model_features_train.csv
    - output/model_features_validation.csv
    - In-memory DataFrames for downstream processing

DEPENDENCIES:
    - pandas>=1.5.0
    - numpy>=1.24.0
    - scikit-learn>=1.3.0
    - category_encoders>=2.5.0
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
try:
    from category_encoders import WOEEncoder
    HAS_CATEGORY_ENCODERS = True
except ImportError:
    print("Warning: category_encoders not available, using manual WOE implementation")
    HAS_CATEGORY_ENCODERS = False
    WOEEncoder = None
import os
import sys
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    Load credit data sample and prepare for feature engineering.
    
    Returns:
        pd.DataFrame: Cleaned and prepared data
    """
    print("Loading credit data sample...")
    
    # Load the data
    data_path = Path('output/credit_data_sample.csv')
    if not data_path.exists():
        raise FileNotFoundError(f"Credit data file not found at {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} records with {df.shape[1]} columns")
    
    # Clean monetary fields (remove $ and commas, convert to float)
    monetary_cols = ['monthly_income', 'loan_amount', 'monthly_payment', 'existing_monthly_debt', 'total_monthly_debt']
    for col in monetary_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
    
    # Convert application_date to datetime
    try:
        df['application_date'] = pd.to_datetime(df['application_date'], format='%d%b%Y')
    except ValueError:
        # Try alternative formats
        try:
            df['application_date'] = pd.to_datetime(df['application_date'])
        except:
            print("Warning: Could not parse application_date, using original format")
            pass
    
    print("Data loaded and cleaned successfully")
    return df

def create_derived_features(df):
    """
    Create derived features matching SAS implementation.
    
    Args:
        df (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Data with derived features
    """
    print("Creating derived features...")
    
    df = df.copy()
    
    # Financial ratios  
    df['payment_to_income_ratio'] = (df['monthly_payment'] / df['monthly_income'].replace(0, 1)) * 100
    df['loan_to_income_ratio'] = df['loan_amount'] / df['annual_income'].replace(0, 1)
    df['debt_service_coverage'] = df['monthly_income'] / df['total_monthly_debt'].replace(0, 1)  # Avoid division by zero
    
    # Employment stability score
    emp_stability_map = {
        'Full-time': 5,
        'Self-employed': 4,
        'Retired': 3,
        'Part-time': 2,
        'Unemployed': 1
    }
    df['emp_stability'] = df['employment_status'].map(emp_stability_map)
    
    # Weighted employment score
    df['employment_score'] = df['emp_stability'] * df['employment_years']
    
    # Credit history quality score
    df['credit_quality_score'] = (df['credit_score'] - 
                                 (df['num_late_payments'] * 50) - 
                                 (df['previous_defaults'] * 150))
    
    # Age groups for risk assessment
    df['age_group'] = pd.cut(df['age'], 
                           bins=[0, 25, 35, 45, 55, 65, 100], 
                           labels=[1, 2, 3, 4, 5, 6])
    df['age_group'] = df['age_group'].astype(int)
    
    # Income stability indicator  
    df['income_stability'] = (df['employment_years'] / df['age'].replace(0, 1)) * 100
    
    # Credit behavior score
    df['credit_util_score'] = pd.cut(df['credit_utilization'], 
                                   bins=[0, 30, 50, 70, 90, 100], 
                                   labels=[5, 4, 3, 2, 1])
    df['credit_util_score'] = df['credit_util_score'].astype(int)
    
    # Delinquency indicator
    df['has_delinquency'] = (df['num_late_payments'] > 0).astype(int)
    
    # High-risk flags
    df['flag_high_dti'] = (df['debt_to_income_ratio'] > 43).astype(int)
    df['flag_low_credit'] = (df['credit_score'] < 620).astype(int)
    df['flag_high_util'] = (df['credit_utilization'] > 75).astype(int)
    df['flag_recent_default'] = (df['previous_defaults'] > 0).astype(int)
    df['flag_unstable_employment'] = (df['employment_years'] < 2).astype(int)
    
    # Total risk flags
    risk_flags = ['flag_high_dti', 'flag_low_credit', 'flag_high_util', 
                  'flag_recent_default', 'flag_unstable_employment']
    df['total_risk_flags'] = df[risk_flags].sum(axis=1)
    
    # Loan affordability score
    df['affordability_score'] = ((df['monthly_income'] - df['total_monthly_debt']) / 
                                df['monthly_payment'].replace(0, 1))  # Avoid division by zero
    
    # Credit age to loan ratio
    df['credit_to_loan_years'] = df['credit_history_years'] / ((df['loan_term_months'] / 12).replace(0, 1))
    
    # Handle any infinite or NaN values created by division operations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    print("Derived features created successfully")
    return df

def create_woe_transformations(df_train, df_val=None):
    """
    Create Weight of Evidence transformations for categorical variables.
    
    Args:
        df_train (pd.DataFrame): Training data
        df_val (pd.DataFrame): Validation data (optional)
        
    Returns:
        tuple: (transformed_train, transformed_val, woe_encoders)
    """
    print("Creating WOE transformations...")
    
    df_train = df_train.copy()
    woe_encoders = {}
    
    # Create credit score bands for WOE transformation
    def create_credit_bands(score):
        if score < 580:
            return 'A. <580'
        elif score < 650:
            return 'B. 580-649'
        elif score < 700:
            return 'C. 650-699'
        elif score < 750:
            return 'D. 700-749'
        else:
            return 'E. 750+'
    
    df_train['credit_band'] = df_train['credit_score'].apply(create_credit_bands)
    
    # Calculate WOE for credit score bands manually (as in SAS)
    woe_credit_dict = {}
    total_bad = df_train['default_flag'].sum()
    total_good = len(df_train) - total_bad
    
    for band in df_train['credit_band'].unique():
        band_data = df_train[df_train['credit_band'] == band]
        bad_count = band_data['default_flag'].sum()
        good_count = len(band_data) - bad_count
        
        bad_rate = bad_count / total_bad if total_bad > 0 else 0.001
        good_rate = good_count / total_good if total_good > 0 else 0.001
        
        # Avoid division by zero and log(0)
        if bad_rate == 0:
            bad_rate = 0.001
        if good_rate == 0:
            good_rate = 0.001
            
        woe = np.log(good_rate / bad_rate)
        woe_credit_dict[band] = woe
    
    df_train['woe_credit_score'] = df_train['credit_band'].map(woe_credit_dict)
    
    # Apply WOE encoding to other categorical variables
    categorical_vars = ['home_ownership', 'employment_status', 'loan_purpose']
    
    for var in categorical_vars:
        if var in df_train.columns:
            if HAS_CATEGORY_ENCODERS:
                encoder = WOEEncoder()
                df_train[f'woe_{var}'] = encoder.fit_transform(df_train[var], df_train['default_flag'])
                woe_encoders[var] = encoder
            else:
                # Manual WOE calculation fallback
                woe_dict = {}
                total_bad = df_train['default_flag'].sum()
                total_good = len(df_train) - total_bad
                
                for category in df_train[var].unique():
                    cat_data = df_train[df_train[var] == category]
                    bad_count = cat_data['default_flag'].sum()
                    good_count = len(cat_data) - bad_count
                    
                    bad_rate = max(bad_count / total_bad, 0.001) if total_bad > 0 else 0.001
                    good_rate = max(good_count / total_good, 0.001) if total_good > 0 else 0.001
                    
                    woe = np.log(good_rate / bad_rate)
                    woe_dict[category] = woe
                
                df_train[f'woe_{var}'] = df_train[var].map(woe_dict)
                woe_encoders[var] = woe_dict
    
    # Transform validation data if provided
    transformed_val = None
    if df_val is not None:
        df_val = df_val.copy()
        df_val['credit_band'] = df_val['credit_score'].apply(create_credit_bands)
        df_val['woe_credit_score'] = df_val['credit_band'].map(woe_credit_dict)
        
        for var in categorical_vars:
            if var in df_val.columns:
                if HAS_CATEGORY_ENCODERS and hasattr(woe_encoders[var], 'transform'):
                    df_val[f'woe_{var}'] = woe_encoders[var].transform(df_val[var])
                else:
                    # Use manual WOE dictionary
                    df_val[f'woe_{var}'] = df_val[var].map(woe_encoders[var]).fillna(0)
        
        transformed_val = df_val
    
    print("WOE transformations completed successfully")
    return df_train, transformed_val, woe_encoders

def create_dummy_variables(df):
    """
    Create dummy variables for categorical features.
    
    Args:
        df (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Data with dummy variables
    """
    print("Creating dummy variables...")
    
    df = df.copy()
    
    # Employment status dummies
    df['emp_fulltime'] = (df['employment_status'] == 'Full-time').astype(int)
    df['emp_selfemployed'] = (df['employment_status'] == 'Self-employed').astype(int)
    df['emp_parttime'] = (df['employment_status'] == 'Part-time').astype(int)
    df['emp_retired'] = (df['employment_status'] == 'Retired').astype(int)
    df['emp_unemployed'] = (df['employment_status'] == 'Unemployed').astype(int)
    
    # Education dummies
    df['edu_highschool'] = (df['education'] == 'High School').astype(int)
    df['edu_bachelors'] = (df['education'] == 'Bachelors').astype(int)
    df['edu_masters'] = (df['education'] == 'Masters').astype(int)
    df['edu_doctorate'] = (df['education'] == 'Doctorate').astype(int)
    
    # Home ownership dummies
    df['home_rent'] = (df['home_ownership'] == 'Rent').astype(int)
    df['home_mortgage'] = (df['home_ownership'] == 'Mortgage').astype(int)
    df['home_own'] = (df['home_ownership'] == 'Own').astype(int)
    
    # Loan purpose dummies  
    df['purpose_debt'] = (df['loan_purpose'] == 'Debt Consolidation').astype(int)
    df['purpose_home'] = (df['loan_purpose'] == 'Home Improvement').astype(int)
    df['purpose_auto'] = (df['loan_purpose'] == 'Auto').astype(int)
    df['purpose_personal'] = (df['loan_purpose'] == 'Personal').astype(int)
    df['purpose_medical'] = (df['loan_purpose'] == 'Medical').astype(int)
    
    print("Dummy variables created successfully")
    return df

def create_interaction_terms(df):
    """
    Create interaction terms for feature engineering.
    
    Args:
        df (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Data with interaction terms
    """
    print("Creating interaction terms...")
    
    df = df.copy()
    
    # Income-credit interaction
    df['income_credit_interaction'] = df['monthly_income'] * df['credit_score'] / 1000
    
    # Age-employment interaction  
    df['age_employment_interaction'] = df['age'] * df['employment_years']
    
    # Credit utilization-debt ratio interaction
    df['util_debt_interaction'] = df['credit_utilization'] * df['debt_to_income_ratio'] / 100
    
    print("Interaction terms created successfully")
    return df

def create_binned_features(df):
    """
    Create binned features for continuous variables.
    
    Args:
        df (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Data with binned features
    """
    print("Creating binned features...")
    
    df = df.copy()
    
    # Income bins
    df['income_bins'] = pd.qcut(df['monthly_income'], 
                               q=5, 
                               labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Credit score bins (already done in WOE, but creating numeric version)
    df['credit_score_bins'] = pd.cut(df['credit_score'],
                                   bins=[0, 580, 650, 700, 750, 850],
                                   labels=[1, 2, 3, 4, 5])
    df['credit_score_bins'] = df['credit_score_bins'].astype(int)
    
    # Age bins
    df['age_bins'] = pd.cut(df['age'],
                           bins=[0, 30, 40, 50, 60, 100],
                           labels=[1, 2, 3, 4, 5])
    df['age_bins'] = df['age_bins'].astype(int)
    
    print("Binned features created successfully")
    return df

def apply_feature_scaling(df_train, df_val, features_to_scale):
    """
    Apply standardization to numeric features.
    
    Args:
        df_train (pd.DataFrame): Training data
        df_val (pd.DataFrame): Validation data
        features_to_scale (list): Features to standardize
        
    Returns:
        tuple: (scaled_train, scaled_val, scaler)
    """
    print("Applying feature scaling...")
    
    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Scale training data
    features_present = [f for f in features_to_scale if f in df_train.columns]
    df_train_scaled[features_present] = scaler.fit_transform(df_train[features_present])
    
    # Scale validation data using training parameters
    df_val_scaled[features_present] = scaler.transform(df_val[features_present])
    
    print(f"Scaled {len(features_present)} features successfully")
    return df_train_scaled, df_val_scaled, scaler

def select_final_features(df):
    """
    Select final features for modeling.
    
    Args:
        df (pd.DataFrame): Input data with all features
        
    Returns:
        pd.DataFrame: Data with selected features
    """
    print("Selecting final features...")
    
    # Define final feature set matching SAS implementation
    final_features = [
        'customer_id', 'default_flag',
        
        # Original features
        'age', 'employment_years', 'monthly_income', 'credit_score',
        'credit_utilization', 'debt_to_income_ratio', 'num_late_payments',
        'num_credit_accounts', 'credit_history_years', 'previous_defaults',
        'loan_amount', 'loan_term_months',
        
        # Engineered features
        'payment_to_income_ratio', 'loan_to_income_ratio', 'employment_score',
        'credit_quality_score', 'affordability_score', 'total_risk_flags',
        'woe_credit_score',
        
        # WOE transformations
        'woe_home_ownership', 'woe_employment_status', 'woe_loan_purpose',
        
        # Interaction terms
        'income_credit_interaction', 'age_employment_interaction',
        
        # Binned features
        'credit_score_bins', 'age_bins',
        
        # Binary flags
        'flag_high_dti', 'flag_low_credit', 'flag_high_util',
        'flag_recent_default', 'flag_unstable_employment', 'has_delinquency',
        
        # Categorical dummies
        'emp_fulltime', 'emp_selfemployed', 'emp_unemployed',
        'edu_bachelors', 'edu_masters', 'edu_doctorate',
        'home_rent', 'home_mortgage',
        'purpose_debt', 'purpose_auto', 'purpose_personal'
    ]
    
    # Keep only features that exist in the dataframe
    available_features = [f for f in final_features if f in df.columns]
    df_final = df[available_features].copy()
    
    print(f"Selected {len(available_features)} features for modeling")
    return df_final

def main():
    """
    Main feature engineering pipeline.
    """
    print("=" * 60)
    print("BANK CREDIT RISK MODEL - FEATURE ENGINEERING")
    print("=" * 60)
    
    try:
        # Step 1: Load and prepare data
        print("Step 1: Loading and preparing data...")
        df = load_and_prepare_data()
        print(f"✓ Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Step 2: Create derived features
        print("\nStep 2: Creating derived features...")
        df = create_derived_features(df)
        print("✓ Derived features created")
        
        df = create_interaction_terms(df)
        print("✓ Interaction terms created")
        
        df = create_binned_features(df)
        print("✓ Binned features created")
        
        df = create_dummy_variables(df)
        print("✓ Dummy variables created")
        
        print(f"Dataset now has {len(df.columns)} columns")
        
        # Step 3: Train/validation split with stratification
        print("\nStep 3: Performing train/validation split (70/30)...")
        
        X = df.drop(['default_flag'], axis=1)
        y = df['default_flag']
        
        print(f"Target variable distribution: {y.value_counts().to_dict()}")
        print(f"Default rate: {y.mean():.3f}")
        
        # Check if we have enough samples in each class for stratification
        if y.sum() < 2 or (len(y) - y.sum()) < 2:
            print("Warning: Not enough samples in each class for stratification, using random split")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=0.3,
                random_state=42,
                stratify=None
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=0.3,
                random_state=42,
                stratify=y
            )
        
        # Recombine with target variable
        df_train = pd.concat([X_train, y_train], axis=1)
        df_val = pd.concat([X_val, y_val], axis=1)
        
        print(f"Training set: {len(df_train):,} records ({y_train.mean():.3f} default rate)")
        print(f"Validation set: {len(df_val):,} records ({y_val.mean():.3f} default rate)")
        
        # Step 4: Apply WOE transformations
        print("\nStep 4: Applying WOE transformations...")
        df_train_woe, df_val_woe, woe_encoders = create_woe_transformations(df_train, df_val)
        print("✓ WOE transformations completed")
        
        # Step 5: Feature scaling
        print("\nStep 5: Applying feature scaling...")
        features_to_scale = [
            'age', 'employment_years', 'monthly_income', 'annual_income', 'loan_amount',
            'credit_utilization', 'debt_to_income_ratio', 'num_late_payments',
            'payment_to_income_ratio', 'loan_to_income_ratio', 'employment_score',
            'credit_quality_score', 'affordability_score'
        ]
        
        df_train_final, df_val_final, scaler = apply_feature_scaling(
            df_train_woe, df_val_woe, features_to_scale
        )
        print("✓ Feature scaling completed")
        
        # Step 6: Select final features
        print("\nStep 6: Selecting final features...")
        model_features_train = select_final_features(df_train_final)
        model_features_validation = select_final_features(df_val_final)
        print("✓ Feature selection completed")
        
        # Step 7: Export to CSV files
        print("\nStep 7: Exporting feature-engineered datasets...")
        
        # Create output directory if it doesn't exist
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        # Export training and validation sets
        train_path = output_dir / 'model_features_train.csv'
        val_path = output_dir / 'model_features_validation.csv'
        
        model_features_train.to_csv(train_path, index=False)
        model_features_validation.to_csv(val_path, index=False)
        
        print(f"✓ Training features exported to: {train_path}")
        print(f"✓ Validation features exported to: {val_path}")
        
        # Step 8: Validation checks
        print("\n" + "="*50)
        print("FEATURE ENGINEERING VALIDATION SUMMARY")
        print("="*50)
        
        print(f"Training set shape: {model_features_train.shape}")
        print(f"Validation set shape: {model_features_validation.shape}")
        print(f"Total features: {model_features_train.shape[1] - 2}")  # Excluding customer_id and target
        
        print(f"\nTarget distribution:")
        print(f"Training default rate: {model_features_train['default_flag'].mean():.3f}")
        print(f"Validation default rate: {model_features_validation['default_flag'].mean():.3f}")
        
        print(f"\nSample of engineered features:")
        feature_cols = [col for col in model_features_train.columns 
                       if col not in ['customer_id', 'default_flag']][:10]
        print(model_features_train[feature_cols].describe())
        
        print("\n✓ Feature engineering completed successfully!")
        print("✓ Datasets ready for model training")
        
        return model_features_train, model_features_validation
        
    except Exception as e:
        import traceback
        print(f"\n❌ Error during feature engineering: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    model_features_train, model_features_validation = main()
