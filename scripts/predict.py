"""
Prediction Script for Credit Risk Model

This script performs:
- Loading new customer applications without default_flag
- Replicating feature engineering pipeline from training
- Handling edge cases (unseen categories, out-of-range values)
- Generating predictions with risk scores, grades, recommendations, interest rates
- Saving predictions to CSV

Author: Risk Analytics Team
Date: 2025
"""

import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
import sys

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent))
from logging_config import setup_logging


def load_model_artifacts(models_dir, logger):
    """
    Load saved model artifacts (model, scaler, WOE mapping, selected features).
    
    Parameters
    ----------
    models_dir : Path
        Models directory path
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    tuple
        (model, scaler, woe_mapping, selected_features)
    """
    logger.info("Loading model artifacts")
    
    # Load logistic model
    model_path = models_dir / 'logistic_model.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    logger.info(f"Loaded model from: {model_path}")
    logger.info(f"Model expects {model.n_features_in_} features")
    
    # Load scaler
    scaler_path = models_dir / 'scaler.pkl'
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    logger.info(f"Loaded scaler from: {scaler_path}")
    logger.info(f"Scaler configured for {scaler.n_features_in_} features")
    
    # Load WOE mapping
    woe_path = models_dir / 'woe_mapping.pkl'
    if not woe_path.exists():
        raise FileNotFoundError(f"WOE mapping file not found: {woe_path}")
    with open(woe_path, 'rb') as f:
        woe_mapping = pickle.load(f)
    logger.info(f"Loaded WOE mapping from: {woe_path}")
    logger.info(f"WOE bins: {list(woe_mapping.keys())}")
    
    # Load selected features
    features_path = models_dir / 'selected_features.pkl'
    if not features_path.exists():
        raise FileNotFoundError(f"Selected features file not found: {features_path}")
    with open(features_path, 'rb') as f:
        selected_features = pickle.load(f)
    logger.info(f"Loaded selected features from: {features_path}")
    logger.info(f"Number of selected features: {len(selected_features)}")
    logger.info(f"Selected features: {selected_features}")
    
    return model, scaler, woe_mapping, selected_features


def load_new_applications(data_path, logger):
    """
    Load new customer applications CSV file.
    
    Parameters
    ----------
    data_path : Path
        Path to new applications CSV file
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        New applications dataframe
    """
    logger.info(f"Loading new applications from: {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"New applications file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Verify customer_id exists
    if 'customer_id' not in df.columns:
        raise ValueError("customer_id column not found in new_applications.csv")
    
    logger.info(f"Customer IDs found: {df['customer_id'].nunique()} unique IDs")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        logger.warning("Missing values detected in new applications:")
        for col, count in missing_counts[missing_counts > 0].items():
            logger.warning(f"  {col}: {count} missing values")
    
    return df


def create_ratio_features(df, logger):
    """
    Create ratio features for financial analysis.
    Replicates logic from feature_engineering.py
    
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
    
    # payment_to_income_ratio = monthly_payment / monthly_income
    df['payment_to_income_ratio'] = np.where(
        df['monthly_income'] > 0,
        df['monthly_payment'] / df['monthly_income'],
        0
    )
    
    # loan_to_income_ratio = loan_amount / annual_income
    df['loan_to_income_ratio'] = np.where(
        df['annual_income'] > 0,
        df['loan_amount'] / df['annual_income'],
        0
    )
    
    logger.info("Ratio features created: payment_to_income_ratio, loan_to_income_ratio")
    
    return df


def create_composite_scores(df, logger):
    """
    Create composite score features based on multiple variables.
    Replicates logic from feature_engineering.py
    
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
    
    # Handle unseen employment status values
    unseen_emp = df['emp_stability'].isna()
    if unseen_emp.any():
        logger.warning(f"Found {unseen_emp.sum()} unseen employment_status values, using default stability of 2")
        df.loc[unseen_emp, 'emp_stability'] = 2  # Default to Part-time level
    
    df['employment_score'] = df['emp_stability'] * df['employment_years']
    
    # Credit quality score
    df['credit_quality_score'] = (
        df['credit_score'] 
        - (df['num_late_payments'] * 50) 
        - (df['previous_defaults'] * 150)
    )
    
    # Affordability score
    df['affordability_score'] = np.where(
        df['monthly_payment'] > 0,
        (df['monthly_income'] - df['total_monthly_debt']) / df['monthly_payment'],
        0
    )
    
    # Drop intermediate emp_stability column
    df = df.drop('emp_stability', axis=1)
    
    logger.info("Composite scores created: employment_score, credit_quality_score, affordability_score")
    
    return df


def create_binary_flags(df, logger):
    """
    Create binary flag features for risk indicators.
    Replicates logic from feature_engineering.py
    
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
    
    # Flag: has delinquency
    df['has_delinquency'] = (df['num_late_payments'] > 0).astype(int)
    
    logger.info("Binary flags created: flag_high_dti, flag_low_credit, has_delinquency")
    
    return df


def apply_woe_transformation(df, woe_mapping, logger):
    """
    Apply WOE transformation to credit_score using saved mapping.
    Handles out-of-range credit scores.
    
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
    
    # Define bins matching training
    bins = [0, 580, 650, 700, 750, np.inf]
    labels = ['<580', '580-649', '650-699', '700-749', '750+']
    
    # Check for out-of-range values
    out_of_range = (df['credit_score'] < 580) | (df['credit_score'] >= 750)
    if out_of_range.any():
        count_low = (df['credit_score'] < 580).sum()
        count_high = (df['credit_score'] >= 750).sum()
        if count_low > 0:
            logger.warning(f"Found {count_low} credit_score values < 580 (boundary bin WOE will be applied)")
        if count_high > 0:
            logger.info(f"Found {count_high} credit_score values >= 750 (highest bin WOE will be applied)")
    
    # Create binned credit score
    df['credit_bin'] = pd.cut(df['credit_score'], bins=bins, labels=labels, right=False)
    
    # Map WOE values
    df['credit_score_woe'] = df['credit_bin'].map(woe_mapping)
    
    # Check for any missing WOE values (shouldn't happen with proper bins)
    if df['credit_score_woe'].isna().any():
        logger.error(f"Found {df['credit_score_woe'].isna().sum()} records with missing WOE values")
        # This shouldn't happen but handle it
        df['credit_score_woe'].fillna(0, inplace=True)
    
    # Remove temporary column
    df.drop('credit_bin', axis=1, inplace=True)
    
    logger.info("WOE transformation applied: credit_score_woe created")
    
    return df


def create_dummy_variables(df, logger):
    """
    Create dummy variables for categorical features.
    Handles unseen categorical values.
    Replicates logic from feature_engineering.py
    
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
    
    # Known categories from training
    known_employment = ['Full-time', 'Self-employed', 'Retired', 'Part-time', 'Unemployed']
    known_education = ['Bachelors', 'Masters', 'Doctorate', 'High School', 'Other']
    
    # Check for unseen employment status values
    unseen_emp = ~df['employment_status'].isin(known_employment)
    if unseen_emp.any():
        unseen_values = df.loc[unseen_emp, 'employment_status'].unique()
        logger.warning(f"Found unseen employment_status values: {unseen_values}")
        logger.warning(f"These {unseen_emp.sum()} records will have all employment dummies set to 0")
    
    # Check for unseen education values
    unseen_edu = ~df['education'].isin(known_education)
    if unseen_edu.any():
        unseen_values = df.loc[unseen_edu, 'education'].unique()
        logger.warning(f"Found unseen education values: {unseen_values}")
        logger.warning(f"These {unseen_edu.sum()} records will have all education dummies set to 0")
    
    # Employment status dummies (keep: fulltime, selfemployed, unemployed; drop: Part-time, Retired)
    df['emp_fulltime'] = (df['employment_status'] == 'Full-time').astype(int)
    df['emp_selfemployed'] = (df['employment_status'] == 'Self-employed').astype(int)
    df['emp_unemployed'] = (df['employment_status'] == 'Unemployed').astype(int)
    
    # Education dummies (keep: bachelors, masters, doctorate; drop: High School, Other)
    df['edu_bachelors'] = (df['education'] == 'Bachelors').astype(int)
    df['edu_masters'] = (df['education'] == 'Masters').astype(int)
    df['edu_doctorate'] = (df['education'] == 'Doctorate').astype(int)
    
    logger.info("Dummy variables created for employment_status and education")
    
    return df


def validate_and_clean_features(df, continuous_vars, logger):
    """
    Validate and clean features by checking for infinity and NaN values.
    Replicates logic from feature_engineering.py
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    continuous_vars : list
        List of continuous variable names to validate
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe
    """
    logger.info("Validating and cleaning features for inf/NaN values")
    
    for var in continuous_vars:
        if var in df.columns:
            # Check for infinity
            inf_count = np.isinf(df[var]).sum()
            if inf_count > 0:
                logger.warning(f"Found {inf_count} infinity values in {var}, replacing with finite values")
                df[var] = df[var].replace([np.inf], 999)
                df[var] = df[var].replace([-np.inf], -999)
            
            # Check for NaN
            nan_count = df[var].isna().sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values in {var}, replacing with 0")
                df[var] = df[var].fillna(0)
    
    logger.info("Feature validation and cleaning complete")
    
    return df


def standardize_features(df, continuous_vars, scaler, logger):
    """
    Standardize continuous features using saved StandardScaler.
    IMPORTANT: Only transform, do NOT refit the scaler.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    continuous_vars : list
        List of continuous variable names to standardize
    scaler : StandardScaler
        Fitted scaler from training
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized features
    """
    logger.info("Standardizing continuous features using saved scaler")
    logger.info(f"Features to standardize: {continuous_vars}")
    
    # Transform using saved scaler (do NOT fit)
    df[continuous_vars] = scaler.transform(df[continuous_vars])
    
    logger.info("Standardization complete using saved scaler")
    
    return df


def prepare_features_for_prediction(df, model, selected_features, logger):
    """
    Prepare final feature matrix for prediction.
    Use only the selected features from training.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with all features
    model : LogisticRegression
        Trained model
    selected_features : list
        List of selected feature names from training
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    tuple
        (X, feature_names) - feature matrix and feature names list
    """
    logger.info("Preparing features for prediction")
    logger.info(f"Using {len(selected_features)} selected features from training")
    
    # Check if all selected features exist in the dataframe
    missing_features = [col for col in selected_features if col not in df.columns]
    if missing_features:
        logger.error(f"Missing selected features: {missing_features}")
        raise ValueError(f"Missing selected features: {missing_features}")
    
    # Extract only the selected features in the same order as training
    X = df[selected_features].copy()
    
    # Verify feature count matches model expectations
    if X.shape[1] != model.n_features_in_:
        logger.error(f"Feature count mismatch: got {X.shape[1]}, model expects {model.n_features_in_}")
        raise ValueError(f"Feature count mismatch: got {X.shape[1]}, model expects {model.n_features_in_}")
    
    logger.info(f"Features prepared: {X.shape[1]} columns, {X.shape[0]} rows")
    logger.info(f"Feature count matches model expectation: {model.n_features_in_}")
    
    return X, selected_features


def generate_predictions(df, X, model, logger):
    """
    Generate predictions with risk scores, grades, recommendations, and interest rates.
    Replicates logic from train.py
    
    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe with customer_id
    X : pd.DataFrame
        Feature matrix for prediction
    model : LogisticRegression
        Trained model
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        DataFrame with predictions
    """
    logger.info("Generating predictions")
    
    # Get predictions (probability of default)
    probabilities = model.predict_proba(X)[:, 1]
    logger.info(f"Predictions generated for {len(probabilities)} records")
    
    # Calculate risk scores: round(600 + 250 * (1 - probability))
    risk_scores = np.round(600 + 250 * (1 - probabilities)).astype(int)
    
    # Ensure scores are within valid range [300, 850]
    risk_scores = np.clip(risk_scores, 300, 850)
    logger.info(f"Risk score range: {risk_scores.min()} to {risk_scores.max()}")
    
    # Assign risk grades
    risk_grades = pd.cut(
        risk_scores,
        bins=[0, 550, 600, 650, 700, 750, 1000],
        labels=['F', 'E', 'D', 'C', 'B', 'A'],
        right=False
    )
    
    # Assign recommendations based on grades
    recommendations = risk_grades.map({
        'A': 'Approve',
        'B': 'Approve',
        'C': 'Review',
        'D': 'Decline',
        'E': 'Decline',
        'F': 'Decline'
    })
    
    # Calculate interest rates
    base_rate = 0.05
    rate_premiums = {
        'A': 0.00,
        'B': 0.02,
        'C': 0.04,
        'D': 0.07,
        'E': 0.10,
        'F': 0.15
    }
    interest_rates = risk_grades.map(rate_premiums).astype(float) + base_rate
    
    # Create output DataFrame - include default_flag if present in input
    output_df = pd.DataFrame({
        'customer_id': df['customer_id']
    })
    
    # Add default_flag if it exists in the input dataframe
    if 'default_flag' in df.columns:
        output_df['default_flag'] = df['default_flag'].values
        logger.info("default_flag found in input and included in output")
    
    # Add prediction columns
    output_df['probability'] = probabilities
    output_df['risk_score'] = risk_scores
    output_df['risk_grade'] = risk_grades
    output_df['recommendation'] = recommendations
    output_df['interest_rate'] = interest_rates
    
    # Log distribution
    logger.info("\nRisk Grade Distribution:")
    for grade in ['A', 'B', 'C', 'D', 'E', 'F']:
        count = (output_df['risk_grade'] == grade).sum()
        pct = count / len(output_df) * 100 if len(output_df) > 0 else 0
        logger.info(f"  Grade {grade}: {count} ({pct:.1f}%)")
    
    logger.info("\nRecommendation Distribution:")
    for rec in ['Approve', 'Review', 'Decline']:
        count = (output_df['recommendation'] == rec).sum()
        pct = count / len(output_df) * 100 if len(output_df) > 0 else 0
        logger.info(f"  {rec}: {count} ({pct:.1f}%)")
    
    return output_df


def save_predictions(predictions_df, output_path, logger):
    """
    Save predictions to CSV file.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions dataframe
    output_path : Path
        Output file path
    logger : logging.Logger
        Logger instance
    """
    logger.info(f"Saving predictions to: {output_path}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(exist_ok=True)
    
    # Save to CSV
    predictions_df.to_csv(output_path, index=False)
    
    logger.info(f"Predictions saved: {len(predictions_df)} rows")
    logger.info(f"Output columns: {list(predictions_df.columns)}")


def main():
    """
    Main execution function for prediction pipeline.
    """
    import argparse
    
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Prediction Pipeline for Credit Risk Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  python predict.py
  
  # Specify custom input file
  python predict.py --input data/applications.csv
  
  # Specify custom output file
  python predict.py --output results/predictions.csv
  
  # Specify models directory
  python predict.py --models-dir /path/to/models
  
  # Specify all paths
  python predict.py --input my_apps.csv --output my_preds.csv --models-dir my_models
        """
    )
    
    # Define default paths
    project_root = Path(__file__).parent.parent
    
    parser.add_argument(
        '--input',
        type=str,
        default=str(project_root / 'data' / 'new_applications.csv'),
        help='Path to new applications CSV file (default: data/new_applications.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(project_root / 'output' / 'new_predictions.csv'),
        help='Path to output predictions CSV file (default: output/new_predictions.csv)'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default=str(project_root / 'models'),
        help='Directory containing model artifacts (default: models/)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging('credit_risk_model.log')
    logger.info("=" * 80)
    logger.info("Starting Prediction Pipeline")
    logger.info("=" * 80)
    
    try:
        # Convert paths to Path objects
        data_path = Path(args.input)
        models_dir = Path(args.models_dir)
        output_path = Path(args.output)
        
        # Log the paths being used
        logger.info(f"Input paths:")
        logger.info(f"  New applications: {data_path}")
        logger.info(f"  Models directory: {models_dir}")
        logger.info(f"  Output file: {output_path}")
        
        # 1. Load model artifacts
        model, scaler, woe_mapping, selected_features = load_model_artifacts(models_dir, logger)
        
        # 2. Load new applications
        df = load_new_applications(data_path, logger)
        
        # Store original customer IDs
        original_customer_ids = df['customer_id'].copy()
        original_row_count = len(df)
        
        # 3. Create ratio features
        df = create_ratio_features(df, logger)
        
        # 4. Create composite scores
        df = create_composite_scores(df, logger)
        
        # 5. Create binary flags
        df = create_binary_flags(df, logger)
        
        # 6. Apply WOE transformation
        df = apply_woe_transformation(df, woe_mapping, logger)
        
        # 7. Create dummy variables
        df = create_dummy_variables(df, logger)
        
        # 8. Define continuous variables to standardize (same as training)
        continuous_vars = [
            'age', 'employment_years', 'monthly_income', 'annual_income', 'loan_amount',
            'credit_utilization', 'debt_to_income_ratio', 'num_late_payments',
            'payment_to_income_ratio', 'loan_to_income_ratio',
            'employment_score', 'credit_quality_score', 'affordability_score'
        ]
        
        # 9. Validate and clean features
        df = validate_and_clean_features(df, continuous_vars, logger)
        
        # 10. Standardize features using saved scaler
        df = standardize_features(df, continuous_vars, scaler, logger)
        
        # 11. Prepare features for prediction
        X, feature_names = prepare_features_for_prediction(df, model, selected_features, logger)
        
        # 12. Generate predictions
        predictions_df = generate_predictions(df, X, model, logger)
        
        # 13. Verify output
        if len(predictions_df) != original_row_count:
            logger.warning(f"Output row count ({len(predictions_df)}) differs from input ({original_row_count})")
        
        # Validate predictions
        invalid_probs = (predictions_df['probability'] < 0) | (predictions_df['probability'] > 1)
        if invalid_probs.any():
            logger.error(f"Found {invalid_probs.sum()} invalid probability values")
        
        invalid_scores = (predictions_df['risk_score'] < 300) | (predictions_df['risk_score'] > 850)
        if invalid_scores.any():
            logger.error(f"Found {invalid_scores.sum()} invalid risk scores")
        
        # 14. Save predictions
        save_predictions(predictions_df, output_path, logger)
        
        logger.info("=" * 80)
        logger.info("Prediction Pipeline Completed Successfully")
        logger.info("=" * 80)
        logger.info(f"\nSummary:")
        logger.info(f"  Input records: {original_row_count}")
        logger.info(f"  Output records: {len(predictions_df)}")
        logger.info(f"  Probability range: {predictions_df['probability'].min():.4f} to {predictions_df['probability'].max():.4f}")
        logger.info(f"  Risk score range: {predictions_df['risk_score'].min()} to {predictions_df['risk_score'].max()}")
        logger.info(f"  Output file: {output_path}")
        
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {str(e)}")
        logger.error("Please ensure all model artifacts and input data are available")
        raise
    except ValueError as e:
        logger.error(f"Value error in prediction pipeline: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
