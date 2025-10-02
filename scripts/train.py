"""
Model Training Script for Credit Risk Model

This script performs:
- Manual stepwise logistic regression selection (forward/backward)
- P-value calculation using Wald test
- Model training and serialization
- Risk score generation (300-850 scale)
- Risk grade assignment (A-F)
- Recommendation generation (Approve/Review/Decline)
- Interest rate calculation

Author: Risk Analytics Team
Date: 2025
"""

import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from scipy import stats
import sys

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent))
from logging_config import setup_logging
import feature_engineering


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
    
    return train_df, val_df


def separate_features_target(df, target_col='default_flag', exclude_cols=None, logger=None):
    """
    Separate features (X) from target (y).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column
    exclude_cols : list
        Additional columns to exclude from features
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    tuple
        (X, y, feature_names) - features DataFrame, target Series, feature names list
    """
    if exclude_cols is None:
        exclude_cols = ['customer_id']
    
    # Columns to exclude: target + exclude_cols
    cols_to_exclude = [target_col] + exclude_cols
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in cols_to_exclude]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    if logger:
        logger.info(f"Features: {len(feature_cols)} columns")
        logger.info(f"Target: {target_col}")
        logger.info(f"Feature columns: {feature_cols}")
    
    return X, y, feature_cols


def calculate_p_values(X, y, feature_subset, logger=None):
    """
    Calculate p-values for logistic regression coefficients using Wald test.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    feature_subset : list
        List of feature names to include in model
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Dictionary mapping feature names to (coefficient, p_value) tuples
    """
    if len(feature_subset) == 0:
        return {}
    
    # Fit logistic regression
    model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    model.fit(X[feature_subset], y)
    
    # Get coefficients
    coefficients = model.coef_[0]
    
    # Calculate standard errors using the Hessian approximation
    # For logistic regression, we can use the predicted probabilities
    predictions = model.predict_proba(X[feature_subset])[:, 1]
    
    # Weight matrix: diag(p * (1 - p))
    weights = predictions * (1 - predictions)
    
    # Ensure weights are not too small to avoid numerical issues
    weights = np.maximum(weights, 1e-10)
    
    # Design matrix
    X_design = X[feature_subset].values
    
    # Weighted design matrix
    X_weighted = X_design * np.sqrt(weights)[:, np.newaxis]
    
    # Covariance matrix: (X'WX)^-1
    try:
        # Add small ridge for numerical stability
        XtWX = X_weighted.T @ X_weighted
        XtWX += np.eye(len(feature_subset)) * 1e-8
        cov_matrix = np.linalg.inv(XtWX)
        std_errors = np.sqrt(np.diag(cov_matrix))
    except np.linalg.LinAlgError:
        # If matrix is singular, use large standard errors
        if logger:
            logger.warning("Singular matrix encountered, using large standard errors")
        std_errors = np.ones(len(feature_subset)) * 1e6
    
    # Calculate Wald statistics and p-values
    result = {}
    for i, feature in enumerate(feature_subset):
        if std_errors[i] > 0 and not np.isnan(std_errors[i]):
            wald_stat = coefficients[i] / std_errors[i]
            p_value = 2 * (1 - stats.norm.cdf(abs(wald_stat)))
        else:
            p_value = 1.0  # Maximum p-value if we can't calculate
        
        result[feature] = (coefficients[i], p_value)
    
    return result


def forward_step(X, y, current_features, remaining_features, slentry=0.05, logger=None):
    """
    Perform one forward selection step.
    
    Try adding each remaining feature individually and add the one with
    the lowest p-value if it's below the entry threshold.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    current_features : list
        Currently selected features
    remaining_features : list
        Features not yet selected
    slentry : float
        Entry threshold for p-value
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    tuple
        (feature_to_add, p_value) or (None, None) if no feature meets criterion
    """
    if len(remaining_features) == 0:
        return None, None
    
    best_feature = None
    best_p_value = 1.0
    
    for feature in remaining_features:
        # Test adding this feature
        test_features = current_features + [feature]
        
        try:
            # Calculate p-values with this feature added
            p_values_dict = calculate_p_values(X, y, test_features, logger=logger)
            
            # Get p-value for the newly added feature
            _, p_value = p_values_dict[feature]
            
            if p_value < best_p_value:
                best_p_value = p_value
                best_feature = feature
        except Exception as e:
            if logger:
                logger.warning(f"Error testing feature {feature}: {str(e)}")
            continue
    
    # Only add if p-value meets entry criterion
    if best_p_value < slentry:
        return best_feature, best_p_value
    else:
        return None, None


def backward_step(X, y, current_features, slstay=0.05, logger=None):
    """
    Perform one backward elimination step.
    
    Calculate p-values for all current features and remove the one with
    the highest p-value if it's above the stay threshold.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    current_features : list
        Currently selected features
    slstay : float
        Stay threshold for p-value
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    tuple
        (feature_to_remove, p_value) or (None, None) if no feature should be removed
    """
    if len(current_features) <= 1:
        # Don't remove if only one or zero features
        return None, None
    
    try:
        # Calculate p-values for all current features
        p_values_dict = calculate_p_values(X, y, current_features, logger=logger)
        
        # Find feature with highest p-value
        worst_feature = None
        worst_p_value = 0.0
        
        for feature, (coef, p_value) in p_values_dict.items():
            if p_value > worst_p_value:
                worst_p_value = p_value
                worst_feature = feature
        
        # Only remove if p-value exceeds stay criterion
        if worst_p_value > slstay:
            return worst_feature, worst_p_value
        else:
            return None, None
            
    except Exception as e:
        if logger:
            logger.warning(f"Error in backward step: {str(e)}")
        return None, None


def stepwise_selection(X, y, all_features, slentry=0.05, slstay=0.05, max_iterations=100, logger=None):
    """
    Perform stepwise selection (forward + backward) until convergence.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    all_features : list
        List of all available features
    slentry : float
        Entry threshold for p-value
    slstay : float
        Stay threshold for p-value
    max_iterations : int
        Maximum number of iterations
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    list
        Selected features
    """
    if logger:
        logger.info("=" * 80)
        logger.info("Starting Stepwise Selection")
        logger.info(f"Entry threshold (slentry): {slentry}")
        logger.info(f"Stay threshold (slstay): {slstay}")
        logger.info(f"Total available features: {len(all_features)}")
        logger.info("=" * 80)
    
    current_features = []
    remaining_features = all_features.copy()
    
    for iteration in range(max_iterations):
        if logger:
            logger.info(f"\n--- Iteration {iteration + 1} ---")
            logger.info(f"Current features: {len(current_features)}")
        
        changes_made = False
        
        # Forward step
        feature_to_add, add_p_value = forward_step(
            X, y, current_features, remaining_features, slentry=slentry, logger=logger
        )
        
        if feature_to_add is not None:
            current_features.append(feature_to_add)
            remaining_features.remove(feature_to_add)
            changes_made = True
            if logger:
                logger.info(f"FORWARD: Added {feature_to_add} (p-value: {add_p_value:.6f})")
        else:
            if logger:
                logger.info("FORWARD: No features meet entry criterion")
        
        # Backward step
        feature_to_remove, remove_p_value = backward_step(
            X, y, current_features, slstay=slstay, logger=logger
        )
        
        if feature_to_remove is not None:
            current_features.remove(feature_to_remove)
            remaining_features.append(feature_to_remove)
            changes_made = True
            if logger:
                logger.info(f"BACKWARD: Removed {feature_to_remove} (p-value: {remove_p_value:.6f})")
        else:
            if logger:
                logger.info("BACKWARD: No features exceed stay threshold")
        
        # Check for convergence
        if not changes_made:
            if logger:
                logger.info(f"\nConverged after {iteration + 1} iterations")
                logger.info(f"Final selected features: {len(current_features)}")
                logger.info(f"Features: {current_features}")
            break
    else:
        if logger:
            logger.warning(f"Maximum iterations ({max_iterations}) reached without convergence")
    
    return current_features


def train_final_model(X, y, selected_features, logger=None):
    """
    Train final logistic regression model with selected features.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    selected_features : list
        Selected feature names
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    tuple
        (model, coefficients_df) - trained model and DataFrame with coefficients/p-values
    """
    if logger:
        logger.info("=" * 80)
        logger.info("Training Final Model")
        logger.info(f"Selected features: {len(selected_features)}")
        logger.info("=" * 80)
    
    # Train model
    model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    model.fit(X[selected_features], y)
    
    if logger:
        logger.info("Model training complete")
    
    # Calculate final p-values
    p_values_dict = calculate_p_values(X, y, selected_features, logger=logger)
    
    # Create DataFrame with coefficients and p-values
    coef_data = []
    for feature in selected_features:
        coef, p_value = p_values_dict[feature]
        coef_data.append({
            'feature': feature,
            'coefficient': coef,
            'p_value': p_value
        })
    
    # Add intercept
    coef_data.insert(0, {
        'feature': 'Intercept',
        'coefficient': model.intercept_[0],
        'p_value': np.nan  # P-value for intercept not typically reported in this context
    })
    
    coefficients_df = pd.DataFrame(coef_data)
    
    if logger:
        logger.info("\nFinal Model Coefficients:")
        for _, row in coefficients_df.iterrows():
            if pd.notna(row['p_value']):
                logger.info(f"  {row['feature']}: coef={row['coefficient']:.6f}, p-value={row['p_value']:.6f}")
            else:
                logger.info(f"  {row['feature']}: coef={row['coefficient']:.6f}")
    
    return model, coefficients_df


def generate_risk_scores(df, model, selected_features, logger=None):
    """
    Generate predictions, risk scores, grades, recommendations, and interest rates.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with features
    model : LogisticRegression
        Trained model
    selected_features : list
        Selected feature names
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        DataFrame with customer_id, default_flag (if present), probability, risk_score, 
        risk_grade, recommendation, interest_rate
    """
    if logger:
        logger.info("Generating risk scores and grades")
    
    # Get predictions (probability of default)
    probabilities = model.predict_proba(df[selected_features])[:, 1]
    
    # Calculate risk scores: round(600 + 250 * (1 - probability))
    risk_scores = np.round(600 + 250 * (1 - probabilities)).astype(int)
    
    # Ensure scores are within valid range [300, 850]
    risk_scores = np.clip(risk_scores, 300, 850)
    
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
    
    # Add prediction columns
    output_df['probability'] = probabilities
    output_df['risk_score'] = risk_scores
    output_df['risk_grade'] = risk_grades
    output_df['recommendation'] = recommendations
    output_df['interest_rate'] = interest_rates
    
    if logger:
        logger.info(f"Risk scores generated for {len(output_df)} customers")
        logger.info("\nRisk Grade Distribution:")
        for grade in ['A', 'B', 'C', 'D', 'E', 'F']:
            count = (output_df['risk_grade'] == grade).sum()
            pct = count / len(output_df) * 100
            logger.info(f"  Grade {grade}: {count} ({pct:.1f}%)")
        
        logger.info("\nRecommendation Distribution:")
        for rec in ['Approve', 'Review', 'Decline']:
            count = (output_df['recommendation'] == rec).sum()
            pct = count / len(output_df) * 100
            logger.info(f"  {rec}: {count} ({pct:.1f}%)")
    
    return output_df


def save_model_outputs(model, coefficients_df, selected_features, models_dir, logger=None):
    """
    Save trained model, coefficients, and selected features.
    
    Parameters
    ----------
    model : LogisticRegression
        Trained model
    coefficients_df : pd.DataFrame
        Coefficients with p-values
    selected_features : list
        List of selected feature names
    models_dir : Path
        Models directory path
    logger : logging.Logger
        Logger instance
    """
    if logger:
        logger.info("Saving model outputs")
    
    # Save model
    model_path = models_dir / 'logistic_model.pkl'
    joblib.dump(model, model_path)
    if logger:
        logger.info(f"Model saved to: {model_path}")
    
    # Save coefficients
    coef_path = models_dir / 'model_coefficients.csv'
    coefficients_df.to_csv(coef_path, index=False)
    if logger:
        logger.info(f"Coefficients saved to: {coef_path}")
    
    # Save selected features
    features_path = models_dir / 'selected_features.pkl'
    with open(features_path, 'wb') as f:
        pickle.dump(selected_features, f)
    if logger:
        logger.info(f"Selected features saved to: {features_path}")
        logger.info(f"Number of selected features: {len(selected_features)}")


def save_risk_scores(train_scores, val_scores, output_dir, logger=None):
    """
    Save risk scores for train and validation sets.
    
    Parameters
    ----------
    train_scores : pd.DataFrame
        Training set risk scores
    val_scores : pd.DataFrame
        Validation set risk scores
    output_dir : Path
        Output directory path
    logger : logging.Logger
        Logger instance
    """
    if logger:
        logger.info("Saving risk score outputs")
    
    # Save training scores
    train_path = output_dir / 'risk_scores_train.csv'
    train_scores.to_csv(train_path, index=False)
    if logger:
        logger.info(f"Training risk scores saved to: {train_path} ({len(train_scores)} rows)")
    
    # Save validation scores
    val_path = output_dir / 'risk_scores_validation.csv'
    val_scores.to_csv(val_path, index=False)
    if logger:
        logger.info(f"Validation risk scores saved to: {val_path} ({len(val_scores)} rows)")


def check_feature_engineering_outputs(data_dir, models_dir, logger=None):
    """
    Check if feature engineering outputs exist.
    
    Parameters
    ----------
    data_dir : Path
        Data directory path
    models_dir : Path
        Models directory path
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    bool
        True if all required outputs exist, False otherwise
    """
    required_files = [
        data_dir / 'model_features_train.csv',
        data_dir / 'model_features_validation.csv',
        models_dir / 'scaler.pkl',
        models_dir / 'woe_mapping.pkl'
    ]
    
    all_exist = True
    for file_path in required_files:
        if not file_path.exists():
            if logger:
                logger.warning(f"Required file not found: {file_path}")
            all_exist = False
    
    return all_exist


def verify_consistency(models_dir, logger=None):
    """
    Load and verify consistency of saved scaler and WOE mapping.
    
    Parameters
    ----------
    models_dir : Path
        Models directory path
    logger : logging.Logger
        Logger instance
    """
    if logger:
        logger.info("Verifying consistency of saved artifacts")
    
    # Load scaler
    scaler_path = models_dir / 'scaler.pkl'
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        if logger:
            logger.info(f"Scaler loaded: {scaler.n_features_in_} features")
    else:
        if logger:
            logger.warning(f"Scaler not found at: {scaler_path}")
    
    # Load WOE mapping
    woe_path = models_dir / 'woe_mapping.pkl'
    if woe_path.exists():
        with open(woe_path, 'rb') as f:
            woe_mapping = pickle.load(f)
        if logger:
            logger.info(f"WOE mapping loaded: {len(woe_mapping)} bins")
            logger.info(f"WOE bins: {list(woe_mapping.keys())}")
    else:
        if logger:
            logger.warning(f"WOE mapping not found at: {woe_path}")


def main():
    """
    Main execution function for model training pipeline.
    """
    import argparse
    
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Model Training Pipeline for Credit Risk Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  python train.py
  
  # Specify custom input files
  python train.py --train-input data/train_features.csv --val-input data/val_features.csv
  
  # Specify custom directories
  python train.py --models-dir /path/to/models --output-dir /path/to/output
  
  # Specify all paths
  python train.py --train-input my_train.csv --val-input my_val.csv --models-dir models --output-dir output
        """
    )
    
    # Define default paths
    project_root = Path(__file__).parent.parent
    
    parser.add_argument(
        '--train-input',
        type=str,
        default=str(project_root / 'data' / 'model_features_train.csv'),
        help='Path to training features CSV file (default: data/model_features_train.csv)'
    )
    parser.add_argument(
        '--val-input',
        type=str,
        default=str(project_root / 'data' / 'model_features_validation.csv'),
        help='Path to validation features CSV file (default: data/model_features_validation.csv)'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default=str(project_root / 'models'),
        help='Directory for saving model artifacts (default: models/)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(project_root / 'output'),
        help='Directory for saving output files (default: output/)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging('credit_risk_model.log')
    logger.info("=" * 80)
    logger.info("Starting Model Training Pipeline")
    logger.info("=" * 80)
    
    try:
        # Convert paths to Path objects
        train_path = Path(args.train_input)
        val_path = Path(args.val_input)
        models_dir = Path(args.models_dir)
        output_dir = Path(args.output_dir)
        data_dir = train_path.parent  # For checking feature engineering outputs
        
        # Log the paths being used
        logger.info(f"Input paths:")
        logger.info(f"  Training features: {train_path}")
        logger.info(f"  Validation features: {val_path}")
        logger.info(f"  Models directory: {models_dir}")
        logger.info(f"  Output directory: {output_dir}")
        
        # Check if feature engineering outputs exist
        logger.info("Checking for feature engineering outputs...")
        if not check_feature_engineering_outputs(data_dir, models_dir, logger=logger):
            logger.info("=" * 80)
            logger.info("Feature engineering outputs not found. Running feature engineering first...")
            logger.info("=" * 80)
            # Run feature engineering
            feature_engineering.main()
            logger.info("=" * 80)
            logger.info("Feature engineering completed. Resuming model training...")
            logger.info("=" * 80)
        else:
            logger.info("All feature engineering outputs found. Proceeding with training.")
        
        # Verify consistency of saved artifacts
        verify_consistency(models_dir, logger=logger)
        
        # 1. Load data
        train_df, val_df = load_data(train_path, val_path, logger)
        
        # 2. Separate features from target
        X_train, y_train, all_features = separate_features_target(
            train_df, target_col='default_flag', exclude_cols=['customer_id'], logger=logger
        )
        
        X_val, y_val, _ = separate_features_target(
            val_df, target_col='default_flag', exclude_cols=['customer_id'], logger=logger
        )
        
        # 3. Perform stepwise selection
        selected_features = stepwise_selection(
            X_train, y_train, all_features,
            slentry=0.05, slstay=0.05,
            max_iterations=100,
            logger=logger
        )
        
        if len(selected_features) == 0:
            logger.error("No features selected by stepwise selection")
            raise ValueError("No features selected by stepwise selection")
        
        # 4. Train final model
        model, coefficients_df = train_final_model(
            X_train, y_train, selected_features, logger=logger
        )
        
        # 5. Save model, coefficients, and selected features
        save_model_outputs(model, coefficients_df, selected_features, models_dir, logger=logger)
        
        # 6. Generate risk scores for training set
        train_scores = generate_risk_scores(
            train_df, model, selected_features, logger=logger
        )
        
        # 7. Generate risk scores for validation set
        logger.info("\nGenerating validation set risk scores")
        val_scores = generate_risk_scores(
            val_df, model, selected_features, logger=logger
        )
        
        # 8. Save risk scores
        save_risk_scores(train_scores, val_scores, output_dir, logger=logger)
        
        logger.info("=" * 80)
        logger.info("Model Training Pipeline Completed Successfully")
        logger.info("=" * 80)
        logger.info(f"\nFinal Results Summary:")
        logger.info(f"  Selected features: {len(selected_features)}")
        logger.info(f"  Training samples: {len(train_scores)}")
        logger.info(f"  Validation samples: {len(val_scores)}")
        logger.info(f"  Risk score range: 300-850")
        logger.info(f"  Risk grades: A, B, C, D, E, F")
        logger.info(f"  Model saved: models/logistic_model.pkl")
        logger.info(f"  Coefficients saved: models/model_coefficients.csv")
        logger.info(f"  Training scores saved: output/risk_scores_train.csv")
        logger.info(f"  Validation scores saved: output/risk_scores_validation.csv")
        
    except Exception as e:
        logger.error(f"Error in model training pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
