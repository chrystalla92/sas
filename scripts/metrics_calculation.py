"""
Metrics Calculation Script for Credit Risk Model

This script performs comprehensive model validation metrics:
- ROC curve and AUC calculation
- Gini coefficient
- KS statistic (Kolmogorov-Smirnov)
- Confusion matrices at multiple thresholds
- Decile analysis
- PSI (Population Stability Index)
- Calibration metrics

Author: Risk Analytics Team
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import sys

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent))
from logging_config import setup_logging


def load_risk_scores(output_dir, logger):
    """
    Load risk scores for training and validation sets.
    
    Parameters
    ----------
    output_dir : Path
        Output directory path containing risk score CSV files
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    tuple
        (train_df, validation_df) DataFrames with risk scores
    """
    logger.info("Loading risk score files")
    
    # Load validation data
    validation_path = output_dir / 'risk_scores_validation.csv'
    if not validation_path.exists():
        raise FileNotFoundError(f"Validation file not found: {validation_path}")
    
    validation_df = pd.read_csv(validation_path)
    logger.info(f"Loaded validation data: {validation_df.shape[0]} rows, {validation_df.shape[1]} columns")
    logger.info(f"Validation file path: {validation_path}")
    
    # Load training data
    train_path = output_dir / 'risk_scores_train.csv'
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    
    train_df = pd.read_csv(train_path)
    logger.info(f"Loaded training data: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    logger.info(f"Training file path: {train_path}")
    
    return train_df, validation_df


def validate_required_columns(df, dataset_name, logger):
    """
    Validate that required columns exist in the dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    dataset_name : str
        Name of dataset for logging
    logger : logging.Logger
        Logger instance
        
    Raises
    ------
    ValueError
        If required columns are missing
    """
    required_cols = ['customer_id', 'default_flag', 'pd_logistic', 'credit_risk_score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"{dataset_name}: Missing required columns: {missing_cols}")
    
    logger.info(f"{dataset_name}: All required columns present")
    
    # Check for missing values in key columns
    for col in required_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            logger.warning(f"{dataset_name}: {col} has {missing_count} missing values")


def calculate_roc_auc_gini(y_true, y_pred, dataset_name, logger):
    """
    Calculate ROC curve, AUC, and Gini coefficient.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted probabilities
    dataset_name : str
        Name of dataset for logging
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Dictionary with auc and gini values
    """
    logger.info(f"Calculating ROC/AUC/Gini for {dataset_name}")
    
    # Calculate AUC
    auc = roc_auc_score(y_true, y_pred)
    
    # Calculate Gini coefficient
    gini = 2 * auc - 1
    
    logger.info(f"{dataset_name} - AUC: {auc:.6f}, Gini: {gini:.6f}")
    
    return {'auc': auc, 'gini': gini}


def calculate_ks_statistic(y_true, y_pred, logger):
    """
    Calculate KS statistic with cumulative distributions.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels (0=good, 1=bad/default)
    y_pred : array-like
        Predicted probabilities
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    tuple
        (ks_statistic, ks_df) - KS value and DataFrame with cumulative distributions
    """
    logger.info("Calculating KS statistic")
    
    # Create dataframe and sort by predicted probability
    df = pd.DataFrame({
        'default_flag': y_true,
        'pd_logistic': y_pred
    }).sort_values('pd_logistic').reset_index(drop=True)
    
    # Calculate total counts
    total_good = (df['default_flag'] == 0).sum()
    total_bad = (df['default_flag'] == 1).sum()
    
    logger.info(f"Total good (non-defaults): {total_good}, Total bad (defaults): {total_bad}")
    
    # Calculate cumulative counts
    df['cum_good'] = (df['default_flag'] == 0).cumsum()
    df['cum_bad'] = (df['default_flag'] == 1).cumsum()
    
    # Calculate cumulative percentages
    df['cum_good_rate'] = df['cum_good'] / total_good
    df['cum_bad_rate'] = df['cum_bad'] / total_bad
    
    # Calculate KS value
    df['ks_value'] = np.abs(df['cum_good_rate'] - df['cum_bad_rate'])
    
    # Find maximum KS
    max_ks_idx = df['ks_value'].idxmax()
    ks_statistic = df.loc[max_ks_idx, 'ks_value']
    ks_cutoff = df.loc[max_ks_idx, 'pd_logistic']
    
    logger.info(f"KS Statistic: {ks_statistic:.6f} at probability cutoff: {ks_cutoff:.6f}")
    
    # Create bins for output (using deciles)
    df['probability_bin'] = pd.qcut(df['pd_logistic'], q=10, labels=False, duplicates='drop') + 1
    
    # Aggregate by bin for output
    ks_output = df.groupby('probability_bin').agg({
        'cum_good_rate': 'max',
        'cum_bad_rate': 'max',
        'ks_value': 'max'
    }).reset_index()
    
    return ks_statistic, ks_output


def calculate_confusion_matrices(y_true, y_pred, thresholds, logger):
    """
    Calculate confusion matrices and metrics at multiple thresholds.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted probabilities
    thresholds : list
        List of probability thresholds to evaluate
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        DataFrame with metrics for each threshold
    """
    logger.info(f"Calculating confusion matrices at thresholds: {thresholds}")
    
    results = []
    
    for threshold in thresholds:
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # Handle division by zero
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = sensitivity  # Recall is same as sensitivity
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        })
        
        logger.info(f"Threshold {threshold:.1f}: TP={tp}, TN={tn}, FP={fp}, FN={fn}, "
                   f"Accuracy={accuracy:.4f}, F1={f1_score:.4f}")
    
    return pd.DataFrame(results)


def calculate_decile_analysis(df, logger):
    """
    Calculate decile analysis based on credit_risk_score.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with default_flag, pd_logistic, credit_risk_score
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        DataFrame with decile metrics
    """
    logger.info("Calculating decile analysis")
    
    # Create deciles based on pd_logistic (descending - highest risk first)
    # Use pd_logistic instead of credit_risk_score for decile ranking
    df = df.copy()
    df['decile'] = pd.qcut(df['pd_logistic'], q=10, labels=False, duplicates='drop')
    # Reverse so that decile 10 = highest risk (highest pd_logistic)
    df['decile'] = 10 - df['decile']
    
    # Calculate metrics by decile
    decile_metrics = df.groupby('decile').agg({
        'customer_id': 'count',
        'default_flag': ['sum', 'mean'],
        'pd_logistic': 'mean',
        'credit_risk_score': ['min', 'max']
    }).reset_index()
    
    # Flatten column names
    decile_metrics.columns = ['decile', 'n_records', 'n_defaults', 'default_rate', 
                              'avg_predicted_prob', 'min_score', 'max_score']
    
    # Sort by decile
    decile_metrics = decile_metrics.sort_values('decile').reset_index(drop=True)
    
    logger.info(f"Decile analysis completed for {len(decile_metrics)} deciles")
    logger.info(f"Default rates range: {decile_metrics['default_rate'].min():.4f} to {decile_metrics['default_rate'].max():.4f}")
    
    return decile_metrics


def calculate_psi(train_pred, validation_pred, logger):
    """
    Calculate Population Stability Index (PSI).
    
    Parameters
    ----------
    train_pred : array-like
        Predicted probabilities from training set
    validation_pred : array-like
        Predicted probabilities from validation set
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    float
        PSI value
    """
    logger.info("Calculating Population Stability Index (PSI)")
    
    # Create 10 equal-frequency bins based on training data
    train_df = pd.DataFrame({'pd_logistic': train_pred})
    validation_df = pd.DataFrame({'pd_logistic': validation_pred})
    
    # Use qcut on training data to get bin edges
    train_df['bin'] = pd.qcut(train_df['pd_logistic'], q=10, labels=False, duplicates='drop')
    
    # Get the bin edges from training data
    bin_edges = pd.qcut(train_df['pd_logistic'], q=10, retbins=True, duplicates='drop')[1]
    
    # Apply same bins to validation data
    validation_df['bin'] = pd.cut(validation_df['pd_logistic'], bins=bin_edges, labels=False, include_lowest=True)
    
    # Handle any values outside training range
    validation_df['bin'] = validation_df['bin'].fillna(0).astype(int)
    
    # Calculate distributions
    train_dist = train_df['bin'].value_counts(normalize=True).sort_index()
    validation_dist = validation_df['bin'].value_counts(normalize=True).sort_index()
    
    # Ensure all bins are present
    all_bins = range(train_dist.index.min(), train_dist.index.max() + 1)
    train_dist = train_dist.reindex(all_bins, fill_value=0.0001)  # Small value to avoid log(0)
    validation_dist = validation_dist.reindex(all_bins, fill_value=0.0001)
    
    # Calculate PSI for each bin
    psi_values = (validation_dist - train_dist) * np.log(validation_dist / train_dist)
    psi = psi_values.sum()
    
    logger.info(f"PSI: {psi:.6f}")
    
    if psi < 0.1:
        interpretation = "No significant change"
    elif psi < 0.25:
        interpretation = "Some change"
    else:
        interpretation = "Significant change"
    
    logger.info(f"PSI Interpretation: {interpretation}")
    
    return psi


def calculate_calibration_metrics(y_true, y_pred, logger):
    """
    Calculate calibration metrics and plot data.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted probabilities
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    tuple
        (calibration_df, mean_calibration_error)
    """
    logger.info("Calculating calibration metrics")
    
    # Create dataframe
    df = pd.DataFrame({
        'default_flag': y_true,
        'pd_logistic': y_pred
    })
    
    # Create 10 bins based on predicted probability
    df['probability_bin'] = pd.qcut(df['pd_logistic'], q=10, labels=False, duplicates='drop') + 1
    
    # Calculate observed vs predicted by bin
    calibration_data = df.groupby('probability_bin').agg({
        'pd_logistic': ['mean', 'count'],
        'default_flag': 'mean'
    }).reset_index()
    
    # Flatten column names
    calibration_data.columns = ['probability_bin', 'avg_predicted_prob', 'n_records', 'observed_default_rate']
    
    # Calculate mean absolute calibration error
    calibration_data['calibration_error'] = np.abs(
        calibration_data['observed_default_rate'] - calibration_data['avg_predicted_prob']
    )
    mean_calibration_error = calibration_data['calibration_error'].mean()
    
    logger.info(f"Mean absolute calibration error: {mean_calibration_error:.6f}")
    
    # Reorder columns for output
    calibration_output = calibration_data[['probability_bin', 'n_records', 'avg_predicted_prob', 'observed_default_rate']]
    
    return calibration_output, mean_calibration_error


def save_outputs(validation_summary, decile_analysis, threshold_analysis, 
                ks_statistic, calibration_plot, model_performance_metrics,
                output_dir, logger):
    """
    Save all output CSV files.
    
    Parameters
    ----------
    validation_summary : pd.DataFrame
        Validation summary metrics
    decile_analysis : pd.DataFrame
        Decile analysis results
    threshold_analysis : pd.DataFrame
        Threshold analysis results
    ks_statistic : pd.DataFrame
        KS statistic data
    calibration_plot : pd.DataFrame
        Calibration plot data
    model_performance_metrics : pd.DataFrame
        Comprehensive model metrics
    output_dir : Path
        Output directory path
    logger : logging.Logger
        Logger instance
    """
    logger.info("Saving output files")
    
    # Save validation_summary.csv
    output_path = output_dir / 'validation_summary.csv'
    validation_summary.to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path}")
    
    # Save decile_analysis.csv
    output_path = output_dir / 'decile_analysis.csv'
    decile_analysis.to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path}")
    
    # Save threshold_analysis.csv
    output_path = output_dir / 'threshold_analysis.csv'
    threshold_analysis.to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path}")
    
    # Save ks_statistic.csv
    output_path = output_dir / 'ks_statistic.csv'
    ks_statistic.to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path}")
    
    # Save calibration_plot.csv
    output_path = output_dir / 'calibration_plot.csv'
    calibration_plot.to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path}")
    
    # Save model_performance_metrics.csv
    output_path = output_dir / 'model_performance_metrics.csv'
    model_performance_metrics.to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path}")
    
    logger.info("All output files saved successfully")


def main():
    """
    Main execution function for metrics calculation.
    """
    # Setup logging
    logger = setup_logging()
    logger.info("="*80)
    logger.info("Starting metrics calculation")
    logger.info("="*80)
    
    try:
        # Define paths
        project_root = Path(__file__).parent.parent
        output_dir = project_root / 'output'
        
        # Load data
        train_df, validation_df = load_risk_scores(output_dir, logger)
        
        # Validate columns
        validate_required_columns(train_df, "Training data", logger)
        validate_required_columns(validation_df, "Validation data", logger)
        
        # Extract arrays for calculations
        y_train = train_df['default_flag'].values
        y_pred_train = train_df['pd_logistic'].values
        
        y_validation = validation_df['default_flag'].values
        y_pred_validation = validation_df['pd_logistic'].values
        
        logger.info(f"Training set: {len(y_train)} records, {y_train.sum()} defaults ({y_train.mean():.4f} rate)")
        logger.info(f"Validation set: {len(y_validation)} records, {y_validation.sum()} defaults ({y_validation.mean():.4f} rate)")
        
        # 1. Calculate ROC/AUC/Gini for both sets
        logger.info("-" * 80)
        logger.info("SECTION 1: ROC Curve and AUC Analysis")
        logger.info("-" * 80)
        
        train_metrics = calculate_roc_auc_gini(y_train, y_pred_train, "Training", logger)
        validation_metrics = calculate_roc_auc_gini(y_validation, y_pred_validation, "Validation", logger)
        
        # 2. Calculate KS statistic
        logger.info("-" * 80)
        logger.info("SECTION 2: Kolmogorov-Smirnov (KS) Statistic")
        logger.info("-" * 80)
        
        ks_value, ks_df = calculate_ks_statistic(y_validation, y_pred_validation, logger)
        
        # 3. Calculate confusion matrices at multiple thresholds
        logger.info("-" * 80)
        logger.info("SECTION 3: Confusion Matrix and Accuracy Metrics")
        logger.info("-" * 80)
        
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        threshold_analysis = calculate_confusion_matrices(y_validation, y_pred_validation, thresholds, logger)
        
        # 4. Calculate decile analysis
        logger.info("-" * 80)
        logger.info("SECTION 4: Decile Analysis and Lift Chart")
        logger.info("-" * 80)
        
        decile_analysis = calculate_decile_analysis(validation_df, logger)
        
        # 5. Calculate PSI
        logger.info("-" * 80)
        logger.info("SECTION 5: Population Stability Index (PSI)")
        logger.info("-" * 80)
        
        psi_value = calculate_psi(y_pred_train, y_pred_validation, logger)
        
        # 6. Calculate calibration metrics
        logger.info("-" * 80)
        logger.info("SECTION 6: Calibration Plot")
        logger.info("-" * 80)
        
        calibration_plot, mean_calibration_error = calculate_calibration_metrics(
            y_validation, y_pred_validation, logger
        )
        
        # Create validation summary
        logger.info("-" * 80)
        logger.info("SECTION 7: Generate Validation Report")
        logger.info("-" * 80)
        
        validation_summary = pd.DataFrame([{
            'auc_train': train_metrics['auc'],
            'auc_validation': validation_metrics['auc'],
            'gini_train': train_metrics['gini'],
            'gini_validation': validation_metrics['gini'],
            'ks_statistic': ks_value,
            'psi': psi_value,
            'mean_calibration_error': mean_calibration_error
        }])
        
        # Create comprehensive model performance metrics
        # Get metrics at threshold 0.5
        metrics_at_50 = threshold_analysis[threshold_analysis['threshold'] == 0.5].iloc[0]
        
        model_performance_metrics = pd.DataFrame([{
            'model_name': 'Logistic Regression',
            'auc_train': train_metrics['auc'],
            'auc_validation': validation_metrics['auc'],
            'gini_train': train_metrics['gini'],
            'gini_validation': validation_metrics['gini'],
            'ks_statistic': ks_value,
            'psi': psi_value,
            'mean_calibration_error': mean_calibration_error,
            'accuracy_at_50': metrics_at_50['accuracy'],
            'precision_at_50': metrics_at_50['precision'],
            'recall_at_50': metrics_at_50['recall'],
            'f1_score_at_50': metrics_at_50['f1_score'],
            'specificity_at_50': metrics_at_50['specificity'],
            'validation_records': len(validation_df),
            'train_records': len(train_df)
        }])
        
        # Log summary statistics
        logger.info("Summary Statistics:")
        logger.info(f"  Training AUC: {train_metrics['auc']:.6f}")
        logger.info(f"  Validation AUC: {validation_metrics['auc']:.6f}")
        logger.info(f"  Training Gini: {train_metrics['gini']:.6f}")
        logger.info(f"  Validation Gini: {validation_metrics['gini']:.6f}")
        logger.info(f"  KS Statistic: {ks_value:.6f}")
        logger.info(f"  PSI: {psi_value:.6f}")
        logger.info(f"  Mean Calibration Error: {mean_calibration_error:.6f}")
        
        # Save all outputs
        logger.info("-" * 80)
        logger.info("SECTION 8: Export Validation Results")
        logger.info("-" * 80)
        
        save_outputs(
            validation_summary=validation_summary,
            decile_analysis=decile_analysis,
            threshold_analysis=threshold_analysis,
            ks_statistic=ks_df,
            calibration_plot=calibration_plot,
            model_performance_metrics=model_performance_metrics,
            output_dir=output_dir,
            logger=logger
        )
        
        logger.info("="*80)
        logger.info("Metrics calculation completed successfully")
        logger.info("="*80)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        logger.error("Please ensure train.py has been run successfully to generate risk score files")
        raise
    
    except ValueError as e:
        logger.error(f"Data validation error: {str(e)}")
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error during metrics calculation: {str(e)}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    main()
