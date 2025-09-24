#!/usr/bin/env python3
"""
Bank Credit Risk Scoring Model - Step 4: Model Training

Purpose: Train logistic regression and decision tree credit risk scoring models
Author: Risk Analytics Team
Date: 2025

This script trains machine learning models for credit risk scoring including:
- Logistic regression with feature selection (primary model)
- Decision tree classifier with specific hyperparameters
- Model persistence using joblib
- Risk scoring and grading system
- Model evaluation and validation

MIGRATION FROM SAS:
This Python implementation replicates the functionality from 04_train_credit_model.sas,
including PROC LOGISTIC and PROC HPSPLIT equivalent implementations.

USAGE:
    python 04_train_credit_model.py

INPUT:
    - output/model_features_train.csv (from Script 3)
    - output/model_features_validation.csv (from Script 3)

OUTPUT:
    - models/logistic_model.joblib
    - models/decision_tree_model.joblib
    - output/scored_applications.csv
    - Model performance metrics and summaries

DEPENDENCIES:
    - pandas>=1.5.0
    - numpy>=1.24.0
    - scikit-learn>=1.3.0
    - joblib>=1.3.0
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, roc_curve, precision_recall_curve,
                            accuracy_score, precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys
import subprocess
from pathlib import Path
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

def ensure_feature_engineering():
    """
    Ensure feature engineering has been completed and datasets are available.
    
    Returns:
        bool: True if datasets are available, False otherwise
    """
    print("Checking for feature engineering outputs...")
    
    output_dir = Path('output')
    train_file = output_dir / 'model_features_train.csv'
    val_file = output_dir / 'model_features_validation.csv'
    
    if train_file.exists() and val_file.exists():
        print("✓ Feature engineering datasets found")
        return True
    
    print("Feature engineering datasets not found. Running feature engineering...")
    
    try:
        # Import and run feature engineering using subprocess for proper execution
        result = subprocess.run([sys.executable, '03_feature_engineering.py'], 
                              capture_output=True, text=True, cwd='.')
        
        if result.returncode != 0:
            print(f"❌ Feature engineering script failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error output: {result.stderr}")
            if result.stdout:
                print(f"Standard output: {result.stdout}")
            return False
        else:
            print("Feature engineering script executed successfully")
            if result.stdout:
                print("Script output:", result.stdout)
            
        # Check again
        if train_file.exists() and val_file.exists():
            print("✓ Feature engineering completed successfully")
            return True
        else:
            raise Exception("Feature engineering outputs not created")
            
    except Exception as e:
        print(f"❌ Error running feature engineering: {str(e)}")
        return False

def load_model_data():
    """
    Load training and validation datasets for model training.
    
    Returns:
        tuple: (train_df, validation_df)
    """
    print("Loading model training data...")
    
    try:
        train_df = pd.read_csv('output/model_features_train.csv')
        val_df = pd.read_csv('output/model_features_validation.csv')
        
        print(f"Training set: {len(train_df):,} records")
        print(f"Validation set: {len(val_df):,} records")
        print(f"Training default rate: {train_df['default_flag'].mean():.3f}")
        print(f"Validation default rate: {val_df['default_flag'].mean():.3f}")
        
        return train_df, val_df
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        raise

def prepare_model_features(train_df, val_df):
    """
    Prepare features for model training, matching SAS implementation.
    
    Args:
        train_df (pd.DataFrame): Training dataset
        val_df (pd.DataFrame): Validation dataset
        
    Returns:
        tuple: (X_train, X_val, y_train, y_val, feature_names)
    """
    print("Preparing model features...")
    
    # Define features matching SAS PROC LOGISTIC implementation
    core_features = [
        # Core financial metrics
        'credit_score', 'debt_to_income_ratio', 'credit_utilization',
        'payment_to_income_ratio', 'loan_to_income_ratio',
        
        # Employment and stability
        'employment_years', 'employment_score',
        
        # Credit history
        'num_late_payments', 'previous_defaults', 'has_delinquency',
        'credit_quality_score', 'credit_history_years',
        
        # Risk flags
        'flag_high_dti', 'flag_low_credit', 'flag_high_util',
        'flag_recent_default', 'total_risk_flags',
        
        # Demographics and loan
        'age', 'monthly_income', 'loan_amount', 'affordability_score'
    ]
    
    # Categorical indicators (dummy variables)
    categorical_features = [
        'emp_fulltime', 'emp_unemployed',
        'home_rent', 'purpose_debt'
    ]
    
    # Combine all features
    all_features = core_features + categorical_features
    
    # Keep only features that exist in both datasets
    available_features = [f for f in all_features 
                         if f in train_df.columns and f in val_df.columns]
    
    print(f"Using {len(available_features)} features for modeling")
    
    # Prepare feature matrices
    X_train = train_df[available_features].copy()
    X_val = val_df[available_features].copy()
    y_train = train_df['default_flag'].copy()
    y_val = val_df['default_flag'].copy()
    
    # Handle any missing values (fill with median for numeric, mode for categorical)
    for col in X_train.columns:
        if X_train[col].dtype in ['float64', 'int64']:
            fill_value = X_train[col].median()
            X_train[col] = X_train[col].fillna(fill_value)
            X_val[col] = X_val[col].fillna(fill_value)
        else:
            fill_value = X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else 0
            X_train[col] = X_train[col].fillna(fill_value)
            X_val[col] = X_val[col].fillna(fill_value)
    
    print("✓ Model features prepared successfully")
    return X_train, X_val, y_train, y_val, available_features

def train_logistic_regression(X_train, y_train, X_val, y_val, feature_names):
    """
    Train logistic regression model with feature selection (replicating PROC LOGISTIC).
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        feature_names: List of feature names
        
    Returns:
        dict: Model results including fitted model, predictions, and metrics
    """
    print("Training Logistic Regression model...")
    
    # Initialize logistic regression (similar to SAS defaults)
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='liblinear',  # Good for small datasets and feature selection
        penalty='l1',  # L1 regularization for feature selection
        C=1.0  # Default regularization strength
    )
    
    # Feature selection using Recursive Feature Elimination (mimics stepwise)
    print("Performing feature selection (stepwise equivalent)...")
    selector = RFECV(
        estimator=lr_model,
        step=1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1
    )
    
    # Fit with feature selection
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    
    # Get selected features
    selected_features = [feature_names[i] for i in range(len(feature_names)) 
                        if selector.support_[i]]
    print(f"Selected {len(selected_features)} features: {selected_features}")
    
    # Train final model with selected features
    final_lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='liblinear',
        penalty='l1',
        C=1.0
    )
    
    final_lr_model.fit(X_train_selected, y_train)
    
    # Generate predictions
    train_pred_proba = final_lr_model.predict_proba(X_train_selected)[:, 1]
    val_pred_proba = final_lr_model.predict_proba(X_val_selected)[:, 1]
    
    train_pred = (train_pred_proba >= 0.5).astype(int)
    val_pred = (val_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    train_auc = roc_auc_score(y_train, train_pred_proba)
    val_auc = roc_auc_score(y_val, val_pred_proba)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"Training AUC: {train_auc:.4f}, Accuracy: {train_acc:.4f}")
    print(f"Validation AUC: {val_auc:.4f}, Accuracy: {val_acc:.4f}")
    
    # Feature importance (coefficients)
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'coefficient': final_lr_model.coef_[0],
        'abs_coefficient': np.abs(final_lr_model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return {
        'model': final_lr_model,
        'selector': selector,
        'selected_features': selected_features,
        'train_predictions': train_pred_proba,
        'val_predictions': val_pred_proba,
        'train_binary_pred': train_pred,
        'val_binary_pred': val_pred,
        'train_auc': train_auc,
        'val_auc': val_auc,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'feature_importance': feature_importance
    }

def train_decision_tree(X_train, y_train, X_val, y_val, feature_names):
    """
    Train decision tree classifier with exact SAS PROC HPSPLIT parameters.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data  
        feature_names: List of feature names
        
    Returns:
        dict: Model results including fitted model, predictions, and metrics
    """
    print("Training Decision Tree model (PROC HPSPLIT equivalent)...")
    
    # Initialize decision tree with exact SAS parameters
    dt_model = DecisionTreeClassifier(
        max_depth=10,  # Specified in task requirements
        min_samples_leaf=50,  # Specified in task requirements
        criterion='gini',  # Specified in task requirements
        random_state=42,
        min_samples_split=100,  # Conservative splitting
        max_features=None  # Use all features
    )
    
    # Train the model
    dt_model.fit(X_train, y_train)
    
    # Generate predictions
    train_pred_proba = dt_model.predict_proba(X_train)[:, 1]
    val_pred_proba = dt_model.predict_proba(X_val)[:, 1]
    
    train_pred = dt_model.predict(X_train)
    val_pred = dt_model.predict(X_val)
    
    # Calculate metrics
    train_auc = roc_auc_score(y_train, train_pred_proba)
    val_auc = roc_auc_score(y_val, val_pred_proba)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"Training AUC: {train_auc:.4f}, Accuracy: {train_acc:.4f}")
    print(f"Validation AUC: {val_auc:.4f}, Accuracy: {val_acc:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': dt_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return {
        'model': dt_model,
        'train_predictions': train_pred_proba,
        'val_predictions': val_pred_proba,
        'train_binary_pred': train_pred,
        'val_binary_pred': val_pred,
        'train_auc': train_auc,
        'val_auc': val_auc,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'feature_importance': feature_importance
    }

def main():
    """
    Main model training pipeline.
    """
    print("=" * 60)
    print("BANK CREDIT RISK MODEL - MODEL TRAINING")
    print("=" * 60)
    
    try:
        # Step 1: Ensure feature engineering is complete
        if not ensure_feature_engineering():
            raise Exception("Feature engineering failed")
        
        # Update todo
        print("\n✓ Feature engineering datasets confirmed")
        
        # Step 2: Load data
        train_df, val_df = load_model_data()
        
        # Step 3: Prepare features
        X_train, X_val, y_train, y_val, feature_names = prepare_model_features(train_df, val_df)
        
        # Create models directory
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        # Step 4: Train logistic regression
        lr_results = train_logistic_regression(X_train, y_train, X_val, y_val, feature_names)
        
        # Step 5: Train decision tree  
        dt_results = train_decision_tree(X_train, y_train, X_val, y_val, feature_names)
        
        return lr_results, dt_results, train_df, val_df, X_train, X_val, y_train, y_val
        
    except Exception as e:
        print(f"\n❌ Error during model training: {str(e)}")
        raise

def calibrate_probabilities(lr_results, X_train, y_train, X_val):
    """
    Implement probability calibration using Platt scaling.
    
    Args:
        lr_results: Logistic regression results
        X_train, y_train: Training data
        X_val: Validation features
        
    Returns:
        dict: Calibrated model and predictions
    """
    print("Implementing probability calibration (Platt scaling)...")
    
    try:
        # Get selected features for calibration
        X_train_selected = lr_results['selector'].transform(X_train)
        X_val_selected = lr_results['selector'].transform(X_val)
        
        # Create calibrated classifier
        calibrated_clf = CalibratedClassifierCV(
            lr_results['model'], 
            method='sigmoid',  # Platt scaling
            cv=3
        )
        
        # Fit calibration
        calibrated_clf.fit(X_train_selected, y_train)
        
        # Generate calibrated predictions
        cal_train_proba = calibrated_clf.predict_proba(X_train_selected)[:, 1]
        cal_val_proba = calibrated_clf.predict_proba(X_val_selected)[:, 1]
        
        print("✓ Probability calibration completed")
        
        return {
            'calibrated_model': calibrated_clf,
            'cal_train_proba': cal_train_proba,
            'cal_val_proba': cal_val_proba
        }
        
    except Exception as e:
        print(f"❌ Error during calibration: {str(e)}")
        return None

def create_risk_scoring_system(predictions, customer_ids):
    """
    Create risk scoring system (300-850 scale) and risk grades (A-F).
    
    Args:
        predictions: Probability predictions
        customer_ids: Customer IDs
        
    Returns:
        pd.DataFrame: Risk scores and grades
    """
    print("Creating risk scoring system...")
    
    # Convert probabilities to 300-850 credit score scale (inverse relationship)
    base_score = 600
    score_range = 250
    credit_risk_scores = np.round(base_score + score_range * (1 - predictions)).astype(int)
    
    # Assign risk grades based on scores
    def assign_risk_grade(score):
        if score >= 750:
            return 'A'
        elif score >= 700:
            return 'B'
        elif score >= 650:
            return 'C'
        elif score >= 600:
            return 'D'
        elif score >= 550:
            return 'E'
        else:
            return 'F'
    
    risk_grades = [assign_risk_grade(score) for score in credit_risk_scores]
    
    # Generate recommendations
    def get_recommendation(grade):
        if grade in ['A', 'B']:
            return 'Approve'
        elif grade == 'C':
            return 'Review'
        else:
            return 'Decline'
    
    recommendations = [get_recommendation(grade) for grade in risk_grades]
    
    # Risk-based pricing (interest rates)
    def get_interest_rate(grade):
        base_rate = 0.05
        rate_mapping = {
            'A': base_rate,
            'B': base_rate + 0.02,
            'C': base_rate + 0.04,
            'D': base_rate + 0.07,
            'E': base_rate + 0.10,
            'F': base_rate + 0.15
        }
        return rate_mapping[grade]
    
    interest_rates = [get_interest_rate(grade) for grade in risk_grades]
    
    # Create results dataframe
    risk_results = pd.DataFrame({
        'customer_id': customer_ids,
        'default_probability': predictions,
        'credit_risk_score': credit_risk_scores,
        'risk_grade': risk_grades,
        'recommendation': recommendations,
        'interest_rate': interest_rates
    })
    
    print("✓ Risk scoring system created")
    return risk_results

def save_models(lr_results, dt_results, calibration_results=None):
    """
    Save trained models using joblib for persistence.
    
    Args:
        lr_results: Logistic regression results
        dt_results: Decision tree results
        calibration_results: Calibration results (optional)
    """
    print("Saving trained models...")
    
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    try:
        # Save logistic regression
        lr_model_path = models_dir / 'logistic_model.joblib'
        joblib.dump({
            'model': lr_results['model'],
            'selector': lr_results['selector'],
            'selected_features': lr_results['selected_features'],
            'feature_importance': lr_results['feature_importance']
        }, lr_model_path)
        print(f"✓ Logistic regression model saved to {lr_model_path}")
        
        # Save decision tree
        dt_model_path = models_dir / 'decision_tree_model.joblib'
        joblib.dump({
            'model': dt_results['model'],
            'feature_importance': dt_results['feature_importance']
        }, dt_model_path)
        print(f"✓ Decision tree model saved to {dt_model_path}")
        
        # Save calibrated model if available
        if calibration_results:
            cal_model_path = models_dir / 'calibrated_model.joblib'
            joblib.dump(calibration_results, cal_model_path)
            print(f"✓ Calibrated model saved to {cal_model_path}")
        
        # Save model metadata
        metadata = {
            'logistic_regression': {
                'train_auc': lr_results['train_auc'],
                'val_auc': lr_results['val_auc'],
                'train_accuracy': lr_results['train_accuracy'],
                'val_accuracy': lr_results['val_accuracy'],
                'n_features': len(lr_results['selected_features'])
            },
            'decision_tree': {
                'train_auc': dt_results['train_auc'],
                'val_auc': dt_results['val_auc'],
                'train_accuracy': dt_results['train_accuracy'],
                'val_accuracy': dt_results['val_accuracy'],
                'max_depth': 10,
                'min_samples_leaf': 50
            }
        }
        
        metadata_path = models_dir / 'model_metadata.joblib'
        joblib.dump(metadata, metadata_path)
        print(f"✓ Model metadata saved to {metadata_path}")
        
    except Exception as e:
        print(f"❌ Error saving models: {str(e)}")
        raise

def generate_scored_applications(train_df, val_df, lr_results, dt_results, calibration_results=None):
    """
    Generate scored_applications.csv with predictions and probability scores.
    
    Args:
        train_df, val_df: Original datasets
        lr_results: Logistic regression results
        dt_results: Decision tree results
        calibration_results: Calibration results (optional)
    """
    print("Generating scored applications output...")
    
    try:
        # Combine training and validation sets for complete scoring
        all_data = pd.concat([train_df, val_df], ignore_index=True)
        
        # Get predictions from both models
        all_lr_pred = np.concatenate([lr_results['train_predictions'], 
                                    lr_results['val_predictions']])
        all_dt_pred = np.concatenate([dt_results['train_predictions'], 
                                    dt_results['val_predictions']])
        
        # Use calibrated predictions if available
        if calibration_results:
            all_cal_pred = np.concatenate([calibration_results['cal_train_proba'],
                                         calibration_results['cal_val_proba']])
        else:
            all_cal_pred = all_lr_pred
        
        # Create risk scoring for primary model (logistic regression)
        risk_scores = create_risk_scoring_system(all_cal_pred, all_data['customer_id'])
        
        # Create comprehensive output dataset
        scored_applications = pd.DataFrame({
            'customer_id': all_data['customer_id'],
            'default_flag': all_data['default_flag'],
            
            # Model predictions
            'logistic_probability': all_lr_pred,
            'decision_tree_probability': all_dt_pred,
            'calibrated_probability': all_cal_pred,
            
            # Binary predictions (0.5 threshold)
            'logistic_prediction': (all_lr_pred >= 0.5).astype(int),
            'decision_tree_prediction': (all_dt_pred >= 0.5).astype(int),
            
            # Risk scoring system
            'credit_risk_score': risk_scores['credit_risk_score'],
            'risk_grade': risk_scores['risk_grade'],
            'recommendation': risk_scores['recommendation'],
            'interest_rate': risk_scores['interest_rate'],
            
            # Key features for interpretability
            'credit_score': all_data['credit_score'] if 'credit_score' in all_data.columns else np.nan,
            'debt_to_income_ratio': all_data['debt_to_income_ratio'] if 'debt_to_income_ratio' in all_data.columns else np.nan,
            'total_risk_flags': all_data['total_risk_flags'] if 'total_risk_flags' in all_data.columns else np.nan
        })
        
        # Export to CSV
        output_path = Path('output/scored_applications.csv')
        scored_applications.to_csv(output_path, index=False)
        print(f"✓ Scored applications exported to {output_path}")
        
        return scored_applications
        
    except Exception as e:
        print(f"❌ Error generating scored applications: {str(e)}")
        raise

def generate_model_performance_summary(lr_results, dt_results, y_train, y_val):
    """
    Generate comprehensive model performance metrics and validation summary.
    
    Args:
        lr_results: Logistic regression results
        dt_results: Decision tree results
        y_train, y_val: True labels
    """
    print("Generating model performance summary...")
    
    try:
        # Create performance summary
        performance_summary = {
            'Model': ['Logistic Regression', 'Decision Tree'],
            'Train_AUC': [lr_results['train_auc'], dt_results['train_auc']],
            'Validation_AUC': [lr_results['val_auc'], dt_results['val_auc']],
            'Train_Accuracy': [lr_results['train_accuracy'], dt_results['train_accuracy']],
            'Validation_Accuracy': [lr_results['val_accuracy'], dt_results['val_accuracy']],
            'AUC_Difference': [lr_results['train_auc'] - lr_results['val_auc'],
                             dt_results['train_auc'] - dt_results['val_auc']]
        }
        
        performance_df = pd.DataFrame(performance_summary)
        
        # Export performance metrics
        performance_path = Path('output/model_performance_metrics.csv')
        performance_df.to_csv(performance_path, index=False)
        print(f"✓ Performance metrics exported to {performance_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*50)
        print(performance_df.to_string(index=False))
        
        # Check for overfitting
        print(f"\nOverfitting Check:")
        for i, model in enumerate(['Logistic Regression', 'Decision Tree']):
            auc_diff = performance_summary['AUC_Difference'][i]
            if auc_diff > 0.05:
                print(f"⚠️  {model}: Potential overfitting detected (AUC diff: {auc_diff:.4f})")
            else:
                print(f"✓ {model}: Good generalization (AUC diff: {auc_diff:.4f})")
        
        return performance_df
        
    except Exception as e:
        print(f"❌ Error generating performance summary: {str(e)}")
        raise

def main():
    """
    Main model training pipeline.
    """
    print("=" * 60)
    print("BANK CREDIT RISK MODEL - MODEL TRAINING")
    print("=" * 60)
    
    try:
        # Step 1: Ensure feature engineering is complete
        if not ensure_feature_engineering():
            raise Exception("Feature engineering failed")
        
        print("\n✓ Feature engineering datasets confirmed")
        
        # Step 2: Load data
        train_df, val_df = load_model_data()
        
        # Step 3: Prepare features
        X_train, X_val, y_train, y_val, feature_names = prepare_model_features(train_df, val_df)
        
        # Create models directory
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        # Step 4: Train logistic regression
        lr_results = train_logistic_regression(X_train, y_train, X_val, y_val, feature_names)
        
        # Step 5: Train decision tree  
        dt_results = train_decision_tree(X_train, y_train, X_val, y_val, feature_names)
        
        # Step 6: Probability calibration
        calibration_results = calibrate_probabilities(lr_results, X_train, y_train, X_val)
        
        # Step 7: Save models
        save_models(lr_results, dt_results, calibration_results)
        
        # Step 8: Generate scored applications
        scored_apps = generate_scored_applications(train_df, val_df, lr_results, dt_results, calibration_results)
        
        # Step 9: Generate performance summary
        performance_summary = generate_model_performance_summary(lr_results, dt_results, y_train, y_val)
        
        print("\n" + "="*60)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        print("✓ Logistic regression model trained and saved")
        print("✓ Decision tree model trained and saved") 
        print("✓ Probability calibration completed")
        print("✓ Risk scoring system implemented")
        print("✓ Models saved using joblib")
        print("✓ Scored applications generated")
        print("✓ Performance metrics calculated")
        
        return lr_results, dt_results, calibration_results, scored_apps, performance_summary
        
    except Exception as e:
        print(f"\n❌ Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    lr_results, dt_results, calibration_results, scored_apps, performance_summary = main()
