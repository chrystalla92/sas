#!/usr/bin/env python3
"""
Bank Credit Risk Scoring Model - Step 4: Train Credit Model with Probability Calibration

Purpose: Train logistic regression model with feature selection, cross-validation, and probability calibration
Author: Risk Analytics Team
Date: 2025

This script performs:
- Feature engineering (if not already done)
- Multiple feature selection methods (RFE, SelectKBest, SelectFromModel)
- Hyperparameter optimization using GridSearchCV
- Stratified cross-validation
- Model training and evaluation
- Probability calibration using CalibratedClassifierCV with sigmoid method (Platt scaling)
- Calibration quality assessment using reliability diagrams, calibration curves, and Brier scores
- Bin-wise validation of calibrated probabilities vs observed default rates
- Model coefficient interpretation
- Model persistence (both uncalibrated and calibrated models)
"""

import numpy as np
import pandas as pd
import joblib
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, SelectKBest, SelectFromModel, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Plotting imports for reliability diagrams
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class CreditModelTrainer:
    """
    Comprehensive credit model training with feature selection and cross-validation.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.calibrated_model = None
        self.best_features = None
        self.feature_selection_results = {}
        self.cv_results = {}
        self.calibration_results = {}
        
    def load_data(self, data_path: str = "output/credit_data_sample.csv") -> pd.DataFrame:
        """
        Load credit data from CSV file.
        
        Args:
            data_path: Path to the credit data file
            
        Returns:
            DataFrame with credit data
        """
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Clean monetary columns that have $ and commas
        monetary_cols = ['monthly_income', 'loan_amount']
        for col in monetary_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].str.replace('[$,]', '', regex=True).astype(float)
        
        print(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for modeling (based on SAS feature engineering script).
        
        Args:
            df: Raw credit data DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        print("Engineering features...")
        
        # Create a copy to avoid modifying original data
        df_eng = df.copy()
        
        # Financial ratios
        df_eng['payment_to_income_ratio'] = (df_eng['monthly_payment'] / df_eng['monthly_income']) * 100
        df_eng['loan_to_income_ratio'] = df_eng['loan_amount'] / df_eng['annual_income']
        df_eng['debt_service_coverage'] = df_eng['monthly_income'] / df_eng['total_monthly_debt']
        
        # Handle infinite values
        df_eng['debt_service_coverage'] = df_eng['debt_service_coverage'].replace([np.inf, -np.inf], np.nan)
        df_eng['debt_service_coverage'] = df_eng['debt_service_coverage'].fillna(df_eng['debt_service_coverage'].median())
        
        # Employment stability score
        emp_stability_map = {
            'Full-time': 5,
            'Self-employed': 4,
            'Retired': 3,
            'Part-time': 2,
            'Unemployed': 1
        }
        df_eng['emp_stability'] = df_eng['employment_status'].map(emp_stability_map).fillna(1)
        
        # Weighted employment score
        df_eng['employment_score'] = df_eng['emp_stability'] * df_eng['employment_years']
        
        # Credit history quality score
        df_eng['credit_quality_score'] = (df_eng['credit_score'] - 
                                         (df_eng['num_late_payments'] * 50) - 
                                         (df_eng['previous_defaults'] * 150))
        
        # Age groups for risk assessment
        df_eng['age_group'] = pd.cut(df_eng['age'], 
                                   bins=[0, 25, 35, 45, 55, 65, 100], 
                                   labels=[1, 2, 3, 4, 5, 6])
        df_eng['age_group'] = df_eng['age_group'].astype(int)
        
        # Income stability indicator
        df_eng['income_stability'] = (df_eng['employment_years'] / df_eng['age']) * 100
        
        # Credit behavior score
        df_eng['credit_util_score'] = pd.cut(df_eng['credit_utilization'], 
                                           bins=[0, 30, 50, 70, 90, 100], 
                                           labels=[5, 4, 3, 2, 1]).astype(int)
        
        # Delinquency indicator
        df_eng['has_delinquency'] = (df_eng['num_late_payments'] > 0).astype(int)
        
        # High-risk flags
        df_eng['flag_high_dti'] = (df_eng['debt_to_income_ratio'] > 43).astype(int)
        df_eng['flag_low_credit'] = (df_eng['credit_score'] < 620).astype(int)
        df_eng['flag_high_util'] = (df_eng['credit_utilization'] > 75).astype(int)
        df_eng['flag_recent_default'] = (df_eng['previous_defaults'] > 0).astype(int)
        df_eng['flag_unstable_employment'] = (df_eng['employment_years'] < 2).astype(int)
        
        # Total risk flags
        risk_flags = ['flag_high_dti', 'flag_low_credit', 'flag_high_util', 
                     'flag_recent_default', 'flag_unstable_employment']
        df_eng['total_risk_flags'] = df_eng[risk_flags].sum(axis=1)
        
        # Loan affordability score
        df_eng['affordability_score'] = ((df_eng['monthly_income'] - df_eng['existing_monthly_debt']) / 
                                        df_eng['monthly_payment'])
        df_eng['affordability_score'] = df_eng['affordability_score'].replace([np.inf, -np.inf], np.nan)
        df_eng['affordability_score'] = df_eng['affordability_score'].fillna(df_eng['affordability_score'].median())
        
        # Credit age to loan ratio
        df_eng['credit_to_loan_years'] = df_eng['credit_history_years'] / (df_eng['loan_term_months'] / 12)
        
        # Create dummy variables for categorical features
        categorical_features = ['employment_status', 'education', 'home_ownership', 'loan_purpose']
        df_encoded = pd.get_dummies(df_eng, columns=categorical_features, prefix=categorical_features, drop_first=True)
        
        print(f"Feature engineering complete. Created {len(df_encoded.columns)} total features")
        return df_encoded
        
    def prepare_modeling_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare data for modeling by selecting relevant features and target.
        
        Args:
            df: Engineered features DataFrame
            
        Returns:
            Tuple of (features_df, target_series, feature_names)
        """
        # Define target variable
        target = 'default_flag'
        
        # Define features to exclude from modeling
        exclude_features = [
            'customer_id', 'application_date', 'risk_rating', 'interest_rate',
            target, 'monthly_payment', 'existing_monthly_debt', 'total_monthly_debt',
            'emp_stability'  # Remove intermediate calculation column
        ]
        
        # Select feature columns
        feature_cols = [col for col in df.columns if col not in exclude_features]
        
        # Prepare feature matrix and target vector
        X = df[feature_cols].copy()
        y = df[target].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        print(f"Prepared modeling dataset with {len(feature_cols)} features")
        print(f"Target variable distribution: {y.value_counts().to_dict()}")
        
        return X, y, feature_cols
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series, 
                                    n_features_range: List[int] = None) -> Dict[str, Any]:
        """
        Perform Recursive Feature Elimination with cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_features_range: Range of number of features to test
            
        Returns:
            Dictionary with RFE results
        """
        print("Performing Recursive Feature Elimination with CV...")
        
        if n_features_range is None:
            n_features_range = [5, 10, 15, 20, 25]
        
        # Use logistic regression with L2 regularization for RFE
        estimator = LogisticRegression(
            solver='liblinear', 
            random_state=self.random_state,
            max_iter=1000
        )
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        best_score = 0
        best_features = None
        best_n_features = None
        
        rfe_results = {}
        
        for n_features in n_features_range:
            # Perform RFE
            rfe = RFE(estimator=estimator, n_features_to_select=n_features)
            X_rfe = rfe.fit_transform(X, y)
            
            # Cross-validation score
            cv_scores = cross_val_score(estimator, X_rfe, y, cv=cv, scoring='roc_auc')
            mean_score = cv_scores.mean()
            
            rfe_results[n_features] = {
                'mean_cv_score': mean_score,
                'std_cv_score': cv_scores.std(),
                'selected_features': X.columns[rfe.support_].tolist(),
                'feature_rankings': dict(zip(X.columns, rfe.ranking_))
            }
            
            print(f"RFE with {n_features} features: CV ROC-AUC = {mean_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_features = X.columns[rfe.support_].tolist()
                best_n_features = n_features
        
        print(f"Best RFE configuration: {best_n_features} features with CV ROC-AUC = {best_score:.4f}")
        
        return {
            'method': 'RFE',
            'best_score': best_score,
            'best_features': best_features,
            'best_n_features': best_n_features,
            'all_results': rfe_results
        }
    
    def univariate_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                   k_range: List[int] = None) -> Dict[str, Any]:
        """
        Perform univariate feature selection using SelectKBest.
        
        Args:
            X: Feature matrix
            y: Target vector
            k_range: Range of k values to test
            
        Returns:
            Dictionary with SelectKBest results
        """
        print("Performing univariate feature selection with SelectKBest...")
        
        if k_range is None:
            k_range = [5, 10, 15, 20, 25]
        
        # Use f_classif for continuous features
        scoring_function = f_classif
        
        estimator = LogisticRegression(
            solver='liblinear', 
            random_state=self.random_state,
            max_iter=1000
        )
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        best_score = 0
        best_features = None
        best_k = None
        
        kbest_results = {}
        
        for k in k_range:
            if k > X.shape[1]:
                continue
                
            # Perform SelectKBest
            selector = SelectKBest(score_func=scoring_function, k=k)
            X_selected = selector.fit_transform(X, y)
            
            # Cross-validation score
            cv_scores = cross_val_score(estimator, X_selected, y, cv=cv, scoring='roc_auc')
            mean_score = cv_scores.mean()
            
            kbest_results[k] = {
                'mean_cv_score': mean_score,
                'std_cv_score': cv_scores.std(),
                'selected_features': X.columns[selector.get_support()].tolist(),
                'feature_scores': dict(zip(X.columns, selector.scores_))
            }
            
            print(f"SelectKBest with k={k}: CV ROC-AUC = {mean_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_features = X.columns[selector.get_support()].tolist()
                best_k = k
        
        print(f"Best SelectKBest configuration: k={best_k} with CV ROC-AUC = {best_score:.4f}")
        
        return {
            'method': 'SelectKBest',
            'best_score': best_score,
            'best_features': best_features,
            'best_k': best_k,
            'all_results': kbest_results
        }
    
    def l1_regularization_selection(self, X: pd.DataFrame, y: pd.Series, 
                                  C_range: List[float] = None) -> Dict[str, Any]:
        """
        Perform feature selection using L1 regularization (Lasso).
        
        Args:
            X: Feature matrix
            y: Target vector
            C_range: Range of C values for L1 regularization
            
        Returns:
            Dictionary with L1 regularization results
        """
        print("Performing L1 regularization feature selection...")
        
        if C_range is None:
            C_range = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        best_score = 0
        best_features = None
        best_C = None
        
        l1_results = {}
        
        for C in C_range:
            # Create L1 regularized logistic regression
            l1_model = LogisticRegression(
                penalty='l1',
                C=C,
                solver='liblinear',
                random_state=self.random_state,
                max_iter=1000
            )
            
            # Fit model and get feature selection
            selector = SelectFromModel(l1_model, prefit=False)
            X_selected = selector.fit_transform(X, y)
            
            # Skip if no features selected
            if X_selected.shape[1] == 0:
                print(f"L1 with C={C}: No features selected")
                continue
            
            # Cross-validation score
            cv_scores = cross_val_score(l1_model, X_selected, y, cv=cv, scoring='roc_auc')
            mean_score = cv_scores.mean()
            
            selected_features = X.columns[selector.get_support()].tolist()
            
            l1_results[C] = {
                'mean_cv_score': mean_score,
                'std_cv_score': cv_scores.std(),
                'selected_features': selected_features,
                'n_features': len(selected_features)
            }
            
            print(f"L1 with C={C}: {len(selected_features)} features, CV ROC-AUC = {mean_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_features = selected_features
                best_C = C
        
        print(f"Best L1 configuration: C={best_C} with CV ROC-AUC = {best_score:.4f}")
        
        return {
            'method': 'L1_Regularization',
            'best_score': best_score,
            'best_features': best_features,
            'best_C': best_C,
            'all_results': l1_results
        }
    
    def compare_feature_selection_methods(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Compare all feature selection methods and select the best one.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary with comparison results and best method
        """
        print("Comparing feature selection methods...")
        
        # Run all feature selection methods
        rfe_results = self.recursive_feature_elimination(X, y)
        kbest_results = self.univariate_feature_selection(X, y)
        l1_results = self.l1_regularization_selection(X, y)
        
        # Store all results
        self.feature_selection_results = {
            'RFE': rfe_results,
            'SelectKBest': kbest_results,
            'L1_Regularization': l1_results
        }
        
        # Compare results
        methods_comparison = {}
        for method_name, results in self.feature_selection_results.items():
            methods_comparison[method_name] = {
                'score': results['best_score'],
                'features': results['best_features'],
                'n_features': len(results['best_features'])
            }
        
        # Select best method
        best_method = max(methods_comparison.keys(), 
                         key=lambda k: methods_comparison[k]['score'])
        
        self.best_features = methods_comparison[best_method]['features']
        
        print("\nFeature Selection Comparison:")
        print("-" * 50)
        for method, results in methods_comparison.items():
            print(f"{method}: {results['score']:.4f} ROC-AUC ({results['n_features']} features)")
        print(f"\nBest method: {best_method}")
        print(f"Selected features ({len(self.best_features)}): {self.best_features[:10]}{'...' if len(self.best_features) > 10 else ''}")
        
        return {
            'best_method': best_method,
            'best_features': self.best_features,
            'comparison': methods_comparison,
            'all_results': self.feature_selection_results
        }
    
    def hyperparameter_optimization(self, X: pd.DataFrame, y: pd.Series, 
                                   selected_features: List[str]) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization using GridSearchCV.
        
        Args:
            X: Feature matrix
            y: Target vector
            selected_features: List of selected feature names
            
        Returns:
            Dictionary with optimization results
        """
        print("Performing hyperparameter optimization...")
        
        # Select only the chosen features
        X_selected = X[selected_features]
        
        # Define parameter grid
        param_grid = {
            'C': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],  # Compatible with both l1 and l2
            'max_iter': [1000]
        }
        
        # Create logistic regression model
        logistic_model = LogisticRegression(random_state=self.random_state)
        
        # Set up stratified cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=logistic_model,
            param_grid=param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_selected, y)
        
        # Get best results
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"Best parameters: {best_params}")
        print(f"Best CV ROC-AUC score: {best_score:.4f}")
        
        # Store CV results
        self.cv_results = {
            'best_params': best_params,
            'best_score': best_score,
            'grid_search_results': grid_search.cv_results_
        }
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'grid_search': grid_search
        }
    
    def train_final_model(self, X: pd.DataFrame, y: pd.Series, 
                         selected_features: List[str], 
                         best_params: Dict[str, Any]) -> LogisticRegression:
        """
        Train the final logistic regression model with optimal parameters.
        
        Args:
            X: Feature matrix
            y: Target vector
            selected_features: List of selected feature names
            best_params: Optimal hyperparameters
            
        Returns:
            Trained LogisticRegression model
        """
        print("Training final model...")
        
        # Select features and scale data
        X_selected = X[selected_features]
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Create and train final model
        self.model = LogisticRegression(
            random_state=self.random_state,
            **best_params
        )
        
        self.model.fit(X_scaled, y)
        
        # Evaluate on training data (for reference)
        train_predictions = self.model.predict_proba(X_scaled)[:, 1]
        train_auc = roc_auc_score(y, train_predictions)
        
        print(f"Training ROC-AUC: {train_auc:.4f}")
        
        return self.model
    
    def split_data_for_calibration(self, X: pd.DataFrame, y: pd.Series, 
                                  selected_features: List[str], 
                                  test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and calibration/validation sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            selected_features: List of selected feature names
            test_size: Proportion of data for calibration/validation
            
        Returns:
            Tuple of (X_train, X_cal, y_train, y_cal)
        """
        print(f"Splitting data for calibration (validation size: {test_size:.1%})...")
        
        # Select only the chosen features
        X_selected = X[selected_features]
        
        # Split data stratified by target
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_selected, y, 
            test_size=test_size, 
            stratify=y, 
            random_state=self.random_state
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Calibration/Validation set: {len(X_cal)} samples")
        print(f"Training set default rate: {y_train.mean():.3f}")
        print(f"Validation set default rate: {y_cal.mean():.3f}")
        
        return X_train, X_cal, y_train, y_cal
    
    def calibrate_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_cal: pd.DataFrame, y_cal: pd.Series,
                       selected_features: List[str], 
                       best_params: Dict[str, Any],
                       cv_folds: int = 5) -> Any:
        """
        Calibrate the logistic regression model using CalibratedClassifierCV with sigmoid method.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            X_cal: Calibration feature matrix  
            y_cal: Calibration target vector
            selected_features: List of selected feature names
            best_params: Optimal hyperparameters for base model
            cv_folds: Number of CV folds for calibration
            
        Returns:
            Calibrated classifier
        """
        print("Calibrating model using CalibratedClassifierCV with sigmoid method...")
        
        # Scale training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_cal_scaled = self.scaler.transform(X_cal)
        
        # Create base logistic regression model
        base_model = LogisticRegression(
            random_state=self.random_state,
            **best_params
        )
        
        # Train base model on training set
        base_model.fit(X_train_scaled, y_train)
        
        # Create calibrated classifier using sigmoid method (Platt scaling)
        calibrated_model = CalibratedClassifierCV(
            base_estimator=base_model,
            method='sigmoid',  # Platt scaling equivalent to SAS
            cv=cv_folds
        )
        
        # Fit calibration using the training data (CalibratedClassifierCV handles CV internally)
        calibrated_model.fit(X_train_scaled, y_train)
        
        # Store the calibrated model
        self.calibrated_model = calibrated_model
        
        # Get predictions from both models for comparison
        uncalibrated_probs = base_model.predict_proba(X_cal_scaled)[:, 1]
        calibrated_probs = calibrated_model.predict_proba(X_cal_scaled)[:, 1]
        
        print(f"Base model validation AUC: {roc_auc_score(y_cal, uncalibrated_probs):.4f}")
        print(f"Calibrated model validation AUC: {roc_auc_score(y_cal, calibrated_probs):.4f}")
        
        return calibrated_model
    
    def generate_calibration_curve(self, y_true: pd.Series, y_prob_uncalibrated: np.ndarray, 
                                 y_prob_calibrated: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
        """
        Generate calibration curve data using sklearn.calibration.calibration_curve.
        
        Args:
            y_true: True binary labels
            y_prob_uncalibrated: Predicted probabilities from uncalibrated model
            y_prob_calibrated: Predicted probabilities from calibrated model
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary with calibration curve data
        """
        print(f"Generating calibration curves with {n_bins} bins...")
        
        # Generate calibration curves
        fraction_pos_uncal, mean_pred_uncal = calibration_curve(
            y_true, y_prob_uncalibrated, n_bins=n_bins, strategy='uniform'
        )
        
        fraction_pos_cal, mean_pred_cal = calibration_curve(
            y_true, y_prob_calibrated, n_bins=n_bins, strategy='uniform'
        )
        
        calibration_data = {
            'uncalibrated': {
                'fraction_of_positives': fraction_pos_uncal,
                'mean_predicted_value': mean_pred_uncal
            },
            'calibrated': {
                'fraction_of_positives': fraction_pos_cal,
                'mean_predicted_value': mean_pred_cal
            },
            'n_bins': n_bins
        }
        
        # Calculate calibration error (deviation from diagonal)
        uncal_error = np.mean(np.abs(fraction_pos_uncal - mean_pred_uncal))
        cal_error = np.mean(np.abs(fraction_pos_cal - mean_pred_cal))
        
        calibration_data['uncalibrated_error'] = uncal_error
        calibration_data['calibrated_error'] = cal_error
        
        print(f"Uncalibrated model calibration error: {uncal_error:.4f}")
        print(f"Calibrated model calibration error: {cal_error:.4f}")
        
        return calibration_data
    
    def create_reliability_diagram(self, calibration_data: Dict[str, Any], 
                                 save_path: str = "output/reliability_diagram.png") -> None:
        """
        Create and save reliability diagram to visualize calibration quality.
        
        Args:
            calibration_data: Dictionary with calibration curve data
            save_path: Path to save the reliability diagram
        """
        print(f"Creating reliability diagram and saving to {save_path}...")
        
        # Create the reliability diagram
        plt.figure(figsize=(10, 8))
        
        # Plot perfect calibration line (diagonal)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        
        # Plot uncalibrated model
        plt.plot(
            calibration_data['uncalibrated']['mean_predicted_value'],
            calibration_data['uncalibrated']['fraction_of_positives'],
            'o-', color='red', linewidth=2, markersize=8,
            label=f"Uncalibrated (Error: {calibration_data['uncalibrated_error']:.3f})"
        )
        
        # Plot calibrated model
        plt.plot(
            calibration_data['calibrated']['mean_predicted_value'],
            calibration_data['calibrated']['fraction_of_positives'],
            's-', color='blue', linewidth=2, markersize=8,
            label=f"Calibrated (Error: {calibration_data['calibrated_error']:.3f})"
        )
        
        # Formatting
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Reliability Diagram (Calibration Curve)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        # Add annotations
        plt.text(0.02, 0.98, f'Bins: {calibration_data["n_bins"]}', 
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        print("Reliability diagram saved successfully!")
    
    def calculate_brier_score(self, y_true: pd.Series, y_prob_uncalibrated: np.ndarray, 
                            y_prob_calibrated: np.ndarray) -> Dict[str, float]:
        """
        Calculate Brier score to quantify calibration performance.
        
        Args:
            y_true: True binary labels
            y_prob_uncalibrated: Predicted probabilities from uncalibrated model
            y_prob_calibrated: Predicted probabilities from calibrated model
            
        Returns:
            Dictionary with Brier scores and improvement metrics
        """
        print("Calculating Brier scores for calibration assessment...")
        
        # Calculate Brier scores
        brier_uncalibrated = brier_score_loss(y_true, y_prob_uncalibrated)
        brier_calibrated = brier_score_loss(y_true, y_prob_calibrated)
        
        # Calculate improvement
        brier_improvement = brier_uncalibrated - brier_calibrated
        brier_improvement_pct = (brier_improvement / brier_uncalibrated) * 100
        
        results = {
            'brier_uncalibrated': brier_uncalibrated,
            'brier_calibrated': brier_calibrated,
            'brier_improvement': brier_improvement,
            'brier_improvement_pct': brier_improvement_pct
        }
        
        print(f"Uncalibrated Brier Score: {brier_uncalibrated:.4f}")
        print(f"Calibrated Brier Score: {brier_calibrated:.4f}")
        print(f"Brier Score Improvement: {brier_improvement:.4f} ({brier_improvement_pct:.1f}%)")
        
        return results
    
    def validate_probability_bins(self, y_true: pd.Series, y_prob_calibrated: np.ndarray, 
                                 n_bins: int = 10) -> pd.DataFrame:
        """
        Validate that calibrated probabilities align with observed default rates across probability bins.
        
        Args:
            y_true: True binary labels
            y_prob_calibrated: Predicted probabilities from calibrated model
            n_bins: Number of probability bins to create
            
        Returns:
            DataFrame with bin-wise validation results
        """
        print(f"Validating calibrated probabilities across {n_bins} probability bins...")
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        # Assign each prediction to a bin
        bin_indices = np.digitize(y_prob_calibrated, bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Handle edge cases
        
        # Calculate statistics for each bin
        validation_results = []
        
        for i in range(n_bins):
            bin_mask = (bin_indices == i)
            
            if np.sum(bin_mask) == 0:
                continue
                
            bin_count = np.sum(bin_mask)
            observed_rate = np.mean(y_true[bin_mask])
            predicted_rate = np.mean(y_prob_calibrated[bin_mask])
            
            # Calculate confidence intervals for observed rate (Wilson score interval)
            if bin_count > 0:
                p = observed_rate
                n = bin_count
                z = 1.96  # 95% confidence interval
                
                denominator = 1 + z**2/n
                center = (p + z**2/(2*n)) / denominator
                width = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
                
                ci_lower = max(0, center - width)
                ci_upper = min(1, center + width)
            else:
                ci_lower = ci_upper = np.nan
            
            validation_results.append({
                'bin_number': i + 1,
                'bin_range': f"[{bin_boundaries[i]:.2f}, {bin_boundaries[i+1]:.2f})",
                'bin_center': bin_centers[i],
                'sample_count': bin_count,
                'observed_default_rate': observed_rate,
                'predicted_default_rate': predicted_rate,
                'absolute_error': abs(observed_rate - predicted_rate),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'within_ci': ci_lower <= predicted_rate <= ci_upper if not np.isnan(ci_lower) else False
            })
        
        validation_df = pd.DataFrame(validation_results)
        
        # Calculate overall validation statistics
        overall_mae = validation_df['absolute_error'].mean()
        bins_within_ci = validation_df['within_ci'].sum()
        total_bins = len(validation_df)
        
        print(f"Overall Mean Absolute Error: {overall_mae:.4f}")
        print(f"Bins within 95% CI: {bins_within_ci}/{total_bins} ({bins_within_ci/total_bins*100:.1f}%)")
        
        # Display bin validation summary
        print("\nBin Validation Summary:")
        print("-" * 100)
        print(f"{'Bin':<4} {'Range':<15} {'Count':<6} {'Observed':<10} {'Predicted':<10} {'Error':<8} {'In CI':<6}")
        print("-" * 100)
        
        for _, row in validation_df.head(n_bins).iterrows():
            print(f"{int(row['bin_number']):<4} "
                  f"{row['bin_range']:<15} "
                  f"{int(row['sample_count']):<6} "
                  f"{row['observed_default_rate']:.3f}     "
                  f"{row['predicted_default_rate']:.3f}      "
                  f"{row['absolute_error']:.3f}    "
                  f"{'Yes' if row['within_ci'] else 'No':<6}")
        
        return validation_df
    
    def evaluate_calibration(self, X_cal: pd.DataFrame, y_cal: pd.Series) -> Dict[str, Any]:
        """
        Comprehensive calibration evaluation including all metrics and visualizations.
        
        Args:
            X_cal: Calibration/validation feature matrix
            y_cal: Calibration/validation target vector
            
        Returns:
            Dictionary with comprehensive calibration evaluation results
        """
        if self.model is None or self.calibrated_model is None:
            raise ValueError("Both uncalibrated and calibrated models must be trained first")
        
        print("Performing comprehensive calibration evaluation...")
        
        # Scale validation data
        X_cal_scaled = self.scaler.transform(X_cal)
        
        # Get predictions from both models
        uncalibrated_probs = self.model.predict_proba(X_cal_scaled)[:, 1]
        calibrated_probs = self.calibrated_model.predict_proba(X_cal_scaled)[:, 1]
        
        # Generate calibration curves
        calibration_data = self.generate_calibration_curve(
            y_cal, uncalibrated_probs, calibrated_probs
        )
        
        # Create reliability diagram
        self.create_reliability_diagram(calibration_data)
        
        # Calculate Brier scores
        brier_results = self.calculate_brier_score(
            y_cal, uncalibrated_probs, calibrated_probs
        )
        
        # Validate probability bins
        bin_validation = self.validate_probability_bins(y_cal, calibrated_probs)
        
        # Calculate AUC for both models
        uncalibrated_auc = roc_auc_score(y_cal, uncalibrated_probs)
        calibrated_auc = roc_auc_score(y_cal, calibrated_probs)
        
        # Compile comprehensive results
        evaluation_results = {
            'calibration_curves': calibration_data,
            'brier_scores': brier_results,
            'bin_validation': bin_validation,
            'auc_scores': {
                'uncalibrated_auc': uncalibrated_auc,
                'calibrated_auc': calibrated_auc,
                'auc_change': calibrated_auc - uncalibrated_auc
            },
            'predictions': {
                'uncalibrated_probs': uncalibrated_probs,
                'calibrated_probs': calibrated_probs,
                'true_labels': y_cal.values
            }
        }
        
        # Store results
        self.calibration_results = evaluation_results
        
        # Print summary
        print("\n" + "="*80)
        print("CALIBRATION EVALUATION SUMMARY")
        print("="*80)
        print(f"Validation samples: {len(y_cal)}")
        print(f"Uncalibrated AUC: {uncalibrated_auc:.4f}")
        print(f"Calibrated AUC: {calibrated_auc:.4f} (Δ: {calibrated_auc - uncalibrated_auc:+.4f})")
        print(f"Uncalibrated Brier Score: {brier_results['brier_uncalibrated']:.4f}")
        print(f"Calibrated Brier Score: {brier_results['brier_calibrated']:.4f}")
        print(f"Brier Score Improvement: {brier_results['brier_improvement_pct']:.1f}%")
        print(f"Calibration Error (Uncalibrated): {calibration_data['uncalibrated_error']:.4f}")
        print(f"Calibration Error (Calibrated): {calibration_data['calibrated_error']:.4f}")
        
        overall_mae = bin_validation['absolute_error'].mean()
        bins_within_ci = bin_validation['within_ci'].sum()
        total_bins = len(bin_validation)
        print(f"Bin Validation MAE: {overall_mae:.4f}")
        print(f"Bins within 95% CI: {bins_within_ci}/{total_bins} ({bins_within_ci/total_bins*100:.1f}%)")
        print("="*80)
        
        return evaluation_results
    
    def interpret_model_coefficients(self, selected_features: List[str]) -> pd.DataFrame:
        """
        Extract and interpret model coefficients.
        
        Args:
            selected_features: List of selected feature names
            
        Returns:
            DataFrame with coefficient interpretations
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        print("Interpreting model coefficients...")
        
        # Get coefficients
        coefficients = self.model.coef_[0]
        
        # Create coefficient DataFrame
        coef_df = pd.DataFrame({
            'feature': selected_features,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients),
            'odds_ratio': np.exp(coefficients)
        })
        
        # Sort by absolute coefficient value
        coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
        
        # Add interpretation
        def interpret_coefficient(row):
            if row['coefficient'] > 0:
                return f"Increases default risk (OR: {row['odds_ratio']:.3f})"
            else:
                return f"Decreases default risk (OR: {row['odds_ratio']:.3f})"
        
        coef_df['interpretation'] = coef_df.apply(interpret_coefficient, axis=1)
        
        print(f"\nModel Intercept: {self.model.intercept_[0]:.4f}")
        print("\nTop 10 Most Important Features:")
        print("-" * 70)
        for idx, row in coef_df.head(10).iterrows():
            print(f"{row['feature']:<25}: {row['coefficient']:>8.4f} ({row['interpretation']})")
        
        return coef_df
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series, 
                      selected_features: List[str]) -> Dict[str, Any]:
        """
        Evaluate the trained model using cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            selected_features: List of selected feature names
            
        Returns:
            Dictionary with evaluation results
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        print("Evaluating model performance...")
        
        # Select features and scale data
        X_selected = X[selected_features]
        X_scaled = self.scaler.transform(X_selected)
        
        # Stratified cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Calculate cross-validation scores
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='roc_auc')
        
        # Make predictions for additional metrics
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Calculate additional metrics
        auc_score = roc_auc_score(y, probabilities)
        
        evaluation_results = {
            'cv_scores': cv_scores,
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'auc_score': auc_score,
            'classification_report': classification_report(y, predictions),
            'confusion_matrix': confusion_matrix(y, predictions)
        }
        
        print(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Full dataset ROC-AUC: {auc_score:.4f}")
        print("\nClassification Report:")
        print(evaluation_results['classification_report'])
        
        return evaluation_results
    
    def save_model(self, model_path: str = "output/trained_credit_model.pkl", 
                  calibrated_model_path: str = "output/calibrated_credit_model.pkl",
                  metadata_path: str = "output/model_metadata.pkl"):
        """
        Save the trained models (both uncalibrated and calibrated) and metadata using joblib.
        
        Args:
            model_path: Path to save the uncalibrated model
            calibrated_model_path: Path to save the calibrated model
            metadata_path: Path to save metadata
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        print(f"Saving uncalibrated model to {model_path}...")
        print(f"Saving calibrated model to {calibrated_model_path}...")
        
        # Create output directory if it doesn't exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save uncalibrated model
        joblib.dump(self.model, model_path)
        
        # Save calibrated model (if available)
        if self.calibrated_model is not None:
            joblib.dump(self.calibrated_model, calibrated_model_path)
            print("Calibrated model saved successfully!")
        else:
            print("Warning: No calibrated model to save")
        
        # Prepare metadata
        metadata = {
            'model_type': 'LogisticRegression',
            'has_calibrated_model': self.calibrated_model is not None,
            'best_features': self.best_features,
            'scaler': self.scaler,
            'feature_selection_results': self.feature_selection_results,
            'cv_results': self.cv_results,
            'calibration_results': self.calibration_results,
            'training_date': datetime.now().isoformat(),
            'random_state': self.random_state,
            'model_paths': {
                'uncalibrated': model_path,
                'calibrated': calibrated_model_path if self.calibrated_model is not None else None
            }
        }
        
        # Save metadata
        joblib.dump(metadata, metadata_path)
        
        print(f"Model metadata saved to {metadata_path}")
        print("Model training and saving completed successfully!")


def main():
    """
    Main function to execute the complete model training pipeline with calibration.
    """
    print("=" * 80)
    print("CREDIT RISK MODEL TRAINING PIPELINE WITH PROBABILITY CALIBRATION")
    print("=" * 80)
    
    # Initialize trainer
    trainer = CreditModelTrainer(random_state=42)
    
    try:
        # Step 1: Load data
        df = trainer.load_data("output/credit_data_sample.csv")
        
        # Step 2: Engineer features
        df_engineered = trainer.engineer_features(df)
        
        # Step 3: Prepare modeling data
        X, y, feature_names = trainer.prepare_modeling_data(df_engineered)
        
        # Step 4: Compare feature selection methods
        feature_selection_results = trainer.compare_feature_selection_methods(X, y)
        
        # Step 5: Hyperparameter optimization
        optimization_results = trainer.hyperparameter_optimization(X, y, trainer.best_features)
        
        # Step 6: Split data for calibration evaluation
        X_train, X_cal, y_train, y_cal = trainer.split_data_for_calibration(
            X, y, trainer.best_features, test_size=0.2
        )
        
        # Step 7: Train base model on training set
        print("Training base model on training set...")
        final_model = trainer.train_final_model(X_train, y_train, trainer.best_features, 
                                              optimization_results['best_params'])
        
        # Step 8: Calibrate model using sigmoid method (Platt scaling)
        calibrated_model = trainer.calibrate_model(
            X_train, y_train, X_cal, y_cal, 
            trainer.best_features, optimization_results['best_params']
        )
        
        # Step 9: Comprehensive calibration evaluation
        calibration_evaluation = trainer.evaluate_calibration(X_cal, y_cal)
        
        # Step 10: Interpret coefficients (from base model)
        coefficient_df = trainer.interpret_model_coefficients(trainer.best_features)
        
        # Step 11: Evaluate uncalibrated model performance
        print("\nEvaluating uncalibrated model performance...")
        evaluation_results = trainer.evaluate_model(X_cal, y_cal, trainer.best_features)
        
        # Step 12: Save models and results
        trainer.save_model()
        
        # Save additional results
        coefficient_df.to_csv("output/model_coefficients.csv", index=False)
        
        # Save calibration evaluation results
        calibration_evaluation['bin_validation'].to_csv("output/calibration_bin_validation.csv", index=False)
        
        # Save calibration summary
        calibration_summary = {
            'validation_samples': len(y_cal),
            'uncalibrated_auc': calibration_evaluation['auc_scores']['uncalibrated_auc'],
            'calibrated_auc': calibration_evaluation['auc_scores']['calibrated_auc'],
            'auc_change': calibration_evaluation['auc_scores']['auc_change'],
            'brier_uncalibrated': calibration_evaluation['brier_scores']['brier_uncalibrated'],
            'brier_calibrated': calibration_evaluation['brier_scores']['brier_calibrated'],
            'brier_improvement_pct': calibration_evaluation['brier_scores']['brier_improvement_pct'],
            'calibration_error_uncalibrated': calibration_evaluation['calibration_curves']['uncalibrated_error'],
            'calibration_error_calibrated': calibration_evaluation['calibration_curves']['calibrated_error']
        }
        
        calibration_summary_df = pd.DataFrame([calibration_summary])
        calibration_summary_df.to_csv("output/calibration_summary.csv", index=False)
        
        print("=" * 80)
        print("MODEL TRAINING WITH CALIBRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("Files saved:")
        print("- output/trained_credit_model.pkl (uncalibrated model)")
        print("- output/calibrated_credit_model.pkl (calibrated model for production)")
        print("- output/model_metadata.pkl (model metadata including calibration results)")
        print("- output/model_coefficients.csv (coefficient interpretations)")
        print("- output/reliability_diagram.png (calibration visualization)")
        print("- output/calibration_bin_validation.csv (bin-wise calibration validation)")
        print("- output/calibration_summary.csv (calibration performance summary)")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
