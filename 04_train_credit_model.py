#!/usr/bin/env python3
"""
Bank Credit Risk Scoring Model - Step 4: Train Credit Model

Purpose: Train logistic regression model with feature selection and cross-validation
Author: Risk Analytics Team
Date: 2025

This script performs:
- Feature engineering (if not already done)
- Multiple feature selection methods (RFE, SelectKBest, SelectFromModel)
- Hyperparameter optimization using GridSearchCV
- Stratified cross-validation
- Model training and evaluation
- Model coefficient interpretation
- Model persistence
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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

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
        self.best_features = None
        self.feature_selection_results = {}
        self.cv_results = {}
        
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
                  metadata_path: str = "output/model_metadata.pkl"):
        """
        Save the trained model and metadata using joblib.
        
        Args:
            model_path: Path to save the model
            metadata_path: Path to save metadata
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        print(f"Saving model to {model_path}...")
        
        # Create output directory if it doesn't exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_path)
        
        # Prepare metadata
        metadata = {
            'model_type': 'LogisticRegression',
            'best_features': self.best_features,
            'scaler': self.scaler,
            'feature_selection_results': self.feature_selection_results,
            'cv_results': self.cv_results,
            'training_date': datetime.now().isoformat(),
            'random_state': self.random_state
        }
        
        # Save metadata
        joblib.dump(metadata, metadata_path)
        
        print(f"Model metadata saved to {metadata_path}")
        print("Model training and saving completed successfully!")


def main():
    """
    Main function to execute the complete model training pipeline.
    """
    print("=" * 80)
    print("CREDIT RISK MODEL TRAINING PIPELINE")
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
        
        # Step 6: Train final model
        final_model = trainer.train_final_model(X, y, trainer.best_features, 
                                              optimization_results['best_params'])
        
        # Step 7: Interpret coefficients
        coefficient_df = trainer.interpret_model_coefficients(trainer.best_features)
        
        # Step 8: Evaluate model
        evaluation_results = trainer.evaluate_model(X, y, trainer.best_features)
        
        # Step 9: Save model
        trainer.save_model()
        
        # Save coefficient interpretations
        coefficient_df.to_csv("output/model_coefficients.csv", index=False)
        
        print("=" * 80)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
