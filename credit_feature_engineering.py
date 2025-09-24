#!/usr/bin/env python3
"""
Bank Credit Risk Scoring Model - Step 3: Feature Engineering

Purpose: Create derived features and transform variables for modeling
Author: Risk Analytics Team
Date: 2025

This script performs feature engineering including:
- Creating risk indicators and financial ratios
- Implementing business rule-based risk flags
- Feature scaling and transformation
- One-hot encoding for categorical variables
- Feature selection and importance analysis
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Tuple, Dict, List
import logging

# sklearn imports
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class CreditFeatureEngineer:
    """Credit risk feature engineering pipeline."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_selector = None
        self.feature_names = None
        self.feature_importance = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load training data from CSV file."""
        logger.info(f"Loading data from {file_path}")
        
        # Try different possible file locations
        possible_paths = [
            file_path,
            f"output/{file_path}",
            f"output/credit_train.csv", 
            f"output/credit_data_train.csv",
            f"output/credit_data_sample.csv"  # fallback to sample
        ]
        
        data = None
        for path in possible_paths:
            try:
                if Path(path).exists():
                    data = pd.read_csv(path)
                    logger.info(f"Successfully loaded data from {path}")
                    logger.info(f"Data shape: {data.shape}")
                    break
            except Exception as e:
                logger.warning(f"Could not load from {path}: {e}")
                continue
                
        if data is None:
            raise FileNotFoundError(f"Could not find training data in any of: {possible_paths}")
            
        return data
    
    def create_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived financial ratio features."""
        logger.info("Creating financial ratios...")
        
        df = df.copy()
        
        # Convert currency strings to numeric if needed
        for col in ['monthly_income', 'loan_amount']:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace('$', '').str.replace(',', '').astype(float)
        
        # Financial ratios from SAS script
        df['payment_to_income_ratio'] = (df['monthly_payment'] / df['monthly_income']) * 100
        df['loan_to_income_ratio'] = df['loan_amount'] / df['annual_income'] 
        df['debt_service_coverage'] = df['monthly_income'] / df['total_monthly_debt']
        
        # Handle division by zero
        df['debt_service_coverage'] = df['debt_service_coverage'].replace([np.inf, -np.inf], np.nan)
        df['debt_service_coverage'].fillna(0, inplace=True)
        
        logger.info("Financial ratios created successfully")
        return df
    
    def create_employment_stability(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create employment stability scoring."""
        logger.info("Creating employment stability scores...")
        
        df = df.copy()
        
        # Employment stability score (1-5 scale)
        employment_stability_map = {
            'Full-time': 5,
            'Self-employed': 4, 
            'Retired': 3,
            'Part-time': 2,
            'Unemployed': 1
        }
        
        df['emp_stability'] = df['employment_status'].map(employment_stability_map)
        df['employment_score'] = df['emp_stability'] * df['employment_years']
        
        # Income stability indicator
        df['income_stability'] = (df['employment_years'] / df['age']) * 100
        
        logger.info("Employment stability scores created successfully")
        return df
    
    def create_credit_quality_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create credit quality and behavioral scores."""
        logger.info("Creating credit quality scores...")
        
        df = df.copy()
        
        # Credit quality score (adjusted for negatives)
        df['credit_quality_score'] = (df['credit_score'] - 
                                    (df['num_late_payments'] * 50) - 
                                    (df['previous_defaults'] * 150))
        
        # Credit utilization scoring (1-5 scale)
        conditions = [
            df['credit_utilization'] < 30,
            df['credit_utilization'] < 50,
            df['credit_utilization'] < 70,
            df['credit_utilization'] < 90,
            df['credit_utilization'] >= 90
        ]
        choices = [5, 4, 3, 2, 1]
        df['credit_util_score'] = np.select(conditions, choices, default=1)
        
        # Loan affordability score
        df['affordability_score'] = (df['monthly_income'] - df['total_monthly_debt']) / df['monthly_payment']
        df['affordability_score'] = df['affordability_score'].replace([np.inf, -np.inf], np.nan)
        df['affordability_score'].fillna(0, inplace=True)
        
        # Credit age to loan ratio
        df['credit_to_loan_years'] = df['credit_history_years'] / (df['loan_term_months'] / 12)
        df['credit_to_loan_years'] = df['credit_to_loan_years'].replace([np.inf, -np.inf], np.nan)
        df['credit_to_loan_years'].fillna(0, inplace=True)
        
        logger.info("Credit quality scores created successfully")
        return df
    
    def create_age_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create age group categorization."""
        logger.info("Creating age groups...")
        
        df = df.copy()
        
        # Age groups for risk assessment
        conditions = [
            df['age'] < 25,
            df['age'] < 35,
            df['age'] < 45,
            df['age'] < 55,
            df['age'] < 65,
            df['age'] >= 65
        ]
        choices = [1, 2, 3, 4, 5, 6]
        df['age_group'] = np.select(conditions, choices, default=6)
        
        logger.info("Age groups created successfully")
        return df
    
    def create_risk_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create business rule-based risk flags."""
        logger.info("Creating risk flags...")
        
        df = df.copy()
        
        # Delinquency indicators
        df['has_delinquency'] = (df['num_late_payments'] > 0).astype(int)
        
        # High-risk flags based on business rules
        df['flag_high_dti'] = (df['debt_to_income_ratio'] > 43).astype(int)
        df['flag_low_credit'] = (df['credit_score'] < 620).astype(int)
        df['flag_high_util'] = (df['credit_utilization'] > 75).astype(int)
        df['flag_recent_default'] = (df['previous_defaults'] > 0).astype(int)
        df['flag_unstable_employment'] = (df['employment_years'] < 2).astype(int)
        
        # Total risk flags count
        risk_columns = ['flag_high_dti', 'flag_low_credit', 'flag_high_util', 
                       'flag_recent_default', 'flag_unstable_employment']
        df['total_risk_flags'] = df[risk_columns].sum(axis=1)
        
        logger.info("Risk flags created successfully")
        return df

    def create_interaction_terms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction terms between key variables."""
        logger.info("Creating interaction terms...")
        
        df = df.copy()
        
        # Key interaction terms
        df['income_credit_score'] = df['monthly_income'] * df['credit_score']
        df['age_employment_years'] = df['age'] * df['employment_years']
        
        # Additional meaningful interactions
        df['credit_score_utilization'] = df['credit_score'] * (100 - df['credit_utilization'])
        df['income_dti_ratio'] = df['monthly_income'] * (100 - df['debt_to_income_ratio'])
        df['loan_credit_history'] = df['loan_amount'] * df['credit_history_years']
        
        logger.info("Interaction terms created successfully")
        return df
    
    def create_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate composite financial stability and creditworthiness scores."""
        logger.info("Creating composite scores...")
        
        df = df.copy()
        
        # Financial stability index (0-100 scale)
        # Components: employment stability, income level, debt management
        employment_component = (df['emp_stability'] / 5) * 30  # 30% weight
        income_component = np.minimum(df['monthly_income'] / 10000, 1) * 25  # 25% weight, capped
        debt_management = np.maximum(0, (50 - df['debt_to_income_ratio']) / 50) * 25  # 25% weight
        savings_capacity = np.maximum(0, df['affordability_score'] / 10) * 20  # 20% weight, normalized
        
        df['financial_stability_index'] = (employment_component + income_component + 
                                         debt_management + savings_capacity)
        
        # Creditworthiness score (0-100 scale)
        # Components: credit score, payment history, credit age, utilization
        credit_score_component = (df['credit_score'] - 300) / (850 - 300) * 40  # 40% weight
        payment_history_component = np.maximum(0, (5 - df['num_late_payments']) / 5) * 25  # 25% weight
        credit_age_component = np.minimum(df['credit_history_years'] / 20, 1) * 20  # 20% weight
        utilization_component = np.maximum(0, (100 - df['credit_utilization']) / 100) * 15  # 15% weight
        
        df['creditworthiness_score'] = (credit_score_component + payment_history_component +
                                       credit_age_component + utilization_component)
        
        logger.info("Composite scores created successfully")
        return df
    
    def apply_categorical_encoding(self, df: pd.DataFrame, fit_encoders=True) -> pd.DataFrame:
        """Apply one-hot encoding to categorical variables."""
        logger.info("Applying categorical encoding...")
        
        df = df.copy()
        
        categorical_columns = ['employment_status', 'education', 'home_ownership', 'loan_purpose']
        
        if fit_encoders:
            self.encoders = {}
        
        for col in categorical_columns:
            if col in df.columns:
                if fit_encoders:
                    # Create and fit encoder
                    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                    encoded = encoder.fit_transform(df[[col]])
                    self.encoders[col] = encoder
                else:
                    # Use existing encoder
                    encoder = self.encoders.get(col)
                    if encoder:
                        encoded = encoder.transform(df[[col]])
                    else:
                        continue
                
                # Create feature names
                if hasattr(encoder, 'get_feature_names_out'):
                    feature_names = encoder.get_feature_names_out([col])
                else:
                    # Fallback for older versions
                    categories = encoder.categories_[0][1:]  # drop first category
                    feature_names = [f"{col}_{cat}" for cat in categories]
                
                # Add encoded features to dataframe
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
                df = pd.concat([df, encoded_df], axis=1)
        
        logger.info("Categorical encoding completed successfully")
        return df
    
    def apply_feature_scaling(self, df: pd.DataFrame, fit_scaler=True) -> pd.DataFrame:
        """Apply feature normalization/standardization for continuous variables."""
        logger.info("Applying feature scaling...")
        
        df = df.copy()
        
        # Continuous features to scale
        continuous_features = [
            'age', 'employment_years', 'monthly_income', 'annual_income', 'loan_amount',
            'credit_utilization', 'debt_to_income_ratio', 'num_late_payments',
            'payment_to_income_ratio', 'loan_to_income_ratio', 'employment_score',
            'credit_quality_score', 'affordability_score', 'income_credit_score',
            'age_employment_years', 'financial_stability_index', 'creditworthiness_score'
        ]
        
        # Filter to only columns that exist in dataframe
        available_features = [col for col in continuous_features if col in df.columns]
        
        if fit_scaler:
            scaled_data = self.scaler.fit_transform(df[available_features])
        else:
            scaled_data = self.scaler.transform(df[available_features])
        
        # Create scaled feature names
        scaled_feature_names = [f"{col}_scaled" for col in available_features]
        scaled_df = pd.DataFrame(scaled_data, columns=scaled_feature_names, index=df.index)
        
        # Add scaled features to dataframe
        df = pd.concat([df, scaled_df], axis=1)
        
        logger.info("Feature scaling completed successfully")
        return df
    
    def apply_feature_selection(self, df: pd.DataFrame, target_col='default_flag', k=20) -> pd.DataFrame:
        """Add univariate feature selection using SelectKBest."""
        logger.info(f"Applying feature selection (selecting top {k} features)...")
        
        df = df.copy()
        
        # Get numeric features for selection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # Remove ID columns
        numeric_cols = [col for col in numeric_cols if 'customer_id' not in col.lower()]
        
        X = df[numeric_cols]
        y = df[target_col]
        
        # Use f_classif for classification
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, len(numeric_cols)))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        selected_features = [col for col, selected in zip(numeric_cols, selected_mask) if selected]
        
        # Store feature scores for reporting
        feature_scores = self.feature_selector.scores_
        self.feature_importance['selectkbest'] = dict(zip(numeric_cols, feature_scores))
        
        logger.info(f"Selected {len(selected_features)} features: {selected_features[:10]}...")
        return df, selected_features
    
    def calculate_feature_importance(self, df: pd.DataFrame, target_col='default_flag') -> Dict:
        """Generate feature importance analysis using correlation and mutual information."""
        logger.info("Calculating feature importance...")
        
        # Get numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # Remove ID columns
        numeric_cols = [col for col in numeric_cols if 'customer_id' not in col.lower()]
        
        X = df[numeric_cols]
        y = df[target_col]
        
        # Correlation with target
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        self.feature_importance['correlation'] = correlations.to_dict()
        
        # Mutual information
        mi_scores = mutual_info_classif(X.fillna(0), y, random_state=self.random_state)
        mi_importance = dict(zip(numeric_cols, mi_scores))
        mi_importance = dict(sorted(mi_importance.items(), key=lambda x: x[1], reverse=True))
        self.feature_importance['mutual_info'] = mi_importance
        
        logger.info("Feature importance analysis completed")
        return self.feature_importance
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Complete feature engineering pipeline - fit and transform."""
        logger.info("Starting complete feature engineering pipeline...")
        
        # Step 1: Load and validate data
        if 'default_flag' not in df.columns:
            raise ValueError("Target column 'default_flag' not found in data")
        
        # Step 2: Create financial ratios
        df = self.create_financial_ratios(df)
        
        # Step 3: Create employment stability scores
        df = self.create_employment_stability(df)
        
        # Step 4: Create credit quality scores
        df = self.create_credit_quality_scores(df)
        
        # Step 5: Create age groups
        df = self.create_age_groups(df)
        
        # Step 6: Create risk flags
        df = self.create_risk_flags(df)
        
        # Step 7: Create interaction terms
        df = self.create_interaction_terms(df)
        
        # Step 8: Create composite scores
        df = self.create_composite_scores(df)
        
        # Step 9: Apply categorical encoding
        df = self.apply_categorical_encoding(df, fit_encoders=True)
        
        # Step 10: Apply feature scaling
        df = self.apply_feature_scaling(df, fit_scaler=True)
        
        # Step 11: Feature selection
        df, selected_features = self.apply_feature_selection(df)
        
        # Step 12: Calculate feature importance
        self.calculate_feature_importance(df)
        
        logger.info("Feature engineering pipeline completed successfully")
        return df, selected_features
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline."""
        logger.info("Transforming new data using fitted pipeline...")
        
        # Apply same transformations as fit_transform but without fitting
        df = self.create_financial_ratios(df)
        df = self.create_employment_stability(df)
        df = self.create_credit_quality_scores(df)
        df = self.create_age_groups(df)
        df = self.create_risk_flags(df)
        df = self.create_interaction_terms(df)
        df = self.create_composite_scores(df)
        df = self.apply_categorical_encoding(df, fit_encoders=False)
        df = self.apply_feature_scaling(df, fit_scaler=False)
        
        logger.info("Data transformation completed")
        return df

def main():
    """Main execution function."""
    logger.info("Starting Credit Feature Engineering Pipeline")
    
    # Initialize feature engineer
    fe = CreditFeatureEngineer(random_state=42)
    
    # Load training data
    try:
        train_data = fe.load_data("credit_train.csv")
    except FileNotFoundError:
        logger.warning("Training data not found, using sample data for demonstration")
        train_data = fe.load_data("credit_data_sample.csv")
        # Create a train/validation split for demo
        train_data, val_data = train_test_split(
            train_data, test_size=0.3, random_state=42, stratify=train_data['default_flag']
        )
        logger.info(f"Created train/validation split: {train_data.shape}, {val_data.shape}")
    
    # Apply feature engineering
    engineered_data, selected_features = fe.fit_transform(train_data)
    
    # Save results to output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Export engineered features
    output_file = output_dir / "credit_features_engineered.csv"
    engineered_data.to_csv(output_file, index=False)
    logger.info(f"Engineered features saved to {output_file}")
    
    # Save selected features for model training
    selected_data = engineered_data[selected_features + ['customer_id', 'default_flag']]
    selected_output_file = output_dir / "credit_features_selected.csv"
    selected_data.to_csv(selected_output_file, index=False)
    logger.info(f"Selected features saved to {selected_output_file}")
    
    # Save feature engineering objects
    joblib.dump(fe.scaler, output_dir / "feature_scaler.pkl")
    joblib.dump(fe.encoders, output_dir / "feature_encoders.pkl")
    joblib.dump(fe.feature_selector, output_dir / "feature_selector.pkl")
    logger.info("Feature engineering objects saved")
    
    # Create feature engineering summary report
    create_summary_report(engineered_data, fe.feature_importance, selected_features, output_dir)
    
    logger.info("Feature engineering pipeline completed successfully!")

def create_summary_report(data: pd.DataFrame, feature_importance: Dict, 
                         selected_features: List[str], output_dir: Path):
    """Create feature engineering summary report with transformation details."""
    logger.info("Creating feature engineering summary report...")
    
    report_file = output_dir / "feature_engineering_report.md"
    
    with open(report_file, 'w') as f:
        f.write("# Credit Risk Feature Engineering Report\n\n")
        
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Dataset Summary\n\n")
        f.write(f"- **Total Samples:** {len(data):,}\n")
        f.write(f"- **Total Features:** {len(data.columns)}\n")
        f.write(f"- **Selected Features:** {len(selected_features)}\n")
        f.write(f"- **Default Rate:** {data['default_flag'].mean():.2%}\n\n")
        
        f.write("## Feature Categories\n\n")
        
        # Risk flags
        risk_flags = [col for col in data.columns if col.startswith('flag_')]
        f.write(f"### Risk Flags ({len(risk_flags)} features)\n")
        for flag in risk_flags[:10]:  # Top 10
            if flag in data.columns:
                rate = data[flag].mean()
                f.write(f"- **{flag}**: {rate:.1%} of customers flagged\n")
        f.write("\n")
        
        # Composite scores
        composite_scores = ['financial_stability_index', 'creditworthiness_score', 
                          'employment_score', 'credit_quality_score']
        f.write(f"### Composite Scores\n")
        for score in composite_scores:
            if score in data.columns:
                stats = data[score].describe()
                f.write(f"- **{score}**: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}\n")
        f.write("\n")
        
        # Feature importance
        f.write("## Feature Importance Analysis\n\n")
        
        if 'correlation' in feature_importance:
            f.write("### Top 10 Features by Correlation with Default\n")
            corr_importance = feature_importance['correlation']
            top_corr = sorted(corr_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for feature, score in top_corr:
                f.write(f"- **{feature}**: {score:.4f}\n")
            f.write("\n")
        
        if 'mutual_info' in feature_importance:
            f.write("### Top 10 Features by Mutual Information\n")
            mi_importance = feature_importance['mutual_info']
            top_mi = sorted(mi_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for feature, score in top_mi:
                f.write(f"- **{feature}**: {score:.4f}\n")
            f.write("\n")
        
        f.write("## Selected Features for Model Training\n\n")
        for i, feature in enumerate(selected_features[:20], 1):  # Top 20
            f.write(f"{i}. {feature}\n")
        if len(selected_features) > 20:
            f.write(f"... and {len(selected_features) - 20} more features\n")
        f.write("\n")
        
        f.write("## Business Rules Implemented\n\n")
        business_rules = [
            "**High DTI Risk**: Debt-to-income ratio > 43%",
            "**Low Credit Risk**: Credit score < 620",
            "**High Utilization Risk**: Credit utilization > 75%", 
            "**Recent Default Risk**: Previous defaults > 0",
            "**Employment Instability**: Employment years < 2"
        ]
        for rule in business_rules:
            f.write(f"- {rule}\n")
        f.write("\n")
        
        f.write("## Transformations Applied\n\n")
        transformations = [
            "1. **Financial Ratios**: Payment-to-income, loan-to-income, debt service coverage",
            "2. **Employment Stability**: 5-point scale scoring based on employment status",
            "3. **Credit Quality**: Adjusted credit score accounting for payment history",
            "4. **Risk Flags**: Binary indicators based on business rule thresholds",
            "5. **Interaction Terms**: Income×credit_score, age×employment_years",
            "6. **Composite Scores**: Financial stability index, creditworthiness score",
            "7. **One-Hot Encoding**: Categorical variables (employment, education, etc.)",
            "8. **Feature Scaling**: StandardScaler normalization for continuous variables",
            "9. **Feature Selection**: SelectKBest using F-statistic for top predictive features"
        ]
        for transformation in transformations:
            f.write(f"{transformation}\n")
        f.write("\n")
    
    logger.info(f"Feature engineering report saved to {report_file}")

if __name__ == "__main__":
    main()
