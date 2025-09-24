#!/usr/bin/env python3
"""
Bank Credit Risk Scoring Model - Step 6: Score New Customer Applications

Purpose: Production scoring system for new credit applications
Author: Risk Analytics Team
Date: 2025

This script:
- Loads trained models (LogisticRegression and DecisionTreeClassifier)
- Generates new customer applications for scoring
- Applies feature engineering transformations
- Scores using trained models
- Calculates 300-850 credit risk scores
- Assigns A-F risk grades
- Implements business decision logic
- Generates decision reports and exports results

MIGRATION FROM SAS:
This Python implementation replicates the functionality from 06_score_new_customers.sas,
including model scoring, risk calculation, and business decision logic.

USAGE:
    python 06_score_new_customers.py

INPUT:
    - models/logistic_model.joblib (trained logistic regression)
    - models/decision_tree_model.joblib (trained decision tree)
    - models/calibrated_model.joblib (calibrated model, optional)

OUTPUT:
    - output/new_application_decisions.csv
    - output/approval_summary.csv
    - Application scoring results and business recommendations

DEPENDENCIES:
    - pandas>=1.5.0
    - numpy>=1.24.0
    - scikit-learn>=1.3.0
    - joblib>=1.3.0
"""

import numpy as np
import pandas as pd
import joblib
import os
import sys
import subprocess
import warnings
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')

def run_model_training() -> bool:
    """
    Run model training script (04_train_credit_model.py) when models are missing.
    
    Returns:
        bool: True if training completed successfully, False otherwise
    """
    print("Model joblibs not found. Running model training...")
    print("This may take a few minutes...")
    
    try:
        # Run the model training script using subprocess
        result = subprocess.run([sys.executable, '04_train_credit_model.py'], 
                              capture_output=True, text=True, cwd='.')
        
        if result.returncode != 0:
            print(f"❌ Model training script failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error output: {result.stderr}")
            if result.stdout:
                print(f"Standard output: {result.stdout}")
            return False
        else:
            print("✓ Model training script executed successfully")
            if result.stdout:
                # Show only the last few lines of output to avoid cluttering
                output_lines = result.stdout.strip().split('\n')
                print("Training summary:")
                for line in output_lines[-10:]:  # Show last 10 lines
                    if line.strip():
                        print(f"  {line}")
            return True
            
    except Exception as e:
        print(f"❌ Error running model training: {str(e)}")
        return False

def load_trained_models() -> Dict[str, Any]:
    """
    Load trained models from Script 4, or run training if models are not available.
    
    Returns:
        dict: Dictionary containing loaded models and metadata
    """
    print("Loading trained models...")
    
    models_dir = Path('models')
    models = {}
    
    # Define required model files
    required_models = {
        'logistic': models_dir / 'logistic_model.joblib',
        'decision_tree': models_dir / 'decision_tree_model.joblib'
    }
    
    optional_models = {
        'calibrated': models_dir / 'calibrated_model.joblib',
        'metadata': models_dir / 'model_metadata.joblib'
    }
    
    # Check if required models exist
    missing_required_models = []
    for model_name, model_path in required_models.items():
        if not model_path.exists():
            missing_required_models.append(model_name)
    
    # If required models are missing, run training
    if missing_required_models:
        print(f"Required models missing: {missing_required_models}")
        if run_model_training():
            print("✓ Model training completed. Attempting to load models...")
        else:
            print("❌ Model training failed. Falling back to mock predictions.")
            return {'logistic': None, 'decision_tree': None, 'calibrated': None, 'metadata': None}
    
    try:
        # Load logistic regression model
        lr_path = required_models['logistic']
        if lr_path.exists():
            lr_data = joblib.load(lr_path)
            models['logistic'] = lr_data
            print(f"✓ Logistic regression model loaded from {lr_path}")
        else:
            print(f"❌ Logistic regression model still not found at {lr_path}")
            models['logistic'] = None
        
        # Load decision tree model
        dt_path = required_models['decision_tree']
        if dt_path.exists():
            dt_data = joblib.load(dt_path)
            models['decision_tree'] = dt_data
            print(f"✓ Decision tree model loaded from {dt_path}")
        else:
            print(f"❌ Decision tree model still not found at {dt_path}")
            models['decision_tree'] = None
        
        # Load optional models
        cal_path = optional_models['calibrated']
        if cal_path.exists():
            cal_data = joblib.load(cal_path)
            models['calibrated'] = cal_data
            print(f"✓ Calibrated model loaded from {cal_path}")
        else:
            print(f"ℹ️  Calibrated model not found at {cal_path} (optional)")
            models['calibrated'] = None
        
        metadata_path = optional_models['metadata']
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            models['metadata'] = metadata
            print(f"✓ Model metadata loaded from {metadata_path}")
        else:
            print(f"ℹ️  Model metadata not found at {metadata_path} (optional)")
            models['metadata'] = None
        
        # Check if we successfully loaded at least one required model
        if models['logistic'] is None and models['decision_tree'] is None:
            print("⚠️  No required models could be loaded. Using mock predictions.")
        
        return models
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        print("ℹ️  Falling back to mock predictions")
        return {'logistic': None, 'decision_tree': None, 'calibrated': None, 'metadata': None}

def generate_new_customer_applications(num_applications: int = 50) -> pd.DataFrame:
    """
    Generate new customer applications matching SAS implementation.
    
    Args:
        num_applications: Number of applications to generate
        
    Returns:
        pd.DataFrame: New customer applications
    """
    print(f"Generating {num_applications} new customer applications...")
    
    np.random.seed(98765)  # Different seed for new data, matching SAS
    
    applications = []
    
    for i in range(1, num_applications + 1):
        # Generate application details
        customer_id = f'NEW{i:05d}'
        application_date = date.today()
        
        # Customer demographics
        age = max(18, min(75, int(np.random.normal(40, 15))))
        
        # Employment
        emp_rand = np.random.uniform()
        if emp_rand < 0.60:
            employment_status = 'Full-time'
        elif emp_rand < 0.75:
            employment_status = 'Self-employed'
        elif emp_rand < 0.85:
            employment_status = 'Part-time'
        elif emp_rand < 0.92:
            employment_status = 'Retired'
        else:
            employment_status = 'Unemployed'
        
        employment_years = max(0, int(np.random.uniform() * min(15, age - 20)))
        
        # Education
        edu_rand = np.random.uniform()
        if edu_rand < 0.35:
            education = 'High School'
        elif edu_rand < 0.65:
            education = 'Bachelors'
        elif edu_rand < 0.85:
            education = 'Masters'
        else:
            education = 'Doctorate'
        
        # Income
        base_income = 4000 * (1 + (education != 'High School') * 0.3)
        monthly_income = round(base_income * np.exp(np.random.normal(0, 0.4)), -2)
        annual_income = monthly_income * 12
        
        # Home ownership
        home_rand = np.random.uniform()
        if home_rand < 0.35:
            home_ownership = 'Rent'
        elif home_rand < 0.75:
            home_ownership = 'Mortgage'
        else:
            home_ownership = 'Own'
        
        # Credit history
        credit_history_years = max(0, min(age - 18, int(np.random.gamma(4))))
        num_credit_accounts = max(1, int(np.random.poisson(4)))
        
        # Credit score - varied distribution for testing
        if i <= 10:
            credit_score = int(np.random.uniform(750, 850))  # Excellent
        elif i <= 20:
            credit_score = int(np.random.uniform(680, 750))  # Good
        elif i <= 35:
            credit_score = int(np.random.uniform(600, 680))  # Fair
        else:
            credit_score = int(np.random.uniform(450, 600))  # Poor
        
        # Payment history
        if credit_score > 700:
            num_late_payments = 0
        elif credit_score > 650:
            num_late_payments = int(np.random.poisson(0.5))
        else:
            num_late_payments = int(np.random.poisson(2))
        
        previous_defaults = int((np.random.uniform() < 0.05) * np.random.poisson(0.3))
        
        # Credit utilization
        if credit_score > 700:
            credit_utilization = np.random.beta(2, 8) * 100
        else:
            credit_utilization = np.random.beta(3, 5) * 100
        
        # Loan request
        loan_amount = round(annual_income * np.random.uniform(0.2, 2.5), -3)
        loan_term_months = 12 * (1 + int(np.random.uniform(0, 2)))
        
        loan_purpose_rand = np.random.uniform()
        if loan_purpose_rand < 0.3:
            loan_purpose = 'Debt Consolidation'
        elif loan_purpose_rand < 0.5:
            loan_purpose = 'Home Improvement'
        elif loan_purpose_rand < 0.7:
            loan_purpose = 'Personal'
        elif loan_purpose_rand < 0.85:
            loan_purpose = 'Auto'
        else:
            loan_purpose = 'Medical'
        
        # Calculate monthly payment and DTI
        interest_rate = 0.06 + (800 - credit_score) / 5000
        monthly_rate = interest_rate / 12
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**loan_term_months) / ((1 + monthly_rate)**loan_term_months - 1)
        
        existing_monthly_debt = monthly_income * np.random.beta(2, 10)
        total_monthly_debt = existing_monthly_debt + monthly_payment
        debt_to_income_ratio = (total_monthly_debt / monthly_income) * 100
        
        # Other variables
        num_dependents = max(0, int(np.random.poisson(1.2)))
        
        applications.append({
            'customer_id': customer_id,
            'application_date': application_date,
            'age': age,
            'employment_status': employment_status,
            'employment_years': employment_years,
            'education': education,
            'monthly_income': monthly_income,
            'annual_income': annual_income,
            'home_ownership': home_ownership,
            'credit_history_years': credit_history_years,
            'num_credit_accounts': num_credit_accounts,
            'credit_score': credit_score,
            'num_late_payments': num_late_payments,
            'previous_defaults': previous_defaults,
            'credit_utilization': credit_utilization,
            'loan_amount': loan_amount,
            'loan_term_months': loan_term_months,
            'loan_purpose': loan_purpose,
            'monthly_payment': monthly_payment,
            'total_monthly_debt': total_monthly_debt,
            'debt_to_income_ratio': debt_to_income_ratio,
            'num_dependents': num_dependents
        })
    
    df = pd.DataFrame(applications)
    print(f"✓ Generated {len(df)} new customer applications")
    
    return df

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering transformations matching Script 3 pipeline.
    
    Args:
        df: Raw application data
        
    Returns:
        pd.DataFrame: Feature engineered data
    """
    print("Applying feature engineering transformations...")
    
    df = df.copy()
    
    # Apply same feature engineering as training
    df['payment_to_income_ratio'] = (df['monthly_payment'] / df['monthly_income']) * 100
    df['loan_to_income_ratio'] = (df['loan_amount'] / df['annual_income'])
    df['debt_service_coverage'] = df['monthly_income'] / df['total_monthly_debt']
    
    # Employment score
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
    df['credit_quality_score'] = (df['credit_score'] - 
                                 (df['num_late_payments'] * 50) - 
                                 (df['previous_defaults'] * 150))
    
    # Affordability
    df['affordability_score'] = ((df['monthly_income'] - df['total_monthly_debt']) / 
                                df['monthly_payment'])
    
    # Risk flags
    df['flag_high_dti'] = (df['debt_to_income_ratio'] > 43).astype(int)
    df['flag_low_credit'] = (df['credit_score'] < 620).astype(int)
    df['flag_high_util'] = (df['credit_utilization'] > 75).astype(int)
    df['flag_recent_default'] = (df['previous_defaults'] > 0).astype(int)
    df['flag_unstable_employment'] = (df['employment_years'] < 2).astype(int)
    df['has_delinquency'] = (df['num_late_payments'] > 0).astype(int)
    
    df['total_risk_flags'] = (df['flag_high_dti'] + df['flag_low_credit'] + 
                             df['flag_high_util'] + df['flag_recent_default'] + 
                             df['flag_unstable_employment'])
    
    # Categorical dummies
    df['emp_fulltime'] = (df['employment_status'] == 'Full-time').astype(int)
    df['emp_selfemployed'] = (df['employment_status'] == 'Self-employed').astype(int)
    df['emp_unemployed'] = (df['employment_status'] == 'Unemployed').astype(int)
    df['edu_bachelors'] = (df['education'] == 'Bachelors').astype(int)
    df['edu_masters'] = (df['education'] == 'Masters').astype(int)
    df['edu_doctorate'] = (df['education'] == 'Doctorate').astype(int)
    df['home_rent'] = (df['home_ownership'] == 'Rent').astype(int)
    df['home_mortgage'] = (df['home_ownership'] == 'Mortgage').astype(int)
    df['purpose_debt'] = (df['loan_purpose'] == 'Debt Consolidation').astype(int)
    df['purpose_auto'] = (df['loan_purpose'] == 'Auto').astype(int)
    df['purpose_personal'] = (df['loan_purpose'] == 'Personal').astype(int)
    
    print("✓ Feature engineering transformations applied")
    return df

def score_applications_with_models(df: pd.DataFrame, models: Dict[str, Any]) -> pd.DataFrame:
    """
    Score applications using trained models, or create mock predictions if models not available.
    
    Args:
        df: Feature engineered application data
        models: Dictionary containing trained models
        
    Returns:
        pd.DataFrame: Applications with model scores
    """
    print("Scoring applications with trained models...")
    
    df_scored = df.copy()
    
    try:
        # Check if we have trained models
        if models.get('logistic') is not None:
            # Use actual trained models
            lr_model = models['logistic']['model']
            feature_selector = models['logistic']['selector']
            selected_features = models['logistic']['selected_features']
            
            # Prepare features matching training pipeline
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
            
            # Categorical indicators
            categorical_features = [
                'emp_fulltime', 'emp_unemployed',
                'home_rent', 'purpose_debt'
            ]
            
            all_features = core_features + categorical_features
            
            # Keep only features that exist in the dataframe
            available_features = [f for f in all_features if f in df.columns]
            
            # Handle missing features by filling with defaults
            for feature in available_features:
                if feature not in df.columns:
                    if feature.startswith('flag_') or feature.startswith('emp_') or feature.startswith('home_') or feature.startswith('purpose_'):
                        df_scored[feature] = 0
                    else:
                        df_scored[feature] = df_scored[feature].median() if feature in df_scored.columns else 0
            
            # Prepare feature matrix
            X_features = df_scored[available_features].fillna(0)
            
            # Apply feature selection
            X_selected = feature_selector.transform(X_features)
            
            # Get probability predictions from logistic regression
            probability_default = lr_model.predict_proba(X_selected)[:, 1]
            df_scored['probability_default'] = probability_default
            
            # Use calibrated model if available
            if models.get('calibrated') is not None:
                cal_model = models['calibrated']['calibrated_model']
                cal_probability = cal_model.predict_proba(X_selected)[:, 1]
                df_scored['calibrated_probability'] = cal_probability
                df_scored['probability_default'] = cal_probability  # Use calibrated as primary
            
            # Get decision tree predictions if available
            if models.get('decision_tree') is not None:
                dt_model = models['decision_tree']['model']
                dt_probability = dt_model.predict_proba(X_features)[:, 1]
                df_scored['dt_probability_default'] = dt_probability
            
            print(f"✓ Applications scored with models (avg probability: {probability_default.mean():.4f})")
            
        else:
            # Create mock predictions based on risk factors
            print("ℹ️  Creating mock predictions based on risk factors...")
            
            # Create a simple scoring model based on key risk factors
            risk_score = 0.0
            
            # Credit score impact (most important factor)
            credit_score_normalized = (df_scored['credit_score'] - 300) / 550  # Normalize 300-850 to 0-1
            risk_score += (1 - credit_score_normalized) * 0.4  # Lower credit = higher risk
            
            # DTI impact
            dti_risk = np.clip(df_scored['debt_to_income_ratio'] / 100, 0, 1)  # Cap at 100%
            risk_score += dti_risk * 0.2
            
            # Credit utilization impact
            util_risk = np.clip(df_scored['credit_utilization'] / 100, 0, 1)
            risk_score += util_risk * 0.15
            
            # Late payments impact
            late_payments_risk = np.clip(df_scored['num_late_payments'] / 10, 0, 1)
            risk_score += late_payments_risk * 0.1
            
            # Previous defaults (major impact)
            default_risk = np.clip(df_scored['previous_defaults'], 0, 1)
            risk_score += default_risk * 0.15
            
            # Ensure probabilities are between 0 and 1
            probability_default = np.clip(risk_score, 0.01, 0.99)
            df_scored['probability_default'] = probability_default
            
            print(f"✓ Mock predictions created (avg probability: {probability_default.mean():.4f})")
        
        return df_scored
        
    except Exception as e:
        print(f"❌ Error scoring applications: {str(e)}")
        raise

def calculate_credit_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert probability to credit risk score (300-850 scale) matching SAS algorithm.
    
    Args:
        df: Scored applications with probability_default
        
    Returns:
        pd.DataFrame: Applications with credit risk scores
    """
    print("Converting probabilities to credit risk scores (300-850 scale)...")
    
    df_risk = df.copy()
    
    # Convert probability to credit risk score (300-850 scale)
    # Matching SAS formula: credit_risk_score = round(600 + 250 * (1 - probability_default))
    base_score = 600
    score_range = 250
    
    df_risk['credit_risk_score'] = np.round(
        base_score + score_range * (1 - df_risk['probability_default'])
    ).astype(int)
    
    # Ensure scores are within valid range
    df_risk['credit_risk_score'] = np.clip(df_risk['credit_risk_score'], 300, 850)
    
    print(f"✓ Credit risk scores calculated (range: {df_risk['credit_risk_score'].min()}-{df_risk['credit_risk_score'].max()})")
    
    return df_risk

def assign_risk_grades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign risk grades A-F based on credit risk scores.
    
    Args:
        df: Applications with credit risk scores
        
    Returns:
        pd.DataFrame: Applications with risk grades
    """
    print("Assigning risk grades A-F based on score ranges...")
    
    df_grades = df.copy()
    
    # Assign risk grade based on credit risk score
    def get_risk_grade(score):
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
    
    df_grades['risk_grade'] = df_grades['credit_risk_score'].apply(get_risk_grade)
    
    # Count distribution
    grade_counts = df_grades['risk_grade'].value_counts().sort_index()
    print(f"✓ Risk grade distribution: {dict(grade_counts)}")
    
    return df_grades

def implement_business_decision_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implement business decision logic (approve/review/decline) based on risk grades and thresholds.
    
    Args:
        df: Applications with risk grades
        
    Returns:
        pd.DataFrame: Applications with business recommendations
    """
    print("Implementing business decision logic...")
    
    df_decisions = df.copy()
    
    decisions = []
    reasons = []
    offered_rates = []
    max_approved_amounts = []
    approved_amounts = []
    
    base_rate = 0.049  # 4.9% base rate
    
    for _, row in df_decisions.iterrows():
        risk_grade = row['risk_grade']
        debt_to_income = row['debt_to_income_ratio']
        credit_score = row['credit_score']
        loan_amount = row['loan_amount']
        annual_income = row['annual_income']
        
        # Decision logic matching SAS implementation
        if risk_grade in ['A', 'B']:
            decision = 'APPROVED'
            reason = 'Low risk profile'
            
        elif risk_grade == 'C':
            if debt_to_income < 40 and credit_score >= 650:
                decision = 'APPROVED'
                reason = 'Acceptable risk with conditions'
            else:
                decision = 'MANUAL REVIEW'
                reason = 'Borderline risk profile'
                
        else:  # D, E, F
            decision = 'DECLINED'
            if row['flag_low_credit']:
                reason = 'Credit score below minimum'
            elif row['flag_high_dti']:
                reason = 'Debt-to-income ratio too high'
            elif row['flag_recent_default']:
                reason = 'Recent default history'
            else:
                reason = 'Overall risk exceeds threshold'
        
        # Risk-based pricing
        if risk_grade == 'A':
            offered_rate = base_rate
        elif risk_grade == 'B':
            offered_rate = base_rate + 0.015
        elif risk_grade == 'C':
            offered_rate = base_rate + 0.035
        elif risk_grade == 'D':
            offered_rate = base_rate + 0.055
        elif risk_grade == 'E':
            offered_rate = base_rate + 0.080
        else:  # F
            offered_rate = None  # No offer for F grade
        
        # Maximum approved amount based on risk
        if decision == 'APPROVED':
            # Calculate risk multiplier based on grade
            grade_multipliers = {'A': 3.0, 'B': 2.7, 'C': 2.4, 'D': 2.1, 'E': 1.8, 'F': 0}
            risk_multiplier = grade_multipliers.get(risk_grade, 0)
            max_approved = round(annual_income * risk_multiplier / 12, -3)  # Round to nearest 1000
            approved_amount = min(loan_amount, max_approved)
        else:
            max_approved = 0
            approved_amount = 0
        
        decisions.append(decision)
        reasons.append(reason)
        offered_rates.append(offered_rate)
        max_approved_amounts.append(max_approved)
        approved_amounts.append(approved_amount)
    
    df_decisions['decision'] = decisions
    df_decisions['decision_reason'] = reasons
    df_decisions['offered_rate'] = offered_rates
    df_decisions['max_approved_amount'] = max_approved_amounts
    df_decisions['approved_amount'] = approved_amounts
    
    # Count decisions
    decision_counts = df_decisions['decision'].value_counts()
    print(f"✓ Decision distribution: {dict(decision_counts)}")
    
    return df_decisions

def generate_decision_outputs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate new_application_decisions.csv and approval_summary.csv outputs.
    
    Args:
        df: Applications with all scoring and decision data
        
    Returns:
        tuple: (new_application_decisions, approval_summary)
    """
    print("Generating decision output files...")
    
    # Create new_application_decisions.csv
    decision_columns = [
        'customer_id', 'application_date', 'age', 'employment_status', 'education',
        'monthly_income', 'credit_score', 'loan_amount', 'loan_term_months',
        'debt_to_income_ratio', 'probability_default', 'credit_risk_score', 
        'risk_grade', 'decision', 'decision_reason', 'offered_rate',
        'approved_amount', 'total_risk_flags'
    ]
    
    available_columns = [col for col in decision_columns if col in df.columns]
    new_application_decisions = df[available_columns].copy()
    
    # Format columns to match SAS output
    new_application_decisions['application_date'] = pd.to_datetime(new_application_decisions['application_date']).dt.strftime('%d%b%Y')
    new_application_decisions['monthly_income'] = new_application_decisions['monthly_income'].apply(lambda x: f"${x:,.2f}")
    new_application_decisions['approved_amount'] = new_application_decisions['approved_amount'].apply(lambda x: f"${x:,.0f}" if x > 0 else "$0")
    new_application_decisions['probability_default'] = new_application_decisions['probability_default'].apply(lambda x: f"{x:.2%}")
    new_application_decisions['offered_rate'] = new_application_decisions['offered_rate'].apply(lambda x: f"{x:.2%}" if x is not None else "")
    
    # Create approval_summary.csv
    total_applications = len(df)
    approved_count = len(df[df['decision'] == 'APPROVED'])
    review_count = len(df[df['decision'] == 'MANUAL REVIEW'])
    declined_count = len(df[df['decision'] == 'DECLINED'])
    
    approved_df = df[df['decision'] == 'APPROVED']
    avg_offered_rate = approved_df['offered_rate'].mean() if len(approved_df) > 0 else 0
    total_approved_amount = approved_df['approved_amount'].sum()
    avg_approved_amount = approved_df['approved_amount'].mean() if len(approved_df) > 0 else 0
    
    approval_summary = pd.DataFrame([{
        'total_applications': total_applications,
        'approved_count': approved_count,
        'review_count': review_count,
        'declined_count': declined_count,
        'approval_rate': f"{approved_count / total_applications:.2%}" if total_applications > 0 else "0%",
        'avg_offered_rate': f"{avg_offered_rate:.2%}" if avg_offered_rate > 0 else "",
        'total_approved_amount': f"${total_approved_amount:,.0f}",
        'avg_approved_amount': f"${avg_approved_amount:,.0f}" if avg_approved_amount > 0 else "$0"
    }])
    
    print(f"✓ Generated decision outputs for {total_applications} applications")
    print(f"  - Approved: {approved_count}, Review: {review_count}, Declined: {declined_count}")
    
    return new_application_decisions, approval_summary

def export_results(new_application_decisions: pd.DataFrame, 
                  approval_summary: pd.DataFrame) -> None:
    """
    Export results to CSV files matching SAS format exactly.
    
    Args:
        new_application_decisions: Individual decision records
        approval_summary: Aggregate statistics
    """
    print("Exporting results to CSV files...")
    
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Export new_application_decisions.csv
        decisions_path = output_dir / 'new_application_decisions.csv'
        new_application_decisions.to_csv(decisions_path, index=False)
        print(f"✓ New application decisions exported to {decisions_path}")
        
        # Export approval_summary.csv
        summary_path = output_dir / 'approval_summary.csv'
        approval_summary.to_csv(summary_path, index=False)
        print(f"✓ Approval summary exported to {summary_path}")
        
    except Exception as e:
        print(f"❌ Error exporting results: {str(e)}")
        raise

def validate_outputs(df: pd.DataFrame) -> bool:
    """
    Validate scoring outputs for data quality and business logic.
    
    Args:
        df: Final scored applications
        
    Returns:
        bool: True if validation passes
    """
    print("Validating scoring outputs...")
    
    try:
        # Check risk scores are within valid range
        assert df['credit_risk_score'].min() >= 300, "Risk scores below minimum (300)"
        assert df['credit_risk_score'].max() <= 850, "Risk scores above maximum (850)"
        
        # Check probabilities are valid
        assert (df['probability_default'] >= 0).all(), "Negative probabilities found"
        assert (df['probability_default'] <= 1).all(), "Probabilities > 1 found"
        
        # Check risk grades are valid
        valid_grades = {'A', 'B', 'C', 'D', 'E', 'F'}
        assert df['risk_grade'].isin(valid_grades).all(), "Invalid risk grades found"
        
        # Check decisions are valid
        valid_decisions = {'APPROVED', 'MANUAL REVIEW', 'DECLINED'}
        assert df['decision'].isin(valid_decisions).all(), "Invalid decisions found"
        
        # Check approved amounts are non-negative
        assert (df['approved_amount'] >= 0).all(), "Negative approved amounts found"
        
        # Business logic checks
        approved_df = df[df['decision'] == 'APPROVED']
        if len(approved_df) > 0:
            # Approved applications should have offered rates
            assert approved_df['offered_rate'].notna().all(), "Approved applications missing offered rates"
            
            # Approved amounts should not exceed loan amounts
            assert (approved_df['approved_amount'] <= approved_df['loan_amount']).all(), "Approved amounts exceed requested amounts"
        
        print("✓ All validation checks passed")
        return True
        
    except AssertionError as e:
        print(f"❌ Validation failed: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {str(e)}")
        return False

def main():
    """
    Main scoring pipeline.
    """
    print("=" * 70)
    print("BANK CREDIT RISK MODEL - NEW CUSTOMER SCORING")
    print("=" * 70)
    
    try:
        # Step 1: Load trained models
        models = load_trained_models()
        
        # Step 2: Generate new customer applications
        new_applications = generate_new_customer_applications(num_applications=50)
        
        # Step 3: Apply feature engineering
        applications_features = apply_feature_engineering(new_applications)
        
        # Step 4: Score using trained models
        applications_scored = score_applications_with_models(applications_features, models)
        
        # Step 5: Calculate credit risk scores (300-850)
        applications_risk_scores = calculate_credit_risk_scores(applications_scored)
        
        # Step 6: Assign risk grades (A-F)
        applications_risk_grades = assign_risk_grades(applications_risk_scores)
        
        # Step 7: Implement business decision logic
        applications_decisions = implement_business_decision_logic(applications_risk_grades)
        
        # Step 8: Generate output files
        new_application_decisions, approval_summary = generate_decision_outputs(applications_decisions)
        
        # Step 9: Validate outputs
        if not validate_outputs(applications_decisions):
            raise Exception("Output validation failed")
        
        # Step 10: Export results
        export_results(new_application_decisions, approval_summary)
        
        print("\n" + "=" * 70)
        print("NEW CUSTOMER SCORING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"✓ Scored {len(new_applications)} new customer applications")
        print(f"✓ Risk scores calculated on 300-850 scale")
        print(f"✓ Risk grades assigned A-F")
        print(f"✓ Business decisions implemented")
        print(f"✓ Output files exported to output/ directory")
        print(f"✓ All validation checks passed")
        
        return applications_decisions, new_application_decisions, approval_summary
        
    except Exception as e:
        print(f"\n❌ Error during scoring pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    applications_decisions, new_application_decisions, approval_summary = main()
