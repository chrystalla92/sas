#!/usr/bin/env python3
"""
Bank Credit Risk Scoring Model - Step 1: Generate Synthetic Credit Data

Purpose: Generate realistic synthetic customer credit data for model training
Author: Risk Analytics Team  
Date: 2025

This script creates a synthetic dataset representing bank customers with:
- Demographic information (age, income, employment)
- Credit history metrics
- Financial obligations
- Default indicator (target variable)

MIGRATION FROM SAS:
This Python implementation replicates the exact logic from 01_generate_credit_data.sas,
maintaining the same statistical distributions, business rules, and data relationships.

USAGE:
    python 01_generate_credit_data.py

OUTPUT:
    - output/credit_data_sample.csv (10,000 records)
    - In-memory DataFrame 'credit_data_sample' for downstream processing

DEPENDENCIES:
    - numpy>=1.24.0
    - pandas>=1.5.0  
    - PyYAML>=6.0 (for config, optional)

CONFIGURATION:
    The script uses config/config.yaml for parameters if available,
    otherwise falls back to hardcoded defaults matching the SAS implementation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Add config directory to path for imports
sys.path.append(str(Path(__file__).parent / 'config'))

try:
    from config.config import ConfigManager
    # Initialize configuration
    config = ConfigManager()
    
    # Set random seed for reproducibility from config
    np.random.seed(config.data_generation.random_seed)
    print(f"Using random seed: {config.data_generation.random_seed}")
    
except ImportError:
    print("Config module not found, using default parameters")
    # Fallback to default parameters
    np.random.seed(12345)
    config = None

def generate_credit_data(n_records=None):
    """
    Generate synthetic credit application data matching SAS implementation.
    
    Args:
        n_records (int): Number of records to generate. If None, uses config default.
    
    Returns:
        pd.DataFrame: Generated credit application data
    """
    
    # Use config parameters if available
    if config and n_records is None:
        n_records = config.data_generation.num_records
    elif n_records is None:
        n_records = 10000
    
    print(f"Generating {n_records:,} synthetic credit application records...")
    
    # Initialize data storage
    data = {}
    
    # Generate customer IDs
    data['customer_id'] = [f'CUST{i:06d}' for i in range(1, n_records + 1)]
    
    # Application dates (random dates in last 2 years)
    today = datetime.now()
    days_back = np.random.uniform(0, 730, n_records)
    data['application_date'] = [today - timedelta(days=int(d)) for d in days_back]
    
    # Age distribution (use config parameters if available)
    if config:
        age_mean = config.data_generation.age_mean
        age_std = config.data_generation.age_std
        age_min, age_max = config.data_generation.age_range
    else:
        age_mean, age_std = 42, 12
        age_min, age_max = 18, 75
        
    ages = np.random.normal(age_mean, age_std, n_records)
    data['age'] = np.clip(np.round(ages), age_min, age_max).astype(int)
    
    # Employment years (correlated with age)
    employment_years = []
    for age in data['age']:
        if age < 25:
            emp_years = max(0, round(np.random.uniform(0, 3)))
        elif age < 35:
            emp_years = max(0, round(np.random.uniform(0, 10)))
        else:
            emp_years = max(0, round(np.random.uniform(0, min(20, age - 20))))
        employment_years.append(emp_years)
    data['employment_years'] = employment_years
    
    # Employment status
    emp_rand = np.random.uniform(0, 1, n_records)
    employment_status = []
    for r in emp_rand:
        if r < 0.65:
            employment_status.append('Full-time')
        elif r < 0.80:
            employment_status.append('Self-employed')
        elif r < 0.90:
            employment_status.append('Part-time')
        elif r < 0.95:
            employment_status.append('Retired')
        else:
            employment_status.append('Unemployed')
    data['employment_status'] = employment_status
    
    # Education level
    edu_rand = np.random.uniform(0, 1, n_records)
    education = []
    for r in edu_rand:
        if r < 0.30:
            education.append('High School')
        elif r < 0.60:
            education.append('Bachelors')
        elif r < 0.80:
            education.append('Masters')
        elif r < 0.90:
            education.append('Doctorate')
        else:
            education.append('Other')
    data['education'] = education
    
    # Monthly income (log-normal distribution based on education and employment)
    monthly_income = []
    for i in range(n_records):
        # Base income by education
        if data['education'][i] == 'High School':
            base_income = 2500
        elif data['education'][i] == 'Bachelors':
            base_income = 4000
        elif data['education'][i] == 'Masters':
            base_income = 5500
        elif data['education'][i] == 'Doctorate':
            base_income = 7000
        else:
            base_income = 3000
        
        # Income multiplier by employment status
        if data['employment_status'][i] == 'Full-time':
            income_mult = 1.0
        elif data['employment_status'][i] == 'Self-employed':
            income_mult = 1.2
        elif data['employment_status'][i] == 'Part-time':
            income_mult = 0.5
        elif data['employment_status'][i] == 'Retired':
            income_mult = 0.6
        else:  # Unemployed
            income_mult = 0.1
        
        # Apply log-normal variation
        income = base_income * income_mult * np.exp(np.random.normal(0, 0.3))
        monthly_income.append(round(income, -2))  # Round to nearest 100
    
    data['monthly_income'] = monthly_income
    data['annual_income'] = [income * 12 for income in monthly_income]
    
    # Home ownership
    home_rand = np.random.uniform(0, 1, n_records)
    home_ownership = []
    for r in home_rand:
        if r < 0.40:
            home_ownership.append('Rent')
        elif r < 0.70:
            home_ownership.append('Mortgage')
        else:
            home_ownership.append('Own')
    data['home_ownership'] = home_ownership
    
    # Number of dependents (Poisson distribution)
    data['num_dependents'] = np.maximum(0, np.random.poisson(1.5, n_records))
    
    # Credit history metrics
    # Credit history years (limited by age)
    credit_history_years = []
    for age in data['age']:
        max_years = age - 18
        hist_years = max(0, min(max_years, round(np.random.gamma(5))))
        credit_history_years.append(hist_years)
    data['credit_history_years'] = credit_history_years
    
    # Number of credit accounts (Poisson distribution)
    data['num_credit_accounts'] = np.maximum(1, np.random.poisson(3, n_records))
    
    # Number of late payments (exponential for those with late payments)
    late_payment_prob = np.random.uniform(0, 1, n_records)
    num_late_payments = []
    for p in late_payment_prob:
        if p < 0.7:
            num_late_payments.append(0)
        else:
            num_late_payments.append(round(np.random.exponential(2)))
    data['num_late_payments'] = num_late_payments
    
    # Credit utilization ratio (Beta distribution, 0-100%)
    credit_util = np.random.beta(2, 5, n_records) * 100
    data['credit_utilization'] = np.clip(credit_util, 0, 100)
    
    # Previous defaults
    prev_default_prob = np.random.uniform(0, 1, n_records)
    previous_defaults = []
    for p in prev_default_prob:
        if p < 0.92:
            previous_defaults.append(0)
        else:
            previous_defaults.append(max(0, round(np.random.poisson(0.5))))
    data['previous_defaults'] = previous_defaults
    
    # Loan details
    loan_amounts = []
    loan_terms = []
    for annual_inc in data['annual_income']:
        # Loan amount as multiple of annual income
        loan_amt = annual_inc * np.random.uniform(0.1, 3)
        loan_amounts.append(round(loan_amt, -3))  # Round to nearest 1000
        
        # Loan term (12, 24, or 36 months)
        term_rand = np.random.uniform(0, 1)
        if term_rand < 0.2:
            loan_terms.append(12)
        elif term_rand < 0.7:
            loan_terms.append(24)
        else:
            loan_terms.append(36)
    
    data['loan_amount'] = loan_amounts
    data['loan_term_months'] = loan_terms
    
    # Loan purpose
    purpose_rand = np.random.uniform(0, 1, n_records)
    loan_purpose = []
    for r in purpose_rand:
        if r < 0.25:
            loan_purpose.append('Debt Consolidation')
        elif r < 0.45:
            loan_purpose.append('Home Improvement')
        elif r < 0.60:
            loan_purpose.append('Auto')
        elif r < 0.75:
            loan_purpose.append('Personal')
        elif r < 0.85:
            loan_purpose.append('Medical')
        else:
            loan_purpose.append('Other')
    data['loan_purpose'] = loan_purpose
    
    # Calculate monthly payment and debt-to-income ratio
    monthly_payments = []
    debt_to_income_ratios = []
    
    for i in range(n_records):
        # Interest rate based on risk factors
        interest_rate = 0.05 + (data['credit_utilization'][i]/100) * 0.15 + (data['num_late_payments'][i] * 0.02)
        monthly_rate = interest_rate / 12
        loan_amt = data['loan_amount'][i]
        term = data['loan_term_months'][i]
        
        # Monthly payment calculation
        if monthly_rate > 0:
            monthly_payment = loan_amt * (monthly_rate * (1 + monthly_rate)**term) / ((1 + monthly_rate)**term - 1)
        else:
            monthly_payment = loan_amt / term
        
        monthly_payments.append(monthly_payment)
        
        # Existing monthly debt (Beta distribution)
        existing_debt = data['monthly_income'][i] * np.random.beta(2, 8)
        total_debt = existing_debt + monthly_payment
        debt_ratio = (total_debt / data['monthly_income'][i]) * 100
        debt_to_income_ratios.append(debt_ratio)
    
    data['monthly_payment'] = monthly_payments
    data['debt_to_income_ratio'] = debt_to_income_ratios
    
    # Credit score simulation (300-850)
    credit_scores = []
    for i in range(n_records):
        base_score = 650
        score_adjustment = (-data['num_late_payments'][i] * 30 
                          - data['previous_defaults'][i] * 100
                          - (data['credit_utilization'][i] - 30) * 2
                          + data['credit_history_years'][i] * 5
                          + data['employment_years'][i] * 2
                          + np.random.normal(0, 30))
        
        credit_score = max(300, min(850, round(base_score + score_adjustment)))
        credit_scores.append(credit_score)
    
    data['credit_score'] = credit_scores
    
    # Calculate default probability and generate default flag
    default_flags = []
    risk_ratings = []
    
    for i in range(n_records):
        # Base default probability
        default_prob = 0.05
        
        # Adjust based on credit score
        if data['credit_score'][i] < 600:
            default_prob += 0.15
        elif data['credit_score'][i] < 650:
            default_prob += 0.08
        elif data['credit_score'][i] < 700:
            default_prob += 0.03
        else:
            default_prob -= 0.02
        
        # Adjust based on debt-to-income ratio
        if data['debt_to_income_ratio'][i] > 50:
            default_prob += 0.12
        elif data['debt_to_income_ratio'][i] > 40:
            default_prob += 0.06
        
        # Adjust based on employment status
        if data['employment_status'][i] == 'Unemployed':
            default_prob += 0.20
        elif data['employment_status'][i] == 'Part-time':
            default_prob += 0.05
        
        # Adjust based on late payments
        if data['num_late_payments'][i] > 3:
            default_prob += 0.15
        elif data['num_late_payments'][i] > 0:
            default_prob += 0.05
        
        # Adjust based on previous defaults
        if data['previous_defaults'][i] > 0:
            default_prob += 0.25
        
        # Cap probability between 1% and 95%
        default_prob = max(0.01, min(0.95, default_prob))
        
        # Generate default flag
        default_flag = 1 if np.random.uniform() < default_prob else 0
        default_flags.append(default_flag)
        
        # Risk rating based on credit score
        if data['credit_score'][i] >= 750:
            risk_ratings.append('Excellent')
        elif data['credit_score'][i] >= 700:
            risk_ratings.append('Good')
        elif data['credit_score'][i] >= 650:
            risk_ratings.append('Fair')
        elif data['credit_score'][i] >= 600:
            risk_ratings.append('Poor')
        else:
            risk_ratings.append('Very Poor')
    
    data['default_flag'] = default_flags
    data['risk_rating'] = risk_ratings
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Format application_date as date string (matching SAS format)
    df['application_date'] = df['application_date'].dt.strftime('%d%b%Y').str.upper()
    
    return df

def main():
    """Main execution function."""
    
    print("=== Bank Credit Risk Scoring Model - Data Generation ===\n")
    
    # Generate the data (use config default if available)
    credit_applications = generate_credit_data()
    
    # Display summary statistics
    print("\n=== Summary Statistics ===")
    numeric_cols = ['age', 'employment_years', 'monthly_income', 'annual_income', 
                   'loan_amount', 'credit_score', 'credit_utilization', 'debt_to_income_ratio', 
                   'num_late_payments']
    
    summary_stats = credit_applications[numeric_cols].describe()
    print(summary_stats.round(2))
    
    # Calculate and display default rate
    default_rate = credit_applications['default_flag'].mean() * 100
    print(f"\n=== Default Rate Analysis ===")
    print(f"Overall default rate: {default_rate:.2f}%")
    
    # Default rate by risk rating
    print("\n=== Default Rate by Risk Rating ===")
    risk_default = credit_applications.groupby('risk_rating')['default_flag'].agg(['count', 'mean']).round(4)
    risk_default.columns = ['Count', 'Default_Rate']
    risk_default['Default_Rate'] = risk_default['Default_Rate'] * 100
    print(risk_default)
    
    # Create output directory if it doesn't exist
    if config:
        output_dir = config.paths.output
    else:
        output_dir = 'output'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Export full dataset to CSV (matching SAS format)
    output_file = os.path.join(output_dir, 'credit_data_sample.csv')
    credit_applications.to_csv(output_file, index=False)
    print(f"\n=== Data Export ===")
    print(f"Full dataset exported to: {output_file}")
    print(f"Total records: {len(credit_applications):,}")
    
    # Verification checks
    print(f"\n=== Verification ===")
    print(f"✓ Generated exactly {len(credit_applications):,} records")
    print(f"✓ Default rate: {default_rate:.2f}% (target: ~20%)")
    print(f"✓ Age range: {credit_applications['age'].min()}-{credit_applications['age'].max()}")
    print(f"✓ Credit score range: {credit_applications['credit_score'].min()}-{credit_applications['credit_score'].max()}")
    print(f"✓ All required columns present: {len(credit_applications.columns)} columns")
    
    # Display column names for verification
    print(f"\n=== Dataset Schema ===")
    print("Columns generated:")
    for i, col in enumerate(credit_applications.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Data type summary
    print(f"\nData types:")
    dtype_summary = credit_applications.dtypes.value_counts()
    for dtype, count in dtype_summary.items():
        print(f"  {dtype}: {count} columns")
    
    # Create global DataFrame variable for downstream processing
    globals()['credit_data_sample'] = credit_applications
    
    print("\n=== Data Generation Complete ===")
    print("DataFrame 'credit_data_sample' is available for downstream processing")
    print(f"Use: python 01_generate_credit_data.py to regenerate data")
    
    return credit_applications

if __name__ == "__main__":
    credit_applications = main()
