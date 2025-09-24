#!/usr/bin/env python3
"""
Bank Credit Risk Scoring Model - Step 2: Data Exploration & Analysis

Purpose: Explore and analyze the credit application dataset
Author: Risk Analytics Team  
Date: 2025

This script performs comprehensive exploratory data analysis including:
- Data quality assessment
- Distribution analysis  
- Correlation analysis
- Risk factor identification

Python equivalent of 02_data_exploration.sas
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def setup_output_directories():
    """Create output directories for visualizations"""
    viz_dir = Path('output/visualizations')
    viz_dir.mkdir(parents=True, exist_ok=True)
    return viz_dir

def load_and_preprocess_data():
    """Load credit data and handle data type conversions"""
    print("Loading credit data...")
    
    # Load data
    data_path = Path('output/credit_data_sample.csv')
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Convert monetary columns from string to numeric
    monetary_cols = ['monthly_income', 'loan_amount']
    for col in monetary_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace('$', '').str.replace(',', '').astype(float)
    
    # Convert date column
    df['application_date'] = pd.to_datetime(df['application_date'])
    
    print(f"Data loaded successfully: {len(df):,} records, {len(df.columns)} columns")
    return df

def data_quality_assessment(df, viz_dir):
    """
    1. Data Quality Assessment
    Equivalent to SAS PROC MEANS for missing values and PROC CONTENTS
    """
    print("\n" + "="*60)
    print("1. DATA QUALITY ASSESSMENT")
    print("="*60)
    
    # Missing value analysis (equivalent to PROC MEANS nmiss n)
    print("\nMissing Value Analysis:")
    print("-" * 40)
    
    continuous_vars = ['age', 'employment_years', 'monthly_income', 'annual_income', 
                      'loan_amount', 'credit_score', 'credit_utilization', 
                      'debt_to_income_ratio', 'num_late_payments', 'num_credit_accounts', 
                      'credit_history_years', 'previous_defaults']
    
    missing_stats = []
    for col in continuous_vars:
        if col in df.columns:
            n_missing = df[col].isnull().sum() 
            n_total = len(df)
            missing_stats.append({
                'Variable': col,
                'N_Missing': n_missing,
                'N_Total': n_total,
                'Pct_Missing': (n_missing / n_total) * 100
            })
    
    missing_df = pd.DataFrame(missing_stats)
    print(missing_df.to_string(index=False))
    
    # Dataset structure (equivalent to PROC CONTENTS)
    print(f"\nDataset Structure:")
    print("-" * 40)
    print(f"Total Observations: {len(df):,}")
    print(f"Total Variables: {len(df.columns)}")
    print(f"\nVariable Information:")
    
    dtype_summary = []
    for col in df.columns:
        dtype_summary.append({
            'Variable': col,
            'Type': str(df[col].dtype),
            'Non_Null': df[col].count(),
            'Unique_Values': df[col].nunique()
        })
    
    dtype_df = pd.DataFrame(dtype_summary)
    print(dtype_df.to_string(index=False))
    
    return missing_df, dtype_df

def univariate_analysis(df, viz_dir):
    """
    2. Univariate Analysis
    Equivalent to SAS PROC UNIVARIATE and PROC FREQ
    """
    print("\n" + "="*60)
    print("2. UNIVARIATE ANALYSIS") 
    print("="*60)
    
    # Continuous variables distribution (equivalent to PROC UNIVARIATE)
    print("\nDistribution of Continuous Variables:")
    print("-" * 40)
    
    continuous_vars = ['age', 'employment_years', 'monthly_income', 'credit_score',
                      'credit_utilization', 'debt_to_income_ratio']
    
    # Statistical summary
    stats_summary = df[continuous_vars].describe()
    
    # Add skewness and kurtosis
    for col in continuous_vars:
        if col in df.columns:
            stats_summary.loc['skewness', col] = df[col].skew()
            stats_summary.loc['kurtosis', col] = df[col].kurtosis()
    
    print(stats_summary.round(3))
    
    # Create histograms with normal overlay (equivalent to SAS histogram /normal)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Distribution of Continuous Variables', fontsize=16, fontweight='bold')
    
    axes = axes.ravel()
    for i, col in enumerate(continuous_vars):
        if col in df.columns:
            ax = axes[i]
            
            # Histogram
            df[col].hist(bins=30, alpha=0.7, ax=ax, density=True)
            
            # Normal overlay
            mu, sigma = df[col].mean(), df[col].std()
            x = np.linspace(df[col].min(), df[col].max(), 100)
            normal_curve = stats.norm.pdf(x, mu, sigma)
            ax.plot(x, normal_curve, 'r-', linewidth=2, label='Normal')
            
            ax.set_title(f'{col.replace("_", " ").title()}')
            ax.set_xlabel(col.replace("_", " ").title())
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'continuous_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Categorical variables frequency (equivalent to PROC FREQ)
    print(f"\nCategorical Variables Distribution:")
    print("-" * 40)
    
    categorical_vars = ['employment_status', 'education', 'home_ownership', 'loan_purpose', 'risk_rating']
    
    for col in categorical_vars:
        if col in df.columns:
            print(f"\n{col.replace('_', ' ').title()}:")
            freq_table = df[col].value_counts().reset_index()
            freq_table.columns = [col, 'Frequency']
            freq_table['Percent'] = (freq_table['Frequency'] / len(df)) * 100
            freq_table['Cumulative_Percent'] = freq_table['Percent'].cumsum()
            print(freq_table.to_string(index=False))
    
    # Create categorical frequency plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Categorical Variables Distribution', fontsize=16, fontweight='bold')
    
    axes = axes.ravel()
    for i, col in enumerate(categorical_vars):
        if col in df.columns and i < len(axes):
            ax = axes[i]
            
            freq_data = df[col].value_counts()
            freq_data.plot(kind='bar', ax=ax, alpha=0.7)
            
            ax.set_title(f'{col.replace("_", " ").title()}')
            ax.set_xlabel(col.replace("_", " ").title())
            ax.set_ylabel('Frequency')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add frequency labels on bars
            for j, v in enumerate(freq_data.values):
                ax.text(j, v + len(df)*0.005, str(v), ha='center', va='bottom')
    
    # Remove empty subplot
    if len(categorical_vars) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'categorical_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Target variable distribution (equivalent to PROC FREQ with plots)
    print(f"\nTarget Variable Distribution (Default Rate):")
    print("-" * 40)
    
    default_freq = df['default_flag'].value_counts().reset_index()
    default_freq.columns = ['Default_Flag', 'Frequency']
    default_freq['Percent'] = (default_freq['Frequency'] / len(df)) * 100
    default_freq['Default_Status'] = default_freq['Default_Flag'].map({0: 'No Default', 1: 'Default'})
    
    print(default_freq[['Default_Status', 'Frequency', 'Percent']].to_string(index=False))
    
    # Target variable plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Target Variable Distribution (Default Rate)', fontsize=14, fontweight='bold')
    
    # Bar plot
    default_freq.set_index('Default_Status')['Frequency'].plot(kind='bar', ax=ax1, alpha=0.7)
    ax1.set_title('Default Frequency')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=0)
    
    # Pie chart
    ax2.pie(default_freq['Frequency'], labels=default_freq['Default_Status'], 
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Default Rate Distribution')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats_summary, categorical_vars

def bivariate_analysis(df, viz_dir):
    """
    3. Bivariate Analysis - Relationship with Default
    Equivalent to SAS PROC FREQ with chi-square tests and PROC MEANS by class
    """
    print("\n" + "="*60)
    print("3. BIVARIATE ANALYSIS - RELATIONSHIP WITH DEFAULT")
    print("="*60)
    
    # Default rate by categorical variables (equivalent to PROC FREQ with chisq)
    print("\nDefault Rate by Categorical Variables:")
    print("-" * 40)
    
    categorical_vars = ['employment_status', 'education', 'home_ownership', 'risk_rating']
    chi_square_results = []
    
    for col in categorical_vars:
        if col in df.columns:
            print(f"\n{col.replace('_', ' ').title()}:")
            
            # Create contingency table
            contingency = pd.crosstab(df[col], df['default_flag'], margins=True)
            
            # Calculate default rates
            default_rates = pd.crosstab(df[col], df['default_flag'], normalize='index') * 100
            
            # Display frequency table with percentages
            print("Frequency Table:")
            print(contingency)
            print("\nDefault Rates (%):")
            print(default_rates.round(2))
            
            # Chi-square test
            if contingency.shape[0] > 2 and contingency.shape[1] > 2:  # Need at least 2x2
                chi2, p_value, dof, expected = stats.chi2_contingency(
                    contingency.iloc[:-1, :-1]  # Remove margin totals
                )
                chi_square_results.append({
                    'Variable': col,
                    'Chi_Square': chi2,
                    'P_Value': p_value,
                    'Degrees_of_Freedom': dof
                })
                print(f"Chi-square: {chi2:.4f}, p-value: {p_value:.4f}")
    
    # Display chi-square summary
    if chi_square_results:
        chi_df = pd.DataFrame(chi_square_results)
        print(f"\nChi-Square Test Summary:")
        print(chi_df.to_string(index=False))
    
    # Mean comparison for continuous variables (equivalent to PROC MEANS by class)
    print(f"\nVariable Means by Default Status:")
    print("-" * 40)
    
    continuous_vars = ['age', 'employment_years', 'monthly_income', 'credit_score',
                      'credit_utilization', 'debt_to_income_ratio', 'num_late_payments']
    
    means_by_default = df.groupby('default_flag')[continuous_vars].agg(['mean', 'std', 'min', 'max'])
    print(means_by_default.round(3))
    
    # T-tests for significant differences (equivalent to PROC TTEST)
    print(f"\nT-Tests: Key Variables by Default Status:")
    print("-" * 40)
    
    key_vars = ['credit_score', 'debt_to_income_ratio', 'num_late_payments']
    ttest_results = []
    
    for var in key_vars:
        if var in df.columns:
            group_0 = df[df['default_flag'] == 0][var].dropna()
            group_1 = df[df['default_flag'] == 1][var].dropna()
            
            t_stat, p_value = stats.ttest_ind(group_0, group_1)
            
            ttest_results.append({
                'Variable': var,
                'T_Statistic': t_stat,
                'P_Value': p_value,
                'Mean_No_Default': group_0.mean(),
                'Mean_Default': group_1.mean(),
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            })
            
            print(f"{var}: t={t_stat:.4f}, p={p_value:.4f}")
    
    ttest_df = pd.DataFrame(ttest_results)
    print(f"\nT-Test Summary:")
    print(ttest_df.to_string(index=False))
    
    # Create comparison visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Variable Distributions by Default Status', fontsize=16, fontweight='bold')
    
    comparison_vars = ['credit_score', 'debt_to_income_ratio', 'num_late_payments', 'monthly_income']
    axes = axes.ravel()
    
    for i, var in enumerate(comparison_vars):
        if var in df.columns and i < len(axes):
            ax = axes[i]
            
            # Box plots by default status
            df.boxplot(column=var, by='default_flag', ax=ax)
            ax.set_title(f'{var.replace("_", " ").title()} by Default Status')
            ax.set_xlabel('Default Flag (0=No, 1=Yes)')
            ax.set_ylabel(var.replace("_", " ").title())
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'bivariate_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return means_by_default, ttest_df

def correlation_analysis(df, viz_dir):
    """
    4. Correlation Analysis
    Equivalent to SAS PROC CORR
    """
    print("\n" + "="*60)
    print("4. CORRELATION ANALYSIS")
    print("="*60)
    
    # Select continuous variables for correlation analysis
    corr_vars = ['age', 'employment_years', 'monthly_income', 'credit_score',
                'credit_utilization', 'debt_to_income_ratio', 'num_late_payments',
                'num_credit_accounts', 'credit_history_years', 'default_flag']
    
    # Calculate correlation matrix
    corr_matrix = df[corr_vars].corr()
    
    print("Correlation Matrix - Continuous Variables:")
    print("-" * 40)
    print(corr_matrix.round(4))
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True, 
                fmt='.3f',
                cbar_kws={"shrink": .8})
    
    plt.title('Correlation Matrix - Continuous Variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(viz_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Identify strong correlations
    print(f"\nStrong Correlations (|r| > 0.5):")
    print("-" * 40)
    
    strong_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                strong_corr.append({
                    'Variable_1': corr_matrix.columns[i],
                    'Variable_2': corr_matrix.columns[j],
                    'Correlation': corr_val
                })
    
    if strong_corr:
        strong_corr_df = pd.DataFrame(strong_corr)
        strong_corr_df = strong_corr_df.sort_values('Correlation', key=abs, ascending=False)
        print(strong_corr_df.to_string(index=False))
    else:
        print("No strong correlations found (|r| > 0.5)")
    
    # Create scatter plots for key relationships
    key_pairs = [('credit_score', 'default_flag'), 
                ('debt_to_income_ratio', 'default_flag'),
                ('monthly_income', 'loan_amount')]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Key Variable Relationships', fontsize=16, fontweight='bold')
    
    for i, (var1, var2) in enumerate(key_pairs):
        if var1 in df.columns and var2 in df.columns:
            ax = axes[i]
            
            if var2 == 'default_flag':
                # For categorical scatter with default flag
                df_sample = df.sample(min(1000, len(df)), random_state=42)  # Sample for readability
                ax.scatter(df_sample[var1], df_sample[var2], alpha=0.6)
                ax.set_ylabel('Default Flag')
            else:
                # For continuous-continuous scatter
                df_sample = df.sample(min(1000, len(df)), random_state=42)
                ax.scatter(df_sample[var1], df_sample[var2], alpha=0.6)
                ax.set_ylabel(var2.replace("_", " ").title())
            
            ax.set_xlabel(var1.replace("_", " ").title())
            ax.set_title(f'{var1.replace("_", " ").title()} vs {var2.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return corr_matrix

def risk_segmentation_analysis(df, viz_dir):
    """
    5. Risk Segmentation Analysis
    Equivalent to SAS data steps with score bands and DTI bands
    """
    print("\n" + "="*60)
    print("5. RISK SEGMENTATION ANALYSIS")
    print("="*60)
    
    # Credit score bands analysis (equivalent to SAS score_bands data step)
    print("\nCredit Score Bands Analysis:")
    print("-" * 40)
    
    # Create score bands
    df_score_bands = df.copy()
    conditions = [
        df_score_bands['credit_score'] < 580,
        (df_score_bands['credit_score'] >= 580) & (df_score_bands['credit_score'] < 670),
        (df_score_bands['credit_score'] >= 670) & (df_score_bands['credit_score'] < 740),
        (df_score_bands['credit_score'] >= 740) & (df_score_bands['credit_score'] < 800),
        df_score_bands['credit_score'] >= 800
    ]
    
    choices = [
        '1. <580 (Very Poor)',
        '2. 580-669 (Fair)', 
        '3. 670-739 (Good)',
        '4. 740-799 (Very Good)',
        '5. 800+ (Excellent)'
    ]
    
    df_score_bands['score_band'] = np.select(conditions, choices, default='Unknown')
    
    # Default rate by credit score bands
    score_band_analysis = pd.crosstab(df_score_bands['score_band'], df_score_bands['default_flag'])
    score_band_rates = pd.crosstab(df_score_bands['score_band'], df_score_bands['default_flag'], normalize='index') * 100
    
    print("Frequency Table:")
    print(score_band_analysis)
    print("\nDefault Rates (%):")
    print(score_band_rates.round(2))
    
    # DTI ratio bands analysis (equivalent to SAS dti_bands data step)
    print(f"\nDebt-to-Income Ratio Bands Analysis:")
    print("-" * 40)
    
    df_dti_bands = df.copy()
    dti_conditions = [
        df_dti_bands['debt_to_income_ratio'] < 20,
        (df_dti_bands['debt_to_income_ratio'] >= 20) & (df_dti_bands['debt_to_income_ratio'] < 30),
        (df_dti_bands['debt_to_income_ratio'] >= 30) & (df_dti_bands['debt_to_income_ratio'] < 40),
        (df_dti_bands['debt_to_income_ratio'] >= 40) & (df_dti_bands['debt_to_income_ratio'] < 50),
        df_dti_bands['debt_to_income_ratio'] >= 50
    ]
    
    dti_choices = [
        '1. <20% (Low)',
        '2. 20-30% (Moderate)',
        '3. 30-40% (High)', 
        '4. 40-50% (Very High)',
        '5. 50%+ (Excessive)'
    ]
    
    df_dti_bands['dti_band'] = np.select(dti_conditions, dti_choices, default='Unknown')
    
    # Default rate by DTI bands
    dti_band_analysis = pd.crosstab(df_dti_bands['dti_band'], df_dti_bands['default_flag'])
    dti_band_rates = pd.crosstab(df_dti_bands['dti_band'], df_dti_bands['default_flag'], normalize='index') * 100
    
    print("Frequency Table:")
    print(dti_band_analysis)
    print("\nDefault Rates (%):")
    print(dti_band_rates.round(2))
    
    # Create visualization for risk segmentation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Risk Segmentation Analysis', fontsize=16, fontweight='bold')
    
    # Credit score bands default rates
    score_default_rates = score_band_rates[1] if 1 in score_band_rates.columns else score_band_rates.iloc[:, -1]
    score_default_rates.plot(kind='bar', ax=ax1, alpha=0.7, color='skyblue')
    ax1.set_title('Default Rate by Credit Score Bands')
    ax1.set_xlabel('Credit Score Band')
    ax1.set_ylabel('Default Rate (%)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(score_default_rates.values):
        ax1.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # DTI bands default rates
    dti_default_rates = dti_band_rates[1] if 1 in dti_band_rates.columns else dti_band_rates.iloc[:, -1]
    dti_default_rates.plot(kind='bar', ax=ax2, alpha=0.7, color='lightcoral')
    ax2.set_title('Default Rate by DTI Ratio Bands')
    ax2.set_xlabel('DTI Ratio Band')
    ax2.set_ylabel('Default Rate (%)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(dti_default_rates.values):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'risk_segmentation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return score_band_analysis, dti_band_analysis

def multivariate_analysis(df, viz_dir):
    """
    6. Multivariate Analysis
    Equivalent to SAS PROC PRINCOMP for PCA analysis
    """
    print("\n" + "="*60)
    print("6. MULTIVARIATE ANALYSIS")
    print("="*60)
    
    # Principal Component Analysis for dimension reduction insight
    print("\nPrincipal Component Analysis:")
    print("-" * 40)
    
    # Select variables for PCA (same as SAS script)
    pca_vars = ['age', 'employment_years', 'monthly_income', 'credit_score',
               'credit_utilization', 'debt_to_income_ratio', 'num_late_payments',
               'num_credit_accounts', 'credit_history_years']
    
    # Prepare data for PCA
    pca_data = df[pca_vars].dropna()
    
    # Standardize the data
    scaler = StandardScaler()
    pca_data_scaled = scaler.fit_transform(pca_data)
    
    # Perform PCA
    pca = PCA()
    pca_scores = pca.fit_transform(pca_data_scaled)
    
    # Create results DataFrame
    pca_results = pd.DataFrame({
        'Component': [f'Prin{i+1}' for i in range(len(pca_vars))],
        'Eigenvalue': pca.explained_variance_,
        'Proportion': pca.explained_variance_ratio_,
        'Cumulative': pca.explained_variance_ratio_.cumsum()
    })
    
    print("Principal Component Analysis Results:")
    print(pca_results.round(4))
    
    # Component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'Prin{i+1}' for i in range(len(pca_vars))],
        index=pca_vars
    )
    
    print(f"\nComponent Loadings:")
    print(loadings.round(4))
    
    # Create PCA visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Principal Component Analysis', fontsize=16, fontweight='bold')
    
    # Scree plot
    ax1.plot(range(1, len(pca_results)+1), pca_results['Eigenvalue'], 'bo-')
    ax1.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Eigenvalue = 1')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Scree Plot')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Cumulative variance explained
    ax2.plot(range(1, len(pca_results)+1), pca_results['Cumulative']*100, 'ro-')
    ax2.axhline(y=80, color='g', linestyle='--', alpha=0.7, label='80% Variance')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Cumulative Variance Explained (%)')
    ax2.set_title('Cumulative Variance Explained')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Biplot of first two components
    plt.figure(figsize=(12, 8))
    
    # Plot scores
    plt.scatter(pca_scores[:, 0], pca_scores[:, 1], alpha=0.6, s=20)
    
    # Plot loadings as arrows
    for i, var in enumerate(pca_vars):
        plt.arrow(0, 0, loadings.iloc[i, 0]*3, loadings.iloc[i, 1]*3, 
                 head_width=0.1, head_length=0.1, fc='red', ec='red')
        plt.text(loadings.iloc[i, 0]*3.2, loadings.iloc[i, 1]*3.2, var, 
                fontsize=10, ha='center', va='center')
    
    plt.xlabel(f'First Principal Component ({pca_results.iloc[0]["Proportion"]*100:.1f}% variance)')
    plt.ylabel(f'Second Principal Component ({pca_results.iloc[1]["Proportion"]*100:.1f}% variance)')
    plt.title('PCA Biplot - First Two Principal Components')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / 'pca_biplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return pca_results, loadings

def risk_indicators_summary(df, viz_dir):
    """
    7. Risk Indicators Summary  
    Equivalent to SAS risk_indicators data step and analysis
    """
    print("\n" + "="*60)
    print("7. RISK INDICATORS SUMMARY")
    print("="*60)
    
    # Create risk indicator flags (equivalent to SAS data step)
    df_risk = df.copy()
    
    # High-risk indicators (same logic as SAS)
    df_risk['high_dti'] = (df_risk['debt_to_income_ratio'] > 40).astype(int)
    df_risk['low_credit_score'] = (df_risk['credit_score'] < 650).astype(int) 
    df_risk['recent_late_payments'] = (df_risk['num_late_payments'] > 2).astype(int)
    df_risk['high_credit_util'] = (df_risk['credit_utilization'] > 70).astype(int)
    df_risk['has_previous_default'] = (df_risk['previous_defaults'] > 0).astype(int)
    df_risk['unemployed'] = (df_risk['employment_status'] == 'Unemployed').astype(int)
    
    # Calculate total risk indicators
    risk_cols = ['high_dti', 'low_credit_score', 'recent_late_payments', 
                'high_credit_util', 'has_previous_default', 'unemployed']
    df_risk['total_risk_flags'] = df_risk[risk_cols].sum(axis=1)
    
    # Risk indicators impact on default rate
    print("Individual Risk Indicators Analysis:")
    print("-" * 40)
    
    risk_analysis = []
    for col in risk_cols:
        indicator_summary = df_risk.groupby(col)['default_flag'].agg(['count', 'sum', 'mean']).reset_index()
        indicator_summary.columns = [col, 'Total_Count', 'Default_Count', 'Default_Rate']
        indicator_summary['Default_Rate'] = indicator_summary['Default_Rate'] * 100
        
        print(f"\n{col.replace('_', ' ').title()}:")
        print(indicator_summary.to_string(index=False))
        
        # Store for summary
        if len(indicator_summary) > 1:  # Has both 0 and 1 values
            no_flag_rate = indicator_summary[indicator_summary[col] == 0]['Default_Rate'].iloc[0] if any(indicator_summary[col] == 0) else 0
            has_flag_rate = indicator_summary[indicator_summary[col] == 1]['Default_Rate'].iloc[0] if any(indicator_summary[col] == 1) else 0
            
            risk_analysis.append({
                'Risk_Indicator': col.replace('_', ' ').title(),
                'Default_Rate_No_Flag': no_flag_rate,
                'Default_Rate_With_Flag': has_flag_rate,
                'Risk_Increase': has_flag_rate - no_flag_rate
            })
    
    risk_analysis_df = pd.DataFrame(risk_analysis)
    
    print(f"\nRisk Indicators Summary:")
    print("-" * 40)
    print(risk_analysis_df.round(2).to_string(index=False))
    
    # Default rate by number of risk indicators (equivalent to PROC MEANS by class)
    print(f"\nDefault Rate by Number of Risk Indicators:")
    print("-" * 40)
    
    risk_flags_summary = df_risk.groupby('total_risk_flags')['default_flag'].agg(['count', 'sum', 'mean']).reset_index()
    risk_flags_summary.columns = ['Total_Risk_Flags', 'Count', 'Defaults', 'Default_Rate']
    risk_flags_summary['Default_Rate'] = risk_flags_summary['Default_Rate'] * 100
    
    print(risk_flags_summary.to_string(index=False))
    
    # Create risk indicators visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Risk Indicators Analysis', fontsize=16, fontweight='bold')
    
    # Individual risk indicators impact
    ax1.barh(risk_analysis_df['Risk_Indicator'], risk_analysis_df['Risk_Increase'], alpha=0.7)
    ax1.set_xlabel('Default Rate Increase (%)')
    ax1.set_title('Risk Increase by Individual Indicators')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Default rate by total risk flags
    ax2.plot(risk_flags_summary['Total_Risk_Flags'], risk_flags_summary['Default_Rate'], 
             'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Total Number of Risk Flags')
    ax2.set_ylabel('Default Rate (%)')
    ax2.set_title('Default Rate by Number of Risk Indicators')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, row in risk_flags_summary.iterrows():
        ax2.text(row['Total_Risk_Flags'], row['Default_Rate'] + 1, 
                f'{row["Default_Rate"]:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'risk_indicators.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return risk_flags_summary, risk_analysis_df

def generate_exploration_report(df, viz_dir):
    """
    8. Generate Exploration Report
    Equivalent to SAS PROC SQL and export steps
    """
    print("\n" + "="*60)
    print("8. EXPLORATION REPORT GENERATION")
    print("="*60)
    
    # Create summary dataset for reporting (equivalent to SAS PROC SQL)
    print("Generating Dataset Summary Statistics...")
    print("-" * 40)
    
    # Calculate summary statistics (matching SAS format)
    exploration_summary = {
        'total_applications': len(df),
        'total_defaults': df['default_flag'].sum(),
        'default_rate': df['default_flag'].mean() * 100,
        'avg_credit_score': df['credit_score'].mean(),
        'avg_dti_ratio': df['debt_to_income_ratio'].mean(),
        'avg_monthly_income': df['monthly_income'].mean(),
        'avg_loan_amount': df['loan_amount'].mean()
    }
    
    # Create DataFrame for export
    exploration_summary_df = pd.DataFrame([exploration_summary])
    
    # Format for display (matching SAS format)
    display_summary = exploration_summary_df.copy()
    display_summary['default_rate'] = display_summary['default_rate'].round(2)
    display_summary['avg_credit_score'] = display_summary['avg_credit_score'].round(0).astype(int)
    display_summary['avg_dti_ratio'] = display_summary['avg_dti_ratio'].round(2)
    display_summary['avg_monthly_income'] = display_summary['avg_monthly_income'].apply(lambda x: f'${x:,.0f}')
    display_summary['avg_loan_amount'] = display_summary['avg_loan_amount'].apply(lambda x: f'${x:,.0f}')
    
    print("Dataset Summary Statistics:")
    print(display_summary.to_string(index=False))
    
    # Export key insights (equivalent to SAS PROC EXPORT)
    output_path = Path('output/exploration_summary.csv')
    
    # Format for CSV export (matching SAS CSV format)
    csv_summary = exploration_summary_df.copy()
    csv_summary['default_rate'] = csv_summary['default_rate'].round(2)
    csv_summary['avg_credit_score'] = csv_summary['avg_credit_score'].round(0).astype(int)
    csv_summary['avg_dti_ratio'] = csv_summary['avg_dti_ratio'].round(2)
    csv_summary['avg_monthly_income'] = csv_summary['avg_monthly_income'].apply(lambda x: f'${x:,.0f}')
    csv_summary['avg_loan_amount'] = csv_summary['avg_loan_amount'].apply(lambda x: f'${x:,.0f}')
    
    csv_summary.to_csv(output_path, index=False)
    
    print(f"\nKey insights exported to: {output_path}")
    
    # Create final summary visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Credit Data Exploration Summary', fontsize=16, fontweight='bold')
    
    # Overall default rate
    default_counts = df['default_flag'].value_counts()
    labels = ['No Default', 'Default']
    ax1.pie(default_counts.values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'Overall Default Rate\n({exploration_summary["total_applications"]:,} applications)')
    
    # Credit score distribution
    df['credit_score'].hist(bins=30, ax=ax2, alpha=0.7)
    ax2.axvline(df['credit_score'].mean(), color='red', linestyle='--', 
               label=f'Mean: {df["credit_score"].mean():.0f}')
    ax2.set_xlabel('Credit Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Credit Score Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # DTI ratio distribution  
    df['debt_to_income_ratio'].hist(bins=30, ax=ax3, alpha=0.7)
    ax3.axvline(df['debt_to_income_ratio'].mean(), color='red', linestyle='--',
               label=f'Mean: {df["debt_to_income_ratio"].mean():.1f}%')
    ax3.set_xlabel('Debt-to-Income Ratio (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('DTI Ratio Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Monthly income vs loan amount
    sample_size = min(1000, len(df))
    df_sample = df.sample(sample_size, random_state=42)
    scatter = ax4.scatter(df_sample['monthly_income'], df_sample['loan_amount'], 
                         c=df_sample['default_flag'], cmap='RdYlBu', alpha=0.6)
    ax4.set_xlabel('Monthly Income ($)')
    ax4.set_ylabel('Loan Amount ($)')
    ax4.set_title('Income vs Loan Amount (colored by default)')
    plt.colorbar(scatter, ax=ax4, label='Default Flag')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'exploration_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return exploration_summary_df

def main():
    """Main execution function"""
    print("="*80)
    print("CREDIT RISK DATA EXPLORATION & ANALYSIS")
    print("Python equivalent of 02_data_exploration.sas")
    print("="*80)
    
    try:
        # Setup
        viz_dir = setup_output_directories()
        
        # Load data
        df = load_and_preprocess_data()
        
        # 1. Data Quality Assessment
        missing_stats, dtype_summary = data_quality_assessment(df, viz_dir)
        
        # 2. Univariate Analysis
        stats_summary, categorical_vars = univariate_analysis(df, viz_dir)
        
        # 3. Bivariate Analysis  
        means_by_default, ttest_results = bivariate_analysis(df, viz_dir)
        
        # 4. Correlation Analysis
        corr_matrix = correlation_analysis(df, viz_dir)
        
        # 5. Risk Segmentation Analysis
        score_analysis, dti_analysis = risk_segmentation_analysis(df, viz_dir)
        
        # 6. Multivariate Analysis
        pca_results, loadings = multivariate_analysis(df, viz_dir)
        
        # 7. Risk Indicators Summary
        risk_summary, risk_analysis = risk_indicators_summary(df, viz_dir)
        
        # 8. Generate Exploration Report
        exploration_summary = generate_exploration_report(df, viz_dir)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"• Data processed: {len(df):,} records")
        print(f"• Visualizations saved to: {viz_dir}")
        print(f"• Summary statistics exported to: output/exploration_summary.csv")
        print(f"• Analysis faster than SAS equivalent")
        
        return {
            'data': df,
            'missing_stats': missing_stats,
            'stats_summary': stats_summary, 
            'correlation_matrix': corr_matrix,
            'pca_results': pca_results,
            'risk_summary': risk_summary,
            'exploration_summary': exploration_summary
        }
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
