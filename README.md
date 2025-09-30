# Bank Credit Risk Scoring Model

A complete SAS-based credit risk scoring system for evaluating loan applications and predicting default probability.

## Overview

This repository contains a production-ready credit risk scoring model that:
- Generates synthetic credit application data
- Performs feature engineering and data transformation
- Trains a logistic regression model for default prediction
- Validates model performance with comprehensive metrics
- Scores new applications with risk grades and recommendations

## Project Structure

```
├── output/
│   └── generate_credit_data.sas     # Generate synthetic training/validation data
├── feature_engineering.sas          # Feature engineering and transformation
├── train.sas                        # Train logistic regression model
├── metrics_calculation.sas          # Comprehensive model validation
└── predict.sas                      # Score new applications
```

## Prerequisites

- SAS 9.4 or later
- Write access to: `/home/u64352077/sasuser.v94/output/`

## Quick Start

### 1. Generate Training Data

```sas
%include "output/generate_credit_data.sas";
```

**Outputs:**
- `credit_train.csv` (7,000 records)
- `credit_validation.csv` (3,000 records)
- `credit_applications_full.csv` (10,000 records)

### 2. Feature Engineering

```sas
%include "feature_engineering.sas";
```

**Inputs:**
- `credit_train.csv`
- `credit_validation.csv`

**Outputs:**
- `model_features_train.csv`
- `model_features_validation.csv`

**Features Created:**
- Financial ratios (payment-to-income, loan-to-income, debt service coverage)
- Employment stability scores
- Credit quality scores
- Risk flags (high DTI, low credit, high utilization, etc.)
- Weight of Evidence (WOE) transformations
- One-hot encoded categorical variables
- Standardized continuous variables

### 3. Train Model

```sas
%include "train.sas";
```

**Inputs:**
- `model_features_train.csv`
- `model_features_validation.csv`

**Outputs:**
- `risk_scores_train.csv`
- `risk_scores_validation.csv`
- `final_model_output.csv`
- `model_coefficients.csv`
- Model stored in `work.logit_model`

**Model Features:**
- Logistic regression with stepwise selection
- Risk score scale: 300-850 (FICO-like)
- Risk grades: A-F
- Risk-based pricing (interest rates)

### 4. Validate Model

```sas
%include "metrics_calculation.sas";
```

**Inputs:**
- `risk_scores_train.csv`
- `risk_scores_validation.csv`

**Outputs:**
- `validation_summary.csv` - Overall model performance
- `decile_analysis.csv` - Lift and capture rates by decile
- `threshold_analysis.csv` - Metrics at different thresholds
- `ks_statistic.csv` - Kolmogorov-Smirnov statistic
- `calibration_plot.csv` - Predicted vs actual probabilities
- `model_performance_metrics.csv` - Comprehensive metrics

**Validation Metrics:**
- ROC curve and AUC
- Gini coefficient
- KS statistic
- Confusion matrix (multiple thresholds)
- Precision, Recall, F1-score
- Decile analysis and lift charts
- Population Stability Index (PSI)
- Calibration plots

### 5. Score New Applications

```sas
%include "predict.sas";
```

**Inputs:**
- `new_applications.csv` (customer applications)
- Model from training script (`work.logit_model`)

**Outputs:**
- `new_predictions.csv`

**Output Columns:**
- `customer_id` - Customer identifier
- `pd_logistic` - Probability of default (0-1)
- `credit_risk_score` - Risk score (300-850)
- `risk_grade` - Risk grade (A-F)
- `recommendation` - Approve/Review/Decline
- `interest_rate` - Suggested interest rate
- Additional credit metrics

## Data Flow

```
output/generate_credit_data.sas
    ↓
credit_train.csv, credit_validation.csv
    ↓
feature_engineering.sas
    ↓
model_features_train.csv, model_features_validation.csv
    ↓
train.sas
    ↓
risk_scores_train.csv, risk_scores_validation.csv, work.logit_model
    ↓
metrics_calculation.sas (validation metrics)
    ↓
predict.sas + new_applications.csv
    ↓
new_predictions.csv
```

## Model Performance

Expected performance metrics:
- **AUC**: 0.70-0.80
- **KS Statistic**: 0.30-0.40
- **Gini Coefficient**: 0.40-0.60
- **PSI**: < 0.25 (stable)

## Risk Grading System

| Risk Grade | Score Range | Default Probability | Interest Rate | Recommendation |
|-----------|-------------|---------------------|---------------|----------------|
| A | 750+ | < 10% | 5.0% | Approve |
| B | 700-749 | 10-15% | 7.0% | Approve |
| C | 650-699 | 15-25% | 9.0% | Review |
| D | 600-649 | 25-35% | 12.0% | Decline |
| E | 550-599 | 35-50% | 15.0% | Decline |
| F | < 550 | > 50% | 20.0% | Decline |

## Key Features Used in Model

### Core Financial Metrics
- Credit score
- Debt-to-income ratio
- Credit utilization
- Payment-to-income ratio
- Loan-to-income ratio

### Employment & Stability
- Employment years
- Employment score
- Employment status (unemployed flag)

### Credit History
- Number of late payments
- Previous defaults
- Has delinquency flag
- Credit quality score
- Credit history years

### Risk Flags
- High DTI (> 43%)
- Low credit score (< 620)
- High utilization (> 75%)
- Recent default
- Unstable employment (< 2 years)
- Total risk flags

### Demographics
- Age
- Monthly income
- Loan amount
- Affordability score

### Categorical Indicators
- Employment status (full-time, self-employed, unemployed)
- Home ownership (rent, mortgage)
- Loan purpose (debt consolidation, auto, personal)
- Education level (bachelors, masters, doctorate)

## File Formats

### Input Format for New Applications (`new_applications.csv`)

Required columns:
- `customer_id` - Unique identifier
- `age` - Customer age
- `employment_status` - Full-time/Self-employed/Part-time/Retired/Unemployed
- `employment_years` - Years in current employment
- `education` - High School/Bachelors/Masters/Doctorate
- `monthly_income` - Monthly income (numeric, no formatting)
- `annual_income` - Annual income (numeric, no formatting)
- `home_ownership` - Rent/Mortgage/Own
- `loan_purpose` - Debt Consolidation/Home Improvement/Auto/Personal/Medical
- `loan_amount` - Requested loan amount (numeric, no formatting)
- `loan_term_months` - Loan term in months
- `monthly_payment` - Monthly payment amount
- `total_monthly_debt` - Total monthly debt obligations
- `credit_score` - Credit score (300-850)
- `credit_utilization` - Credit utilization percentage (0-100)
- `debt_to_income_ratio` - DTI ratio percentage
- `num_late_payments` - Number of late payments
- `num_credit_accounts` - Number of credit accounts
- `credit_history_years` - Years of credit history
- `previous_defaults` - Number of previous defaults

## Configuration

All scripts use the output directory:
```sas
/home/u64352077/sasuser.v94/output/
```

Update this path in all scripts if you need to change the location.

## Important Notes

- **Numeric Variables**: All numeric variables (income, loan_amount, etc.) must be plain numbers without dollar signs, commas, or other formatting
- **Probabilities**: The model converts logits to probabilities using: `p = 1 / (1 + exp(-logit))`
- **Feature Engineering**: Includes standardization of continuous variables and WOE transformation
- **Model Type**: Stepwise logistic regression for automatic feature selection

## Troubleshooting

### Issue: Variables are character instead of numeric
**Solution**: In `output/generate_credit_data.sas`, remove dollar formatting from income and loan amount variables

### Issue: pd_logistic values are too large (not between 0-1)
**Solution**: Ensure inverse logit transformation is applied in scoring:
```sas
pd_logistic = 1 / (1 + exp(-predicted_logit));
```

### Issue: Model performance is poor
**Solution**:
- Check data quality
- Verify default rate is between 5-15%
- Ensure proper feature engineering was applied
- Check for missing values

### Issue: Training and validation sets have different columns
**Solution**: Ensure feature engineering script applies the same transformations to both datasets

## Output Directory Structure

```
output/
├── credit_train.csv                    # Training data (7,000 records)
├── credit_validation.csv               # Validation data (3,000 records)
├── credit_applications_full.csv        # Full dataset (10,000 records)
├── model_features_train.csv            # Engineered training features
├── model_features_validation.csv       # Engineered validation features
├── risk_scores_train.csv               # Scored training set
├── risk_scores_validation.csv          # Scored validation set
├── final_model_output.csv              # Final model results
├── model_coefficients.csv              # Model parameters
├── validation_summary.csv              # Overall validation metrics
├── decile_analysis.csv                 # Decile lift analysis
├── threshold_analysis.csv              # Performance at different thresholds
├── ks_statistic.csv                    # KS statistic
├── calibration_plot.csv                # Calibration data
├── model_performance_metrics.csv       # Comprehensive metrics
└── new_predictions.csv                 # Predictions for new applications
```

## Pipeline Execution

To run the complete pipeline:

```sas
/* Step 1: Generate data */
%include "output/generate_credit_data.sas";

/* Step 2: Feature engineering */
%include "feature_engineering.sas";

/* Step 3: Train model */
%include "train.sas";

/* Step 4: Validate model */
%include "metrics_calculation.sas";

/* Step 5: Score new applications (requires new_applications.csv) */
%include "predict.sas";
```