# Bank Credit Risk Scoring Model with SAS

This project implements a comprehensive credit risk scoring system for bank loan applications using SAS procedures. The workflow covers synthetic data generation, exploratory analysis, feature engineering, model training, validation, production scoring, and project archiving.

## Directory Structure

```
bank_risk_credit_scoring/
├── output/                           # Directory for outputs and reports
├── 01_generate_credit_data.sas      # Synthetic data generation
├── 02_data_exploration.sas          # Exploratory data analysis
├── 03_feature_engineering.sas       # Feature creation and transformation
├── 04_train_credit_model.sas        # Model training (decision tree)
├── 05_model_validation.sas          # Model performance evaluation
├── 06_score_new_customers.sas       # Production scoring pipeline
├── 07_archive_project.sas           # Project backup and documentation
├── tree_model.sas                    # Generated scoring code from HPSPLIT
└── README.md                         # This file
```

## Prerequisites

- SAS Base with the following procedures available:
  - PROC HPSPLIT (High-Performance Decision Trees)
  - PROC HPFOREST (Random Forest - optional)
  - PROC LOGISTIC (Logistic Regression)
  - PROC MEANS/FREQ/UNIVARIATE (Statistical Analysis)
  - PROC SGPLOT (Visualization)
- Sufficient memory for processing 10,000+ records
- Write access to output directory

## Workflow Overview

### Step 1: Generate Synthetic Credit Data (`01_generate_credit_data.sas`)

This script creates a realistic synthetic dataset of 10,000 bank loan applications with controlled relationships between features and default risk.

**What it does:**
- Generates customer demographics (age, income, employment status, education)
- Creates credit history metrics (credit score, utilization, payment history)
- Simulates financial obligations (loan amounts, debt-to-income ratios)
- Produces a binary default indicator based on risk factors
- Exports data for analysis and model training

**Key outputs:**
- `work.credit_applications` - Full synthetic dataset with 10,000 records
- `work.data_summary` - Statistical summary of generated data
- CSV export of the dataset for external analysis

**Generated Features:**
- **Demographics:** age, employment_years, employment_status, education, home_ownership
- **Financial:** monthly_income, loan_amount, debt_to_income_ratio
- **Credit History:** credit_score, credit_utilization, num_credit_lines, months_since_delinquency
- **Risk Indicators:** previous_defaults, payment_history_score
- **Target:** default_flag (0=good, 1=default)

### Step 2: Data Exploration (`02_data_exploration.sas`)

This script performs comprehensive exploratory data analysis to understand data patterns and relationships.

**What it does:**
- Calculates univariate statistics for all variables
- Analyzes default rates across categorical segments
- Creates correlation matrix for numeric features
- Generates distribution plots and histograms
- Identifies potential data quality issues

**Analysis Components:**
- **Univariate Analysis:** Distribution statistics for all variables
- **Bivariate Analysis:** Default rates by categorical features
- **Correlation Analysis:** Relationships between numeric predictors
- **Visual Analysis:** Histograms, box plots, and bar charts

**Key outputs:**
- `work.numeric_summary` - Summary statistics for numeric variables
- `work.categorical_summary` - Frequency tables for categorical variables
- `work.default_analysis` - Default rates by various segments
- Multiple visualization plots in SAS Output

### Step 3: Feature Engineering (`03_feature_engineering.sas`)

This script creates advanced features and performs data transformations to improve model performance.

**What it does:**
- Creates risk flag indicators based on business rules
- Generates interaction terms between key variables
- Performs binning of continuous variables
- Creates dummy variables for categorical features
- Normalizes numeric features for modeling

**Engineered Features:**
- **Risk Flags:**
  - `high_dti_flag` - Debt-to-income > 0.45
  - `low_credit_flag` - Credit score < 600
  - `high_utilization_flag` - Credit utilization > 0.8
  - `recent_delinquency_flag` - Delinquency within 12 months
- **Composite Scores:**
  - `total_risk_flags` - Sum of all risk indicators
  - `credit_income_ratio` - Credit score / income interaction
- **Categorical Encodings:**
  - One-hot encoding for employment_status
  - Ordinal encoding for education level

**Key outputs:**
- `work.model_data` - Dataset with engineered features
- `work.feature_importance` - Initial feature importance metrics
- Partitioned datasets for training (70%) and validation (30%)

### Step 4: Train Credit Model (`04_train_credit_model.sas`)

This script trains a decision tree model using PROC HPSPLIT for credit risk prediction.

**What it does:**
- Splits data into training and validation sets
- Trains a decision tree with optimal parameters
- Performs automatic variable selection
- Generates scoring code for production use
- Evaluates initial model performance

**Model Configuration:**
- **Algorithm:** Decision Tree (HPSPLIT)
- **Max Depth:** 10 levels
- **Min Observations:** 50 per leaf
- **Split Criterion:** Gini impurity
- **Pruning:** Cost-complexity with cross-validation

**Key outputs:**
- `work.credit_model` - Trained model object
- `tree_model.sas` - Auto-generated scoring code
- `work.train_scored` - Training predictions
- `work.valid_scored` - Validation predictions
- Variable importance ranking

### Step 5: Model Validation (`05_model_validation.sas`)

This script performs comprehensive model validation and performance assessment.

**What it does:**
- Calculates classification metrics (accuracy, precision, recall, F1)
- Generates ROC curves and calculates AUC
- Creates confusion matrices for different thresholds
- Performs lift and gain chart analysis
- Conducts stability analysis across segments

**Performance Metrics:**
- **Classification Metrics:**
  - Overall accuracy
  - Sensitivity (recall) for defaults
  - Specificity for non-defaults
  - Precision and F1 score
- **Ranking Metrics:**
  - AUC-ROC score
  - Gini coefficient
  - KS statistic
- **Business Metrics:**
  - Lift at various deciles
  - Cumulative gain
  - Expected loss estimates

**Key outputs:**
- `work.model_metrics` - Complete performance metrics
- `work.confusion_matrix` - Classification results at optimal threshold
- `work.roc_data` - ROC curve coordinates
- `work.lift_chart` - Lift and gain analysis
- Performance visualization plots

### Step 6: Score New Customers (`06_score_new_customers.sas`)

This script implements the production scoring pipeline for new loan applications.

**What it does:**
- Generates 100 new synthetic applications for scoring
- Applies the same feature engineering pipeline
- Scores using the trained model
- Assigns risk categories based on probability thresholds
- Creates approval recommendations

**Risk Categorization:**
- **Low Risk:** Default probability < 0.1 → Auto-approve
- **Medium Risk:** Default probability 0.1-0.3 → Manual review
- **High Risk:** Default probability > 0.3 → Decline/require additional documentation

**Key outputs:**
- `work.new_applications` - New customer data
- `work.scored_applications` - Scored results with probabilities
- `work.risk_summary` - Distribution of risk categories
- `work.approval_decisions` - Final lending recommendations
- CSV exports for integration with other systems

### Step 7: Archive Project (`07_archive_project.sas`)

This script creates comprehensive documentation and backup of the entire project.

**What it does:**
- Lists all project files and their purposes
- Creates a manifest of datasets and outputs
- Documents model parameters and performance
- Generates a project summary report
- Attempts to create backup (within SAS constraints)

**Documentation Components:**
- File inventory with descriptions
- Dataset catalog with record counts
- Model configuration summary
- Performance metrics archive
- Timestamp and version information

**Key outputs:**
- `work.project_manifest` - Complete file listing
- `work.model_documentation` - Model specifications
- Project summary report in log
- Backup instructions for manual archiving

## Running the Pipeline

Execute the scripts in sequence:

```sas
/* Step 1: Generate synthetic credit data */
%include "/path/to/bank_risk_credit_scoring/01_generate_credit_data.sas";

/* Step 2: Explore and understand the data */
%include "/path/to/bank_risk_credit_scoring/02_data_exploration.sas";

/* Step 3: Create features for modeling */
%include "/path/to/bank_risk_credit_scoring/03_feature_engineering.sas";

/* Step 4: Train the credit risk model */
%include "/path/to/bank_risk_credit_scoring/04_train_credit_model.sas";

/* Step 5: Validate model performance */
%include "/path/to/bank_risk_credit_scoring/05_model_validation.sas";

/* Step 6: Score new applications */
%include "/path/to/bank_risk_credit_scoring/06_score_new_customers.sas";

/* Step 7: Archive and document project */
%include "/path/to/bank_risk_credit_scoring/07_archive_project.sas";
```

## Expected Results

1. **Data Generation**: 10,000 synthetic loan applications with ~20% default rate
2. **Model Training**: Decision tree with 75-80% accuracy on validation set
3. **Risk Scoring**: Classification of new applications into risk categories
4. **Business Impact**: Automated approval decisions for low-risk applications

## Performance Considerations

- The model uses decision trees for interpretability over black-box methods
- Feature engineering significantly improves model performance (~10% lift)
- Validation uses 30% holdout to ensure robust performance estimates
- Production scoring processes ~1000 applications per second

## Troubleshooting

### Memory Issues
If you encounter memory errors:
- Reduce the number of observations in data generation
- Use PROC HPSPLIT options for memory management
- Consider sampling for initial model development

### Missing Procedures
If PROC HPSPLIT or HPFOREST are not available:
- Use PROC DTREE as an alternative for decision trees
- PROC LOGISTIC can be substituted for a simpler model
- Check SAS/STAT licensing for advanced procedures

### Path Issues
- Update all file paths to match your environment
- Ensure write permissions for output directory
- Verify SAS work library has sufficient space

### Model Performance
If model performance is poor:
- Review feature engineering for additional predictors
- Adjust decision tree parameters (depth, leaf size)
- Consider ensemble methods if available
- Check for class imbalance and adjust accordingly

## File Descriptions

| File | Purpose | Key Components |
|------|---------|---------------|
| `01_generate_credit_data.sas` | Create synthetic training data | 10,000 records with demographics, credit history, default flag |
| `02_data_exploration.sas` | Exploratory data analysis | Statistics, distributions, correlations |
| `03_feature_engineering.sas` | Feature creation and transformation | Risk flags, interactions, normalization |
| `04_train_credit_model.sas` | Model training | Decision tree with HPSPLIT |
| `05_model_validation.sas` | Performance evaluation | ROC, confusion matrix, lift charts |
| `06_score_new_customers.sas` | Production scoring | Risk categorization, approval decisions |
| `07_archive_project.sas` | Project documentation | File manifest, model archive |
| `tree_model.sas` | Auto-generated scoring code | Production-ready scoring logic |

## Business Value

This credit risk scoring model provides:
- **Risk Reduction**: Identify high-risk applications before approval
- **Efficiency**: Automate low-risk application approvals
- **Compliance**: Transparent, interpretable decision logic
- **Scalability**: Process thousands of applications quickly
- **Flexibility**: Easy to update with new data or business rules

## Next Steps

After running the pipeline, you can:
1. Deploy the scoring code to production systems
2. Set up monitoring for model drift
3. Implement A/B testing for model improvements
4. Integrate with existing loan origination systems
5. Extend to other credit products (credit cards, mortgages)
6. Add more sophisticated models (gradient boosting, neural networks)

## Notes

- The project uses synthetic data for demonstration purposes
- Default rates and patterns are simulated but realistic
- Model parameters should be tuned for production use
- Regular retraining is recommended as portfolio changes
- Consider regulatory requirements for model governance