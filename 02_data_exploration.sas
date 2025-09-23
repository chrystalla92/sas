/*****************************************************************************
 * Bank Credit Risk Scoring Model - Step 2: Data Exploration & Analysis
 *
 * Purpose: Explore and analyze the credit application dataset
 * Author: Risk Analytics Team
 * Date: 2025
 *
 * This script performs comprehensive exploratory data analysis including:
 * - Data quality assessment
 * - Distribution analysis
 * - Correlation analysis
 * - Risk factor identification
 *****************************************************************************/

/* Ensure data is loaded */
%if not %sysfunc(exist(work.credit_data_full)) %then %do;
   %include "/home/u64345824/sasuser.v94/bank_risk_credit_scoring/01_generate_credit_data.sas";
%end;

/* Set output options */
ods graphics on / width=800 height=600;
title "Credit Risk Data Exploration";

/*****************************************************************************
 * 1. Data Quality Assessment
 *****************************************************************************/

/* Check for missing values */
proc means data=work.credit_data_full nmiss n;
   var age employment_years monthly_income annual_income loan_amount
       credit_score credit_utilization debt_to_income_ratio num_late_payments
       num_credit_accounts credit_history_years previous_defaults;
   title2 "Missing Value Analysis";
run;

/* Data type and length analysis */
proc contents data=work.credit_data_full;
   title2 "Dataset Structure";
run;

/*****************************************************************************
 * 2. Univariate Analysis
 *****************************************************************************/

/* Continuous variables distribution */
proc univariate data=work.credit_data_full;
   var age employment_years monthly_income credit_score
       credit_utilization debt_to_income_ratio;
   histogram / normal;
   title2 "Distribution of Continuous Variables";
run;

/* Categorical variables frequency */
proc freq data=work.credit_data_full;
   tables employment_status education home_ownership loan_purpose risk_rating;
   title2 "Categorical Variables Distribution";
run;

/* Target variable distribution */
proc freq data=work.credit_data_full;
   tables default_flag / plots=freqplot;
   title2 "Target Variable Distribution (Default Rate)";
run;

/*****************************************************************************
 * 3. Bivariate Analysis - Relationship with Default
 *****************************************************************************/

/* Default rate by categorical variables */
proc freq data=work.credit_data_full;
   tables (employment_status education home_ownership risk_rating) * default_flag
          / chisq expected cellchi2 nocol nopercent;
   title2 "Default Rate by Categorical Variables";
run;

/* Mean comparison for continuous variables */
proc means data=work.credit_data_full mean std min max;
   class default_flag;
   var age employment_years monthly_income credit_score
       credit_utilization debt_to_income_ratio num_late_payments;
   title2 "Variable Means by Default Status";
run;

/* T-tests for significant differences */
proc ttest data=work.credit_data_full;
   class default_flag;
   var credit_score debt_to_income_ratio num_late_payments;
   title2 "T-Tests: Key Variables by Default Status";
run;

/*****************************************************************************
 * 4. Correlation Analysis
 *****************************************************************************/

/* Pearson correlation matrix */
proc corr data=work.credit_data_full pearson;
   var age employment_years monthly_income credit_score
       credit_utilization debt_to_income_ratio num_late_payments
       num_credit_accounts credit_history_years default_flag;
   title2 "Correlation Matrix - Continuous Variables";
run;

/* Create correlation heatmap dataset */
proc corr data=work.credit_data_full outp=work.corr_matrix noprint;
   var age employment_years monthly_income credit_score
       credit_utilization debt_to_income_ratio num_late_payments default_flag;
run;

/*****************************************************************************
 * 5. Risk Segmentation Analysis
 *****************************************************************************/

/* Credit score bands analysis */
data work.score_bands;
   set work.credit_data_full;
   if credit_score < 580 then score_band = '1. <580 (Very Poor)';
   else if credit_score < 670 then score_band = '2. 580-669 (Fair)';
   else if credit_score < 740 then score_band = '3. 670-739 (Good)';
   else if credit_score < 800 then score_band = '4. 740-799 (Very Good)';
   else score_band = '5. 800+ (Excellent)';
run;

proc freq data=work.score_bands;
   tables score_band * default_flag / nocol nopercent;
   title2 "Default Rate by Credit Score Bands";
run;

/* DTI ratio bands analysis */
data work.dti_bands;
   set work.credit_data_full;
   if debt_to_income_ratio < 20 then dti_band = '1. <20% (Low)';
   else if debt_to_income_ratio < 30 then dti_band = '2. 20-30% (Moderate)';
   else if debt_to_income_ratio < 40 then dti_band = '3. 30-40% (High)';
   else if debt_to_income_ratio < 50 then dti_band = '4. 40-50% (Very High)';
   else dti_band = '5. 50%+ (Excessive)';
run;

proc freq data=work.dti_bands;
   tables dti_band * default_flag / nocol nopercent;
   title2 "Default Rate by DTI Ratio Bands";
run;

/*****************************************************************************
 * 6. Multivariate Analysis
 *****************************************************************************/

/* Principal Component Analysis for dimension reduction insight */
proc princomp data=work.credit_data_full out=work.pca_scores;
   var age employment_years monthly_income credit_score
       credit_utilization debt_to_income_ratio num_late_payments
       num_credit_accounts credit_history_years;
   title2 "Principal Component Analysis";
run;

/*****************************************************************************
 * 7. Risk Indicators Summary
 *****************************************************************************/

/* Create risk indicator flags */
data work.risk_indicators;
   set work.credit_data_full;

   /* High-risk indicators */
   high_dti = (debt_to_income_ratio > 40);
   low_credit_score = (credit_score < 650);
   recent_late_payments = (num_late_payments > 2);
   high_credit_util = (credit_utilization > 70);
   has_previous_default = (previous_defaults > 0);
   unemployed = (employment_status = 'Unemployed');

   /* Calculate total risk indicators */
   total_risk_flags = sum(high_dti, low_credit_score, recent_late_payments,
                          high_credit_util, has_previous_default, unemployed);
run;

/* Risk indicators impact on default rate */
proc means data=work.risk_indicators mean;
   class total_risk_flags;
   var default_flag;
   title2 "Default Rate by Number of Risk Indicators";
run;

/*****************************************************************************
 * 8. Generate Exploration Report
 *****************************************************************************/

/* Create summary dataset for reporting */
proc sql;
   create table work.exploration_summary as
   select
      count(*) as total_applications,
      sum(default_flag) as total_defaults,
      mean(default_flag) * 100 as default_rate format=8.2,
      mean(credit_score) as avg_credit_score format=8.0,
      mean(debt_to_income_ratio) as avg_dti_ratio format=8.2,
      mean(monthly_income) as avg_monthly_income format=dollar12.0,
      mean(loan_amount) as avg_loan_amount format=dollar12.0
   from work.credit_data_full;
quit;

/* Print summary */
proc print data=work.exploration_summary noobs;
   title2 "Dataset Summary Statistics";
run;

/* Export key insights */
proc export data=work.exploration_summary
   outfile="/home/u64345824/sasuser.v94/bank_risk_credit_scoring/output/exploration_summary.csv"
   dbms=csv
   replace;
run;

/* Reset graphics */
ods graphics off;
title;

%put NOTE: Data exploration completed successfully;
%put NOTE: Key insights exported to output folder;