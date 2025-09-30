/*****************************************************************************
 * Bank Credit Risk Scoring Model - Step 1: Generate Synthetic Credit Data
 *
 * Purpose: Generate realistic synthetic customer credit data for model training
 * Author: Risk Analytics Team
 * Date: 2025
 *
 * This script creates a synthetic dataset representing bank customers with:
 * - Demographic information (age, income, employment)
 * - Credit history metrics
 * - Financial obligations
 * - Default indicator (target variable)
 *****************************************************************************/

/* Set random seed for reproducibility */
data _null_;
   call streaminit(12345);
run;

/* Generate 10,000 synthetic customer records */
data work.credit_applications;
   /* Customer demographics */
   length customer_id $10 employment_status $20 education $20 home_ownership $10;
   format application_date date9.;

   do i = 1 to 10000;
      /* Generate unique customer ID */
      customer_id = cat('CUST', put(i, z6.));

      /* Application date (random dates in last 2 years) */
      application_date = today() - floor(rand('uniform') * 730);

      /* Age distribution (18-75, normal distribution centered at 42) */
      age = max(18, min(75, round(rand('normal', 42, 12))));

      /* Employment years (correlated with age) */
      if age < 25 then employment_years = max(0, round(rand('uniform') * 3));
      else if age < 35 then employment_years = max(0, round(rand('uniform') * 10));
      else employment_years = max(0, round(rand('uniform') * min(20, age - 20)));

      /* Employment status */
      emp_rand = rand('uniform');
      if emp_rand < 0.65 then employment_status = 'Full-time';
      else if emp_rand < 0.80 then employment_status = 'Self-employed';
      else if emp_rand < 0.90 then employment_status = 'Part-time';
      else if emp_rand < 0.95 then employment_status = 'Retired';
      else employment_status = 'Unemployed';

      /* Education level */
      edu_rand = rand('uniform');
      if edu_rand < 0.30 then education = 'High School';
      else if edu_rand < 0.60 then education = 'Bachelors';
      else if edu_rand < 0.80 then education = 'Masters';
      else if edu_rand < 0.90 then education = 'Doctorate';
      else education = 'Other';

      /* Monthly income (log-normal distribution) */
      base_income = 3000;
      if education = 'High School' then base_income = 2500;
      else if education = 'Bachelors' then base_income = 4000;
      else if education = 'Masters' then base_income = 5500;
      else if education = 'Doctorate' then base_income = 7000;

      if employment_status = 'Full-time' then income_mult = 1.0;
      else if employment_status = 'Self-employed' then income_mult = 1.2;
      else if employment_status = 'Part-time' then income_mult = 0.5;
      else if employment_status = 'Retired' then income_mult = 0.6;
      else income_mult = 0.1;

      monthly_income = round(base_income * income_mult * exp(rand('normal', 0, 0.3)), 100);
      annual_income = monthly_income * 12;

      /* Home ownership */
      home_rand = rand('uniform');
      if home_rand < 0.40 then home_ownership = 'Rent';
      else if home_rand < 0.70 then home_ownership = 'Mortgage';
      else home_ownership = 'Own';

      /* Number of dependents */
      num_dependents = max(0, round(rand('poisson', 1.5)));

      /* Credit history metrics */
      credit_history_years = max(0, min(age - 18, round(rand('gamma', 5))));

      /* Number of credit accounts */
      num_credit_accounts = max(1, round(rand('poisson', 3)));

      /* Number of late payments in last 2 years */
      if rand('uniform') < 0.7 then num_late_payments = 0;
      else num_late_payments = round(rand('exponential', 2));

      /* Credit utilization ratio (0-100%) */
      credit_utilization = min(100, max(0, rand('beta', 2, 5) * 100));

      /* Previous defaults */
      if rand('uniform') < 0.92 then previous_defaults = 0;
      else previous_defaults = max(0, round(rand('poisson', 0.5)));

      /* Loan details */
      loan_amount = round(annual_income * rand('uniform', 0.1, 3), 1000);
      loan_term_months = rand('table', 0.2, 0.5, 0.3) * 12 + 12; /* 12, 24, or 36 months */
      loan_purpose_rand = rand('uniform');
      if loan_purpose_rand < 0.25 then loan_purpose = 'Debt Consolidation';
      else if loan_purpose_rand < 0.45 then loan_purpose = 'Home Improvement';
      else if loan_purpose_rand < 0.60 then loan_purpose = 'Auto';
      else if loan_purpose_rand < 0.75 then loan_purpose = 'Personal';
      else if loan_purpose_rand < 0.85 then loan_purpose = 'Medical';
      else loan_purpose = 'Other';

      /* Monthly loan payment */
      interest_rate = 0.05 + (credit_utilization/100) * 0.15 + (num_late_payments * 0.02);
      monthly_rate = interest_rate / 12;
      monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**loan_term_months) /
                       ((1 + monthly_rate)**loan_term_months - 1);

      /* Debt-to-income ratio */
      existing_monthly_debt = monthly_income * rand('beta', 2, 8);
      total_monthly_debt = existing_monthly_debt + monthly_payment;
      debt_to_income_ratio = (total_monthly_debt / monthly_income) * 100;

      /* Credit score simulation (300-850) */
      base_score = 650;
      score_adjustment = -num_late_payments * 30
                        - previous_defaults * 100
                        - (credit_utilization - 30) * 2
                        + credit_history_years * 5
                        + (employment_years * 2)
                        + rand('normal', 0, 30);
      credit_score = max(300, min(850, round(base_score + score_adjustment)));

      /* Determine default probability based on risk factors */
      default_prob = 0.05; /* Base rate */

      /* Adjust probability based on risk factors */
      if credit_score < 600 then default_prob = default_prob + 0.15;
      else if credit_score < 650 then default_prob = default_prob + 0.08;
      else if credit_score < 700 then default_prob = default_prob + 0.03;
      else default_prob = default_prob - 0.02;

      if debt_to_income_ratio > 50 then default_prob = default_prob + 0.12;
      else if debt_to_income_ratio > 40 then default_prob = default_prob + 0.06;

      if employment_status = 'Unemployed' then default_prob = default_prob + 0.20;
      else if employment_status = 'Part-time' then default_prob = default_prob + 0.05;

      if num_late_payments > 3 then default_prob = default_prob + 0.15;
      else if num_late_payments > 0 then default_prob = default_prob + 0.05;

      if previous_defaults > 0 then default_prob = default_prob + 0.25;

      /* Generate default indicator (target variable) */
      if rand('uniform') < min(0.95, max(0.01, default_prob)) then default_flag = 1;
      else default_flag = 0;

      /* Risk rating (for reference) */
      if credit_score >= 750 then risk_rating = 'Excellent';
      else if credit_score >= 700 then risk_rating = 'Good';
      else if credit_score >= 650 then risk_rating = 'Fair';
      else if credit_score >= 600 then risk_rating = 'Poor';
      else risk_rating = 'Very Poor';

      output;
   end;

   drop i emp_rand edu_rand home_rand loan_purpose_rand base_income income_mult
        monthly_rate base_score score_adjustment default_prob;
run;

/* Create training and validation datasets (70/30 split) */
proc surveyselect data=work.credit_applications
                  out=work.credit_split
                  samprate=0.7
                  seed=42
                  outall;
run;

data work.credit_train work.credit_validation;
   set work.credit_split;
   if selected = 1 then output work.credit_train;
   else output work.credit_validation;
   drop selected;
run;

/* Summary statistics */
proc means data=work.credit_applications n mean std min max;
   var age employment_years monthly_income annual_income loan_amount
       credit_score credit_utilization debt_to_income_ratio num_late_payments;
   title "Credit Application Data - Summary Statistics";
run;

/* Default rate by risk rating */
proc freq data=work.credit_applications;
   tables risk_rating * default_flag / nocol nopercent;
   title "Default Rate by Risk Rating";
run;

/* Save datasets */
data work.credit_data_full;
   set work.credit_applications;
run;

/* Export all datasets to CSV */
/* Export full credit applications dataset */
proc export data=work.credit_applications
   outfile="/home/u64352077/sasuser.v94/output/credit_applications_full.csv"
   dbms=csv
   replace;
run;

/* Export training dataset */
proc export data=work.credit_train
   outfile="/home/u64352077/sasuser.v94/output/credit_train.csv"
   dbms=csv
   replace;
run;

/* Export validation dataset */
proc export data=work.credit_validation
   outfile="/home/u64352077/sasuser.v94/output/credit_validation.csv"
   dbms=csv
   replace;
run;

/* Export sample for review */
proc export data=work.credit_applications(obs=100)
   outfile="/home/u64352077/sasuser.v94/output/credit_data_sample.csv"
   dbms=csv
   replace;
run;

/* Calculate default rate using PROC SQL */
proc sql noprint;
   select mean(default_flag) * 100 into :default_rate
   from work.credit_applications;
quit;

/* Print confirmation */
%put NOTE: Successfully generated 10,000 synthetic credit records;
%put NOTE: Training set: 7,000 records;
%put NOTE: Validation set: 3,000 records;
%put NOTE: Overall default rate: %sysfunc(putn(&default_rate, 6.2))%;