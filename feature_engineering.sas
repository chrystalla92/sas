/*****************************************************************************
 * Bank Credit Risk Scoring Model - Step 3: Feature Engineering
 *
 * Purpose: Create derived features and transform variables for modeling
 * Author: Risk Analytics Team
 * Date: 2025
 *
 * This script performs feature engineering including:
 * - Creating risk indicators and ratios
 * - Binning continuous variables (Weight of Evidence)
 * - Handling categorical variables
 * - Feature scaling and transformation
 *****************************************************************************/

/* Read input data from CSV */
proc import datafile="/home/u64352077/sasuser.v94/output/credit_train.csv"
    out=work.credit_train
    dbms=csv
    replace;
    getnames=yes;
run;

proc import datafile="/home/u64352077/sasuser.v94/output/credit_validation.csv"
    out=work.credit_validation
    dbms=csv
    replace;
    getnames=yes;
run;

/*****************************************************************************
 * 1. Create Derived Features
 *****************************************************************************/

data work.credit_features;
   set work.credit_train;

   /* Financial ratios */
   payment_to_income_ratio = (monthly_payment / monthly_income) * 100;
   loan_to_income_ratio = (loan_amount / annual_income);
   debt_service_coverage = monthly_income / total_monthly_debt;

   /* Employment stability score */
   if employment_status = 'Full-time' then emp_stability = 5;
   else if employment_status = 'Self-employed' then emp_stability = 4;
   else if employment_status = 'Retired' then emp_stability = 3;
   else if employment_status = 'Part-time' then emp_stability = 2;
   else emp_stability = 1;

   /* Weighted employment score */
   employment_score = emp_stability * employment_years;

   /* Credit history quality score */
   credit_quality_score = credit_score - (num_late_payments * 50) - (previous_defaults * 150);

   /* Age groups for risk assessment */
   if age < 25 then age_group = 1;
   else if age < 35 then age_group = 2;
   else if age < 45 then age_group = 3;
   else if age < 55 then age_group = 4;
   else if age < 65 then age_group = 5;
   else age_group = 6;

   /* Income stability indicator */
   income_stability = employment_years / age * 100;

   /* Credit behavior score */
   if credit_utilization < 30 then credit_util_score = 5;
   else if credit_utilization < 50 then credit_util_score = 4;
   else if credit_utilization < 70 then credit_util_score = 3;
   else if credit_utilization < 90 then credit_util_score = 2;
   else credit_util_score = 1;

   /* Delinquency indicator */
   has_delinquency = (num_late_payments > 0);

   /* High-risk flags */
   flag_high_dti = (debt_to_income_ratio > 43);
   flag_low_credit = (credit_score < 620);
   flag_high_util = (credit_utilization > 75);
   flag_recent_default = (previous_defaults > 0);
   flag_unstable_employment = (employment_years < 2);

   /* Total risk flags */
   total_risk_flags = sum(flag_high_dti, flag_low_credit, flag_high_util,
                          flag_recent_default, flag_unstable_employment);

   /* Loan affordability score */
   affordability_score = (monthly_income - total_monthly_debt) / monthly_payment;

   /* Credit age to loan ratio */
   credit_to_loan_years = credit_history_years / (loan_term_months / 12);
run;

/*****************************************************************************
 * 2. Weight of Evidence (WOE) Transformation for Key Variables
 *****************************************************************************/

/* Calculate WOE for credit score bands */
proc sql;
   create table work.woe_credit_score as
   select
      case
         when credit_score < 580 then 'A. <580'
         when credit_score < 650 then 'B. 580-649'
         when credit_score < 700 then 'C. 650-699'
         when credit_score < 750 then 'D. 700-749'
         else 'E. 750+'
      end as credit_band,
      sum(default_flag = 1) as bad_count,
      sum(default_flag = 0) as good_count,
      calculated bad_count / (select sum(default_flag = 1) from work.credit_features) as bad_rate,
      calculated good_count / (select sum(default_flag = 0) from work.credit_features) as good_rate,
      log(calculated good_rate / calculated bad_rate) as woe
   from work.credit_features
   group by calculated credit_band;
quit;

/* Apply WOE transformation */
proc sql;
   create table work.credit_features_woe as
   select a.*,
      case
         when a.credit_score < 580 then b1.woe
         when a.credit_score < 650 then b2.woe
         when a.credit_score < 700 then b3.woe
         when a.credit_score < 750 then b4.woe
         else b5.woe
      end as woe_credit_score
   from work.credit_features a
   cross join (select woe from work.woe_credit_score where credit_band = 'A. <580') b1
   cross join (select woe from work.woe_credit_score where credit_band = 'B. 580-649') b2
   cross join (select woe from work.woe_credit_score where credit_band = 'C. 650-699') b3
   cross join (select woe from work.woe_credit_score where credit_band = 'D. 700-749') b4
   cross join (select woe from work.woe_credit_score where credit_band = 'E. 750+') b5;
quit;

/*****************************************************************************
 * 3. Create Dummy Variables for Categorical Features
 *****************************************************************************/

data work.credit_features_dummy;
   set work.credit_features_woe;

   /* Employment status dummies */
   emp_fulltime = (employment_status = 'Full-time');
   emp_selfemployed = (employment_status = 'Self-employed');
   emp_parttime = (employment_status = 'Part-time');
   emp_retired = (employment_status = 'Retired');
   emp_unemployed = (employment_status = 'Unemployed');

   /* Education dummies */
   edu_highschool = (education = 'High School');
   edu_bachelors = (education = 'Bachelors');
   edu_masters = (education = 'Masters');
   edu_doctorate = (education = 'Doctorate');

   /* Home ownership dummies */
   home_rent = (home_ownership = 'Rent');
   home_mortgage = (home_ownership = 'Mortgage');
   home_own = (home_ownership = 'Own');

   /* Loan purpose dummies */
   purpose_debt = (loan_purpose = 'Debt Consolidation');
   purpose_home = (loan_purpose = 'Home Improvement');
   purpose_auto = (loan_purpose = 'Auto');
   purpose_personal = (loan_purpose = 'Personal');
   purpose_medical = (loan_purpose = 'Medical');
run;

/*****************************************************************************
 * 4. Feature Scaling and Normalization
 *****************************************************************************/

/* Standardize numeric features */
proc stdize data=work.credit_features_dummy
           out=work.credit_features_scaled
           method=std;
   var age employment_years monthly_income annual_income loan_amount
       credit_utilization debt_to_income_ratio num_late_payments
       payment_to_income_ratio loan_to_income_ratio employment_score
       credit_quality_score affordability_score;
run;

/*****************************************************************************
 * 5. Information Value (IV) Calculation for Feature Selection
 *****************************************************************************/

%macro calculate_iv(var);
   proc sql;
      create table work.iv_&var as
      select
         "&var" as variable,
         sum((good_rate - bad_rate) * woe) as information_value
      from (
         select
            &var as value,
            sum(default_flag = 1) / (select sum(default_flag = 1) from work.credit_features_scaled) as bad_rate,
            sum(default_flag = 0) / (select sum(default_flag = 0) from work.credit_features_scaled) as good_rate,
            log(calculated good_rate / calculated bad_rate) as woe
         from work.credit_features_scaled
         group by &var
      );
   quit;
%mend;

/* Calculate IV for key binary features */
%calculate_iv(flag_high_dti);
%calculate_iv(flag_low_credit);
%calculate_iv(has_delinquency);
%calculate_iv(emp_unemployed);

/*****************************************************************************
 * 6. Create Final Modeling Dataset
 *****************************************************************************/

/* Select final features for modeling */
data work.model_features_train;
   set work.credit_features_scaled;

   /* Keep selected features */
   keep customer_id default_flag

        /* Original features */
        age employment_years monthly_income credit_score
        credit_utilization debt_to_income_ratio num_late_payments
        num_credit_accounts credit_history_years previous_defaults
        loan_amount loan_term_months

        /* Engineered features */
        payment_to_income_ratio loan_to_income_ratio employment_score
        credit_quality_score affordability_score total_risk_flags
        woe_credit_score

        /* Binary flags */
        flag_high_dti flag_low_credit flag_high_util
        flag_recent_default flag_unstable_employment has_delinquency

        /* Categorical dummies */
        emp_fulltime emp_selfemployed emp_unemployed
        edu_bachelors edu_masters edu_doctorate
        home_rent home_mortgage
        purpose_debt purpose_auto purpose_personal;
run;

/* Apply same transformations to validation set */
data work.credit_validation_features;
   set work.credit_validation;

   /* Apply same feature engineering steps */
   payment_to_income_ratio = (monthly_payment / monthly_income) * 100;
   loan_to_income_ratio = (loan_amount / annual_income);
   debt_service_coverage = monthly_income / total_monthly_debt;

   if employment_status = 'Full-time' then emp_stability = 5;
   else if employment_status = 'Self-employed' then emp_stability = 4;
   else if employment_status = 'Retired' then emp_stability = 3;
   else if employment_status = 'Part-time' then emp_stability = 2;
   else emp_stability = 1;

   employment_score = emp_stability * employment_years;
   credit_quality_score = credit_score - (num_late_payments * 50) - (previous_defaults * 150);
   affordability_score = (monthly_income - total_monthly_debt) / monthly_payment;

   /* Risk flags */
   flag_high_dti = (debt_to_income_ratio > 43);
   flag_low_credit = (credit_score < 620);
   flag_high_util = (credit_utilization > 75);
   flag_recent_default = (previous_defaults > 0);
   flag_unstable_employment = (employment_years < 2);
   has_delinquency = (num_late_payments > 0);

   total_risk_flags = sum(flag_high_dti, flag_low_credit, flag_high_util,
                          flag_recent_default, flag_unstable_employment);

   /* Categorical dummies */
   emp_fulltime = (employment_status = 'Full-time');
   emp_selfemployed = (employment_status = 'Self-employed');
   emp_unemployed = (employment_status = 'Unemployed');
   edu_bachelors = (education = 'Bachelors');
   edu_masters = (education = 'Masters');
   edu_doctorate = (education = 'Doctorate');
   home_rent = (home_ownership = 'Rent');
   home_mortgage = (home_ownership = 'Mortgage');
   purpose_debt = (loan_purpose = 'Debt Consolidation');
   purpose_auto = (loan_purpose = 'Auto');
   purpose_personal = (loan_purpose = 'Personal');
run;

/* Apply WOE transformation to validation set */
proc sql;
   create table work.credit_validation_woe as
   select a.*,
      case
         when a.credit_score < 580 then b1.woe
         when a.credit_score < 650 then b2.woe
         when a.credit_score < 700 then b3.woe
         when a.credit_score < 750 then b4.woe
         else b5.woe
      end as woe_credit_score
   from work.credit_validation_features a
   cross join (select woe from work.woe_credit_score where credit_band = 'A. <580') b1
   cross join (select woe from work.woe_credit_score where credit_band = 'B. 580-649') b2
   cross join (select woe from work.woe_credit_score where credit_band = 'C. 650-699') b3
   cross join (select woe from work.woe_credit_score where credit_band = 'D. 700-749') b4
   cross join (select woe from work.woe_credit_score where credit_band = 'E. 750+') b5;
quit;

/* Standardize validation set using training set parameters */
proc stdize data=work.credit_validation_woe
           out=work.credit_validation_scaled
           method=std;
   var age employment_years monthly_income annual_income loan_amount
       credit_utilization debt_to_income_ratio num_late_payments
       payment_to_income_ratio loan_to_income_ratio employment_score
       credit_quality_score affordability_score;
run;

/* Select same features as training set */
data work.model_features_validation;
   set work.credit_validation_scaled;

   /* Keep same columns as training set in same order */
   keep customer_id age employment_years monthly_income
        credit_history_years num_credit_accounts num_late_payments
        credit_utilization previous_defaults loan_amount
        loan_term_months debt_to_income_ratio credit_score
        default_flag payment_to_income_ratio loan_to_income_ratio
        employment_score credit_quality_score has_delinquency
        flag_high_dti flag_low_credit flag_high_util
        flag_recent_default flag_unstable_employment total_risk_flags
        affordability_score woe_credit_score emp_fulltime
        emp_selfemployed emp_unemployed edu_bachelors edu_masters
        edu_doctorate home_rent home_mortgage purpose_debt
        purpose_auto purpose_personal;
run;

/* Summary of engineered features */
proc means data=work.model_features_train n mean std min max;
   var payment_to_income_ratio employment_score credit_quality_score
       affordability_score total_risk_flags;
   title "Summary of Engineered Features";
run;

/* Export final datasets to CSV */
proc export data=work.model_features_train
    outfile="/home/u64352077/sasuser.v94/output/model_features_train.csv"
    dbms=csv
    replace;
run;

proc export data=work.model_features_validation
    outfile="/home/u64352077/sasuser.v94/output/model_features_validation.csv"
    dbms=csv
    replace;
run;

%put NOTE: Feature engineering completed successfully;
%put NOTE: Training set exported to: /home/u64352077/sasuser.v94/output/model_features_train.csv;
%put NOTE: Validation set exported to: /home/u64352077/sasuser.v94/output/model_features_validation.csv;