/*****************************************************************************
 * Bank Credit Risk Scoring Model - Step 6: Score New Customer Applications
 *
 * Purpose: Production scoring system for new credit applications
 * Author: Risk Analytics Team
 * Date: 2025
 *
 * This script:
 * - Loads new applications
 * - Applies feature engineering
 * - Scores using trained model
 * - Generates risk ratings and recommendations
 * - Produces decision reports
 *****************************************************************************/

/*****************************************************************************
 * 1. Read New Customer Applications from CSV
 *****************************************************************************/

/* Read new applications from CSV file */
proc import datafile="/home/u64352077/sasuser.v94/output/new_applications.csv"
    out=work.new_applications
    dbms=csv
    replace;
    getnames=yes;
run;

/*****************************************************************************
 * 2. Apply Feature Engineering to New Applications
 *****************************************************************************/

data work.new_applications_features;
   set work.new_applications;

   /* Apply same feature engineering as training */
   payment_to_income_ratio = (monthly_payment / monthly_income) * 100;
   loan_to_income_ratio = (loan_amount / annual_income);
   debt_service_coverage = monthly_income / total_monthly_debt;

   /* Employment score */
   if employment_status = 'Full-time' then emp_stability = 5;
   else if employment_status = 'Self-employed' then emp_stability = 4;
   else if employment_status = 'Retired' then emp_stability = 3;
   else if employment_status = 'Part-time' then emp_stability = 2;
   else emp_stability = 1;

   employment_score = emp_stability * employment_years;

   /* Credit quality score */
   credit_quality_score = credit_score - (num_late_payments * 50) - (previous_defaults * 150);

   /* Affordability */
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

/*****************************************************************************
 * 3. Score New Applications Using Trained Model
 *****************************************************************************/

/* Score new applications with logistic model */
proc plm restore=work.logit_model;
   score data=work.new_applications_features
         out=work.new_applications_scored_temp
         predicted=predicted_logit;
run;

/* Convert logit to probability */
data work.new_applications_scored;
   set work.new_applications_scored_temp;

   /* Apply inverse logit (logistic) function to convert to probability */
   pd_logistic = 1 / (1 + exp(-predicted_logit));

   drop predicted_logit;
run;

/*****************************************************************************
 * 4. Generate Risk Scores and Recommendations
 *****************************************************************************/

data work.application_decisions;
   set work.new_applications_scored;

   /* Convert probability to credit risk score (300-850 scale) */
   base_score = 600;
   score_range = 250;
   credit_risk_score = round(base_score + score_range * (1 - pd_logistic));

   /* Assign risk grades */
   if credit_risk_score >= 750 then risk_grade = 'A';
   else if credit_risk_score >= 700 then risk_grade = 'B';
   else if credit_risk_score >= 650 then risk_grade = 'C';
   else if credit_risk_score >= 600 then risk_grade = 'D';
   else if credit_risk_score >= 550 then risk_grade = 'E';
   else risk_grade = 'F';

   /* Recommended actions */
   if risk_grade in ('A', 'B') then recommendation = 'Approve';
   else if risk_grade = 'C' then recommendation = 'Review';
   else recommendation = 'Decline';

   /* Suggested interest rate (risk-based pricing) */
   base_rate = 0.05;
   if risk_grade = 'A' then interest_rate = base_rate;
   else if risk_grade = 'B' then interest_rate = base_rate + 0.02;
   else if risk_grade = 'C' then interest_rate = base_rate + 0.04;
   else if risk_grade = 'D' then interest_rate = base_rate + 0.07;
   else if risk_grade = 'E' then interest_rate = base_rate + 0.10;
   else interest_rate = base_rate + 0.15;

   format interest_rate percent8.2;
run;

/*****************************************************************************
 * 5. Generate Decision Summary Report
 *****************************************************************************/

/* Risk grade distribution */
proc freq data=work.application_decisions;
   tables risk_grade;
   title "Risk Grade Distribution";
run;

/*****************************************************************************
 * 6. Create Final Output Dataset
 *****************************************************************************/

/* Create final output with key information */
data work.final_predictions;
   set work.application_decisions;

   keep customer_id pd_logistic credit_risk_score
        risk_grade recommendation interest_rate
        credit_score debt_to_income_ratio num_late_payments;
run;

/*****************************************************************************
 * 7. Export Results
 *****************************************************************************/

/* Export predictions to CSV */
proc export data=work.final_predictions
   outfile="/home/u64352077/sasuser.v94/output/new_predictions.csv"
   dbms=csv
   replace;
run;

/* Print sample predictions */
proc print data=work.final_predictions(obs=10) noobs;
   var customer_id pd_logistic credit_risk_score risk_grade recommendation interest_rate;
   title "Sample Predictions";
run;


%put NOTE: Scoring completed successfully;
%put NOTE: Input file: /home/u64352077/sasuser.v94/output/new_applications.csv;
%put NOTE: Decision files exported to: /home/u64352077/sasuser.v94/output/;
%put NOTE: Ready for operational processing;