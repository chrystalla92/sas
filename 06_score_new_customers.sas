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

/* Load the trained model */
%include "/home/u64345824/sasuser.v94/bank_risk_credit_scoring/04_train_credit_model.sas";

/*****************************************************************************
 * 1. Simulate New Customer Applications
 *****************************************************************************/

data work.new_applications;
   /* Generate 50 new credit applications for scoring */
   length customer_id $10 employment_status $20 education $20 home_ownership $10;
   format application_date date9. monthly_income dollar12.2 loan_amount dollar12.2;

   call streaminit(98765); /* Different seed for new data */

   do i = 1 to 50;
      /* Generate application details */
      customer_id = cat('NEW', put(i, z5.));
      application_date = today();

      /* Customer demographics */
      age = max(18, min(75, round(rand('normal', 40, 15))));

      /* Employment */
      emp_rand = rand('uniform');
      if emp_rand < 0.60 then employment_status = 'Full-time';
      else if emp_rand < 0.75 then employment_status = 'Self-employed';
      else if emp_rand < 0.85 then employment_status = 'Part-time';
      else if emp_rand < 0.92 then employment_status = 'Retired';
      else employment_status = 'Unemployed';

      employment_years = max(0, round(rand('uniform') * min(15, age - 20)));

      /* Education */
      edu_rand = rand('uniform');
      if edu_rand < 0.35 then education = 'High School';
      else if edu_rand < 0.65 then education = 'Bachelors';
      else if edu_rand < 0.85 then education = 'Masters';
      else education = 'Doctorate';

      /* Income */
      base_income = 4000 * (1 + (education ne 'High School') * 0.3);
      monthly_income = round(base_income * exp(rand('normal', 0, 0.4)), 100);
      annual_income = monthly_income * 12;

      /* Home ownership */
      home_rand = rand('uniform');
      if home_rand < 0.35 then home_ownership = 'Rent';
      else if home_rand < 0.75 then home_ownership = 'Mortgage';
      else home_ownership = 'Own';

      /* Credit history */
      credit_history_years = max(0, min(age - 18, round(rand('gamma', 4))));
      num_credit_accounts = max(1, round(rand('poisson', 4)));

      /* Credit score - varied distribution for testing */
      if i <= 10 then
         credit_score = round(rand('uniform', 750, 850)); /* Excellent */
      else if i <= 20 then
         credit_score = round(rand('uniform', 680, 750)); /* Good */
      else if i <= 35 then
         credit_score = round(rand('uniform', 600, 680)); /* Fair */
      else
         credit_score = round(rand('uniform', 450, 600)); /* Poor */

      /* Payment history */
      if credit_score > 700 then num_late_payments = 0;
      else if credit_score > 650 then num_late_payments = round(rand('poisson', 0.5));
      else num_late_payments = round(rand('poisson', 2));

      previous_defaults = (rand('uniform') < 0.05) * round(rand('poisson', 0.3));

      /* Credit utilization */
      if credit_score > 700 then credit_utilization = rand('beta', 2, 8) * 100;
      else credit_utilization = rand('beta', 3, 5) * 100;

      /* Loan request */
      loan_amount = round(annual_income * rand('uniform', 0.2, 2.5), 1000);
      loan_term_months = 12 * (1 + round(rand('uniform', 0, 2)));

      loan_purpose_rand = rand('uniform');
      if loan_purpose_rand < 0.3 then loan_purpose = 'Debt Consolidation';
      else if loan_purpose_rand < 0.5 then loan_purpose = 'Home Improvement';
      else if loan_purpose_rand < 0.7 then loan_purpose = 'Personal';
      else if loan_purpose_rand < 0.85 then loan_purpose = 'Auto';
      else loan_purpose = 'Medical';

      /* Calculate monthly payment and DTI */
      interest_rate = 0.06 + (800 - credit_score) / 5000;
      monthly_rate = interest_rate / 12;
      monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**loan_term_months) /
                       ((1 + monthly_rate)**loan_term_months - 1);

      existing_monthly_debt = monthly_income * rand('beta', 2, 10);
      total_monthly_debt = existing_monthly_debt + monthly_payment;
      debt_to_income_ratio = (total_monthly_debt / monthly_income) * 100;

      /* Other variables */
      num_dependents = max(0, round(rand('poisson', 1.2)));

      output;
   end;

   drop i emp_rand edu_rand home_rand loan_purpose_rand base_income monthly_rate;
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

/* Apply the logistic regression model */
proc plm source=work.logit_model;
   score data=work.new_applications_features
         out=work.new_applications_scored
         predicted=probability_default;
run;

/*****************************************************************************
 * 4. Generate Risk Scores and Recommendations
 *****************************************************************************/

data work.application_decisions;
   set work.new_applications_scored;

   /* Convert probability to credit risk score (300-850 scale) */
   credit_risk_score = round(600 + 250 * (1 - probability_default));

   /* Assign risk grade */
   if credit_risk_score >= 750 then risk_grade = 'A';
   else if credit_risk_score >= 700 then risk_grade = 'B';
   else if credit_risk_score >= 650 then risk_grade = 'C';
   else if credit_risk_score >= 600 then risk_grade = 'D';
   else if credit_risk_score >= 550 then risk_grade = 'E';
   else risk_grade = 'F';

   /* Decision logic */
   if risk_grade in ('A', 'B') then do;
      decision = 'APPROVED';
      decision_reason = 'Low risk profile';
   end;
   else if risk_grade = 'C' then do;
      if debt_to_income_ratio < 40 and credit_score >= 650 then do;
         decision = 'APPROVED';
         decision_reason = 'Acceptable risk with conditions';
      end;
      else do;
         decision = 'MANUAL REVIEW';
         decision_reason = 'Borderline risk profile';
      end;
   end;
   else do;
      decision = 'DECLINED';
      if flag_low_credit then decision_reason = 'Credit score below minimum';
      else if flag_high_dti then decision_reason = 'Debt-to-income ratio too high';
      else if flag_recent_default then decision_reason = 'Recent default history';
      else decision_reason = 'Overall risk exceeds threshold';
   end;

   /* Risk-based pricing */
   base_rate = 0.049; /* 4.9% base rate */
   if risk_grade = 'A' then offered_rate = base_rate;
   else if risk_grade = 'B' then offered_rate = base_rate + 0.015;
   else if risk_grade = 'C' then offered_rate = base_rate + 0.035;
   else if risk_grade = 'D' then offered_rate = base_rate + 0.055;
   else if risk_grade = 'E' then offered_rate = base_rate + 0.080;
   else offered_rate = .; /* No offer for F grade */

   /* Maximum approved amount based on risk */
   if decision = 'APPROVED' then do;
      risk_multiplier = 3.0 - (rank(risk_grade) - 65) * 0.3;
      max_approved_amount = round(annual_income * risk_multiplier / 12, 1000);
      approved_amount = min(loan_amount, max_approved_amount);
   end;
   else do;
      max_approved_amount = 0;
      approved_amount = 0;
   end;

   format probability_default percent8.2
          offered_rate percent8.2
          approved_amount dollar12.
          application_date date9.;
run;

/*****************************************************************************
 * 5. Generate Decision Summary Report
 *****************************************************************************/

/* Summary by decision */
proc freq data=work.application_decisions;
   tables decision * risk_grade / nocol nopercent;
   title "Application Decisions by Risk Grade";
run;

/* Decision statistics */
proc means data=work.application_decisions mean std min max;
   class decision;
   var credit_score debt_to_income_ratio probability_default credit_risk_score;
   title "Statistics by Decision Category";
run;

/* Approved loans summary */
proc sql;
   create table work.approval_summary as
   select
      count(*) as total_applications,
      sum(decision = 'APPROVED') as approved_count,
      sum(decision = 'MANUAL REVIEW') as review_count,
      sum(decision = 'DECLINED') as declined_count,
      mean(case when decision = 'APPROVED' then offered_rate else . end) as avg_offered_rate format=percent8.2,
      sum(approved_amount) as total_approved_amount format=dollar15.,
      mean(case when decision = 'APPROVED' then approved_amount else . end) as avg_approved_amount format=dollar12.
   from work.application_decisions;
quit;

proc print data=work.approval_summary noobs;
   title "Loan Application Summary";
run;

/*****************************************************************************
 * 6. Generate Individual Decision Letters
 *****************************************************************************/

/* Create detailed decision records */
data work.decision_letters;
   set work.application_decisions;
   length letter_text $500;

   /* Generate personalized decision text */
   if decision = 'APPROVED' then do;
      letter_text = catx(' ',
         'Congratulations! Your loan application for', put(loan_amount, dollar12.),
         'has been approved.',
         'Approved amount:', put(approved_amount, dollar12.),
         'at', put(offered_rate, percent8.2), 'APR.',
         'Risk Grade:', risk_grade);
   end;
   else if decision = 'MANUAL REVIEW' then do;
      letter_text = catx(' ',
         'Your application for', put(loan_amount, dollar12.),
         'requires additional review.',
         'Reason:', decision_reason,
         '. A loan officer will contact you within 2 business days.');
   end;
   else do;
      letter_text = catx(' ',
         'We regret to inform you that your application for', put(loan_amount, dollar12.),
         'has been declined.',
         'Reason:', decision_reason,
         '. Please contact us to discuss alternatives.');
   end;

   keep customer_id application_date decision risk_grade credit_risk_score
        probability_default loan_amount approved_amount offered_rate
        decision_reason letter_text;
run;

/*****************************************************************************
 * 7. Export Results
 *****************************************************************************/

/* Export decision file for operations */
proc export data=work.application_decisions
   outfile="/home/u64345824/sasuser.v94/bank_risk_credit_scoring/output/new_application_decisions.csv"
   dbms=csv
   replace;
run;

/* Export summary for management */
proc export data=work.approval_summary
   outfile="/home/u64345824/sasuser.v94/bank_risk_credit_scoring/output/approval_summary.csv"
   dbms=csv
   replace;
run;

/* Print sample decisions */
proc print data=work.decision_letters(obs=10) noobs;
   var customer_id decision risk_grade credit_risk_score approved_amount offered_rate;
   title "Sample Application Decisions";
run;

/*****************************************************************************
 * 8. Monitoring and Alerts
 *****************************************************************************/

/* Check for unusual patterns */
data work.monitoring_alerts;
   set work.application_decisions end=last;
   retain high_risk_count 0 total_amount 0;

   if probability_default > 0.3 then high_risk_count + 1;
   if decision = 'APPROVED' then total_amount + approved_amount;

   if last then do;
      high_risk_pct = (high_risk_count / _n_) * 100;

      if high_risk_pct > 30 then
         alert = 'WARNING: High proportion of high-risk applications';
      else if total_amount > 1000000 then
         alert = 'INFO: Large exposure in current batch';
      else
         alert = 'OK: Normal application batch';

      output;
   end;

   keep high_risk_count high_risk_pct total_amount alert;
   format high_risk_pct 8.2 total_amount dollar15.;
run;

proc print data=work.monitoring_alerts noobs;
   title "Batch Monitoring Alerts";
run;

%put NOTE: Scoring completed for 50 new applications;
%put NOTE: Decision files exported to output folder;
%put NOTE: Ready for operational processing;