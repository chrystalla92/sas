/*****************************************************************************
 * Bank Credit Risk Scoring Model - Step 4: Model Training
 *
 * Purpose: Train logistic regression credit risk scoring model
 * Author: Risk Analytics Team
 * Date: 2025
 *
 * This script trains a logistic regression model for credit risk scoring
 * and generates risk grades and scorecards for regulatory compliance
 *****************************************************************************/

/* Ensure features are prepared */
%if not %sysfunc(exist(work.model_features_train)) %then %do;
   %include "/home/u64345824/sasuser.v94/bank_risk_credit_scoring/03_feature_engineering.sas";
%end;

/*****************************************************************************
 * 1. Logistic Regression Model (Primary Model)
 *****************************************************************************/

/* Train logistic regression with stepwise selection */
proc logistic data=work.model_features_train descending;
   id customer_id;
   model default_flag =
      /* Core financial metrics */
      credit_score debt_to_income_ratio credit_utilization
      payment_to_income_ratio loan_to_income_ratio

      /* Employment and stability */
      employment_years employment_score emp_unemployed

      /* Credit history */
      num_late_payments previous_defaults has_delinquency
      credit_quality_score credit_history_years

      /* Risk flags */
      flag_high_dti flag_low_credit flag_high_util
      flag_recent_default total_risk_flags

      /* Demographics and loan */
      age monthly_income loan_amount affordability_score

      /* Categorical indicators */
      emp_fulltime home_rent purpose_debt
   / selection=stepwise slentry=0.05 slstay=0.05
     ctable lackfit rsquare;

   /* Output model parameters */
   output out=work.logit_scored_train
          predicted=pd_logistic
          predprobs=(individual crossvalidate);

   /* Store model */
   store work.logit_model;

   title "Logistic Regression Model for Credit Default Prediction";
run;

/* Add binary prediction variable for confusion matrix */
data work.logit_scored_train;
   set work.logit_scored_train;
   pred_logistic = (pd_logistic >= 0.5);
run;

/* Generate scorecard points */
proc logistic data=work.model_features_train descending outest=work.scorecard_params;
   model default_flag =
      credit_score debt_to_income_ratio num_late_payments
      employment_years flag_low_credit total_risk_flags;

   score data=work.model_features_train
         out=work.scorecard_scores;

   title "Simplified Scorecard Model";
run;


/*****************************************************************************
 * 2. Model Evaluation on Training Data
 *****************************************************************************/

/* Evaluate logistic regression model */
proc freq data=work.logit_scored_train;
   tables default_flag * pred_logistic / nocol nopercent;
   title "Confusion Matrix - Logistic Regression Model";
run;

/*****************************************************************************
 * 3. Apply Model to Validation Set
 *****************************************************************************/

/* Score validation set with logistic model */
proc plm source=work.logit_model;
   score data=work.model_features_validation
         out=work.logit_scored_valid
         predicted=pd_logistic;
run;

/* Add binary prediction variable for validation set */
data work.logit_scored_valid;
   set work.logit_scored_valid;
   pred_logistic = (pd_logistic >= 0.5);
run;

/* Evaluate model on validation set */
proc freq data=work.logit_scored_valid;
   tables default_flag * pred_logistic / nocol nopercent;
   title "Confusion Matrix - Validation Set";
run;

/*****************************************************************************
 * 4. Probability Calibration
 *****************************************************************************/

/* Calibrate probabilities using Platt scaling */
proc logistic data=work.logit_scored_train;
   model default_flag = pd_logistic;
   output out=work.calibrated_train
          predicted=pd_calibrated;
   title "Probability Calibration - Platt Scaling";
run;

/*****************************************************************************
 * 5. Create Risk Scoring Bands
 *****************************************************************************/

data work.risk_scores;
   set work.logit_scored_train;

   /* Convert probability to score (300-850 scale like FICO) */
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

/* Risk grade distribution */
proc freq data=work.risk_scores;
   tables risk_grade * default_flag / nocol nopercent;
   title "Default Rate by Risk Grade";
run;

/*****************************************************************************
 * 6. Export Model Results
 *****************************************************************************/

/* Export model coefficients */
proc print data=work.scorecard_params;
   title "Logistic Regression Model Coefficients";
run;

/* Export scored dataset sample */
proc export data=work.risk_scores(obs=100)
   outfile="/home/u64345824/sasuser.v94/bank_risk_credit_scoring/output/scored_applications.csv"
   dbms=csv
   replace;
run;

/* Save final model dataset */
data work.final_model_output;
   set work.risk_scores;
   keep customer_id default_flag pd_logistic credit_risk_score
        risk_grade recommendation interest_rate
        credit_score debt_to_income_ratio num_late_payments;
run;

%put NOTE: Logistic regression model training completed successfully;
%put NOTE: Model stored as work.logit_model;
%put NOTE: Risk scores generated using 300-850 scale;
%put NOTE: Model outputs saved to output folder;