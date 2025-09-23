****************************************************************;
******        HP TREE (PROC HPSPLIT) SCORING CODE        ******;
****************************************************************;
 
******              LABELS FOR NEW VARIABLES              ******;
LABEL _Node_ = 'Node number';
LABEL _Leaf_ = 'Leaf number';
LABEL _WARN_ = 'Warnings';
LABEL P_default_flag1 = 'Predicted: default_flag=1';
LABEL P_default_flag0 = 'Predicted: default_flag=0';
LABEL V_default_flag1 = 'Validated: default_flag=1';
LABEL V_default_flag0 = 'Validated: default_flag=0';
 
 _WARN_ = ' ';
 
******      TEMPORARY VARIABLES FOR FORMATTED VALUES      ******;
LENGTH _RT_10_12 $12;
_RT_10_12 = ' ';
DROP _RT_10_12;
_RT_10_12 = PUT(emp_unemployed, BEST12.);
%DMNORMIP(_RT_10_12);
 
******             ASSIGN OBSERVATION TO NODE             ******;
IF NOT MISSING(credit_score) AND ((credit_score < 608.37))
 THEN DO;
  IF NOT MISSING(credit_score) AND ((credit_score < 456.89))
   THEN DO;
    _Node_ = 3;
    _Leaf_ = 0;
    P_default_flag1 = 0.72222222;
    P_default_flag0 = 0.27777778;
    V_default_flag1 = 0.62962963;
    V_default_flag0 = 0.37037037;
  END;
  ELSE DO;
    IF NOT MISSING(total_risk_flags) AND ((total_risk_flags >= 2.04))
     THEN DO;
      IF NOT MISSING(monthly_income) AND ((monthly_income < -0.8362282325677504))
       THEN DO;
        _Node_ = 13;
        _Leaf_ = 4;
        P_default_flag1 = 0.76923077;
        P_default_flag0 = 0.23076923;
        V_default_flag1 = 0.5;
        V_default_flag0 = 0.5;
      END;
      ELSE DO;
        _Node_ = 14;
        _Leaf_ = 5;
        P_default_flag1 = 0.46099291;
        P_default_flag0 = 0.53900709;
        V_default_flag1 = 0.44444444;
        V_default_flag0 = 0.55555556;
      END;
  END;
  ELSE DO;
    IF NOT MISSING(credit_score) AND ((credit_score >= 581.3200000000001))
     THEN DO;
      _Node_ = 12;
      _Leaf_ = 3;
      P_default_flag1 = 0.27722772;
      P_default_flag0 = 0.72277228;
      V_default_flag1 = 0.38356164;
      V_default_flag0 = 0.61643836;
    END;
    ELSE DO;
      IF NOT MISSING(debt_to_income_ratio) AND ((debt_to_income_ratio < -0.2764993696502716))
       THEN DO;
        _Node_ = 17;
        _Leaf_ = 7;
        P_default_flag1 = 0.33333333;
        P_default_flag0 = 0.66666667;
        V_default_flag1 = 0.34545455;
        V_default_flag0 = 0.65454545;
      END;
      ELSE DO;
        IF NOT MISSING(debt_to_income_ratio) AND ((debt_to_income_ratio >= 2.725590185734137))
         THEN DO;
          _Node_ = 22;
          _Leaf_ = 11;
          P_default_flag1 = 0;
          P_default_flag0 = 1;
          V_default_flag1 = 1;
          V_default_flag0 = 0;
        END;
        ELSE DO;
          _Node_ = 21;
          _Leaf_ = 10;
          P_default_flag1 = 0.5;
          P_default_flag0 = 0.5;
          V_default_flag1 = 0.55932203;
          V_default_flag0 = 0.44067797;
        END;
      END;
    END;
  END;
END;
END;
ELSE DO;
IF NOT MISSING(debt_to_income_ratio) AND ((debt_to_income_ratio < -1.012860958706825))
 THEN DO;
  _Node_ = 5;
  _Leaf_ = 1;
  P_default_flag1 = 0.083979328;
  P_default_flag0 = 0.91602067;
  V_default_flag1 = 0.12140575;
  V_default_flag0 = 0.87859425;
END;
ELSE DO;
  IF NOT MISSING(credit_score) AND ((credit_score >= 694.9300000000001))
   THEN DO;
    _Node_ = 10;
    _Leaf_ = 2;
    P_default_flag1 = 0.15973478;
    P_default_flag0 = 0.84026522;
    V_default_flag1 = 0.1669024;
    V_default_flag0 = 0.8330976;
  END;
  ELSE DO;
    IF NOT MISSING(emp_unemployed) AND (_RT_10_12 IN ('1') )
     THEN DO;
      IF NOT MISSING(credit_utilization) AND ((credit_utilization < -0.6558261628423244))
       THEN DO;
        _Node_ = 19;
        _Leaf_ = 8;
        P_default_flag1 = 0.75;
        P_default_flag0 = 0.25;
        V_default_flag1 = 0.44444444;
        V_default_flag0 = 0.55555556;
      END;
      ELSE DO;
        _Node_ = 20;
        _Leaf_ = 9;
        P_default_flag1 = 0.39393939;
        P_default_flag0 = 0.60606061;
        V_default_flag1 = 0.41935484;
        V_default_flag0 = 0.58064516;
      END;
    END;
    ELSE DO;
      _Node_ = 16;
      _Leaf_ = 6;
      P_default_flag1 = 0.2288383;
      P_default_flag0 = 0.7711617;
      V_default_flag1 = 0.23796791;
      V_default_flag0 = 0.76203209;
    END;
  END;
END;
END;
****************************************************************;
******     END OF HP TREE (PROC HPSPLIT) SCORING CODE    ******;
****************************************************************;
