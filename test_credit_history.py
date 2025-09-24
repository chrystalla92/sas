"""
Test suite for Credit History Generator

This module provides comprehensive validation tests for the credit history generator,
ensuring that all generated variables meet business requirements, show realistic
correlations, and pass range validation.

Test Categories:
- Range and boundary validation
- Correlation tests with demographics
- Business rule compliance
- Statistical distribution validation
- Risk segmentation validation
- Missing value handling
"""

import unittest
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings

from credit_history_generator import CreditHistoryGenerator, RiskConfig, create_sample_demographics


class TestCreditHistoryGenerator(unittest.TestCase):
    """Test suite for CreditHistoryGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = CreditHistoryGenerator(random_state=42)
        self.demographics = create_sample_demographics(1000, random_state=42)
        self.credit_profiles = self.generator.generate_complete_credit_profile(self.demographics)
        
        # Suppress warnings for cleaner test output
        warnings.filterwarnings('ignore')
    
    def test_payment_history_score_range(self):
        """Test that payment history scores are within 300-850 range."""
        scores = self.credit_profiles['payment_history_score']
        
        self.assertTrue(scores.min() >= 300, 
                       f"Minimum score {scores.min()} is below 300")
        self.assertTrue(scores.max() <= 850,
                       f"Maximum score {scores.max()} is above 850")
        
        # Check that we have a reasonable distribution
        self.assertGreater(scores.std(), 30, "Score distribution seems too narrow")
        
    def test_credit_utilization_range(self):
        """Test that credit utilization ratios are within 0-95% range."""
        utilization = self.credit_profiles['credit_utilization']
        
        self.assertTrue(utilization.min() >= 0.0,
                       f"Minimum utilization {utilization.min()} is below 0")
        self.assertTrue(utilization.max() <= 0.95,
                       f"Maximum utilization {utilization.max()} is above 0.95")
        
        # Check reasonable distribution
        self.assertLess(utilization.mean(), 0.5, "Average utilization seems too high")
        
    def test_debt_to_income_range(self):
        """Test that debt-to-income ratios are within acceptable range."""
        dti = self.credit_profiles['debt_to_income_ratio']
        
        self.assertTrue(dti.min() >= 0.0,
                       f"Minimum DTI {dti.min()} is below 0")
        self.assertTrue(dti.max() <= 0.95,
                       f"Maximum DTI {dti.max()} is above 0.95")
        
        # Industry standards: most customers should have DTI < 0.6
        high_dti_pct = (dti > 0.6).mean()
        self.assertLess(high_dti_pct, 0.3, 
                       f"Too many customers ({high_dti_pct:.1%}) have DTI > 60%")
        
    def test_credit_history_length_consistency(self):
        """Test that credit history length is consistent with age."""
        age = self.credit_profiles['age']
        history_years = self.credit_profiles['credit_history_years']
        
        # History cannot exceed age - 18
        max_possible = age - 18
        violations = history_years > max_possible
        
        self.assertEqual(violations.sum(), 0,
                        f"{violations.sum()} customers have impossible credit history length")
        
        # History should be non-negative
        self.assertTrue(history_years.min() >= 0,
                       "Credit history length cannot be negative")
        
    def test_number_of_accounts_range(self):
        """Test that number of accounts is within reasonable range."""
        accounts = self.credit_profiles['num_credit_accounts']
        
        self.assertTrue(accounts.min() >= 0,
                       "Number of accounts cannot be negative")
        self.assertTrue(accounts.max() <= 20,
                       f"Maximum accounts {accounts.max()} exceeds business limit")
        
        # Customers with no credit history should have 0 accounts
        no_history_mask = self.credit_profiles['credit_history_years'] == 0
        no_history_accounts = accounts[no_history_mask]
        
        self.assertTrue((no_history_accounts == 0).all(),
                       "Customers with no credit history should have 0 accounts")
        
    def test_recent_inquiries_range(self):
        """Test that recent inquiries are within 0-10 range."""
        inquiries = self.credit_profiles['recent_inquiries']
        
        self.assertTrue(inquiries.min() >= 0,
                       "Recent inquiries cannot be negative")
        self.assertTrue(inquiries.max() <= 10,
                       f"Maximum inquiries {inquiries.max()} exceeds 10")
        
    def test_income_payment_score_correlation(self):
        """Test that higher income correlates with better payment scores."""
        income = self.credit_profiles['monthly_income']
        payment_score = self.credit_profiles['payment_history_score']
        
        correlation, p_value = pearsonr(income, payment_score)
        
        self.assertGreater(correlation, 0.1,
                          f"Income-payment score correlation {correlation:.3f} is too weak")
        self.assertLess(p_value, 0.05, "Income-payment score correlation is not significant")
        
    def test_age_credit_history_correlation(self):
        """Test that age correlates with credit history length."""
        age = self.credit_profiles['age']
        history_years = self.credit_profiles['credit_history_years']
        
        correlation, p_value = pearsonr(age, history_years)
        
        self.assertGreater(correlation, 0.2,
                          f"Age-credit history correlation {correlation:.3f} is too weak")
        self.assertLess(p_value, 0.05, "Age-credit history correlation is not significant")
        
    def test_utilization_income_correlation(self):
        """Test that credit utilization negatively correlates with income."""
        income = self.credit_profiles['monthly_income']
        utilization = self.credit_profiles['credit_utilization']
        
        # Filter out customers with zero utilization (no credit accounts)
        has_credit_mask = self.credit_profiles['num_credit_accounts'] > 0
        income_filtered = income[has_credit_mask]
        utilization_filtered = utilization[has_credit_mask]
        
        if len(income_filtered) > 50:  # Need sufficient sample size
            correlation, p_value = pearsonr(income_filtered, utilization_filtered)
            
            # Should be negative correlation (higher income -> lower utilization)
            self.assertLess(correlation, -0.05,
                           f"Income-utilization correlation {correlation:.3f} should be negative")
        
    def test_payment_score_utilization_correlation(self):
        """Test that payment scores negatively correlate with utilization."""
        payment_score = self.credit_profiles['payment_history_score']
        utilization = self.credit_profiles['credit_utilization']
        
        # Filter out customers with zero utilization
        has_credit_mask = self.credit_profiles['num_credit_accounts'] > 0
        score_filtered = payment_score[has_credit_mask]
        utilization_filtered = utilization[has_credit_mask]
        
        if len(score_filtered) > 50:
            correlation, p_value = pearsonr(score_filtered, utilization_filtered)
            
            # Should be negative correlation (higher score -> lower utilization)
            self.assertLess(correlation, -0.1,
                           f"Payment score-utilization correlation {correlation:.3f} should be negative")
        
    def test_default_probability_risk_correlation(self):
        """Test that default probability correlates with risk factors."""
        default_prob = self.credit_profiles['default_probability']
        payment_score = self.credit_profiles['payment_history_score']
        dti = self.credit_profiles['debt_to_income_ratio']
        composite_risk = self.credit_profiles['composite_risk_score']
        
        # Default probability should negatively correlate with payment score
        corr_score, p_score = pearsonr(default_prob, payment_score)
        self.assertLess(corr_score, -0.3,
                       f"Default prob-payment score correlation {corr_score:.3f} should be strongly negative")
        
        # Default probability should positively correlate with DTI
        corr_dti, p_dti = pearsonr(default_prob, dti)
        self.assertGreater(corr_dti, 0.2,
                          f"Default prob-DTI correlation {corr_dti:.3f} should be positive")
        
        # Default probability should positively correlate with composite risk
        corr_risk, p_risk = pearsonr(default_prob, composite_risk)
        self.assertGreater(corr_risk, 0.5,
                          f"Default prob-composite risk correlation {corr_risk:.3f} should be strong")
        
    def test_business_rule_compliance(self):
        """Test compliance with key business rules."""
        # Rule 1: Utilization <= 95%
        max_utilization = self.credit_profiles['credit_utilization'].max()
        self.assertLessEqual(max_utilization, 0.95,
                            f"Maximum utilization {max_utilization:.3f} exceeds 95%")
        
        # Rule 2: DTI <= 95%
        max_dti = self.credit_profiles['debt_to_income_ratio'].max()
        self.assertLessEqual(max_dti, 0.95,
                            f"Maximum DTI {max_dti:.3f} exceeds 95%")
        
        # Rule 3: Payment scores 300-850
        scores = self.credit_profiles['payment_history_score']
        self.assertTrue(((scores >= 300) & (scores <= 850)).all(),
                       "Some payment scores are outside 300-850 range")
        
        # Rule 4: Non-negative accounts
        accounts = self.credit_profiles['num_credit_accounts']
        self.assertTrue((accounts >= 0).all(),
                       "Some customers have negative account counts")
        
        # Rule 5: Inquiries 0-10
        inquiries = self.credit_profiles['recent_inquiries']
        self.assertTrue(((inquiries >= 0) & (inquiries <= 10)).all(),
                       "Some customers have invalid inquiry counts")
        
    def test_thin_file_customers(self):
        """Test handling of customers with no credit history."""
        no_history_mask = self.credit_profiles['credit_history_years'] == 0
        thin_file_customers = self.credit_profiles[no_history_mask]
        
        if len(thin_file_customers) > 0:
            # Should have 0 accounts
            self.assertTrue((thin_file_customers['num_credit_accounts'] == 0).all(),
                           "Thin-file customers should have 0 accounts")
            
            # Should have 0 utilization
            self.assertTrue((thin_file_customers['credit_utilization'] == 0).all(),
                           "Thin-file customers should have 0% utilization")
            
            # Should have lower payment scores
            avg_thin_score = thin_file_customers['payment_history_score'].mean()
            avg_regular_score = self.credit_profiles[~no_history_mask]['payment_history_score'].mean()
            
            self.assertLess(avg_thin_score, avg_regular_score,
                           "Thin-file customers should have lower average payment scores")
        
    def test_risk_segmentation(self):
        """Test that generated data supports meaningful risk segmentation."""
        # Create risk segments based on composite risk score
        risk_score = self.credit_profiles['composite_risk_score']
        default_prob = self.credit_profiles['default_probability']
        
        # Define risk segments
        low_risk = default_prob < 0.1
        medium_risk = (default_prob >= 0.1) & (default_prob < 0.3)
        high_risk = default_prob >= 0.3
        
        # Check that we have representation in each segment
        self.assertGreater(low_risk.sum(), 100, "Not enough low-risk customers")
        self.assertGreater(medium_risk.sum(), 50, "Not enough medium-risk customers")
        self.assertGreater(high_risk.sum(), 10, "Not enough high-risk customers")
        
        # Check that risk scores are ordered correctly across segments
        low_risk_scores = risk_score[low_risk].mean()
        medium_risk_scores = risk_score[medium_risk].mean()
        high_risk_scores = risk_score[high_risk].mean()
        
        self.assertLess(low_risk_scores, medium_risk_scores,
                       "Low-risk customers should have lower risk scores than medium-risk")
        self.assertLess(medium_risk_scores, high_risk_scores,
                       "Medium-risk customers should have lower risk scores than high-risk")
        
    def test_statistical_distributions(self):
        """Test that variables follow expected statistical distributions."""
        # Payment history scores should be roughly normal
        scores = self.credit_profiles['payment_history_score']
        _, p_value_normal = stats.normaltest(scores)
        # Note: With large sample, perfect normality is not expected, but shouldn't be extremely skewed
        
        # Credit utilization should be right-skewed (beta distribution)
        utilization = self.credit_profiles['credit_utilization']
        utilization_nonzero = utilization[utilization > 0]
        if len(utilization_nonzero) > 50:
            skewness = stats.skew(utilization_nonzero)
            self.assertGreater(skewness, 0, "Credit utilization should be right-skewed")
        
        # Number of accounts should follow Poisson-like distribution
        accounts = self.credit_profiles['num_credit_accounts']
        accounts_nonzero = accounts[accounts > 0]
        if len(accounts_nonzero) > 50:
            # Mean should be close to variance for Poisson distribution
            mean_accounts = accounts_nonzero.mean()
            var_accounts = accounts_nonzero.var()
            ratio = var_accounts / mean_accounts
            self.assertLess(abs(ratio - 1), 0.5, 
                           f"Accounts distribution variance/mean ratio {ratio:.2f} deviates from Poisson")
        
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with very young customers
        young_demographics = pd.DataFrame({
            'customer_id': ['YOUNG_001', 'YOUNG_002'],
            'age': [18, 19],
            'employment_years': [0, 1],
            'monthly_income': [2000, 2500]
        })
        
        young_profiles = self.generator.generate_complete_credit_profile(young_demographics)
        
        # Young customers should have limited credit history
        max_history = young_profiles['credit_history_years'].max()
        self.assertLessEqual(max_history, 2, "Very young customers shouldn't have long credit history")
        
        # Test with very old customers
        old_demographics = pd.DataFrame({
            'customer_id': ['OLD_001', 'OLD_002'],
            'age': [70, 75],
            'employment_years': [30, 35],
            'monthly_income': [8000, 10000]
        })
        
        old_profiles = self.generator.generate_complete_credit_profile(old_demographics)
        
        # Older customers can have long credit history
        max_history_old = old_profiles['credit_history_years'].max()
        self.assertLessEqual(max_history_old, 57, "Credit history cannot exceed age - 18")
        
    def test_reproducibility(self):
        """Test that generator produces reproducible results."""
        # Generate data twice with same random state
        generator1 = CreditHistoryGenerator(random_state=123)
        generator2 = CreditHistoryGenerator(random_state=123)
        
        test_demographics = create_sample_demographics(100, random_state=456)
        
        profiles1 = generator1.generate_complete_credit_profile(test_demographics)
        profiles2 = generator2.generate_complete_credit_profile(test_demographics)
        
        # Results should be identical
        pd.testing.assert_frame_equal(profiles1, profiles2,
                                     "Generator should produce identical results with same random state")
        
    def test_custom_risk_config(self):
        """Test generator with custom risk configuration."""
        # Create custom config with more conservative parameters
        custom_config = RiskConfig(
            utilization_max=0.80,  # Lower max utilization
            dti_max=0.60,          # Lower max DTI
            payment_history_base=700.0  # Higher base payment score
        )
        
        custom_generator = CreditHistoryGenerator(risk_config=custom_config, random_state=42)
        custom_profiles = custom_generator.generate_complete_credit_profile(self.demographics)
        
        # Check that custom limits are respected
        max_util = custom_profiles['credit_utilization'].max()
        max_dti = custom_profiles['debt_to_income_ratio'].max()
        avg_score = custom_profiles['payment_history_score'].mean()
        
        self.assertLessEqual(max_util, 0.80, "Custom utilization limit not respected")
        self.assertLessEqual(max_dti, 0.60, "Custom DTI limit not respected")
        self.assertGreater(avg_score, 680, "Custom payment score base not reflected")
        

class TestDataQuality(unittest.TestCase):
    """Test data quality and integrity."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = CreditHistoryGenerator(random_state=42)
        self.demographics = create_sample_demographics(1000, random_state=42)
        self.credit_profiles = self.generator.generate_complete_credit_profile(self.demographics)
    
    def test_no_missing_values(self):
        """Test that no critical variables have missing values."""
        critical_vars = [
            'payment_history_score', 'credit_utilization', 'credit_history_years',
            'num_credit_accounts', 'debt_to_income_ratio', 'default_probability'
        ]
        
        for var in critical_vars:
            missing_count = self.credit_profiles[var].isna().sum()
            self.assertEqual(missing_count, 0, f"{var} has {missing_count} missing values")
    
    def test_data_types(self):
        """Test that variables have correct data types."""
        # Integer variables
        int_vars = ['age', 'employment_years', 'payment_history_score', 
                   'num_credit_accounts', 'recent_inquiries']
        
        for var in int_vars:
            if var in self.credit_profiles.columns:
                self.assertTrue(pd.api.types.is_integer_dtype(self.credit_profiles[var]),
                               f"{var} should be integer type")
        
        # Float variables
        float_vars = ['monthly_income', 'credit_utilization', 'debt_to_income_ratio',
                     'default_probability', 'composite_risk_score']
        
        for var in float_vars:
            if var in self.credit_profiles.columns:
                self.assertTrue(pd.api.types.is_numeric_dtype(self.credit_profiles[var]),
                               f"{var} should be numeric type")
    
    def test_logical_consistency(self):
        """Test logical consistency between related variables."""
        # Customers with 0 accounts should have 0 utilization
        zero_accounts = self.credit_profiles['num_credit_accounts'] == 0
        zero_account_util = self.credit_profiles.loc[zero_accounts, 'credit_utilization']
        
        self.assertTrue((zero_account_util == 0).all(),
                       "Customers with 0 accounts should have 0% utilization")
        
        # Customers with 0 credit history should have 0 accounts
        zero_history = self.credit_profiles['credit_history_years'] == 0
        zero_history_accounts = self.credit_profiles.loc[zero_history, 'num_credit_accounts']
        
        self.assertTrue((zero_history_accounts == 0).all(),
                       "Customers with 0 credit history should have 0 accounts")


def run_performance_benchmark():
    """Run performance benchmark to ensure generator scales well."""
    import time
    
    print("\n=== Performance Benchmark ===")
    
    sizes = [1000, 5000, 10000]
    generator = CreditHistoryGenerator(random_state=42)
    
    for size in sizes:
        demographics = create_sample_demographics(size, random_state=42)
        
        start_time = time.time()
        credit_profiles = generator.generate_complete_credit_profile(demographics)
        end_time = time.time()
        
        duration = end_time - start_time
        rate = size / duration
        
        print(f"Size: {size:,} customers")
        print(f"Time: {duration:.2f} seconds")
        print(f"Rate: {rate:.0f} customers/second")
        print("-" * 30)


if __name__ == '__main__':
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Run performance benchmark
    run_performance_benchmark()
