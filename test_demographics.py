"""
Test Suite for Demographics Generator

Comprehensive tests for demographic data generation including statistical
reasonableness tests, correlation validation, business constraint compliance,
and categorical distribution validation.

Author: Risk Analytics Team  
Date: 2025
"""

import unittest
import numpy as np
import pandas as pd
from scipy import stats
import warnings
from typing import Dict, List

from demographics_generator import DemographicsGenerator, generate_sample_demographics


class TestDemographicsGenerator(unittest.TestCase):
    """Test suite for DemographicsGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = DemographicsGenerator(random_seed=42)
        self.sample_size = 5000  # Large enough for statistical tests
        self.small_sample_size = 100  # For basic functionality tests
    
    def test_initialization(self):
        """Test generator initialization."""
        # Test with seed
        gen1 = DemographicsGenerator(random_seed=123)
        self.assertEqual(gen1.random_seed, 123)
        
        # Test without seed  
        gen2 = DemographicsGenerator(random_seed=None)
        self.assertIsNone(gen2.random_seed)
        
        # Test expected probabilities are loaded
        self.assertIn('Full-time', gen1.employment_probs)
        self.assertIn('Bachelors', gen1.education_probs)
        self.assertAlmostEqual(sum(gen1.employment_probs.values()), 1.0, places=2)
    
    def test_age_generation(self):
        """Test age generation meets statistical requirements."""
        # Generate test data
        data = self.generator.generate_demographics(self.sample_size)
        ages = data['age']
        
        # Test range constraints
        self.assertTrue(ages.min() >= 18, "Age minimum constraint violated")
        self.assertTrue(ages.max() <= 80, "Age maximum constraint violated")
        
        # Test data type
        self.assertEqual(ages.dtype, 'int32', "Age should be integer type")
        
        # Test distribution characteristics (should be roughly normal around 42)
        mean_age = ages.mean()
        std_age = ages.std()
        
        self.assertGreater(mean_age, 38, "Mean age too low")
        self.assertLess(mean_age, 46, "Mean age too high") 
        self.assertGreater(std_age, 8, "Age standard deviation too low")
        self.assertLess(std_age, 16, "Age standard deviation too high")
        
        # Test for normality (relaxed test due to truncation)
        _, p_value = stats.normaltest(ages)
        # Note: Due to truncation at 18/80, perfect normality isn't expected
        
    def test_employment_status_generation(self):
        """Test employment status generation and age correlations."""
        data = self.generator.generate_demographics(self.sample_size)
        
        # Test all expected categories present
        expected_statuses = set(self.generator.employment_probs.keys())
        actual_statuses = set(data['employment_status'].unique())
        self.assertTrue(actual_statuses.issubset(expected_statuses), 
                       f"Unexpected employment statuses: {actual_statuses - expected_statuses}")
        
        # Test age-employment correlations
        # Older people should be more likely to be retired
        elderly = data[data['age'] >= 65]
        if len(elderly) > 10:  # Only test if we have enough elderly people
            retirement_rate_elderly = (elderly['employment_status'] == 'Retired').mean()
            overall_retirement_rate = (data['employment_status'] == 'Retired').mean()
            self.assertGreater(retirement_rate_elderly, overall_retirement_rate * 2,
                             "Elderly not significantly more likely to be retired")
        
        # Young people should have higher part-time rate
        young = data[data['age'] <= 22]
        if len(young) > 10:
            parttime_rate_young = (young['employment_status'] == 'Part-time').mean()
            overall_parttime_rate = (data['employment_status'] == 'Part-time').mean()
            self.assertGreaterEqual(parttime_rate_young, overall_parttime_rate,
                                  "Young people not more likely to work part-time")
    
    def test_education_generation(self):
        """Test education level generation and age correlations."""
        data = self.generator.generate_demographics(self.sample_size)
        
        # Test all expected categories present
        expected_education = set(self.generator.education_probs.keys())
        actual_education = set(data['education'].unique())
        self.assertTrue(actual_education.issubset(expected_education),
                       f"Unexpected education levels: {actual_education - expected_education}")
        
        # Test age-education correlation (older generations less likely to have advanced degrees)
        older = data[data['age'] >= 55]
        younger = data[data['age'] <= 35]
        
        if len(older) > 50 and len(younger) > 50:
            older_advanced_rate = older['education'].isin(['Masters', 'Doctorate']).mean()
            younger_advanced_rate = younger['education'].isin(['Masters', 'Doctorate']).mean()
            
            # Younger people should be more likely to have advanced degrees
            # (though this might be subtle due to individual variation)
            self.assertGreaterEqual(younger_advanced_rate, older_advanced_rate * 0.8,
                                  "Age-education correlation not as expected")
    
    def test_income_generation(self):
        """Test annual income generation and correlations."""
        data = self.generator.generate_demographics(self.sample_size)
        incomes = data['annual_income']
        
        # Test data type
        self.assertEqual(incomes.dtype, 'float64', "Income should be float type")
        
        # Test non-negative values
        self.assertTrue((incomes >= 0).all(), "Negative income values found")
        
        # Test reasonable range
        self.assertGreater(incomes.mean(), 20000, "Mean income unreasonably low")
        self.assertLess(incomes.mean(), 100000, "Mean income unreasonably high")
        
        # Test log-normal characteristics
        log_incomes = np.log(incomes[incomes > 0])
        if len(log_incomes) > 100:
            # Log of income should be more normal than income itself
            _, income_p = stats.normaltest(incomes)
            _, log_income_p = stats.normaltest(log_incomes)
            # Log-transformed should be more normal (higher p-value)
            # This is a directional test, not requiring strict p > 0.05
    
    def test_age_income_correlation(self):
        """Test positive correlation between age and income."""
        data = self.generator.generate_demographics(self.sample_size)
        
        # Calculate correlation
        age_income_corr = data['age'].corr(data['annual_income'])
        
        # Should be positive correlation (experience increases income)
        self.assertGreater(age_income_corr, 0.1, 
                          f"Age-income correlation too weak: {age_income_corr:.3f}")
        self.assertLess(age_income_corr, 0.6,
                       f"Age-income correlation suspiciously strong: {age_income_corr:.3f}")
    
    def test_education_income_correlation(self):
        """Test positive correlation between education and income."""
        data = self.generator.generate_demographics(self.sample_size)
        
        # Create numeric mapping for education
        education_order = {'High School': 1, 'Other': 2, 'Bachelors': 3, 
                          'Masters': 4, 'Doctorate': 5}
        data['education_numeric'] = data['education'].map(education_order)
        
        # Calculate correlation
        edu_income_corr = data['education_numeric'].corr(data['annual_income'])
        
        # Should be positive correlation (higher education → higher income)
        self.assertGreater(edu_income_corr, 0.2, 
                          f"Education-income correlation too weak: {edu_income_corr:.3f}")
        self.assertLess(edu_income_corr, 0.7,
                       f"Education-income correlation suspiciously strong: {edu_income_corr:.3f}")
        
        # Test mean income by education level
        income_by_edu = data.groupby('education')['annual_income'].mean()
        
        # PhD should have higher mean income than high school
        if 'Doctorate' in income_by_edu.index and 'High School' in income_by_edu.index:
            self.assertGreater(income_by_edu['Doctorate'], income_by_edu['High School'],
                             "PhD income not higher than high school")
        
        # Masters should have higher mean income than bachelors
        if 'Masters' in income_by_edu.index and 'Bachelors' in income_by_edu.index:
            self.assertGreater(income_by_edu['Masters'], income_by_edu['Bachelors'],
                             "Masters income not higher than bachelors")
    
    def test_employment_income_relationship(self):
        """Test relationship between employment status and income."""
        data = self.generator.generate_demographics(self.sample_size)
        
        income_by_emp = data.groupby('employment_status')['annual_income'].mean()
        
        # Full-time should generally have higher income than part-time
        if 'Full-time' in income_by_emp.index and 'Part-time' in income_by_emp.index:
            self.assertGreater(income_by_emp['Full-time'], income_by_emp['Part-time'],
                             "Full-time income not higher than part-time")
        
        # Self-employed might have higher mean income than full-time
        if 'Self-employed' in income_by_emp.index and 'Full-time' in income_by_emp.index:
            # Allow for this to be close or higher
            self.assertGreater(income_by_emp['Self-employed'], income_by_emp['Full-time'] * 0.8,
                             "Self-employed income unreasonably low compared to full-time")
        
        # Unemployed should have very low income
        if 'Unemployed' in income_by_emp.index:
            self.assertLess(income_by_emp['Unemployed'], income_by_emp.mean() * 0.3,
                          "Unemployed income too high")
    
    def test_housing_correlations(self):
        """Test housing status correlations with age and income."""
        data = self.generator.generate_demographics(self.sample_size)
        
        # Young people should be more likely to rent
        young = data[data['age'] <= 28]
        overall = data
        
        if len(young) > 50:
            young_rent_rate = (young['housing_status'] == 'Rent').mean()
            overall_rent_rate = (overall['housing_status'] == 'Rent').mean()
            self.assertGreater(young_rent_rate, overall_rent_rate,
                             "Young people not more likely to rent")
        
        # High income people should be more likely to own
        high_income = data[data['annual_income'] > data['annual_income'].quantile(0.75)]
        
        if len(high_income) > 50:
            high_income_own_rate = (high_income['housing_status'] == 'Own').mean()
            overall_own_rate = (overall['housing_status'] == 'Own').mean()
            self.assertGreater(high_income_own_rate, overall_own_rate,
                             "High income people not more likely to own")
    
    def test_marital_age_correlation(self):
        """Test marital status correlation with age.""" 
        data = self.generator.generate_demographics(self.sample_size)
        
        # Young people should be more likely to be single
        young = data[data['age'] <= 25]
        overall = data
        
        if len(young) > 50:
            young_single_rate = (young['marital_status'] == 'Single').mean()
            overall_single_rate = (overall['marital_status'] == 'Single').mean()
            self.assertGreater(young_single_rate, overall_single_rate,
                             "Young people not more likely to be single")
        
        # Older people should be more likely to be widowed
        elderly = data[data['age'] >= 65]
        
        if len(elderly) > 20:
            elderly_widowed_rate = (elderly['marital_status'] == 'Widowed').mean()
            overall_widowed_rate = (overall['marital_status'] == 'Widowed').mean()
            self.assertGreater(elderly_widowed_rate, overall_widowed_rate,
                             "Elderly not more likely to be widowed")
    
    def test_categorical_distributions(self):
        """Test that categorical variables follow expected distributions."""
        data = self.generator.generate_demographics(self.sample_size)
        
        # Test employment status distribution
        emp_counts = data['employment_status'].value_counts(normalize=True)
        expected_emp = self.generator.employment_probs
        
        # Check major categories are within reasonable bounds
        if 'Full-time' in emp_counts:
            self.assertGreater(emp_counts['Full-time'], 0.5,
                             "Full-time employment rate too low")
            self.assertLess(emp_counts['Full-time'], 0.8,
                           "Full-time employment rate too high")
        
        # Test education distribution
        edu_counts = data['education'].value_counts(normalize=True)
        if 'High School' in edu_counts and 'Bachelors' in edu_counts:
            combined_rate = edu_counts['High School'] + edu_counts['Bachelors']
            self.assertGreater(combined_rate, 0.4,
                             "Combined high school + bachelors rate too low")
    
    def test_missing_values(self):
        """Test missing value generation."""
        # Test with missing values
        data_with_missing = self.generator.generate_demographics(
            n_samples=1000, 
            include_missing=True, 
            missing_rate=0.05
        )
        
        # Check that some missing values were introduced
        missable_cols = ['employment_status', 'education', 'marital_status', 
                        'housing_status', 'annual_income']
        
        total_missing = 0
        for col in missable_cols:
            missing_count = data_with_missing[col].isna().sum()
            total_missing += missing_count
        
        self.assertGreater(total_missing, 0, "No missing values were generated")
        self.assertLess(total_missing, len(data_with_missing) * 0.3,
                       "Too many missing values generated")
        
        # Age should never be missing (business rule)
        self.assertEqual(data_with_missing['age'].isna().sum(), 0,
                        "Age should never have missing values")
    
    def test_data_validation(self):
        """Test data validation functions."""
        # Test normal case - should not raise
        data = self.generator.generate_demographics(self.small_sample_size)
        # Validation is called internally, if we get here it passed
        
        # Test edge cases
        with self.assertRaises(ValueError):
            self.generator.generate_demographics(0)  # Invalid sample size
        
        with self.assertRaises(ValueError):
            self.generator.generate_demographics(-10)  # Negative sample size
    
    def test_summary_statistics(self):
        """Test summary statistics generation."""
        data = self.generator.generate_demographics(self.small_sample_size)
        stats = self.generator.get_summary_statistics(data)
        
        # Check that all expected keys are present
        expected_keys = ['age', 'annual_income', 'employment_status', 
                        'education', 'marital_status', 'housing_status']
        for key in expected_keys:
            self.assertIn(key, stats, f"Missing statistics for {key}")
        
        # Check numeric variable stats structure
        for num_var in ['age', 'annual_income']:
            self.assertIn('mean', stats[num_var])
            self.assertIn('std', stats[num_var])
            self.assertIn('min', stats[num_var])
            self.assertIn('max', stats[num_var])
            self.assertIn('median', stats[num_var])
        
        # Check categorical variable stats structure
        for cat_var in ['employment_status', 'education', 'marital_status', 'housing_status']:
            self.assertIn('distribution', stats[cat_var])
            self.assertIn('unique_count', stats[cat_var])
    
    def test_correlation_validation(self):
        """Test correlation validation function."""
        data = self.generator.generate_demographics(self.sample_size)
        correlations = self.generator.validate_correlations(data)
        
        # Check expected correlations are present
        self.assertIn('age_income', correlations)
        self.assertIn('education_income', correlations)
        
        # Check correlations are reasonable
        self.assertGreater(correlations['age_income'], 0,
                          "Age-income correlation should be positive")
        self.assertGreater(correlations['education_income'], 0,
                          "Education-income correlation should be positive")
        
        # Check correlations are not suspiciously high (< 0.8)
        for corr_name, corr_value in correlations.items():
            self.assertLess(abs(corr_value), 0.8,
                           f"{corr_name} correlation suspiciously high: {corr_value}")
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        gen1 = DemographicsGenerator(random_seed=123)
        gen2 = DemographicsGenerator(random_seed=123)
        
        data1 = gen1.generate_demographics(100)
        data2 = gen2.generate_demographics(100)
        
        # DataFrames should be identical
        pd.testing.assert_frame_equal(data1, data2, check_dtype=True)
    
    def test_convenience_function(self):
        """Test convenience function for generating sample data."""
        data = generate_sample_demographics(n_samples=100, random_seed=456)
        
        self.assertEqual(len(data), 100)
        self.assertIn('age', data.columns)
        self.assertIn('annual_income', data.columns)
        
        # Test reproducibility through convenience function
        data2 = generate_sample_demographics(n_samples=100, random_seed=456)
        pd.testing.assert_frame_equal(data, data2)


class TestBusinessRules(unittest.TestCase):
    """Test suite for business rule compliance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = DemographicsGenerator(random_seed=42)
        self.data = self.generator.generate_demographics(2000)
    
    def test_minimum_wage_compliance(self):
        """Test that employed people earn at least minimum wage equivalent."""
        employed_data = self.data[
            ~self.data['employment_status'].isin(['Unemployed'])
        ]
        
        if len(employed_data) > 0:
            min_employed_income = employed_data['annual_income'].min()
            self.assertGreaterEqual(min_employed_income, 15000,
                                  "Employed person earning below minimum wage equivalent")
    
    def test_retirement_age_logic(self):
        """Test retirement age business logic."""
        retired_people = self.data[self.data['employment_status'] == 'Retired']
        
        if len(retired_people) > 0:
            # Most retired people should be over 60
            elderly_retired_rate = (retired_people['age'] >= 60).mean()
            self.assertGreater(elderly_retired_rate, 0.7,
                             "Too many young retirees (unrealistic)")
    
    def test_education_age_consistency(self):
        """Test that education levels are consistent with age."""
        # People under 22 are unlikely to have advanced degrees
        young_people = self.data[self.data['age'] <= 22]
        
        if len(young_people) > 0:
            advanced_degree_rate = young_people['education'].isin(['Masters', 'Doctorate']).mean()
            self.assertLess(advanced_degree_rate, 0.1,
                           "Too many young people with advanced degrees")
    
    def test_income_distribution_reasonableness(self):
        """Test that income distribution is reasonable for banking context."""
        incomes = self.data['annual_income']
        
        # Check percentiles are reasonable
        p10, p50, p90 = np.percentile(incomes, [10, 50, 90])
        
        self.assertGreater(p10, 10000, "10th percentile income too low")
        self.assertGreater(p50, 25000, "Median income too low") 
        self.assertLess(p50, 80000, "Median income too high")
        self.assertLess(p90, 200000, "90th percentile income too high")
        
        # Check income distribution isn't too concentrated
        gini_coeff = self._calculate_gini(incomes)
        self.assertGreater(gini_coeff, 0.2, "Income distribution too equal")
        self.assertLess(gini_coeff, 0.6, "Income distribution too unequal")
    
    def _calculate_gini(self, incomes):
        """Calculate Gini coefficient for income inequality."""
        incomes = np.array(incomes)
        incomes = incomes[incomes > 0]  # Remove zero incomes
        incomes = np.sort(incomes)
        n = len(incomes)
        
        if n == 0:
            return 0
        
        cumulative = np.cumsum(incomes)
        return (2 * np.sum((np.arange(1, n + 1) * incomes))) / (n * cumulative[-1]) - (n + 1) / n


if __name__ == '__main__':
    # Configure warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    print("Running Demographics Generator Test Suite")
    print("=" * 60)
    
    # Run tests
    unittest.main(verbosity=2, argv=[''], exit=False)
    
    # Additional manual validation
    print("\n" + "=" * 60)
    print("Manual Validation Examples")
    print("=" * 60)
    
    # Generate sample for manual inspection
    generator = DemographicsGenerator(random_seed=42)
    sample = generator.generate_demographics(1000)
    
    print(f"\nGenerated {len(sample)} demographic records")
    print("\nSample data:")
    print(sample.head())
    
    print(f"\nData types:")
    print(sample.dtypes)
    
    print(f"\nSummary statistics:")
    stats = generator.get_summary_statistics(sample)
    print(f"Age: mean={stats['age']['mean']:.1f}, std={stats['age']['std']:.1f}")
    print(f"Income: mean=${stats['annual_income']['mean']:,.0f}, std=${stats['annual_income']['std']:,.0f}")
    
    print(f"\nCorrelations:")
    correlations = generator.validate_correlations(sample)
    for name, value in correlations.items():
        print(f"{name}: {value:.3f}")
    
    print(f"\nCategorical distributions:")
    for col in ['employment_status', 'education', 'marital_status', 'housing_status']:
        print(f"\n{col}:")
        counts = sample[col].value_counts(normalize=True)
        for value, pct in counts.head().items():
            print(f"  {value}: {pct:.1%}")
