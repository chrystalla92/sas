"""
Statistical Validation Tests for Random Utilities

Comprehensive tests to validate that Python random utilities match SAS
statistical properties and behavior. Tests include distribution goodness-of-fit,
moment matching, reproducibility, and performance validation.

Author: Risk Analytics Team
Date: 2025
"""

import unittest
import numpy as np
import time
from random_utils import (
    RandomGenerator, StatisticalValidator, validate_sas_equivalence,
    rand_uniform, rand_normal, rand_beta, rand_exponential, rand_gamma,
    set_global_seed, get_global_seed
)


class TestRandomGenerator(unittest.TestCase):
    """Test RandomGenerator class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = RandomGenerator()
        self.test_seed = 12345
        
    def test_seed_management(self):
        """Test seed setting and getting functionality."""
        # Test initial seed
        self.assertIsNone(self.generator.get_seed())
        
        # Test seed setting
        self.generator.set_seed(self.test_seed)
        self.assertEqual(self.generator.get_seed(), self.test_seed)
        
        # Test seed reproducibility
        self.generator.set_seed(42)
        val1 = self.generator.uniform()
        self.generator.set_seed(42)
        val2 = self.generator.uniform()
        self.assertEqual(val1, val2)
    
    def test_uniform_distribution(self):
        """Test uniform distribution generation."""
        self.generator.set_seed(self.test_seed)
        
        # Test default parameters (0, 1)
        samples = self.generator.uniform(size=1000)
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples < 1))
        
        # Test custom range
        samples = self.generator.uniform(10, 20, size=1000)
        self.assertTrue(np.all(samples >= 10))
        self.assertTrue(np.all(samples < 20))
        
        # Test parameter validation
        with self.assertRaises(ValueError):
            self.generator.uniform(5, 5)  # low = high
        with self.assertRaises(ValueError):
            self.generator.uniform(10, 5)  # low > high
    
    def test_normal_distribution(self):
        """Test normal distribution generation."""
        self.generator.set_seed(self.test_seed)
        
        # Test default parameters (0, 1)
        samples = self.generator.normal(size=10000)
        self.assertAlmostEqual(np.mean(samples), 0, delta=0.1)
        self.assertAlmostEqual(np.std(samples), 1, delta=0.1)
        
        # Test custom parameters
        mean, std = 42, 12
        samples = self.generator.normal(mean, std, size=10000)
        self.assertAlmostEqual(np.mean(samples), mean, delta=1)
        self.assertAlmostEqual(np.std(samples), std, delta=1)
        
        # Test parameter validation
        with self.assertRaises(ValueError):
            self.generator.normal(0, 0)  # std = 0
        with self.assertRaises(ValueError):
            self.generator.normal(0, -1)  # std < 0
    
    def test_beta_distribution(self):
        """Test beta distribution generation."""
        self.generator.set_seed(self.test_seed)
        
        alpha, beta = 2, 5
        samples = self.generator.beta(alpha, beta, size=10000)
        
        # Test range
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples <= 1))
        
        # Test approximate moments
        expected_mean = alpha / (alpha + beta)
        expected_var = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        self.assertAlmostEqual(np.mean(samples), expected_mean, delta=0.05)
        self.assertAlmostEqual(np.var(samples), expected_var, delta=0.01)
        
        # Test parameter validation
        with self.assertRaises(ValueError):
            self.generator.beta(0, 1)  # alpha = 0
        with self.assertRaises(ValueError):
            self.generator.beta(1, 0)  # beta = 0
        with self.assertRaises(ValueError):
            self.generator.beta(-1, 1)  # alpha < 0
    
    def test_exponential_distribution(self):
        """Test exponential distribution generation."""
        self.generator.set_seed(self.test_seed)
        
        rate = 2.0
        samples = self.generator.exponential(rate, size=10000)
        
        # Test range (should be >= 0)
        self.assertTrue(np.all(samples >= 0))
        
        # Test approximate moments
        expected_mean = 1 / rate
        expected_var = 1 / (rate**2)
        self.assertAlmostEqual(np.mean(samples), expected_mean, delta=0.05)
        self.assertAlmostEqual(np.var(samples), expected_var, delta=0.05)
        
        # Test parameter validation
        with self.assertRaises(ValueError):
            self.generator.exponential(0)  # rate = 0
        with self.assertRaises(ValueError):
            self.generator.exponential(-1)  # rate < 0
    
    def test_gamma_distribution(self):
        """Test gamma distribution generation."""
        self.generator.set_seed(self.test_seed)
        
        shape = 5.0
        scale = 1.0  # Default SAS scale
        samples = self.generator.gamma(shape, scale, size=10000)
        
        # Test range (should be >= 0)
        self.assertTrue(np.all(samples >= 0))
        
        # Test approximate moments
        expected_mean = shape * scale
        expected_var = shape * (scale**2)
        self.assertAlmostEqual(np.mean(samples), expected_mean, delta=0.5)
        self.assertAlmostEqual(np.var(samples), expected_var, delta=1.0)
        
        # Test parameter validation
        with self.assertRaises(ValueError):
            self.generator.gamma(0)  # shape = 0
        with self.assertRaises(ValueError):
            self.generator.gamma(-1)  # shape < 0
        with self.assertRaises(ValueError):
            self.generator.gamma(1, 0)  # scale = 0


class TestStatisticalValidator(unittest.TestCase):
    """Test statistical validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = RandomGenerator(seed=12345)
        self.validator = StatisticalValidator()
        
    def test_ks_test(self):
        """Test Kolmogorov-Smirnov test functionality."""
        # Test uniform distribution
        samples = self.generator.uniform(0, 1, size=1000)
        ks_stat, p_value = self.validator.ks_test(samples, 'uniform', low=0, high=1)
        self.assertIsInstance(ks_stat, float)
        self.assertIsInstance(p_value, float)
        self.assertGreater(p_value, 0)
        self.assertLess(p_value, 1)
        
        # Test normal distribution
        samples = self.generator.normal(0, 1, size=1000)
        ks_stat, p_value = self.validator.ks_test(samples, 'normal', mean=0, std=1)
        self.assertIsInstance(ks_stat, float)
        self.assertIsInstance(p_value, float)
        
        # Test unsupported distribution
        with self.assertRaises(ValueError):
            self.validator.ks_test(samples, 'unsupported')
    
    def test_moment_comparison(self):
        """Test moment comparison functionality."""
        samples = np.random.normal(5, 2, size=1000)
        
        # Test with correct moments
        passed, results = self.validator.compare_moments(samples, 5, 4, tolerance=0.2)
        self.assertIsInstance(passed, bool)
        self.assertIsInstance(results, dict)
        self.assertIn('sample_mean', results)
        self.assertIn('sample_var', results)
        self.assertIn('overall_passed', results)
        
        # Test with incorrect moments
        passed, results = self.validator.compare_moments(samples, 10, 1, tolerance=0.1)
        self.assertFalse(passed)
    
    def test_range_validation(self):
        """Test range validation functionality."""
        samples = np.array([0.1, 0.5, 0.9])
        
        # Test valid range
        passed, results = self.validator.validate_range(samples, 0, 1)
        self.assertTrue(passed)
        self.assertEqual(results['out_of_range_count'], 0)
        
        # Test invalid range
        passed, results = self.validator.validate_range(samples, 0.2, 0.8)
        self.assertFalse(passed)
        self.assertGreater(results['out_of_range_count'], 0)


class TestGlobalFunctions(unittest.TestCase):
    """Test global convenience functions."""
    
    def test_global_seed_management(self):
        """Test global seed setting and getting."""
        # Test seed setting
        set_global_seed(54321)
        self.assertEqual(get_global_seed(), 54321)
        
        # Test reproducibility
        set_global_seed(42)
        val1 = rand_uniform()
        set_global_seed(42)
        val2 = rand_uniform()
        self.assertEqual(val1, val2)
    
    def test_convenience_functions(self):
        """Test global convenience functions."""
        set_global_seed(12345)
        
        # Test all convenience functions
        u_val = rand_uniform()
        self.assertGreaterEqual(u_val, 0)
        self.assertLess(u_val, 1)
        
        n_val = rand_normal(10, 2)
        self.assertIsInstance(n_val, float)
        
        b_val = rand_beta(2, 3)
        self.assertGreaterEqual(b_val, 0)
        self.assertLessEqual(b_val, 1)
        
        e_val = rand_exponential(1.5)
        self.assertGreaterEqual(e_val, 0)
        
        g_val = rand_gamma(3)
        self.assertGreaterEqual(g_val, 0)


class TestSASEquivalence(unittest.TestCase):
    """Test comprehensive SAS equivalence validation."""
    
    def test_full_validation(self):
        """Test complete SAS equivalence validation."""
        generator = RandomGenerator()
        results = validate_sas_equivalence(generator, n_samples=5000, seed=12345)
        
        # Check that all distributions are tested
        expected_distributions = ['uniform', 'normal', 'beta', 'exponential', 'gamma']
        for dist in expected_distributions:
            self.assertIn(dist, results)
            self.assertIn('ks_test', results[dist])
            self.assertIn('moment_test', results[dist])
            
            # Check that tests return reasonable results
            ks_test = results[dist]['ks_test']
            self.assertIn('statistic', ks_test)
            self.assertIn('p_value', ks_test)
            self.assertIn('passed', ks_test)
            
            moment_test = results[dist]['moment_test']
            self.assertIn('overall_passed', moment_test)


class TestPerformance(unittest.TestCase):
    """Test performance requirements."""
    
    def test_performance_requirements(self):
        """Test that generators can produce 10,000+ samples efficiently."""
        generator = RandomGenerator(seed=12345)
        n_samples = 10000
        
        distributions = [
            ('uniform', lambda: generator.uniform(size=n_samples)),
            ('normal', lambda: generator.normal(0, 1, size=n_samples)),
            ('beta', lambda: generator.beta(2, 3, size=n_samples)),
            ('exponential', lambda: generator.exponential(1.0, size=n_samples)),
            ('gamma', lambda: generator.gamma(2.0, size=n_samples))
        ]
        
        for dist_name, func in distributions:
            start_time = time.time()
            samples = func()
            end_time = time.time()
            
            # Check that we get correct number of samples
            self.assertEqual(len(samples), n_samples)
            
            # Check that generation is reasonably fast (< 1 second)
            generation_time = end_time - start_time
            self.assertLess(generation_time, 1.0, 
                          f"{dist_name} distribution too slow: {generation_time:.3f}s")


class TestReproducibility(unittest.TestCase):
    """Test reproducibility across multiple runs."""
    
    def test_seed_reproducibility(self):
        """Test that same seeds produce identical results."""
        seed = 98765
        n_samples = 100
        
        # Test each distribution type
        distributions = [
            ('uniform', lambda gen: gen.uniform(size=n_samples)),
            ('normal', lambda gen: gen.normal(5, 2, size=n_samples)),
            ('beta', lambda gen: gen.beta(1.5, 2.5, size=n_samples)),
            ('exponential', lambda gen: gen.exponential(0.8, size=n_samples)),
            ('gamma', lambda gen: gen.gamma(3.2, size=n_samples))
        ]
        
        for dist_name, func in distributions:
            # Generate samples twice with same seed
            gen1 = RandomGenerator(seed)
            samples1 = func(gen1)
            
            gen2 = RandomGenerator(seed)
            samples2 = func(gen2)
            
            # Check that results are identical
            np.testing.assert_array_equal(samples1, samples2,
                                        err_msg=f"{dist_name} distribution not reproducible")
    
    def test_multiple_calls_reproducibility(self):
        """Test reproducibility across multiple function calls."""
        generator = RandomGenerator(seed=11111)
        
        # Generate sequence of different distributions
        vals1 = [
            generator.uniform(),
            generator.normal(10, 3),
            generator.beta(2, 2),
            generator.exponential(2),
            generator.gamma(4)
        ]
        
        # Reset seed and generate same sequence
        generator.set_seed(11111)
        vals2 = [
            generator.uniform(),
            generator.normal(10, 3),
            generator.beta(2, 2),
            generator.exponential(2),
            generator.gamma(4)
        ]
        
        # Check that sequences are identical
        for i, (v1, v2) in enumerate(zip(vals1, vals2)):
            self.assertEqual(v1, v2, f"Sequence not reproducible at position {i}")


def run_statistical_validation():
    """Run comprehensive statistical validation and print results."""
    print("Running comprehensive SAS equivalence validation...")
    print("=" * 60)
    
    generator = RandomGenerator()
    results = validate_sas_equivalence(generator, n_samples=10000, seed=12345)
    
    for distribution, dist_results in results.items():
        print(f"\n{distribution.upper()} DISTRIBUTION:")
        print("-" * 30)
        
        # KS Test results
        ks_test = dist_results['ks_test']
        print(f"KS Test: statistic={ks_test['statistic']:.4f}, "
              f"p-value={ks_test['p_value']:.4f}, "
              f"passed={ks_test['passed']}")
        
        # Moment test results
        moment_test = dist_results['moment_test']
        print(f"Moments: mean_diff={moment_test['mean_diff_pct']:.2f}%, "
              f"var_diff={moment_test['var_diff_pct']:.2f}%, "
              f"passed={moment_test['overall_passed']}")
        
        # Range test results (if available)
        if 'range_test' in dist_results:
            range_test = dist_results['range_test']
            print(f"Range: min={range_test['sample_min']:.4f}, "
                  f"max={range_test['sample_max']:.4f}, "
                  f"passed={range_test['overall_passed']}")


if __name__ == '__main__':
    # Run statistical validation first
    run_statistical_validation()
    
    print("\n" + "=" * 60)
    print("Running unit tests...")
    print("=" * 60)
    
    # Run unit tests
    unittest.main(verbosity=2)
