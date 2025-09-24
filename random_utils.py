"""
Random Utilities - SAS-Compatible Random Number Generation

This module provides Python equivalents for SAS random functions using numpy.random.
Ensures statistical properties match SAS output and provides reproducible results
with proper seed management.

## SAS to Python Parameter Mappings

### Uniform Distribution
- SAS: `rand('uniform')` → Python: `uniform(0, 1)`
- SAS: `rand('uniform', a, b)` → Python: `uniform(a, b)`
- Returns values in range [a, b) where b is exclusive

### Normal Distribution  
- SAS: `rand('normal', mean, std)` → Python: `normal(mean, std)`
- Uses same parameterization: mean and standard deviation
- No parameter conversion needed

### Beta Distribution
- SAS: `rand('beta', alpha, beta)` → Python: `beta(alpha, beta)` 
- Uses same shape parameters α and β
- Returns values in range [0, 1]
- Mean: α/(α+β), Variance: αβ/[(α+β)²(α+β+1)]

### Exponential Distribution
- SAS: `rand('exponential', rate)` → Python: `exponential(scale=1/rate)`
- **IMPORTANT**: SAS uses rate (λ) while NumPy uses scale (1/λ)
- Conversion: numpy_scale = 1 / sas_rate
- Mean: 1/rate, Variance: 1/rate²

### Gamma Distribution
- SAS: `rand('gamma', shape)` → Python: `gamma(shape, scale=1.0)`
- SAS uses shape parameter with implicit scale=1.0
- Mean: shape*scale, Variance: shape*scale²

## Usage Examples

### Basic Usage with Global Functions
```python
import random_utils as ru

# Set global seed for reproducibility
ru.set_global_seed(12345)

# Generate single values
uniform_val = ru.rand_uniform()        # 0 to 1
uniform_range = ru.rand_uniform(10, 20)  # 10 to 20
normal_val = ru.rand_normal(42, 12)    # mean=42, std=12
beta_val = ru.rand_beta(2, 5)          # shape parameters
exp_val = ru.rand_exponential(2.0)     # rate=2.0
gamma_val = ru.rand_gamma(5)           # shape=5, scale=1.0

# Generate arrays
uniform_array = ru.rand_uniform(size=1000)
normal_array = ru.rand_normal(0, 1, size=(100, 10))
```

### Advanced Usage with RandomGenerator Class
```python
from random_utils import RandomGenerator

# Create generator with seed
generator = RandomGenerator(seed=12345)

# Generate samples
samples = generator.uniform(0, 1, size=10000)

# Change seed for different sequence
generator.set_seed(54321)
new_samples = generator.normal(10, 2, size=5000)

# Parameter validation
try:
    generator.validate_distribution_params('uniform', low=0, high=1)
except ValueError as e:
    print(f"Invalid parameters: {e}")
```

### SAS Code Translation Example
```sas
/* Original SAS Code */
data synthetic_data;
    call streaminit(12345);
    do i = 1 to 10000;
        age = max(18, min(75, round(rand('normal', 42, 12))));
        income_factor = exp(rand('normal', 0, 0.3));
        credit_util = rand('beta', 2, 5) * 100;
        default_prob = rand('uniform');
        output;
    end;
run;
```

```python
# Equivalent Python Code
import numpy as np
from random_utils import RandomGenerator

generator = RandomGenerator(seed=12345)

ages = np.clip(np.round(generator.normal(42, 12, size=10000)), 18, 75)
income_factors = np.exp(generator.normal(0, 0.3, size=10000))
credit_utils = generator.beta(2, 5, size=10000) * 100
default_probs = generator.uniform(size=10000)
```

### Statistical Validation
```python
from random_utils import validate_sas_equivalence, RandomGenerator

# Comprehensive validation
generator = RandomGenerator()
results = validate_sas_equivalence(generator, n_samples=10000)

# Check results
for dist, result in results.items():
    ks_passed = result['ks_test']['passed']
    moments_passed = result['moment_test']['overall_passed']
    print(f"{dist}: KS Test={ks_passed}, Moments={moments_passed}")
```

## Performance Characteristics
- Optimized for generating 10,000+ samples per distribution call
- Vectorized operations using NumPy for efficient computation
- Minimal overhead for parameter validation and conversion

## Statistical Validation
- Kolmogorov-Smirnov tests confirm distribution equivalence
- Moment matching validates mean and variance accuracy
- Anderson-Darling tests for additional goodness-of-fit validation
- Range validation ensures values fall within expected bounds

## Thread Safety
- RandomGenerator instances are thread-safe within individual instances
- Global functions use a shared generator and are not thread-safe
- For multi-threaded applications, create separate RandomGenerator instances

Author: Risk Analytics Team
Date: 2025
"""

import numpy as np
from typing import Union, Optional, Tuple
import warnings
from scipy import stats


class RandomGenerator:
    """
    SAS-compatible random number generator with configurable seed management.
    
    This class provides wrapper functions for numpy random distributions that
    match SAS statistical properties and parameter conventions.
    
    Attributes:
        _rng: numpy.random.Generator instance for random number generation
        _seed: Current seed value for reproducibility
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize RandomGenerator with optional seed.
        
        Args:
            seed: Random seed for reproducibility. If None, uses random seed.
        """
        self._seed = seed
        self._rng = np.random.default_rng(seed)
    
    def set_seed(self, seed: int) -> None:
        """
        Set new random seed for reproducible results.
        
        Args:
            seed: New seed value
        """
        self._seed = seed
        self._rng = np.random.default_rng(seed)
    
    def get_seed(self) -> Optional[int]:
        """
        Get current seed value.
        
        Returns:
            Current seed value or None if not set
        """
        return self._seed
    
    def uniform(self, low: float = 0.0, high: float = 1.0, size: Optional[Union[int, Tuple]] = None) -> Union[float, np.ndarray]:
        """
        Generate random numbers from uniform distribution.
        
        SAS equivalent: rand('uniform') or rand('uniform', a, b)
        
        Args:
            low: Lower bound (inclusive)
            high: Upper bound (exclusive)  
            size: Shape of output. If None, returns scalar
            
        Returns:
            Random samples from uniform distribution
            
        Raises:
            ValueError: If low >= high
        """
        if low >= high:
            raise ValueError(f"low ({low}) must be less than high ({high})")
            
        return self._rng.uniform(low, high, size)
    
    def normal(self, mean: float = 0.0, std: float = 1.0, size: Optional[Union[int, Tuple]] = None) -> Union[float, np.ndarray]:
        """
        Generate random numbers from normal distribution.
        
        SAS equivalent: rand('normal', mean, std)
        
        Args:
            mean: Mean of distribution
            std: Standard deviation of distribution
            size: Shape of output. If None, returns scalar
            
        Returns:
            Random samples from normal distribution
            
        Raises:
            ValueError: If std <= 0
        """
        if std <= 0:
            raise ValueError(f"std ({std}) must be positive")
            
        return self._rng.normal(mean, std, size)
    
    def beta(self, alpha: float, beta: float, size: Optional[Union[int, Tuple]] = None) -> Union[float, np.ndarray]:
        """
        Generate random numbers from beta distribution.
        
        SAS equivalent: rand('beta', alpha, beta)
        
        Args:
            alpha: Alpha shape parameter (must be > 0)
            beta: Beta shape parameter (must be > 0)
            size: Shape of output. If None, returns scalar
            
        Returns:
            Random samples from beta distribution (values in [0,1])
            
        Raises:
            ValueError: If alpha <= 0 or beta <= 0
        """
        if alpha <= 0:
            raise ValueError(f"alpha ({alpha}) must be positive")
        if beta <= 0:
            raise ValueError(f"beta ({beta}) must be positive")
            
        return self._rng.beta(alpha, beta, size)
    
    def exponential(self, rate: float = 1.0, size: Optional[Union[int, Tuple]] = None) -> Union[float, np.ndarray]:
        """
        Generate random numbers from exponential distribution.
        
        SAS equivalent: rand('exponential', rate)
        
        Note: SAS uses rate parameterization while numpy uses scale.
        Conversion: numpy_scale = 1 / sas_rate
        
        Args:
            rate: Rate parameter (lambda, must be > 0)
            size: Shape of output. If None, returns scalar
            
        Returns:
            Random samples from exponential distribution
            
        Raises:
            ValueError: If rate <= 0
        """
        if rate <= 0:
            raise ValueError(f"rate ({rate}) must be positive")
            
        scale = 1.0 / rate
        return self._rng.exponential(scale, size)
    
    def gamma(self, shape: float, scale: float = 1.0, size: Optional[Union[int, Tuple]] = None) -> Union[float, np.ndarray]:
        """
        Generate random numbers from gamma distribution.
        
        SAS equivalent: rand('gamma', shape)
        
        Note: SAS gamma uses shape parameter with implicit scale=1.0
        
        Args:
            shape: Shape parameter (must be > 0)
            scale: Scale parameter (must be > 0, default=1.0 for SAS compatibility)
            size: Shape of output. If None, returns scalar
            
        Returns:
            Random samples from gamma distribution
            
        Raises:
            ValueError: If shape <= 0 or scale <= 0
        """
        if shape <= 0:
            raise ValueError(f"shape ({shape}) must be positive")
        if scale <= 0:
            raise ValueError(f"scale ({scale}) must be positive")
            
        return self._rng.gamma(shape, scale, size)
    
    def validate_distribution_params(self, distribution: str, **params) -> bool:
        """
        Validate parameters for specified distribution.
        
        Args:
            distribution: Distribution name ('uniform', 'normal', 'beta', 'exponential', 'gamma')
            **params: Distribution parameters to validate
            
        Returns:
            True if parameters are valid
            
        Raises:
            ValueError: If parameters are invalid
        """
        if distribution == 'uniform':
            low = params.get('low', 0.0)
            high = params.get('high', 1.0)
            if low >= high:
                raise ValueError(f"uniform: low ({low}) must be less than high ({high})")
                
        elif distribution == 'normal':
            std = params.get('std', 1.0)
            if std <= 0:
                raise ValueError(f"normal: std ({std}) must be positive")
                
        elif distribution == 'beta':
            alpha = params.get('alpha')
            beta = params.get('beta')
            if alpha is None or beta is None:
                raise ValueError("beta: both alpha and beta parameters required")
            if alpha <= 0:
                raise ValueError(f"beta: alpha ({alpha}) must be positive")
            if beta <= 0:
                raise ValueError(f"beta: beta ({beta}) must be positive")
                
        elif distribution == 'exponential':
            rate = params.get('rate', 1.0)
            if rate <= 0:
                raise ValueError(f"exponential: rate ({rate}) must be positive")
                
        elif distribution == 'gamma':
            shape = params.get('shape')
            scale = params.get('scale', 1.0)
            if shape is None:
                raise ValueError("gamma: shape parameter required")
            if shape <= 0:
                raise ValueError(f"gamma: shape ({shape}) must be positive")
            if scale <= 0:
                raise ValueError(f"gamma: scale ({scale}) must be positive")
                
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
            
        return True


# Global instance for convenience functions
_global_generator = RandomGenerator()


def set_global_seed(seed: int) -> None:
    """
    Set seed for global RandomGenerator instance.
    
    Args:
        seed: Random seed value
    """
    _global_generator.set_seed(seed)


def get_global_seed() -> Optional[int]:
    """
    Get seed from global RandomGenerator instance.
    
    Returns:
        Current global seed value
    """
    return _global_generator.get_seed()


def rand_uniform(low: float = 0.0, high: float = 1.0, size: Optional[Union[int, Tuple]] = None) -> Union[float, np.ndarray]:
    """
    Convenience function for uniform distribution using global generator.
    
    SAS equivalent: rand('uniform') or rand('uniform', a, b)
    """
    return _global_generator.uniform(low, high, size)


def rand_normal(mean: float = 0.0, std: float = 1.0, size: Optional[Union[int, Tuple]] = None) -> Union[float, np.ndarray]:
    """
    Convenience function for normal distribution using global generator.
    
    SAS equivalent: rand('normal', mean, std)
    """
    return _global_generator.normal(mean, std, size)


def rand_beta(alpha: float, beta: float, size: Optional[Union[int, Tuple]] = None) -> Union[float, np.ndarray]:
    """
    Convenience function for beta distribution using global generator.
    
    SAS equivalent: rand('beta', alpha, beta)
    """
    return _global_generator.beta(alpha, beta, size)


def rand_exponential(rate: float = 1.0, size: Optional[Union[int, Tuple]] = None) -> Union[float, np.ndarray]:
    """
    Convenience function for exponential distribution using global generator.
    
    SAS equivalent: rand('exponential', rate)
    """
    return _global_generator.exponential(rate, size)


def rand_gamma(shape: float, scale: float = 1.0, size: Optional[Union[int, Tuple]] = None) -> Union[float, np.ndarray]:
    """
    Convenience function for gamma distribution using global generator.
    
    SAS equivalent: rand('gamma', shape)
    """
    return _global_generator.gamma(shape, scale, size)


class StatisticalValidator:
    """
    Statistical validation utilities to compare distributions and test equivalence.
    
    Provides methods to validate that generated samples match expected statistical
    properties using various statistical tests.
    """
    
    @staticmethod
    def ks_test(sample: np.ndarray, distribution: str, **params) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test for distribution goodness of fit.
        
        Args:
            sample: Sample data to test
            distribution: Distribution name ('uniform', 'normal', 'beta', 'exponential', 'gamma')
            **params: Distribution parameters
            
        Returns:
            Tuple of (statistic, p_value)
            
        Raises:
            ValueError: If distribution is not supported
        """
        if distribution == 'uniform':
            low = params.get('low', 0.0)
            high = params.get('high', 1.0)
            return stats.kstest(sample, lambda x: stats.uniform.cdf(x, loc=low, scale=high-low))
            
        elif distribution == 'normal':
            mean = params.get('mean', 0.0)
            std = params.get('std', 1.0)
            return stats.kstest(sample, lambda x: stats.norm.cdf(x, loc=mean, scale=std))
            
        elif distribution == 'beta':
            alpha = params.get('alpha')
            beta = params.get('beta')
            if alpha is None or beta is None:
                raise ValueError("Beta distribution requires alpha and beta parameters")
            return stats.kstest(sample, lambda x: stats.beta.cdf(x, alpha, beta))
            
        elif distribution == 'exponential':
            rate = params.get('rate', 1.0)
            scale = 1.0 / rate
            return stats.kstest(sample, lambda x: stats.expon.cdf(x, scale=scale))
            
        elif distribution == 'gamma':
            shape = params.get('shape')
            scale = params.get('scale', 1.0)
            if shape is None:
                raise ValueError("Gamma distribution requires shape parameter")
            return stats.kstest(sample, lambda x: stats.gamma.cdf(x, shape, scale=scale))
            
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
    
    @staticmethod
    def anderson_darling_test(sample: np.ndarray, distribution: str = 'norm') -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Perform Anderson-Darling test for normality or other distributions.
        
        Args:
            sample: Sample data to test
            distribution: Distribution to test against ('norm', 'expon', 'logistic', 'gumbel', 'extreme1')
            
        Returns:
            Tuple of (statistic, critical_values, significance_levels)
        """
        return stats.anderson(sample, dist=distribution)
    
    @staticmethod
    def compare_moments(sample: np.ndarray, expected_mean: float, expected_var: float, 
                       tolerance: float = 0.1) -> Tuple[bool, dict]:
        """
        Compare sample moments (mean, variance) with expected values.
        
        Args:
            sample: Sample data
            expected_mean: Expected mean value
            expected_var: Expected variance value
            tolerance: Relative tolerance for comparison
            
        Returns:
            Tuple of (passed, results_dict)
        """
        sample_mean = np.mean(sample)
        sample_var = np.var(sample, ddof=1)  # Use sample variance
        
        mean_diff = abs(sample_mean - expected_mean) / max(abs(expected_mean), 1e-10)
        var_diff = abs(sample_var - expected_var) / max(abs(expected_var), 1e-10)
        
        mean_passed = mean_diff <= tolerance
        var_passed = var_diff <= tolerance
        
        results = {
            'sample_mean': sample_mean,
            'expected_mean': expected_mean,
            'mean_diff_pct': mean_diff * 100,
            'mean_passed': mean_passed,
            'sample_var': sample_var,
            'expected_var': expected_var,
            'var_diff_pct': var_diff * 100,
            'var_passed': var_passed,
            'overall_passed': mean_passed and var_passed
        }
        
        return results['overall_passed'], results
    
    @staticmethod
    def validate_range(sample: np.ndarray, min_val: float, max_val: float) -> Tuple[bool, dict]:
        """
        Validate that all sample values fall within expected range.
        
        Args:
            sample: Sample data
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Tuple of (passed, results_dict)
        """
        sample_min = np.min(sample)
        sample_max = np.max(sample)
        
        min_passed = sample_min >= min_val
        max_passed = sample_max <= max_val
        
        results = {
            'sample_min': sample_min,
            'expected_min': min_val,
            'min_passed': min_passed,
            'sample_max': sample_max,
            'expected_max': max_val,
            'max_passed': max_passed,
            'overall_passed': min_passed and max_passed,
            'out_of_range_count': np.sum((sample < min_val) | (sample > max_val))
        }
        
        return results['overall_passed'], results


def validate_sas_equivalence(generator: RandomGenerator, n_samples: int = 10000, 
                           alpha: float = 0.05, seed: int = 12345) -> dict:
    """
    Comprehensive validation of SAS equivalence for all distributions.
    
    Args:
        generator: RandomGenerator instance to test
        n_samples: Number of samples to generate for testing
        alpha: Significance level for statistical tests
        seed: Random seed for reproducible testing
        
    Returns:
        Dictionary with validation results for each distribution
    """
    generator.set_seed(seed)
    results = {}
    
    # Test uniform distribution
    generator.set_seed(seed)
    uniform_samples = generator.uniform(0, 1, n_samples)
    ks_stat, ks_p = StatisticalValidator.ks_test(uniform_samples, 'uniform', low=0, high=1)
    range_passed, range_results = StatisticalValidator.validate_range(uniform_samples, 0, 1)
    moment_passed, moment_results = StatisticalValidator.compare_moments(
        uniform_samples, 0.5, 1/12  # uniform(0,1) mean=0.5, var=1/12
    )
    
    results['uniform'] = {
        'ks_test': {'statistic': ks_stat, 'p_value': ks_p, 'passed': ks_p > alpha},
        'range_test': range_results,
        'moment_test': moment_results
    }
    
    # Test normal distribution
    generator.set_seed(seed)
    normal_samples = generator.normal(42, 12, n_samples)
    ks_stat, ks_p = StatisticalValidator.ks_test(normal_samples, 'normal', mean=42, std=12)
    moment_passed, moment_results = StatisticalValidator.compare_moments(
        normal_samples, 42, 12**2  # mean=42, var=std^2
    )
    
    results['normal'] = {
        'ks_test': {'statistic': ks_stat, 'p_value': ks_p, 'passed': ks_p > alpha},
        'moment_test': moment_results
    }
    
    # Test beta distribution
    generator.set_seed(seed)
    beta_samples = generator.beta(2, 5, n_samples)
    ks_stat, ks_p = StatisticalValidator.ks_test(beta_samples, 'beta', alpha=2, beta=5)
    range_passed, range_results = StatisticalValidator.validate_range(beta_samples, 0, 1)
    # Beta(α,β) mean = α/(α+β), var = αβ/[(α+β)²(α+β+1)]
    expected_mean = 2 / (2 + 5)
    expected_var = (2 * 5) / ((2 + 5)**2 * (2 + 5 + 1))
    moment_passed, moment_results = StatisticalValidator.compare_moments(
        beta_samples, expected_mean, expected_var
    )
    
    results['beta'] = {
        'ks_test': {'statistic': ks_stat, 'p_value': ks_p, 'passed': ks_p > alpha},
        'range_test': range_results,
        'moment_test': moment_results
    }
    
    # Test exponential distribution
    generator.set_seed(seed)
    exp_samples = generator.exponential(2, n_samples)
    ks_stat, ks_p = StatisticalValidator.ks_test(exp_samples, 'exponential', rate=2)
    range_passed, range_results = StatisticalValidator.validate_range(exp_samples, 0, np.inf)
    # Exponential(λ) mean = 1/λ, var = 1/λ²
    expected_mean = 1 / 2
    expected_var = 1 / (2**2)
    moment_passed, moment_results = StatisticalValidator.compare_moments(
        exp_samples, expected_mean, expected_var
    )
    
    results['exponential'] = {
        'ks_test': {'statistic': ks_stat, 'p_value': ks_p, 'passed': ks_p > alpha},
        'range_test': range_results,
        'moment_test': moment_results
    }
    
    # Test gamma distribution
    generator.set_seed(seed)
    gamma_samples = generator.gamma(5, 1, n_samples)
    ks_stat, ks_p = StatisticalValidator.ks_test(gamma_samples, 'gamma', shape=5, scale=1)
    range_passed, range_results = StatisticalValidator.validate_range(gamma_samples, 0, np.inf)
    # Gamma(shape, scale) mean = shape*scale, var = shape*scale²
    expected_mean = 5 * 1
    expected_var = 5 * (1**2)
    moment_passed, moment_results = StatisticalValidator.compare_moments(
        gamma_samples, expected_mean, expected_var
    )
    
    results['gamma'] = {
        'ks_test': {'statistic': ks_stat, 'p_value': ks_p, 'passed': ks_p > alpha},
        'range_test': range_results,
        'moment_test': moment_results
    }
    
    return results
