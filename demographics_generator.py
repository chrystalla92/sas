"""
Demographics Generator for Credit Risk Scoring

This module generates realistic customer demographic variables for synthetic
credit risk datasets, maintaining business-realistic correlations and 
distributions consistent with banking industry patterns.

Author: Risk Analytics Team
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import warnings


class DemographicsGenerator:
    """
    Generate realistic customer demographic data for credit risk modeling.
    
    This class creates synthetic demographic data with realistic correlations
    between variables such as age, income, employment status, education level,
    marital status, and housing status.
    """
    
    def __init__(self, random_seed: Optional[int] = 12345):
        """
        Initialize the Demographics Generator.
        
        Parameters:
        -----------
        random_seed : int, optional
            Random seed for reproducible results (default: 12345)
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Define probability distributions for categorical variables
        self.employment_probs = {
            'Full-time': 0.65,
            'Self-employed': 0.15,
            'Part-time': 0.10,
            'Retired': 0.05,
            'Unemployed': 0.05
        }
        
        self.education_probs = {
            'High School': 0.30,
            'Bachelors': 0.30,
            'Masters': 0.20,
            'Doctorate': 0.10,
            'Other': 0.10
        }
        
        self.marital_probs = {
            'Single': 0.35,
            'Married': 0.45,
            'Divorced': 0.15,
            'Widowed': 0.05
        }
        
        self.housing_probs = {
            'Rent': 0.40,
            'Mortgage': 0.30,
            'Own': 0.30
        }
        
        # Base income levels by education (annual)
        self.base_income_by_education = {
            'High School': 30000,
            'Bachelors': 48000,
            'Masters': 66000,
            'Doctorate': 84000,
            'Other': 36000
        }
        
        # Income multipliers by employment status
        self.income_mult_by_employment = {
            'Full-time': 1.0,
            'Self-employed': 1.2,
            'Part-time': 0.5,
            'Retired': 0.6,
            'Unemployed': 0.1
        }
    
    def _generate_age(self, n_samples: int) -> np.ndarray:
        """
        Generate age distribution (18-80 years, normal distribution around 42).
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        np.ndarray
            Array of ages
        """
        ages = np.random.normal(42, 12, n_samples)
        # Clip to reasonable range and round to integers
        ages = np.clip(ages, 18, 80).round().astype(int)
        return ages
    
    def _generate_employment_status(self, n_samples: int, ages: np.ndarray) -> np.ndarray:
        """
        Generate employment status with age-based adjustments.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        ages : np.ndarray
            Array of ages for correlation
            
        Returns:
        --------
        np.ndarray
            Array of employment statuses
        """
        employment_statuses = []
        
        for age in ages:
            # Adjust probabilities based on age
            probs = self.employment_probs.copy()
            
            # Increase retirement probability for older ages
            if age >= 65:
                probs['Retired'] = min(0.60, probs['Retired'] * 8)
                probs['Full-time'] = max(0.20, probs['Full-time'] * 0.5)
                probs['Self-employed'] = max(0.10, probs['Self-employed'] * 0.7)
            elif age >= 60:
                probs['Retired'] = min(0.25, probs['Retired'] * 4)
                probs['Full-time'] = max(0.45, probs['Full-time'] * 0.8)
            
            # Reduce unemployment for very young ages
            if age < 22:
                probs['Unemployed'] = max(0.02, probs['Unemployed'] * 0.5)
                probs['Part-time'] = min(0.25, probs['Part-time'] * 2)
            
            # Normalize probabilities
            total = sum(probs.values())
            probs = {k: v/total for k, v in probs.items()}
            
            # Sample from distribution
            rand_val = np.random.uniform()
            cumsum = 0
            for status, prob in probs.items():
                cumsum += prob
                if rand_val <= cumsum:
                    employment_statuses.append(status)
                    break
        
        return np.array(employment_statuses)
    
    def _generate_education(self, n_samples: int, ages: np.ndarray) -> np.ndarray:
        """
        Generate education level with age-based adjustments.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        ages : np.ndarray
            Array of ages for correlation
            
        Returns:
        --------
        np.ndarray
            Array of education levels
        """
        education_levels = []
        
        for age in ages:
            # Adjust probabilities based on age (older generations had less access to higher education)
            probs = self.education_probs.copy()
            
            if age >= 55:
                probs['High School'] = min(0.50, probs['High School'] * 1.4)
                probs['Bachelors'] = max(0.20, probs['Bachelors'] * 0.7)
                probs['Masters'] = max(0.10, probs['Masters'] * 0.6)
                probs['Doctorate'] = max(0.05, probs['Doctorate'] * 0.5)
            elif age <= 25:
                # Younger people more likely to have recent degrees
                probs['Bachelors'] = min(0.40, probs['Bachelors'] * 1.2)
                probs['Masters'] = min(0.25, probs['Masters'] * 1.1)
            
            # Normalize probabilities
            total = sum(probs.values())
            probs = {k: v/total for k, v in probs.items()}
            
            # Sample from distribution
            rand_val = np.random.uniform()
            cumsum = 0
            for level, prob in probs.items():
                cumsum += prob
                if rand_val <= cumsum:
                    education_levels.append(level)
                    break
        
        return np.array(education_levels)
    
    def _generate_marital_status(self, n_samples: int, ages: np.ndarray) -> np.ndarray:
        """
        Generate marital status with age-based adjustments.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        ages : np.ndarray
            Array of ages for correlation
            
        Returns:
        --------
        np.ndarray
            Array of marital statuses
        """
        marital_statuses = []
        
        for age in ages:
            # Adjust probabilities based on age
            probs = self.marital_probs.copy()
            
            if age < 25:
                probs['Single'] = min(0.70, probs['Single'] * 1.8)
                probs['Married'] = max(0.20, probs['Married'] * 0.5)
                probs['Divorced'] = max(0.05, probs['Divorced'] * 0.3)
            elif age >= 25 and age < 35:
                probs['Single'] = max(0.25, probs['Single'] * 0.8)
                probs['Married'] = min(0.60, probs['Married'] * 1.2)
            elif age >= 60:
                probs['Widowed'] = min(0.15, probs['Widowed'] * 2.5)
                probs['Married'] = max(0.35, probs['Married'] * 0.8)
            
            # Normalize probabilities
            total = sum(probs.values())
            probs = {k: v/total for k, v in probs.items()}
            
            # Sample from distribution
            rand_val = np.random.uniform()
            cumsum = 0
            for status, prob in probs.items():
                cumsum += prob
                if rand_val <= cumsum:
                    marital_statuses.append(status)
                    break
        
        return np.array(marital_statuses)
    
    def _generate_housing_status(self, n_samples: int, ages: np.ndarray, 
                                incomes: np.ndarray) -> np.ndarray:
        """
        Generate housing status with age and income correlations.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        ages : np.ndarray
            Array of ages for correlation
        incomes : np.ndarray
            Array of incomes for correlation
            
        Returns:
        --------
        np.ndarray
            Array of housing statuses
        """
        housing_statuses = []
        
        for age, income in zip(ages, incomes):
            # Adjust probabilities based on age and income
            probs = self.housing_probs.copy()
            
            # Young people more likely to rent
            if age < 28:
                probs['Rent'] = min(0.70, probs['Rent'] * 1.5)
                probs['Own'] = max(0.10, probs['Own'] * 0.4)
                probs['Mortgage'] = max(0.15, probs['Mortgage'] * 0.6)
            # Older, higher income people more likely to own
            elif age > 50 and income > 60000:
                probs['Own'] = min(0.60, probs['Own'] * 1.8)
                probs['Rent'] = max(0.15, probs['Rent'] * 0.4)
            # Middle-aged people more likely to have mortgages
            elif age >= 28 and age <= 50:
                probs['Mortgage'] = min(0.50, probs['Mortgage'] * 1.4)
            
            # Income adjustments
            if income < 30000:
                probs['Rent'] = min(0.70, probs['Rent'] * 1.4)
                probs['Own'] = max(0.10, probs['Own'] * 0.4)
            elif income > 80000:
                probs['Own'] = min(0.55, probs['Own'] * 1.5)
                probs['Mortgage'] = min(0.35, probs['Mortgage'] * 1.2)
                probs['Rent'] = max(0.15, probs['Rent'] * 0.5)
            
            # Normalize probabilities
            total = sum(probs.values())
            probs = {k: v/total for k, v in probs.items()}
            
            # Sample from distribution
            rand_val = np.random.uniform()
            cumsum = 0
            for status, prob in probs.items():
                cumsum += prob
                if rand_val <= cumsum:
                    housing_statuses.append(status)
                    break
        
        return np.array(housing_statuses)
    
    def _generate_annual_income(self, n_samples: int, ages: np.ndarray, 
                               education_levels: np.ndarray, 
                               employment_statuses: np.ndarray) -> np.ndarray:
        """
        Generate annual income with correlations to age, education, and employment.
        
        Uses log-normal distribution with base amounts adjusted by education and
        employment status, plus experience-based adjustments.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        ages : np.ndarray
            Array of ages for correlation
        education_levels : np.ndarray
            Array of education levels for correlation  
        employment_statuses : np.ndarray
            Array of employment statuses for correlation
            
        Returns:
        --------
        np.ndarray
            Array of annual incomes
        """
        incomes = []
        
        for age, education, employment in zip(ages, education_levels, employment_statuses):
            # Get base income for education level
            base_income = self.base_income_by_education.get(education, 36000)
            
            # Apply employment status multiplier
            employment_mult = self.income_mult_by_employment.get(employment, 0.5)
            
            # Add age/experience bonus (increases with age up to retirement)
            experience_years = max(0, age - 22)  # Assume work starts at 22
            if employment == 'Retired':
                experience_mult = 1.0  # Fixed income
            else:
                # Experience bonus peaks around age 50-55, then declines slightly
                experience_mult = 1.0 + min(0.6, experience_years * 0.02) 
                if age > 55:
                    experience_mult *= 0.95  # Slight decline after 55
            
            # Calculate expected income
            expected_income = base_income * employment_mult * experience_mult
            
            # Add log-normal noise (coefficient of variation around 0.3)
            log_sigma = 0.3
            income_noise = np.random.lognormal(0, log_sigma)
            
            # Final income calculation
            annual_income = expected_income * income_noise
            
            # Apply minimum wage constraints
            if employment != 'Unemployed':
                annual_income = max(annual_income, 15000)  # Minimum wage floor
            else:
                annual_income = max(annual_income, 0)
            
            # Round to nearest $100
            annual_income = round(annual_income / 100) * 100
            
            incomes.append(annual_income)
        
        return np.array(incomes, dtype=float)
    
    def generate_demographics(self, n_samples: int, 
                            include_missing: bool = False,
                            missing_rate: float = 0.02) -> pd.DataFrame:
        """
        Generate complete demographic dataset with realistic correlations.
        
        Parameters:
        -----------
        n_samples : int
            Number of demographic records to generate
        include_missing : bool, optional
            Whether to include missing values (default: False)
        missing_rate : float, optional
            Rate of missing values per variable if include_missing=True (default: 0.02)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing generated demographic data with columns:
            - age: int (18-80)
            - employment_status: category
            - education: category  
            - marital_status: category
            - housing_status: category
            - annual_income: float
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        
        # Reset random seed for reproducibility
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        # Generate core demographics in order to maintain correlations
        ages = self._generate_age(n_samples)
        employment_statuses = self._generate_employment_status(n_samples, ages)
        education_levels = self._generate_education(n_samples, ages)
        marital_statuses = self._generate_marital_status(n_samples, ages)
        
        # Generate income (dependent on age, education, employment)
        annual_incomes = self._generate_annual_income(n_samples, ages, 
                                                     education_levels, 
                                                     employment_statuses)
        
        # Generate housing status (dependent on age and income)
        housing_statuses = self._generate_housing_status(n_samples, ages, annual_incomes)
        
        # Create DataFrame
        df = pd.DataFrame({
            'age': ages,
            'employment_status': employment_statuses,
            'education': education_levels,
            'marital_status': marital_statuses,
            'housing_status': housing_statuses,
            'annual_income': annual_incomes
        })
        
        # Apply data type enforcement
        df = self._enforce_data_types(df)
        
        # Add missing values if requested
        if include_missing and missing_rate > 0:
            df = self._add_missing_values(df, missing_rate)
        
        # Validate generated data
        self._validate_data(df)
        
        return df
    
    def _enforce_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce proper data types for all columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with enforced data types
        """
        # Convert age to int
        df['age'] = df['age'].astype('int32')
        
        # Convert annual income to float
        df['annual_income'] = df['annual_income'].astype('float64')
        
        # Convert categorical variables to pandas categories
        categorical_cols = ['employment_status', 'education', 'marital_status', 'housing_status']
        for col in categorical_cols:
            df[col] = df[col].astype('category')
        
        return df
    
    def _add_missing_values(self, df: pd.DataFrame, missing_rate: float) -> pd.DataFrame:
        """
        Add missing values to simulate real-world data quality issues.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        missing_rate : float
            Rate of missing values per column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with missing values added
        """
        df_with_missing = df.copy()
        n_samples = len(df)
        
        # Define which columns can have missing values (age typically always available)
        missable_cols = ['employment_status', 'education', 'marital_status', 
                        'housing_status', 'annual_income']
        
        for col in missable_cols:
            # Determine number of missing values
            n_missing = int(n_samples * missing_rate)
            if n_missing > 0:
                # Randomly select indices for missing values
                missing_indices = np.random.choice(n_samples, size=n_missing, replace=False)
                df_with_missing.loc[missing_indices, col] = np.nan
        
        return df_with_missing
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate generated data meets business rules and constraints.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Generated demographic data
            
        Raises:
        -------
        ValueError
            If data validation fails
        """
        n_samples = len(df)
        
        # Age validation
        if not df['age'].between(18, 80).all():
            raise ValueError("Age values outside valid range (18-80)")
        
        # Income validation
        if df['annual_income'].min() < 0:
            raise ValueError("Negative income values detected")
        
        if df['annual_income'].max() > 1000000:  # $1M cap
            warnings.warn("Very high income values detected (>$1M)")
        
        # Check categorical values are in expected sets
        expected_employment = set(self.employment_probs.keys())
        actual_employment = set(df['employment_status'].dropna().unique())
        if not actual_employment.issubset(expected_employment):
            raise ValueError(f"Unexpected employment status values: {actual_employment - expected_employment}")
        
        expected_education = set(self.education_probs.keys())
        actual_education = set(df['education'].dropna().unique())
        if not actual_education.issubset(expected_education):
            raise ValueError(f"Unexpected education values: {actual_education - expected_education}")
        
        expected_marital = set(self.marital_probs.keys())
        actual_marital = set(df['marital_status'].dropna().unique())
        if not actual_marital.issubset(expected_marital):
            raise ValueError(f"Unexpected marital status values: {actual_marital - expected_marital}")
        
        expected_housing = set(self.housing_probs.keys())
        actual_housing = set(df['housing_status'].dropna().unique())
        if not actual_housing.issubset(expected_housing):
            raise ValueError(f"Unexpected housing status values: {actual_housing - expected_housing}")
        
        print(f"✓ Successfully validated {n_samples} demographic records")
    
    def get_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for demographic data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Demographic data
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing summary statistics
        """
        stats = {}
        
        # Numeric variables
        numeric_cols = ['age', 'annual_income']
        for col in numeric_cols:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median(),
                'missing_pct': (df[col].isna().sum() / len(df)) * 100
            }
        
        # Categorical variables
        categorical_cols = ['employment_status', 'education', 'marital_status', 'housing_status']
        for col in categorical_cols:
            value_counts = df[col].value_counts(normalize=True, dropna=False)
            stats[col] = {
                'mode': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'unique_count': df[col].nunique(),
                'missing_pct': (df[col].isna().sum() / len(df)) * 100,
                'distribution': value_counts.to_dict()
            }
        
        # Correlations
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            stats['correlations'] = numeric_df.corr().to_dict()
        
        return stats
    
    def validate_correlations(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Validate that expected correlations exist in generated data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Generated demographic data
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of correlation coefficients
        """
        correlations = {}
        
        # Age vs income correlation (should be positive)
        correlations['age_income'] = df['age'].corr(df['annual_income'])
        
        # Create numeric encoding for education to check correlation
        education_order = {'High School': 1, 'Other': 2, 'Bachelors': 3, 
                          'Masters': 4, 'Doctorate': 5}
        df_temp = df.copy()
        df_temp['education_numeric'] = df_temp['education'].map(education_order)
        correlations['education_income'] = df_temp['education_numeric'].corr(df_temp['annual_income'])
        
        return correlations


# Utility functions
def generate_sample_demographics(n_samples: int = 1000, 
                               random_seed: int = 12345) -> pd.DataFrame:
    """
    Convenience function to generate sample demographic data.
    
    Parameters:
    -----------
    n_samples : int
        Number of records to generate
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Generated demographic data
    """
    generator = DemographicsGenerator(random_seed=random_seed)
    return generator.generate_demographics(n_samples)


if __name__ == "__main__":
    # Example usage
    print("Demographics Generator - Example Usage")
    print("=" * 50)
    
    # Generate sample data
    generator = DemographicsGenerator(random_seed=12345)
    sample_data = generator.generate_demographics(n_samples=1000)
    
    # Display basic info
    print(f"\nGenerated {len(sample_data)} demographic records")
    print(f"Columns: {list(sample_data.columns)}")
    print(f"Data types:")
    print(sample_data.dtypes)
    
    # Show first few records
    print(f"\nFirst 10 records:")
    print(sample_data.head(10))
    
    # Show summary statistics
    print(f"\nSummary Statistics:")
    stats = generator.get_summary_statistics(sample_data)
    
    print(f"\nAge: mean={stats['age']['mean']:.1f}, std={stats['age']['std']:.1f}")
    print(f"Income: mean=${stats['annual_income']['mean']:,.0f}, std=${stats['annual_income']['std']:,.0f}")
    
    print(f"\nEmployment distribution:")
    for status, pct in stats['employment_status']['distribution'].items():
        print(f"  {status}: {pct:.1%}")
    
    # Validate correlations
    print(f"\nCorrelation Analysis:")
    correlations = generator.validate_correlations(sample_data)
    print(f"Age-Income correlation: {correlations['age_income']:.3f}")
    print(f"Education-Income correlation: {correlations['education_income']:.3f}")
