"""
Credit History Generator

This module provides functionality to generate realistic credit history and financial
metrics for synthetic customer data used in credit risk modeling.

The generator implements realistic correlations and business rules to create:
- Payment history scores (300-850 scale)
- Credit utilization ratios (0-1)
- Credit history length (months/years)
- Number of credit accounts
- Recent credit inquiries
- Debt-to-income ratios
- Default probability factors

Key Features:
- Configurable risk parameters
- Age and income-based correlations
- Business rule validation
- Missing value strategies for new credit customers
- Realistic distribution patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


@dataclass
class RiskConfig:
    """Configuration class for risk parameters and distribution settings."""
    
    # Payment history score parameters (300-850 scale)
    payment_history_base: float = 650.0
    payment_history_std: float = 80.0
    payment_history_min: float = 300.0
    payment_history_max: float = 850.0
    
    # Credit utilization parameters (0-1)
    utilization_beta_alpha: float = 2.0
    utilization_beta_beta: float = 5.0
    utilization_max: float = 0.95  # Business rule: max 95%
    
    # Credit history length parameters
    credit_history_gamma_shape: float = 3.0
    credit_history_gamma_scale: float = 2.0
    
    # Number of accounts parameters
    accounts_poisson_lambda: float = 3.0
    accounts_min: int = 0
    accounts_max: int = 20
    
    # Credit inquiries parameters
    inquiries_poisson_lambda: float = 1.5
    inquiries_max: int = 10
    
    # Debt-to-income parameters
    dti_base: float = 0.25
    dti_std: float = 0.15
    dti_max: float = 0.95  # Business rule: max 95%
    
    # Risk correlation factors
    income_correlation_strength: float = 0.3
    age_correlation_strength: float = 0.25
    employment_correlation_strength: float = 0.2


class CreditHistoryGenerator:
    """
    Generates realistic credit history and financial metrics for synthetic customer data.
    
    This class implements business logic and correlations to create credit variables
    that reflect real-world relationships between demographics, financial metrics,
    and credit behavior.
    """
    
    def __init__(self, risk_config: Optional[RiskConfig] = None, random_state: Optional[int] = None):
        """
        Initialize the Credit History Generator.
        
        Args:
            risk_config: Configuration object with risk parameters
            random_state: Random seed for reproducible results
        """
        self.risk_config = risk_config or RiskConfig()
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def generate_payment_history_score(self, 
                                     n_customers: int,
                                     income: Optional[np.ndarray] = None,
                                     age: Optional[np.ndarray] = None,
                                     employment_years: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate payment history scores (300-850 scale).
        
        Higher income, age, and employment stability correlate with better scores.
        
        Args:
            n_customers: Number of customers to generate scores for
            income: Monthly income array
            age: Age array
            employment_years: Employment years array
            
        Returns:
            Array of payment history scores (300-850)
        """
        base_scores = np.random.normal(
            self.risk_config.payment_history_base,
            self.risk_config.payment_history_std,
            n_customers
        )
        
        # Apply correlations with demographics
        if income is not None:
            # Higher income -> better scores
            income_normalized = (income - np.mean(income)) / np.std(income)
            income_adjustment = income_normalized * 30 * self.risk_config.income_correlation_strength
            base_scores += income_adjustment
            
        if age is not None:
            # Older customers tend to have better scores (up to a point)
            age_normalized = (age - np.mean(age)) / np.std(age)
            age_adjustment = age_normalized * 25 * self.risk_config.age_correlation_strength
            base_scores += age_adjustment
            
        if employment_years is not None:
            # Longer employment -> better scores
            emp_normalized = (employment_years - np.mean(employment_years)) / np.std(employment_years)
            emp_adjustment = emp_normalized * 20 * self.risk_config.employment_correlation_strength
            base_scores += emp_adjustment
        
        # Apply bounds
        scores = np.clip(base_scores, 
                        self.risk_config.payment_history_min,
                        self.risk_config.payment_history_max)
        
        return np.round(scores).astype(int)
    
    def generate_credit_utilization(self,
                                  n_customers: int,
                                  income: Optional[np.ndarray] = None,
                                  existing_debt: Optional[np.ndarray] = None,
                                  payment_history_score: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate credit utilization ratios (0-1).
        
        Lower income and worse credit scores correlate with higher utilization.
        
        Args:
            n_customers: Number of customers
            income: Monthly income array
            existing_debt: Existing monthly debt obligations
            payment_history_score: Payment history scores
            
        Returns:
            Array of credit utilization ratios (0-1)
        """
        # Base utilization from beta distribution
        base_utilization = np.random.beta(
            self.risk_config.utilization_beta_alpha,
            self.risk_config.utilization_beta_beta,
            n_customers
        )
        
        # Apply correlations
        if income is not None:
            # Lower income -> higher utilization
            income_normalized = (income - np.mean(income)) / np.std(income)
            income_adjustment = -income_normalized * 0.15 * self.risk_config.income_correlation_strength
            base_utilization += income_adjustment
            
        if payment_history_score is not None:
            # Lower scores -> higher utilization
            score_normalized = (payment_history_score - 600) / 100
            score_adjustment = -score_normalized * 0.1
            base_utilization += score_adjustment
            
        if existing_debt is not None and income is not None:
            # Higher debt-to-income -> higher utilization
            dti = existing_debt / income
            dti_adjustment = dti * 0.2
            base_utilization += dti_adjustment
        
        # Apply business rule: max 95% utilization
        utilization = np.clip(base_utilization, 0.0, self.risk_config.utilization_max)
        
        return utilization
    
    def generate_credit_history_length(self,
                                     age: np.ndarray,
                                     min_age_for_credit: int = 18) -> np.ndarray:
        """
        Generate length of credit history in months.
        
        Based on age with some randomness for when people started building credit.
        
        Args:
            age: Customer ages
            min_age_for_credit: Minimum age to have credit history
            
        Returns:
            Array of credit history lengths in months
        """
        n_customers = len(age)
        
        # Maximum possible credit history based on age
        max_possible_months = (age - min_age_for_credit) * 12
        max_possible_months = np.maximum(max_possible_months, 0)
        
        # Generate actual history length using gamma distribution
        # Most people don't start building credit immediately at 18
        gamma_samples = np.random.gamma(
            self.risk_config.credit_history_gamma_shape,
            self.risk_config.credit_history_gamma_scale,
            n_customers
        )
        
        # Scale gamma samples to reasonable range
        history_months = gamma_samples * 12  # Convert to months
        
        # Can't exceed maximum possible based on age
        history_months = np.minimum(history_months, max_possible_months)
        
        # Some people have no credit history (especially young customers)
        no_credit_prob = np.where(age < 25, 0.15, 0.05)  # Higher for young customers
        no_credit_mask = np.random.random(n_customers) < no_credit_prob
        history_months[no_credit_mask] = 0
        
        return np.round(history_months).astype(int)
    
    def generate_number_of_accounts(self,
                                  n_customers: int,
                                  income: Optional[np.ndarray] = None,
                                  credit_history_months: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate number of credit accounts (0-20+).
        
        Correlated with income and credit history length.
        
        Args:
            n_customers: Number of customers
            income: Monthly income array
            credit_history_months: Length of credit history in months
            
        Returns:
            Array of number of credit accounts
        """
        # Base number from Poisson distribution
        base_accounts = np.random.poisson(self.risk_config.accounts_poisson_lambda, n_customers)
        
        # Apply correlations
        if income is not None:
            # Higher income -> more accounts
            income_normalized = (income - np.mean(income)) / np.std(income)
            income_adjustment = income_normalized * 2 * self.risk_config.income_correlation_strength
            base_accounts = base_accounts + income_adjustment.astype(int)
            
        if credit_history_months is not None:
            # Longer history -> more accounts (but with diminishing returns)
            history_years = credit_history_months / 12
            history_adjustment = np.minimum(history_years * 0.5, 3)  # Max 3 extra accounts
            base_accounts = base_accounts + history_adjustment.astype(int)
        
        # Apply bounds and handle no credit history
        accounts = np.clip(base_accounts, self.risk_config.accounts_min, self.risk_config.accounts_max)
        
        # Customers with no credit history have 0 accounts
        if credit_history_months is not None:
            accounts[credit_history_months == 0] = 0
            
        return accounts.astype(int)
    
    def generate_recent_inquiries(self,
                                n_customers: int,
                                credit_utilization: Optional[np.ndarray] = None,
                                payment_history_score: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate number of recent credit inquiries (0-10 in last 2 years).
        
        Higher utilization and lower scores correlate with more inquiries.
        
        Args:
            n_customers: Number of customers
            credit_utilization: Credit utilization ratios
            payment_history_score: Payment history scores
            
        Returns:
            Array of recent credit inquiries
        """
        # Base inquiries from Poisson distribution
        base_inquiries = np.random.poisson(self.risk_config.inquiries_poisson_lambda, n_customers)
        
        # Apply correlations
        if credit_utilization is not None:
            # Higher utilization -> more inquiries (seeking credit)
            util_adjustment = credit_utilization * 2
            base_inquiries = base_inquiries + util_adjustment.astype(int)
            
        if payment_history_score is not None:
            # Lower scores -> more inquiries (difficulty getting approved)
            score_normalized = (600 - payment_history_score) / 100  # Invert so lower scores = higher value
            score_adjustment = np.maximum(score_normalized * 1.5, 0)
            base_inquiries = base_inquiries + score_adjustment.astype(int)
        
        # Apply bounds
        inquiries = np.clip(base_inquiries, 0, self.risk_config.inquiries_max)
        
        return inquiries.astype(int)
    
    def generate_debt_to_income_ratio(self,
                                    n_customers: int,
                                    income: np.ndarray,
                                    age: Optional[np.ndarray] = None,
                                    employment_years: Optional[np.ndarray] = None,
                                    credit_utilization: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate debt-to-income ratios with realistic constraints.
        
        Younger customers and those with higher utilization tend to have higher DTI.
        
        Args:
            n_customers: Number of customers
            income: Monthly income array
            age: Customer ages
            employment_years: Employment years
            credit_utilization: Credit utilization ratios
            
        Returns:
            Array of debt-to-income ratios (0-1)
        """
        # Base DTI from normal distribution
        base_dti = np.random.normal(self.risk_config.dti_base, self.risk_config.dti_std, n_customers)
        
        # Apply correlations
        if age is not None:
            # Younger customers tend to have higher DTI
            age_normalized = (np.mean(age) - age) / np.std(age)  # Invert so younger = higher
            age_adjustment = age_normalized * 0.1 * self.risk_config.age_correlation_strength
            base_dti += age_adjustment
            
        if employment_years is not None:
            # Less employment history -> higher DTI
            emp_normalized = (np.mean(employment_years) - employment_years) / np.std(employment_years)
            emp_adjustment = emp_normalized * 0.05 * self.risk_config.employment_correlation_strength
            base_dti += emp_adjustment
            
        if credit_utilization is not None:
            # Higher credit utilization -> higher DTI
            util_adjustment = credit_utilization * 0.15
            base_dti += util_adjustment
        
        # Apply business rule: max 95% DTI (regulatory/business limit)
        dti = np.clip(base_dti, 0.0, self.risk_config.dti_max)
        
        return dti
    
    def calculate_composite_risk_score(self,
                                     payment_history_score: np.ndarray,
                                     credit_utilization: np.ndarray,
                                     debt_to_income_ratio: np.ndarray,
                                     credit_history_months: np.ndarray,
                                     recent_inquiries: np.ndarray) -> np.ndarray:
        """
        Calculate composite risk score that influences default probability.
        
        Combines multiple credit factors into a single risk metric.
        
        Args:
            payment_history_score: Payment history scores (300-850)
            credit_utilization: Credit utilization ratios (0-1)
            debt_to_income_ratio: Debt-to-income ratios (0-1)
            credit_history_months: Credit history length in months
            recent_inquiries: Number of recent inquiries
            
        Returns:
            Array of composite risk scores (0-100, lower is better)
        """
        n_customers = len(payment_history_score)
        
        # Normalize components to 0-1 scale
        # Payment history (lower score = higher risk)
        payment_risk = (850 - payment_history_score) / 550  # Scale to 0-1
        
        # Credit utilization (higher = higher risk)
        utilization_risk = credit_utilization
        
        # DTI (higher = higher risk)
        dti_risk = debt_to_income_ratio
        
        # Credit history (shorter = higher risk)
        history_years = credit_history_months / 12
        history_risk = np.maximum(0, (10 - history_years) / 10)  # Normalize to 0-1
        
        # Recent inquiries (more = higher risk)
        inquiry_risk = np.minimum(recent_inquiries / 5, 1)  # Cap at 1
        
        # Weighted composite score
        weights = {
            'payment': 0.35,
            'utilization': 0.25,
            'dti': 0.20,
            'history': 0.15,
            'inquiries': 0.05
        }
        
        composite_risk = (
            weights['payment'] * payment_risk +
            weights['utilization'] * utilization_risk +
            weights['dti'] * dti_risk +
            weights['history'] * history_risk +
            weights['inquiries'] * inquiry_risk
        )
        
        # Scale to 0-100
        return (composite_risk * 100).round(2)
    
    def apply_business_rules(self, credit_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply business rule validation and corrections.
        
        Args:
            credit_data: Dictionary of credit variables
            
        Returns:
            Dictionary of validated and corrected credit variables
        """
        validated_data = credit_data.copy()
        
        # Rule 1: Credit utilization cannot exceed 95%
        if 'credit_utilization' in validated_data:
            validated_data['credit_utilization'] = np.clip(
                validated_data['credit_utilization'], 0.0, 0.95
            )
        
        # Rule 2: DTI cannot exceed 95%
        if 'debt_to_income_ratio' in validated_data:
            validated_data['debt_to_income_ratio'] = np.clip(
                validated_data['debt_to_income_ratio'], 0.0, 0.95
            )
        
        # Rule 3: Payment history score must be 300-850
        if 'payment_history_score' in validated_data:
            validated_data['payment_history_score'] = np.clip(
                validated_data['payment_history_score'], 300, 850
            )
        
        # Rule 4: Number of accounts cannot be negative
        if 'num_credit_accounts' in validated_data:
            validated_data['num_credit_accounts'] = np.maximum(
                validated_data['num_credit_accounts'], 0
            )
        
        # Rule 5: Credit history cannot be negative
        if 'credit_history_months' in validated_data:
            validated_data['credit_history_months'] = np.maximum(
                validated_data['credit_history_months'], 0
            )
        
        # Rule 6: Recent inquiries cannot be negative or exceed 10
        if 'recent_inquiries' in validated_data:
            validated_data['recent_inquiries'] = np.clip(
                validated_data['recent_inquiries'], 0, 10
            )
        
        return validated_data
    
    def handle_missing_credit_history(self, 
                                    credit_data: Dict[str, np.ndarray],
                                    age: np.ndarray,
                                    income: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Implement strategy for customers with no credit history.
        
        Args:
            credit_data: Dictionary of credit variables
            age: Customer ages
            income: Customer incomes
            
        Returns:
            Dictionary with missing values handled
        """
        n_customers = len(age)
        handled_data = credit_data.copy()
        
        # Identify customers with no credit history
        no_history_mask = handled_data.get('credit_history_months', np.zeros(n_customers)) == 0
        
        if np.any(no_history_mask):
            # For customers with no history, set appropriate defaults
            if 'payment_history_score' in handled_data:
                # Use a lower default score for thin-file customers
                thin_file_scores = np.random.normal(580, 40, np.sum(no_history_mask))
                thin_file_scores = np.clip(thin_file_scores, 300, 680)  # Cap at 680 for thin files
                handled_data['payment_history_score'][no_history_mask] = thin_file_scores
            
            if 'num_credit_accounts' in handled_data:
                # No credit history = 0 accounts
                handled_data['num_credit_accounts'][no_history_mask] = 0
            
            if 'credit_utilization' in handled_data:
                # No utilization if no accounts
                handled_data['credit_utilization'][no_history_mask] = 0.0
            
            if 'recent_inquiries' in handled_data:
                # Thin-file customers might have some inquiries from shopping
                thin_file_inquiries = np.random.poisson(0.5, np.sum(no_history_mask))
                handled_data['recent_inquiries'][no_history_mask] = np.minimum(thin_file_inquiries, 3)
        
        return handled_data
    
    def generate_complete_credit_profile(self,
                                       demographics: pd.DataFrame,
                                       monthly_income_col: str = 'monthly_income',
                                       age_col: str = 'age',
                                       employment_years_col: str = 'employment_years') -> pd.DataFrame:
        """
        Generate complete credit profile for a dataset of customers.
        
        Args:
            demographics: DataFrame with customer demographics
            monthly_income_col: Column name for monthly income
            age_col: Column name for age
            employment_years_col: Column name for employment years
            
        Returns:
            DataFrame with complete credit profile
        """
        n_customers = len(demographics)
        
        # Extract demographic variables
        income = demographics[monthly_income_col].values
        age = demographics[age_col].values
        employment_years = demographics[employment_years_col].values
        
        # Generate credit variables
        payment_history_score = self.generate_payment_history_score(
            n_customers, income, age, employment_years
        )
        
        credit_history_months = self.generate_credit_history_length(age)
        
        num_credit_accounts = self.generate_number_of_accounts(
            n_customers, income, credit_history_months
        )
        
        # Calculate existing debt for utilization calculation
        existing_debt = income * np.random.beta(2, 8, n_customers) * 0.5
        
        credit_utilization = self.generate_credit_utilization(
            n_customers, income, existing_debt, payment_history_score
        )
        
        recent_inquiries = self.generate_recent_inquiries(
            n_customers, credit_utilization, payment_history_score
        )
        
        debt_to_income_ratio = self.generate_debt_to_income_ratio(
            n_customers, income, age, employment_years, credit_utilization
        )
        
        # Create credit data dictionary
        credit_data = {
            'payment_history_score': payment_history_score,
            'credit_utilization': credit_utilization,
            'credit_history_months': credit_history_months,
            'credit_history_years': credit_history_months / 12,
            'num_credit_accounts': num_credit_accounts,
            'recent_inquiries': recent_inquiries,
            'debt_to_income_ratio': debt_to_income_ratio
        }
        
        # Handle missing credit history cases
        credit_data = self.handle_missing_credit_history(credit_data, age, income)
        
        # Apply business rules validation
        credit_data = self.apply_business_rules(credit_data)
        
        # Calculate composite risk score
        credit_data['composite_risk_score'] = self.calculate_composite_risk_score(
            credit_data['payment_history_score'],
            credit_data['credit_utilization'],
            credit_data['debt_to_income_ratio'],
            credit_data['credit_history_months'],
            credit_data['recent_inquiries']
        )
        
        # Calculate default probability based on risk factors
        credit_data['default_probability'] = self._calculate_default_probability(
            credit_data['composite_risk_score'],
            credit_data['payment_history_score'],
            credit_data['debt_to_income_ratio']
        )
        
        # Combine with original demographics
        result_df = demographics.copy()
        for key, values in credit_data.items():
            result_df[key] = values
        
        return result_df
    
    def _calculate_default_probability(self,
                                     composite_risk_score: np.ndarray,
                                     payment_history_score: np.ndarray,
                                     debt_to_income_ratio: np.ndarray) -> np.ndarray:
        """
        Calculate default probability based on risk factors.
        
        Args:
            composite_risk_score: Composite risk scores (0-100)
            payment_history_score: Payment history scores (300-850)
            debt_to_income_ratio: Debt-to-income ratios (0-1)
            
        Returns:
            Array of default probabilities (0-1)
        """
        # Base probability
        base_prob = 0.05  # 5% base default rate
        
        # Risk score impact (0-100 scale, higher = more risky)
        risk_multiplier = 1 + (composite_risk_score / 100) * 4  # Up to 5x multiplier
        
        # Payment history impact
        score_adjustment = np.where(
            payment_history_score < 600, 
            (600 - payment_history_score) / 300 * 0.15,  # Up to 15% additional risk
            0
        )
        
        # DTI impact
        dti_adjustment = np.where(
            debt_to_income_ratio > 0.4,
            (debt_to_income_ratio - 0.4) / 0.6 * 0.20,  # Up to 20% additional risk
            0
        )
        
        # Calculate final probability
        default_prob = base_prob * risk_multiplier + score_adjustment + dti_adjustment
        
        # Cap at reasonable maximum
        return np.clip(default_prob, 0.01, 0.80)


def create_sample_demographics(n_customers: int = 1000, random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Create sample demographics data for testing the credit history generator.
    
    Args:
        n_customers: Number of customers to generate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with sample demographics
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Age (18-75, normal distribution centered at 42)
    age = np.clip(np.random.normal(42, 12, n_customers), 18, 75).round().astype(int)
    
    # Employment years (correlated with age)
    employment_years = np.zeros(n_customers)
    for i in range(n_customers):
        if age[i] < 25:
            employment_years[i] = max(0, np.random.uniform(0, 3))
        elif age[i] < 35:
            employment_years[i] = max(0, np.random.uniform(0, 10))
        else:
            employment_years[i] = max(0, np.random.uniform(0, min(20, age[i] - 20)))
    
    employment_years = employment_years.round().astype(int)
    
    # Monthly income (log-normal distribution)
    base_income = np.random.lognormal(np.log(4000), 0.5, n_customers)
    monthly_income = np.clip(base_income, 1500, 20000).round(-2)  # Round to nearest $100
    
    # Create DataFrame
    demographics = pd.DataFrame({
        'customer_id': [f'CUST_{i:06d}' for i in range(n_customers)],
        'age': age,
        'employment_years': employment_years,
        'monthly_income': monthly_income
    })
    
    return demographics


# Example usage
if __name__ == "__main__":
    # Create sample demographics
    demographics = create_sample_demographics(1000, random_state=42)
    
    # Initialize generator
    generator = CreditHistoryGenerator(random_state=42)
    
    # Generate complete credit profiles
    credit_profiles = generator.generate_complete_credit_profile(demographics)
    
    print("Sample Credit Profiles:")
    print(credit_profiles.head())
    
    print("\nCredit Variable Statistics:")
    credit_cols = ['payment_history_score', 'credit_utilization', 'credit_history_years',
                  'num_credit_accounts', 'debt_to_income_ratio', 'default_probability']
    print(credit_profiles[credit_cols].describe())
