"""
Quick validation script to verify credit history generator meets requirements.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from credit_history_generator import CreditHistoryGenerator, create_sample_demographics

def main():
    print("=== Credit History Generator Validation ===\n")
    
    # Create test data
    demographics = create_sample_demographics(2000, random_state=42)
    generator = CreditHistoryGenerator(random_state=42)
    credit_profiles = generator.generate_complete_credit_profile(demographics)
    
    print("✓ Successfully generated 2000 customer credit profiles")
    
    # Test 1: Credit variables show realistic correlations with demographics
    print("\n1. Testing correlations with demographics...")
    
    income = credit_profiles['monthly_income']
    payment_score = credit_profiles['payment_history_score']
    age = credit_profiles['age']
    credit_history = credit_profiles['credit_history_years']
    utilization = credit_profiles['credit_utilization']
    
    # Income-payment score correlation (should be positive)
    corr_income_score, p_val = pearsonr(income, payment_score)
    print(f"   Income-Payment Score correlation: {corr_income_score:.3f} (p={p_val:.3f})")
    assert corr_income_score > 0.1 and p_val < 0.05, "Income-payment score correlation too weak"
    
    # Age-credit history correlation (should be positive)  
    corr_age_history, p_val = pearsonr(age, credit_history)
    print(f"   Age-Credit History correlation: {corr_age_history:.3f} (p={p_val:.3f})")
    assert corr_age_history > 0.2 and p_val < 0.05, "Age-credit history correlation too weak"
    
    print("   ✓ Demographics correlations are realistic")
    
    # Test 2: Credit utilization ratios within business-acceptable ranges
    print("\n2. Testing credit utilization ranges...")
    
    util_min = utilization.min()
    util_max = utilization.max()
    util_mean = utilization.mean()
    print(f"   Utilization range: {util_min:.3f} to {util_max:.3f}, mean: {util_mean:.3f}")
    
    assert util_min >= 0.0, "Utilization cannot be negative"
    assert util_max <= 0.95, "Utilization exceeds 95% business limit"
    assert util_mean < 0.5, "Average utilization too high"
    
    # Check distribution
    util_high_count = (utilization > 0.8).sum()
    print(f"   Customers with >80% utilization: {util_high_count} ({util_high_count/len(utilization):.1%})")
    assert util_high_count < len(utilization) * 0.15, "Too many customers with high utilization"
    
    print("   ✓ Credit utilization ratios are within acceptable ranges")
    
    # Test 3: Debt-to-income ratios align with lending industry standards
    print("\n3. Testing debt-to-income ratios...")
    
    dti = credit_profiles['debt_to_income_ratio']
    dti_min = dti.min()
    dti_max = dti.max()
    dti_mean = dti.mean()
    print(f"   DTI range: {dti_min:.3f} to {dti_max:.3f}, mean: {dti_mean:.3f}")
    
    assert dti_min >= 0.0, "DTI cannot be negative"
    assert dti_max <= 0.95, "DTI exceeds 95% business limit"
    
    # Industry standards
    dti_acceptable = (dti <= 0.43).sum()  # 43% is common lending limit
    dti_high = (dti > 0.6).sum()
    print(f"   Customers with DTI ≤ 43%: {dti_acceptable} ({dti_acceptable/len(dti):.1%})")
    print(f"   Customers with DTI > 60%: {dti_high} ({dti_high/len(dti):.1%})")
    
    assert dti_acceptable/len(dti) > 0.6, "Too few customers meet DTI lending standards"
    assert dti_high/len(dti) < 0.3, "Too many customers have very high DTI"
    
    print("   ✓ DTI ratios align with lending industry standards")
    
    # Test 4: Default probability correlation with risk factors
    print("\n4. Testing default probability correlations...")
    
    default_prob = credit_profiles['default_probability']
    composite_risk = credit_profiles['composite_risk_score']
    
    # Should negatively correlate with payment score
    corr_def_score, p_val = pearsonr(default_prob, payment_score)
    print(f"   Default Prob-Payment Score correlation: {corr_def_score:.3f} (p={p_val:.3f})")
    assert corr_def_score < -0.3 and p_val < 0.05, "Default prob-payment score correlation too weak"
    
    # Should positively correlate with DTI
    corr_def_dti, p_val = pearsonr(default_prob, dti)
    print(f"   Default Prob-DTI correlation: {corr_def_dti:.3f} (p={p_val:.3f})")
    assert corr_def_dti > 0.2 and p_val < 0.05, "Default prob-DTI correlation too weak"
    
    # Should strongly correlate with composite risk
    corr_def_risk, p_val = pearsonr(default_prob, composite_risk)
    print(f"   Default Prob-Composite Risk correlation: {corr_def_risk:.3f} (p={p_val:.3f})")
    assert corr_def_risk > 0.5 and p_val < 0.05, "Default prob-risk correlation too weak"
    
    print("   ✓ Default probability shows significant correlation with risk factors")
    
    # Test 5: Range and consistency validation
    print("\n5. Testing range and consistency validation...")
    
    # Payment history scores
    scores = credit_profiles['payment_history_score']
    assert scores.min() >= 300 and scores.max() <= 850, "Payment scores outside 300-850 range"
    print(f"   Payment scores: {scores.min()} to {scores.max()} ✓")
    
    # Number of accounts
    accounts = credit_profiles['num_credit_accounts']
    assert accounts.min() >= 0 and accounts.max() <= 20, "Account counts outside valid range"
    print(f"   Credit accounts: {accounts.min()} to {accounts.max()} ✓")
    
    # Recent inquiries
    inquiries = credit_profiles['recent_inquiries']
    assert inquiries.min() >= 0 and inquiries.max() <= 10, "Inquiries outside 0-10 range"
    print(f"   Recent inquiries: {inquiries.min()} to {inquiries.max()} ✓")
    
    # Credit history consistency with age
    credit_years = credit_profiles['credit_history_years']
    max_possible = credit_profiles['age'] - 18
    violations = (credit_years > max_possible).sum()
    assert violations == 0, f"{violations} customers have impossible credit history length"
    print(f"   Credit history length consistency: ✓")
    
    # Thin file customers consistency
    no_history = credit_profiles['credit_history_years'] == 0
    thin_accounts = credit_profiles[no_history]['num_credit_accounts']
    thin_util = credit_profiles[no_history]['credit_utilization']
    assert (thin_accounts == 0).all(), "Thin file customers should have 0 accounts"
    assert (thin_util == 0).all(), "Thin file customers should have 0% utilization"
    print(f"   Thin file customer consistency: ✓")
    
    print("   ✓ All credit history variables pass range and consistency validation")
    
    # Test 6: Risk segmentation capability
    print("\n6. Testing risk segmentation and scoring...")
    
    # Create risk segments
    low_risk = default_prob < 0.1
    medium_risk = (default_prob >= 0.1) & (default_prob < 0.3)
    high_risk = default_prob >= 0.3
    
    low_count = low_risk.sum()
    medium_count = medium_risk.sum()
    high_count = high_risk.sum()
    
    print(f"   Low risk customers: {low_count} ({low_count/len(credit_profiles):.1%})")
    print(f"   Medium risk customers: {medium_count} ({medium_count/len(credit_profiles):.1%})")
    print(f"   High risk customers: {high_count} ({high_count/len(credit_profiles):.1%})")
    
    assert low_count > 100, "Not enough low-risk customers for segmentation"
    assert medium_count > 50, "Not enough medium-risk customers for segmentation"
    assert high_count > 10, "Not enough high-risk customers for segmentation"
    
    # Check that risk scores are ordered correctly
    low_risk_scores = composite_risk[low_risk].mean()
    medium_risk_scores = composite_risk[medium_risk].mean()
    high_risk_scores = composite_risk[high_risk].mean()
    
    print(f"   Average risk scores - Low: {low_risk_scores:.1f}, Medium: {medium_risk_scores:.1f}, High: {high_risk_scores:.1f}")
    
    assert low_risk_scores < medium_risk_scores < high_risk_scores, "Risk scores not properly ordered across segments"
    
    print("   ✓ Generated data supports meaningful risk segmentation and scoring")
    
    # Summary statistics
    print("\n=== Summary Statistics ===")
    credit_cols = ['payment_history_score', 'credit_utilization', 'credit_history_years',
                  'num_credit_accounts', 'debt_to_income_ratio', 'default_probability']
    
    summary = credit_profiles[credit_cols].describe()
    print(summary.round(3))
    
    print("\n=== VALIDATION COMPLETE ===")
    print("✅ All requirements successfully validated!")
    print(f"✅ Credit History Generator ready for production use")
    print(f"✅ Generated {len(credit_profiles)} customer profiles with realistic correlations")
    print(f"✅ All business rules and constraints properly enforced")
    print(f"✅ Default probability models show statistically significant correlations")
    

if __name__ == "__main__":
    main()
