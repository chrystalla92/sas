"""
Setup configuration for Bank Credit Risk Scoring Model
Python Migration from SAS Implementation

This setup file configures the package installation and dependencies
for the credit risk scoring pipeline.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        install_requires = [
            line.strip() 
            for line in f.readlines() 
            if line.strip() and not line.startswith('#')
        ]
else:
    install_requires = [
        'pandas>=1.5.0',
        'numpy>=1.24.0', 
        'scikit-learn>=1.3.0',
        'matplotlib>=3.6.0',
        'seaborn>=0.12.0',
        'pyyaml>=6.0',
        'joblib>=1.3.0'
    ]

setup(
    name="bank-credit-risk-scoring",
    version="2.0.0",
    description="Python implementation of bank credit risk scoring model migrated from SAS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Risk Analytics Team",
    author_email="risk.analytics@bank.com",
    url="https://github.com/bank/credit-risk-scoring",
    
    # Package configuration
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'config': ['*.yaml', '*.yml'],
        'data': ['*.csv', '*.parquet'],
        'models': ['*.joblib', '*.pkl'],
    },
    
    # Dependencies
    install_requires=install_requires,
    
    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.5.0',
        ],
        'notebook': [
            'jupyter>=1.0.0',
            'notebook>=6.5.0',
            'ipykernel>=6.25.0',
        ],
        'profiling': [
            'great_expectations>=0.17.0',
            'pandas_profiling>=3.6.0',
            'memory_profiler>=0.60.0',
        ],
        'performance': [
            'numba>=0.57.0',
            'cython>=0.29.0',
            'dask>=2023.8.0',
        ]
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence", 
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    # Keywords
    keywords="credit risk scoring machine learning banking finance sas migration",
    
    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            'credit-risk-generate=scripts.01_generate_credit_data:main',
            'credit-risk-explore=scripts.02_data_exploration:main',
            'credit-risk-features=scripts.03_feature_engineering:main',
            'credit-risk-train=scripts.04_train_credit_model:main',
            'credit-risk-validate=scripts.05_model_validation:main',
            'credit-risk-score=scripts.06_score_new_customers:main',
            'credit-risk-pipeline=scripts.run_pipeline:main',
        ],
    },
    
    # Project URLs
    project_urls={
        "Documentation": "https://github.com/bank/credit-risk-scoring/wiki",
        "Source": "https://github.com/bank/credit-risk-scoring",
        "Tracker": "https://github.com/bank/credit-risk-scoring/issues",
    },
)
