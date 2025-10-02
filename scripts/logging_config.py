"""
Logging configuration module for credit risk model.

This module provides a reusable logging setup function that can be imported
and used by all scripts in the credit risk modeling pipeline.
"""

import logging
import os
from pathlib import Path


def setup_logging(log_file='credit_risk_model.log', log_level=logging.INFO):
    """
    Setup logging configuration with both file and console handlers.
    
    Parameters
    ----------
    log_file : str, optional
        Name of the log file (default: 'credit_risk_model.log')
        Log file will be created in the project root directory.
    log_level : int, optional
        Logging level (default: logging.INFO)
    
    Returns
    -------
    logging.Logger
        Configured logger instance
    
    Example
    -------
    >>> from logging_config import setup_logging
    >>> logger = setup_logging()
    >>> logger.info("Starting feature engineering")
    """
    # Get the root directory (parent of scripts/)
    root_dir = Path(__file__).parent.parent
    log_path = root_dir / log_file
    
    # Create logger
    logger = logging.getLogger('credit_risk_model')
    logger.setLevel(log_level)
    
    # Clear any existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name=None):
    """
    Get a logger instance with the specified name.
    
    This function should be called after setup_logging() has been called
    to ensure proper configuration.
    
    Parameters
    ----------
    name : str, optional
        Name for the logger (default: 'credit_risk_model')
    
    Returns
    -------
    logging.Logger
        Logger instance
    
    Example
    -------
    >>> from logging_config import setup_logging, get_logger
    >>> setup_logging()
    >>> logger = get_logger('feature_engineering')
    >>> logger.info("Feature engineering started")
    """
    if name:
        return logging.getLogger(f'credit_risk_model.{name}')
    return logging.getLogger('credit_risk_model')
