"""
Configuration Management Module for Bank Credit Risk Scoring Model
Python Migration from SAS Implementation

This module handles loading, validation, and management of configuration
settings for the credit risk scoring pipeline.

Author: Risk Analytics Team  
Date: 2025
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import warnings


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProjectConfig:
    """Project information configuration"""
    name: str
    version: str
    description: str
    author: str


@dataclass  
class PathConfig:
    """File path configuration"""
    root: str
    data: str
    raw_data: str
    processed_data: str
    models: str
    output: str
    scripts: str
    config: str
    logs: str


@dataclass
class DataGenerationConfig:
    """Data generation parameters"""
    num_records: int
    random_seed: int
    train_split: float
    validation_split: float
    default_rate_target: float
    age_range: list
    age_mean: int
    age_std: int
    base_income: dict
    employment_status_dist: dict
    education_dist: dict


@dataclass
class ModelTrainingConfig:
    """Model training parameters"""
    algorithm: str
    random_seed: int
    decision_tree: dict
    random_forest: dict
    logistic_regression: dict
    xgboost: dict


class ConfigManager:
    """
    Configuration manager for the credit risk scoring system.
    
    Handles loading, validation, and access to configuration parameters
    across the entire pipeline.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default path.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        self._create_directories()
        
        # Create structured config objects
        self.project = ProjectConfig(**self.config['project'])
        self.paths = PathConfig(**self.config['paths'])
        self.data_generation = DataGenerationConfig(**self.config['data_generation'])
        self.model_training = ModelTrainingConfig(**self.config['model_training'])
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary containing configuration parameters
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
                
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _validate_config(self) -> None:
        """
        Validate configuration parameters for correctness and completeness.
        
        Raises:
            ValueError: If configuration validation fails
        """
        required_sections = [
            'project', 'paths', 'data_generation', 'feature_engineering',
            'model_training', 'model_validation', 'production_scoring'
        ]
        
        # Check required sections exist
        missing_sections = [section for section in required_sections 
                          if section not in self.config]
        if missing_sections:
            raise ValueError(f"Missing required configuration sections: {missing_sections}")
        
        # Validate data generation parameters
        data_gen = self.config['data_generation']
        if not (0 < data_gen['train_split'] < 1):
            raise ValueError("train_split must be between 0 and 1")
        
        if not (0 < data_gen['validation_split'] < 1):
            raise ValueError("validation_split must be between 0 and 1")
            
        if abs(data_gen['train_split'] + data_gen['validation_split'] - 1.0) > 0.001:
            warnings.warn("Train and validation splits don't sum to 1.0")
        
        # Validate model parameters
        model_config = self.config['model_training']
        if model_config['algorithm'] not in ['decision_tree', 'random_forest', 'logistic_regression', 'xgboost']:
            raise ValueError(f"Unsupported algorithm: {model_config['algorithm']}")
        
        # Validate risk thresholds
        risk_thresholds = self.config['model_validation']['risk_thresholds']
        if not (0 <= risk_thresholds['low_risk'] <= risk_thresholds['medium_risk'] <= 1):
            raise ValueError("Risk thresholds must be in ascending order between 0 and 1")
        
        logger.info("Configuration validation completed successfully")
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        paths = self.config['paths']
        directories = [
            paths['data'], paths['raw_data'], paths['processed_data'],
            paths['models'], paths['output'], paths['logs']
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        logger.info("Directory structure created successfully")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key using dot notation (e.g., 'model_training.algorithm')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_model_config(self, algorithm: Optional[str] = None) -> Dict[str, Any]:
        """
        Get model configuration for specified algorithm.
        
        Args:
            algorithm: Model algorithm name. If None, uses default from config.
            
        Returns:
            Model configuration dictionary
        """
        if algorithm is None:
            algorithm = self.config['model_training']['algorithm']
            
        return self.config['model_training'].get(algorithm, {})
    
    def get_feature_engineering_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return self.config['feature_engineering']
    
    def get_risk_thresholds(self) -> Dict[str, float]:
        """Get risk categorization thresholds."""
        return self.config['model_validation']['risk_thresholds']
    
    def get_performance_thresholds(self) -> Dict[str, float]:
        """Get model performance thresholds."""
        return {
            'min_accuracy': self.config['model_validation']['min_accuracy'],
            'min_auc': self.config['model_validation']['min_auc'],
            'min_precision': self.config['model_validation']['min_precision'],
            'min_recall': self.config['model_validation']['min_recall']
        }
    
    def setup_logging(self) -> None:
        """Setup logging configuration based on config parameters."""
        log_config = self.config['logging']
        
        # Create logs directory
        log_path = Path(log_config['log_file']).parent
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_config['level'].upper()),
            format=log_config['format'],
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_config['log_file'])
            ]
        )
        
        logger.info("Logging configured successfully")
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        def _update_nested(d: Dict, u: Dict) -> Dict:
            """Recursively update nested dictionaries."""
            for key, value in u.items():
                if isinstance(value, dict):
                    d[key] = _update_nested(d.get(key, {}), value)
                else:
                    d[key] = value
            return d
        
        _update_nested(self.config, updates)
        logger.info("Configuration updated successfully")
    
    def save_config(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save configuration. If None, overwrites original.
        """
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
            
        logger.info(f"Configuration saved to {output_path}")


def load_config(config_path: Optional[Union[str, Path]] = None) -> ConfigManager:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path)


# Global configuration instance
config = None


def get_config() -> ConfigManager:
    """
    Get global configuration instance.
    
    Returns:
        ConfigManager instance
    """
    global config
    if config is None:
        config = ConfigManager()
    return config


if __name__ == "__main__":
    # Test configuration loading
    try:
        config_manager = ConfigManager()
        print("✓ Configuration loaded successfully")
        print(f"Project: {config_manager.project.name} v{config_manager.project.version}")
        print(f"Algorithm: {config_manager.get('model_training.algorithm')}")
        print(f"Training records: {config_manager.get('data_generation.num_records')}")
        
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
