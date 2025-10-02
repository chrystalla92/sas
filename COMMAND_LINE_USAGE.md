# Command Line Usage Guide

All Python scripts in this project now support command-line arguments for specifying input and output paths. This allows for greater flexibility when running the scripts with custom data locations.

## Overview

The following scripts have been updated:
- `scripts/feature_engineering.py` - Feature engineering pipeline
- `scripts/train.py` - Model training pipeline
- `scripts/predict.py` - Prediction pipeline

Each script maintains **backward compatibility** - if no arguments are provided, the scripts will use the default paths as before.

## Feature Engineering Script

### Usage

```bash
python scripts/feature_engineering.py [OPTIONS]
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--train-input` | Path to training CSV file | `credit_train.csv` |
| `--val-input` | Path to validation CSV file | `credit_validation.csv` |
| `--output-dir` | Output directory for processed data and models | Project root directory |

### Examples

```bash
# Use default paths
python scripts/feature_engineering.py

# Specify custom input files
python scripts/feature_engineering.py --train-input data/my_train.csv --val-input data/my_val.csv

# Specify custom output directory
python scripts/feature_engineering.py --output-dir /path/to/output

# Specify all paths
python scripts/feature_engineering.py --train-input data/train.csv --val-input data/val.csv --output-dir output_dir
```

### Output Files

The script creates the following structure in the output directory:
```
<output-dir>/
├── data/
│   ├── model_features_train.csv
│   └── model_features_validation.csv
└── models/
    ├── woe_mapping.pkl
    └── scaler.pkl
```

## Model Training Script

### Usage

```bash
python scripts/train.py [OPTIONS]
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--train-input` | Path to training features CSV file | `data/model_features_train.csv` |
| `--val-input` | Path to validation features CSV file | `data/model_features_validation.csv` |
| `--models-dir` | Directory for saving model artifacts | `models/` |
| `--output-dir` | Directory for saving output files | `output/` |

### Examples

```bash
# Use default paths
python scripts/train.py

# Specify custom input files
python scripts/train.py --train-input data/train_features.csv --val-input data/val_features.csv

# Specify custom directories
python scripts/train.py --models-dir /path/to/models --output-dir /path/to/output

# Specify all paths
python scripts/train.py --train-input my_train.csv --val-input my_val.csv --models-dir models --output-dir output
```

### Output Files

The script creates files in the specified directories:
```
<models-dir>/
├── logistic_model.pkl
└── model_coefficients.csv

<output-dir>/
├── risk_scores_train.csv
└── risk_scores_validation.csv
```

## Prediction Script

### Usage

```bash
python scripts/predict.py [OPTIONS]
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Path to new applications CSV file | `data/new_applications.csv` |
| `--output` | Path to output predictions CSV file | `output/new_predictions.csv` |
| `--models-dir` | Directory containing model artifacts | `models/` |

### Examples

```bash
# Use default paths
python scripts/predict.py

# Specify custom input file
python scripts/predict.py --input data/applications.csv

# Specify custom output file
python scripts/predict.py --output results/predictions.csv

# Specify models directory
python scripts/predict.py --models-dir /path/to/models

# Specify all paths
python scripts/predict.py --input my_apps.csv --output my_preds.csv --models-dir my_models
```

### Output Files

The script creates a single predictions file:
```
<output-file>
```

## Getting Help

All scripts support the `--help` flag to display usage information:

```bash
python scripts/feature_engineering.py --help
python scripts/train.py --help
python scripts/predict.py --help
```

## Common Workflows

### Workflow 1: Using Default Paths

```bash
# Step 1: Feature engineering
python scripts/feature_engineering.py

# Step 2: Train model
python scripts/train.py

# Step 3: Make predictions (requires new_applications.csv in data/)
python scripts/predict.py
```

### Workflow 2: Custom Input/Output Locations

```bash
# Step 1: Feature engineering with custom paths
python scripts/feature_engineering.py \
  --train-input /data/raw/train.csv \
  --val-input /data/raw/validation.csv \
  --output-dir /data/processed

# Step 2: Train model with custom paths
python scripts/train.py \
  --train-input /data/processed/data/model_features_train.csv \
  --val-input /data/processed/data/model_features_validation.csv \
  --models-dir /models \
  --output-dir /results

# Step 3: Make predictions with custom paths
python scripts/predict.py \
  --input /data/new/applications.csv \
  --output /results/predictions.csv \
  --models-dir /models
```

### Workflow 3: Processing Multiple Datasets

```bash
# Process dataset 1
python scripts/feature_engineering.py \
  --train-input dataset1/train.csv \
  --val-input dataset1/val.csv \
  --output-dir dataset1/processed

# Process dataset 2
python scripts/feature_engineering.py \
  --train-input dataset2/train.csv \
  --val-input dataset2/val.csv \
  --output-dir dataset2/processed
```

## Notes

- All paths can be absolute or relative to the current working directory
- Output directories will be created automatically if they don't exist
- The scripts log all operations and will display which paths are being used
- Path validation occurs at runtime - the scripts will fail gracefully if required files are missing
