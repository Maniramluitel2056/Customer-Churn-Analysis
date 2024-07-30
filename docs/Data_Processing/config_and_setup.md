# Configuration and Environment Setup
## Overview

This document provides detailed instructions for setting up the environment and configuring the necessary files to run the data processing and feature engineering pipeline. It includes the setup of configuration files such as `config.json`, `.env`, and `settings.json`.

## Environment Setup

1. Clone the Repository

First, clone the project repository to your local machine:

```
git clone https://github.com/Maniramluitel2056/Customer-Churn-Analysis.git
```

2. Create a Virtual Environment and Install Dependencies

Create and activate a virtual environment to manage project dependencies and install the required dependencies specified in the `environment.yml` file:

```
conda env create -f environment.yml
```

3. Activate the environment

Once the environment is created, activate it using:

```
conda activate churn_analysis
```

## Configuration Files

### `config.json`

The `config.json` file contains paths and settings required for various scripts. Ensure that the paths are correctly set according to your project structure.

```json
{
    "raw_data_path": "data/raw/Dataset (ATS)-1.csv",
    "interim_cleaned_data_path": "data/interim/cleaned_dataset.csv",
    "preprocessed_data_path": "Data_Preparation/preprocessed_dataset/cleaned_dataset.csv",
    "train_data_path": "data/train/train_dataset.csv",
    "test_data_path": "data/test/test_dataset.csv",
    "processed_data_path": "data/processed/processed_dataset_with_features.csv",
    "training_set_path": "Data_Preparation/training_sets/train_dataset_selected.csv",
    "testing_set_path": "Data_Preparation/testing_sets/test_dataset_selected.csv"
}
```

### `.env`

The `.env` file is used to set environment variables required for the project. Ensure the correct paths and settings are provided.

```
PYTHONPATH=${PYTHONPATH}:${workspaceFolder}
```

### `settings.json`

The `settings.json` file is used to configure the project settings in your development environment (e.g., VS Code).

```json
{
        "python.pythonPath": "C:\\Users\\iambh\\anaconda3\\envs\\churn_analysis\\python.exe", // change path if required
        "python.envFile": "${workspaceFolder}/.env",
        "python.autoComplete.extraPaths": [
                "${workspaceFolder}/utils",
                "${workspaceFolder}/scripts"
        ],
        "python.analysis.extraPaths": [
                "${workspaceFolder}/utils",
                "${workspaceFolder}/scripts"
        ],
        "terminal.integrated.env.windows": {
                "PYTHONPATH": "${workspaceFolder}"
        }
}
```

## Running the Scripts

1. Data Preprocessing:

```
python scripts/data_preprocessing.py
```

2. Data Splitting:

```
python scripts/data_splitter.py
```

3. Feature Engineering:

```
python scripts/feature_engineering.py
```

4. Feature Selection:

```
python scripts/feature_selector.py
```

5. Main Workflow:

```
python scripts/main.py
```

**Note**: Ensure all paths and configurations are correctly set before running the scripts.
