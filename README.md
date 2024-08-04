# Customer Churn Analysis Project

## Project Overview

This project aims to analyze customer churn data for a telecommunications company, identify key factors contributing to churn, and develop strategies to improve customer retention.

## Project Structure

- **data_loader.py**: Script to load data from a CSV file.
- **data_cleaner.py**: Script to clean data by handling missing values and encoding categorical variables.
- **handle_missing_and_encode.py**: Advanced script for handling missing data and encoding categorical variables.
- **scaler.py**: Script to apply standard scaling and min-max scaling to the data.
- **data_preprocessing.py**: Main script to load, clean, scale, and save the dataset.
- **feature_engineering.py**: Script to create new features for the dataset.
- **feature_selector.py**: Script to select the top features based on their relevance to the target variable.
- **validate_data.py**: Script to validate data integrity and consistency.
- **settings.json**: Configuration file with paths and settings for the project.
- **.env**: Environment configuration file.

## Tasks Completed

### Task 1: Data Loading and Initial Preprocessing (CCA2)

1. **Objective**: Load and preprocess the dataset.
2. **Scripts Used**: 
    - `data_loader.py`
    - `data_cleaner.py`
3. **Steps**:
    - Loaded the raw dataset from `data/raw/Dataset (ATS)-1.csv`.
    - Handled missing values by dropping rows with missing data.
    - Encoded categorical variables using one-hot encoding.
4. **Output**:
    - Cleaned dataset saved to `data/interim/cleaned_dataset.csv`.

### Task 2: Advanced Handling of Missing Data and Encoding (CCA3)

1. **Objective**: Perform advanced handling of missing data and encoding of categorical variables.
2. **Scripts Used**:
    - `handle_missing_and_encode.py`
3. **Steps**:
    - Handled missing data using mean imputation for numeric columns.
    - Encoded categorical variables using `OneHotEncoder`.
4. **Output**:
    - Processed dataset saved to `data/interim/cleaned_dataset.csv`.

### Task 3: Feature Scaling and Normalization (CCA4)

1. **Objective**: Apply feature scaling and normalization to the dataset.
2. **Scripts Used**:
    - `scaler.py`
3. **Steps**:
    - Applied standard scaling to the cleaned dataset.
    - Applied min-max scaling to the cleaned dataset.
4. **Output**:
    - Standard scaled dataset saved to `data_preparation/scaling_techniques/standard_scaled_dataset.csv`.
    - Min-max scaled dataset saved to `data_preparation/scaling_techniques/min_max_scaled_dataset.csv`.

### Task 4: Feature Engineering

1. **Objective**: Create new features for the dataset.
2. **Scripts Used**:
    - `feature_engineering.py`
3. **Steps**:
    - Created new features such as `Charges_Per_Tenure` and `TotalCharges`.
    - Encoded contract types and payment methods.
4. **Output**:
    - Dataset with new features saved to `data/processed/processed_dataset_with_features.csv`.

### Task 5: Feature Selection

1. **Objective**: Select the top features based on their relevance to the target variable.
2. **Scripts Used**:
    - `feature_selector.py`
3. **Steps**:
    - Applied variance thresholding and SelectKBest to select the top features.
4. **Output**:
    - Selected features dataset saved to `Data_Preparation/training_sets/train_dataset.csv`.

### Task 6: Exploratory Data Analysis (EDA)

1. **Objective**: Conduct exploratory data analysis to understand data distribution and relationships.
2. **Scripts Used**:
    - `eda.py`
3. **Steps**:
    - Plotted distributions, boxplots, and categorical variables.
    - Plotted correlation matrix and pair plots.
4. **Output**:
    - EDA figures saved to `Data_Preparation/eda_visualizations`.

## Configuration

### settings.json

```json
{
     "python.pythonPath": "C:\\Users\\iambh\\anaconda3\\envs\\churn_analysis\\python.exe",
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

### config.json

```json
{
     "raw_data_path": "data/raw/Dataset (ATS)-1.csv",
     "interim_cleaned_data_path": "data/interim/cleaned_dataset.csv",
     "preprocessed_data_path": "Data_Preparation/preprocessed_dataset/cleaned_dataset.csv",
     "processed_data_path": "data/processed/processed_dataset_with_features.csv",
     "train_data_path": "data/train/train_dataset.csv",
     "test_data_path": "data/test/test_dataset.csv",
     "min_max_scaled_path": "Data_Preparation/scaling_techniques/min_max_scaled_dataset.csv",
     "standard_scaled_path": "Data_Preparation/scaling_techniques/standard_scaled_dataset.csv",
     "training_set_path": "Data_Preparation/training_sets/train_dataset.csv",
     "testing_set_path": "Data_Preparation/testing_sets/test_dataset.csv"
}
```

### How to Run

1. Ensure all required packages are installed.
2. Set up the environment variables from the .env file.
3. Execute main.py to run the entire data processing workflow.

```bash
python main.py
```

### Next Steps

1. Review the validation results to confirm data integrity.
2. Proceed with training machine learning models using the processed dataset.
3. Document each step and summarize the results for clarity and reproducibility.

**Note** For detailed documentation on the preprocessing and validation steps, please refer to `Notebooks` or the following files:-
   - `config_and_setup.md` for configuration and setup information.
   - `data_loading_preprocessing.md` for data loading and preprocessing documentation.
   - `data_splitter_documentation.md` for data splitting documentation.
   - `EDA_documentation.md` for exploratory data analysis documentation.
   - `feature_engineering_.md` for feature engineering documentation.
   - `main_workflow.md` for the main workflow documentation.
