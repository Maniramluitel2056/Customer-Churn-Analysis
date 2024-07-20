Data Loading and Preprocessing Documentation

# Task 1: Data Loading and Preprocessing

## Objective:
The goal of Task 1 is to load the raw dataset, preprocess the data by handling missing values and encoding categorical variables, and save the cleaned data for further analysis.

## Steps Taken and Explanations

1. **Setting Up the Configuration File (config.json):**
    - Reason: Using a configuration file allows us to centralize and easily manage the file paths used in the script. This makes the code more flexible and portable, as changes to file paths can be made in one place without altering the code.
    - Snippet:
      ```json
      {
         "raw_data_path": "data/raw/Dataset (ATS)-1.csv",
         "interim_cleaned_data_path": "data/interim/cleaned_dataset.csv",
         "preprocessed_data_path": "Data_Preparation/preprocessed_dataset/cleaned_dataset.csv"
      }
      ```

2. **Loading Configuration File:**
    - Reason: To dynamically load the file paths from config.json, ensuring that the paths used in the script are up-to-date and easily changeable.
    - Snippet:
      ```python
      config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.json')
      with open(config_path, 'r') as f:
            config = json.load(f)

      raw_data_path = config['raw_data_path']
      interim_cleaned_data_path = config['interim_cleaned_data_path']
      preprocessed_data_path = config['preprocessed_data_path']
      ```

3. **Importing Libraries and Modules:**
    - Reason: Essential libraries (pandas, os, sys, json) and custom modules (data_loader, data_cleaner) are imported to handle data processing tasks.
    - Snippet:
      ```python
      import json
      import pandas as pd
      import sys
      import os

      sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))

      from data_loader import load_data
      from data_cleaner import clean_data
      ```

4. **Loading the Raw Data:**
    - Function Used: `load_data(file_path)`
    - Reason: This function reads the raw dataset from the specified path using pandas. It ensures the data is loaded correctly, or provides an appropriate error message if the file is not found.
    - Snippet:
      ```python
      df = load_data(raw_data_path)
      if df is None:
            print(f"File not found at {raw_data_path}. Please check the file path.")
            print("Data loading failed. Exiting the script.")
            return
      ```

5. **Cleaning the Data:**
    - Function Used: `clean_data(df)`
    - Reason: This function handles missing values by dropping rows with missing data and encodes categorical variables using one-hot encoding. It prepares the data for analysis by ensuring it is clean and in the correct format.
    - Snippet:
      ```python
      df_cleaned = clean_data(df)
      if df_cleaned is None:
            print("Data cleaning failed. Exiting the script.")
            return
      ```

6. **Saving the Cleaned Data:**
    - Reason: The cleaned dataset is saved to specified paths for intermediate and final usage. This ensures the cleaned data is readily available for further steps in the analysis pipeline.
    - Snippet:
      ```python
      df_cleaned.to_csv(interim_cleaned_data_path, index=False)
      df_cleaned.to_csv(preprocessed_data_path, index=False)
      print(f"Cleaned data saved to interim at {interim_cleaned_data_path}")
      print(f"Cleaned data saved to preprocessed_dataset at {preprocessed_data_path}")
      ```

## Results Obtained

**Loading Raw Data:**
Successfully loaded the raw dataset from `data/raw/Dataset (ATS)-1.csv`.

**Cleaning Data:**
- Missing values were handled by dropping rows.
- Categorical variables were encoded using one-hot encoding.

**Saving Cleaned Data:**
- Cleaned data was saved to `data/interim/cleaned_dataset.csv`.
- Cleaned data was also saved to `Data_Preparation/preprocessed_dataset/cleaned_dataset.csv`.


# Task 2: Handle Missing Data and Encode Categorical Variables

## Objective
The objective of Task 2 is to further preprocess the data by handling any remaining missing values and encoding categorical variables. This will ensure that the dataset is ready for machine learning models.

## Steps Taken and Explanations

1. Importing Libraries and Modules

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from handle_missing_and_encode import handle_missing_data, encode_categorical_variables
```

2. Handling Missing Data

```python
df_missing_handled = handle_missing_data(df_cleaned)
```

The `handle_missing_data` function is used to handle missing data points in the dataframe using mean imputation. This ensures that all numerical columns are complete and ready for analysis.

3. Encoding Categorical Variables

```python
df_encoded = encode_categorical_variables(df_missing_handled)
```

The `encode_categorical_variables` function is used to encode categorical variables in the dataframe into numerical format using one-hot encoding. This transformation is essential for machine learning algorithms that require numerical input.

4. Saving the Processed Data

```python
df_encoded.to_csv(interim_cleaned_data_path, index=False)
df_encoded.to_csv(preprocessed_data_path, index=False)
```

The processed dataset is saved to the specified paths. This includes saving the dataset with handled missing values and encoded categorical variables for further use in the analysis pipeline.

## Results Obtained

#### Handling Missing Data
- Missing values were handled using mean imputation.

#### Encoding Categorical Variables
- Categorical variables were encoded into numerical format using one-hot encoding.

#### Saving Processed Data
- Processed data was saved to `data/interim/cleaned_dataset.csv`.
- Processed data was also saved to `Data_Preparation/preprocessed_dataset/cleaned_dataset.csv`.

## Summary

In this script, the following steps were completed:

1. Imported necessary libraries and modules.
2. Handled missing data by using mean imputation.
3. Encoded categorical variables using one-hot encoding.
4. Saved the processed data to the interim and preprocessed paths.

## Next Steps

1. Perform feature scaling and normalization to ensure that all features are on a comparable scale, which is crucial for the performance of many machine learning models.
2. Conduct exploratory data analysis (EDA) to understand the data distribution, relationships between variables, and identify any anomalies or patterns.
3. Document each step and summarize the results to ensure clarity and reproducibility for all team members and stakeholders involved in the project.

**Note: For detailed documentation on the notebook, please refer to the notebook itself.**
