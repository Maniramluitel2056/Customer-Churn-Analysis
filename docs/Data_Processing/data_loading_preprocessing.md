# Data Loading and Preprocessing
## Objective:
The goal of Task 1 is to load the raw dataset, preprocess the data by handling missing values and encoding categorical variables, apply scaling techniques, and save the cleaned and scaled data for further analysis.

## Steps Taken and Explanations

### Setting Up the Configuration File (config.json):
Reason: Using a configuration file allows us to centralize and easily manage the file paths used in the script. This makes the code more flexible and portable, as changes to file paths can be made in one place without altering the code.

Snippet:
```json
{
      "raw_data_path": "data/raw/Dataset (ATS)-1.csv",
      "interim_cleaned_data_path": "data/interim/cleaned_dataset.csv",
      "preprocessed_data_path": "Data_Preparation/preprocessed_dataset/cleaned_dataset.csv"
}
```

### Loading Configuration File:
Reason: To dynamically load the file paths from `config.json`, ensuring that the paths used in the script are up-to-date and easily changeable.

Snippet:
```python
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.json')
with open(config_path, 'r') as f:
       config = json.load(f)

raw_data_path = os.path.join(project_root, config['raw_data_path'])
interim_cleaned_data_path = os.path.join(project_root, config['interim_cleaned_data_path'])
preprocessed_data_path = os.path.join(project_root, config['preprocessed_data_path'])
standard_scaled_data_path = os.path.join(project_root, 'data_preparation/scaling_techniques/standard_scaled_dataset.csv')
min_max_scaled_data_path = os.path.join(project_root, 'data_preparation/scaling_techniques/min_max_scaled_dataset.csv')
```

### Importing Libraries and Modules:
Reason: Essential libraries `(pandas, os, sys, json)` and custom modules `(data_loader, data_cleaner, handle_missing_and_encode, scaler)` are imported to handle data processing tasks.

Snippet:
```python
import json
import pandas as pd
import sys
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sys.path.append(os.path.join(project_root, 'utils'))

from data_loader import load_data
from data_cleaner import clean_data
from handle_missing_and_encode import handle_missing_and_encode
from scaler import apply_standard_scaling, apply_min_max_scaling
```

### Loading the Raw Data:
Function Used: load_data(file_path)

Reason: This function reads the raw dataset from the specified path using pandas. It ensures the data is loaded correctly, or provides an appropriate error message if the file is not found.

Snippet:
```python
if not os.path.exists(raw_data_path):
      print(f"File does not exist at {raw_data_path}")
      print("Data loading failed. Exiting the script.")
      return

print(f"Attempting to load raw data from: {raw_data_path}")
df = load_data(raw_data_path)
if df is None:
      print(f"File not found at {raw_data_path}. Please check the file path.")
      print("Data loading failed. Exiting the script.")
      return
```

### Cleaning the Data:
Function Used: handle_missing_and_encode(df)

Reason: This function handles missing values and encodes categorical variables. It prepares the data for analysis by ensuring it is clean and in the correct format.

Snippet:
```python
df_cleaned = handle_missing_and_encode(df)
if df_cleaned is None:
      print("Data cleaning failed. Exiting the script.")
      return
```

### Applying Scaling Techniques:
Functions Used: `apply_standard_scaling(df_cleaned), apply_min_max_scaling(df_cleaned)`

Reason: These functions apply standard scaling and min-max scaling to the cleaned dataset to ensure that the features are on a comparable scale, which is crucial for the performance of many machine learning models.

Snippet:
```python
df_standard_scaled = apply_standard_scaling(df_cleaned)
df_min_max_scaled = apply_min_max_scaling(df_cleaned)
```

### Saving the Cleaned and Scaled Data:
Reason: The cleaned and scaled datasets are saved to specified paths for intermediate and final usage. This ensures the prepared data is readily available for further steps in the analysis pipeline.

Snippet:
```python
df_cleaned.to_csv(interim_cleaned_data_path, index=False)
df_cleaned.to_csv(preprocessed_data_path, index=False)
df_standard_scaled.to_csv(standard_scaled_data_path, index=False)
df_min_max_scaled.to_csv(min_max_scaled_data_path, index=False)

print(f"Cleaned data saved to interim at {interim_cleaned_data_path}")
print(f"Cleaned data saved to preprocessed at {preprocessed_data_path}")
print(f"Standard scaled data saved at {standard_scaled_data_path}")
print(f"Min-Max scaled data saved at {min_max_scaled_data_path}")
```

## Results Obtained
**Loading Raw Data**:
Successfully loaded the raw dataset from data/raw/Dataset (ATS)-1.csv.

**Cleaning Data**:
Missing values were handled and categorical variables were encoded.

**Applying Scaling Techniques**:
Standard scaling and min-max scaling were applied to the cleaned data.

**Saving Cleaned and Scaled Data**:
Cleaned data was saved to `data/interim/cleaned_dataset.csv.`
Cleaned data was also saved to `Data_Preparation/preprocessed_dataset/cleaned_dataset.csv.`
Standard scaled data was saved to `data_preparation/scaling_techniques/standard_scaled_dataset.csv.`
Min-max scaled data was saved to `data_preparation/scaling_techniques/min_max_scaled_dataset.csv.`

## Summary
In Task 1, I successfully set up a configuration file to manage file paths, loaded the raw dataset, performed data cleaning by handling missing values and encoding categorical variables, applied scaling techniques, and saved the cleaned and scaled datasets to designated locations. This structured approach ensures that the data is prepared for subsequent analysis steps in a consistent and reproducible manner.

## Next Steps
1. Handle Remaining Missing Data Points: Implement strategies to handle any remaining missing data points to ensure data quality and completeness.
2. Feature Scaling and Normalization: Apply scaling and normalization techniques to ensure features are on a comparable scale, which is crucial for the performance of many machine learning models.
3. Exploratory Data Analysis (EDA): Conduct EDA to understand the data distributions, relationships between variables, and identify any anomalies or patterns.
4. Model Training: Use the cleaned and preprocessed data to train machine learning models for predicting customer churn.