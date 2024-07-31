# Main Workflow
## Overview

This document provides an overview of the entire data processing and feature engineering workflow. It details each major step, linking to the relevant scripts and documentation for a comprehensive understanding of the pipeline.

## Workflow Steps

1. **Data Loading**
    - Script: `scripts/data_loader.py`
    - Description: This step involves loading the raw data from a CSV file. The data is read into a pandas DataFrame for further processing.

```python
import pandas as pd

def load_data(file_path):
     try:
          data = pd.read_csv(file_path)
          return data
     except Exception as e:
          print(f"Error loading data: {e}")
```

2. **Data Cleaning**
    - Script: `scripts/data_cleaner.py`
    - Description: This step handles missing values and encodes categorical variables. Missing values are typically handled by mean imputation or dropping rows, while categorical variables are encoded using one-hot encoding.

```python
import pandas as pd
from category_encoders import OneHotEncoder

def clean_data(df):
     df = df.dropna()
     categorical_columns = df.select_dtypes(include=['object']).columns
     if len(categorical_columns) > 0:
          encoder = OneHotEncoder(cols=categorical_columns)
          df = encoder.fit_transform(df)
     return df
```

3. **Handling Missing Data and Encoding**
    - Script: `scripts/handle_missing_and_encode.py`
    - Description: This step involves handling any remaining missing values and encoding categorical variables using more advanced methods.

```python
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def handle_missing_data(df):
     imputer = SimpleImputer(strategy='mean')
     df_numeric = df.select_dtypes(include=['float64', 'int64'])
     df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)
     df[df_numeric.columns] = df_numeric_imputed
     return df

def encode_categorical_variables(df):
     df_categorical = df.select_dtypes(include=['object'])
     if not df_categorical.empty:
          encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
          encoded_data = pd.DataFrame(encoder.fit_transform(df_categorical), columns=encoder.get_feature_names_out(df_categorical.columns))
          df = df.drop(columns=df_categorical.columns)
          df = pd.concat([df, encoded_data], axis=1)
     return df
```

4. **Feature Scaling and Normalization**
    - Script: `scripts/scaler.py`
    - Description: This step applies standard scaling and min-max scaling to the data to ensure that all features are on a comparable scale.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def apply_standard_scaling(df):
     numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
     scaler = StandardScaler()
     scaled_df = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
     return scaled_df

def apply_min_max_scaling(df):
     numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
     scaler = MinMaxScaler()
     scaled_df = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
     return scaled_df
```

5. **Data Splitting**
    - Script: `scripts/data_splitter.py`
    - Description: This step splits the cleaned and scaled data into training and testing sets for model development.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, test_size=0.2, random_state=42):
     train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
     return train_df, test_df
```

6. **Feature Engineering**
    - Script: `scripts/feature_engineering.py`
    - Description: This step involves creating new features to enhance the dataset's predictive power.

```python
import pandas as pd

def create_new_features(df):
     if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
          df['Charges_Per_Tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
     if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
          df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']
     return df
```

7. **Feature Selection**
    - Script: `scripts/feature_selector.py`
    - Description: This step selects the top features based on their importance to the target variable.

```python
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

def select_features(df, target_column, k=10):
     y = df[target_column]
     X = df.drop(columns=[target_column])
     selector = SelectKBest(score_func=f_classif, k=k)
     selector.fit(X, y)
     selected_columns = X.columns[selector.get_support()]
     df_selected = X[selected_columns].copy()
     df_selected[target_column] = y.reset_index(drop=True)
     return df_selected
```

8. **Exploratory Data Analysis (EDA)**
    - Script: `scripts/eda.py`
    - Description: This step performs exploratory data analysis to gain insights into the data and visualize important patterns and relationships.

```python 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        print(f"File not found at {file_path}")
    except pd.errors.EmptyDataError:
        print(f"No data found at {file_path}")
    except pd.errors.ParserError:
        print(f"Error parsing data from {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def plot_distribution(df, columns, bins=30):
    df[columns].hist(bins=bins, figsize=(20, 15))
    plt.show()

def plot_boxplots(df, columns):
    plt.figure(figsize=(20, 15))
    sns.boxplot(data=df[columns])
    plt.xticks(rotation=90)
    plt.show()

def plot_categorical(df, columns):
    for column in columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x=column)
        plt.xticks(rotation=90)
        plt.show()

def plot_correlation_matrix(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()

def plot_pairplots(df, hue):
    sns.pairplot(df, hue=hue)
    plt.show()
```

8. **Main Orchestration**
    - Script: `scripts/main.py`
    - Description: This script orchestrates the entire data processing pipeline, calling each of the steps in sequence.

```python
import os
import sys
import json
import pandas as pd

def load_config(config_path='../config.json'):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def ensure_directories(paths):
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)

if __name__ == "__main__":
    config = load_config()
    data = run_data_loader(config)
    cleaned_data = run_data_cleaner(config)
    train_df, test_df = run_data_splitter(config)
    data_with_features = run_feature_engineering(config)
    selected_data = run_feature_selection(config)
    validate_data_integrity(config)

```

## Summary

This document outlines the detailed steps involved in the data processing, feature engineering, and exploratory data analysis pipeline. Each script and notebook plays a crucial role in transforming raw data into a cleaned, processed, and feature-rich dataset ready for machine learning model development. Ensure that all configurations and environment setups are correctly followed to achieve a smooth workflow execution.

