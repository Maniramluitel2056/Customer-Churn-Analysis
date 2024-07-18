# utils/data_cleaner.py

import pandas as pd
from category_encoders import OneHotEncoder

def clean_data(df):
    """
    Clean the data by handling missing values and encoding categorical variables.

    Parameters:
    df (pd.DataFrame): The data to clean.

    Returns:
    pd.DataFrame: Cleaned data.
    """
    try:
        # Handle missing values by dropping rows with missing values
        df = df.dropna()
        print("Missing values handled by dropping rows with missing values.")

        # Identify and encode categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            encoder = OneHotEncoder(cols=categorical_columns)
            df = encoder.fit_transform(df)
            print(f"Categorical columns {list(categorical_columns)} encoded.")
        else:
            print("No categorical columns found to encode.")

        return df
    except Exception as e:
        print(f"An error occurred during data cleaning: {e}")
