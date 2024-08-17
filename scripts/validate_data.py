import json
import pandas as pd # type: ignore
import os

# Load configuration file
config_path = os.path.join(os.path.dirname(os.path.abspath("__file__")), '..', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# Convert relative paths in the configuration file to absolute paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))
raw_data_path = os.path.join(project_root, config['raw_data_path'])

# Define expected data types
expected_types = {
    'customerID': 'object',
    'gender': 'object',
    'SeniorCitizen': 'int64',
    'Partner': 'object',
    'Dependents': 'object',
    'tenure': 'int64',
    'PhoneService': 'object',
    'MultipleLines': 'object',
    'InternetService': 'object',
    'OnlineSecurity': 'object',
    'OnlineBackup': 'object',
    'DeviceProtection': 'object',
    'TechSupport': 'object',
    'StreamingTV': 'object',
    'StreamingMovies': 'object',
    'Contract': 'object',
    'PaperlessBilling': 'object',
    'PaymentMethod': 'object',
    'MonthlyCharges': 'float64',
    'TotalCharges': 'float64',
    'Churn': 'object'
}

# Define value ranges
value_ranges = {
    'tenure': (0, 100),
    'MonthlyCharges': (0, 150),
    'TotalCharges': (0, 10000)
}

# Define columns that should have unique values
unique_columns = ['customerID']

def check_missing_values(df):
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("Missing values found:")
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found.")
    return missing_values

def check_duplicates(df):
    duplicate_rows = df[df.duplicated()]
    if not duplicate_rows.empty:
        print(f"Found {len(duplicate_rows)} duplicate rows:")
        print(duplicate_rows)
    else:
        print("No duplicate rows found.")
    return duplicate_rows

def check_data_types(df, expected_types):
    mismatched_types = {}
    for column, expected_type in expected_types.items():
        if column in df.columns and df[column].dtype != expected_type:
            mismatched_types[column] = df[column].dtype
    if mismatched_types:
        print("Data type mismatches found:")
        print(mismatched_types)
    else:
        print("All data types are as expected.")
    return mismatched_types

def check_value_ranges(df, value_ranges):
    out_of_range = {}
    for column, (min_value, max_value) in value_ranges.items():
        if column in df.columns:
            out_of_bounds = df[(df[column] < min_value) | (df[column] > max_value)]
            if not out_of_bounds.empty:
                out_of_range[column] = out_of_bounds
    if out_of_range:
        print("Out of range values found:")
        for column, values in out_of_range.items():
            print(f"{column}:")
            print(values)
    else:
        print("All values are within the expected range.")
    return out_of_range

def check_unique_columns(df, unique_columns):
    not_unique = {}
    for column in unique_columns:
        if column in df.columns and df[column].duplicated().any():
            not_unique[column] = df[column][df[column].duplicated()]
    if not_unique:
        print("Columns with non-unique values found:")
        for column, values in not_unique.items():
            print(f"{column}:")
            print(values)
    else:
        print("All specified columns have unique values.")
    return not_unique

def check_consistency(df):
    if 'TotalCharges' in df.columns and 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        inconsistent = df[df['TotalCharges'] != df['tenure'] * df['MonthlyCharges']]
        if not inconsistent.empty:
            print("Inconsistent values found:")
            print(inconsistent[['tenure', 'MonthlyCharges', 'TotalCharges']])
        else:
            print("All values are consistent.")
    else:
        print("Skipping consistency check for 'TotalCharges' because it is not present.")



def main():
    # Load the dataset
    df = pd.read_csv(raw_data_path)

    # Perform checks
    check_missing_values(df)
    check_duplicates(df)  # Check for duplicates but do not drop them
    check_data_types(df, expected_types)
    check_value_ranges(df, value_ranges)
    check_unique_columns(df, unique_columns)
    check_consistency(df)

if __name__ == "__main__":
    main()
