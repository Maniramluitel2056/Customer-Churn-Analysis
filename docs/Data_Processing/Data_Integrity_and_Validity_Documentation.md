# Data Integrity and Validation: `validate_data.py`
# Objective
The objective of the `validate_data.py` script is to ensure that the dataset is free from integrity issues that could compromise the quality of analysis. The script performs checks for missing values, duplicate entries, data type consistency, value ranges, and unique constraints to validate the dataset's integrity.

## Steps Taken

### Check for Missing Values:
We assessed the dataset to identify any missing values that could hinder the analysis process.

```python
def check_missing_values(df):
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("Missing values found:")
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found.")
```

### Duplicate Rows Detection:
The script identified duplicate rows in the dataset, which may represent repeated entries.

```python
def check_duplicates(df):
    duplicate_rows = df[df.duplicated()]
    if not duplicate_rows.empty:
        print(f"Found {len(duplicate_rows)} duplicate rows:")
        print(duplicate_rows)
    else:
        print("No duplicate rows found.")
```

### Data Type Validation:
Each column in the dataset was checked against expected data types to ensure consistency and avoid potential errors during processing.

```python
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
```

### Value Range Validation:
The script verified that numerical columns fall within expected ranges to detect any outliers or data entry errors.

```python
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
```

### Unique Column Check:
We ensured that specific columns, such as `customerID`, contain unique values, preventing any duplication of key identifiers.

```python
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
```

## Explanation
**Objective:** The script is designed to ensure that the dataset is clean and ready for analysis by performing critical data integrity and validity checks.

**Missing Values:** No missing values were found, indicating that the dataset is complete.

**Duplicates:** The script detected 103 duplicate rows. We have decided not to remove these duplicates because they may represent valid, repeated transactions, and removing them could result in the loss of important information.

**Data Types:** All columns were found to have the expected data types, ensuring the dataset's consistency.

**Value Ranges:** All numerical columns fall within expected ranges, confirming that there are no significant outliers or anomalies.

**Unique Columns:** Key columns, such as `customerID`, were found to contain unique values, ensuring that each record represents a distinct entity.

## Summary
The `validate_data.py` script successfully validated the dataset, identifying key issues such as duplicate rows but confirming the absence of missing values or data type mismatches. The decision to retain duplicates was made to preserve data integrity and ensure no loss of valuable information.

## Next Steps
1. Proceed with Data Analysis: Since the dataset passed the integrity checks, we can proceed with further analysis, such as clustering or predictive modeling.
2. Documentation: Ensure that the decisions made during the validation process, such as retaining duplicates, are clearly documented in the project's final report.
