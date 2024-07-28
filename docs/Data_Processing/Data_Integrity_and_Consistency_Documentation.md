# Data Integrity and Consistency Documentation
## Overview

This document provides an overview of the steps taken to ensure the integrity and consistency of the dataset throughout the preprocessing workflow. It covers the validation checks performed on the processed data to confirm the correct application of data transformations and the maintenance of dataset integrity.

## Data Validation Steps

### 1. Missing Values Check

**Objective**: Ensure there are no missing values in the processed dataset.

**Method**:
- Count the total number of missing values in the processed dataset.
- Verify that the count of missing values is zero.

**Results**:
- Missing values in the processed dataset: **0**

### 2. Column Validation

**Objective**: Ensure the processed dataset contains all the expected columns.

**Method**:
- Define a list of expected columns in the processed dataset.
- Compare the expected columns with the columns in the processed dataset to check for any missing columns.

**Expected Columns**:
```plaintext
['gender_1', 'gender_2', 'SeniorCitizen', 'Dependents_1', 'Dependents_2', 'tenure', 
 'PhoneService_1', 'PhoneService_2', 'MultipleLines_1', 'MultipleLines_2', 'InternetService_1', 
 'InternetService_2', 'Contract_1', 'Contract_2', 'Contract_3', 'MonthlyCharges', 'Churn_1', 
 'Churn_2', 'Charges_Per_Tenure', 'TotalCharges']
```

**Results**:
- Missing columns in the processed dataset: None

### 3. Statistical Comparison

**Objective**: Compare basic statistics between the original and processed datasets to ensure consistency.

**Method**:
- Calculate descriptive statistics (mean, std, min, 25%, 50%, 75%, max) for key columns in both the original and processed datasets.
- Compare these statistics to identify any significant discrepancies.

**Original Data Statistics**:

```plaintext
  SeniorCitizen       tenure  MonthlyCharges
count    7043.000000  7043.000000     7043.000000
mean        0.162147    32.371149       64.761692
std         0.368612    24.559481       30.090047
min         0.000000     0.000000       18.250000
25%         0.000000     9.000000       35.500000
50%         0.000000    29.000000       70.350000
75%         0.000000    55.000000       89.850000
max         1.000000    72.000000      118.750000
```

**Processed Data Statistics**:

```plaintext
     gender_1     gender_2  SeniorCitizen  Dependents_1  Dependents_2       tenure  ...   Contract_3  MonthlyCharges      Churn_1      Churn_2  Charges_Per_Tenure  TotalCharges    
count  7043.000000  7043.000000    7043.000000   7043.000000   7043.000000  7043.000000  ...  7043.000000     7043.000000  7043.000000  7043.000000         7043.000000   7043.000000    
mean      0.495244     0.504756       0.162147      0.700412      0.299588    32.371149  ...     0.240664       64.761692     0.734630     0.265370            5.770645   2279.581350    
std       0.500013     0.500013       0.368612      0.458110      0.458110    24.559481  ...     0.427517       30.090047     0.441561     0.441561            8.722435   2264.729447    
min       0.000000     0.000000       0.000000      0.000000      0.000000     0.000000  ...     0.000000       18.250000     0.000000     0.000000            0.264384      0.000000    
25%       0.000000     0.000000       0.000000      0.000000      0.000000     9.000000  ...     0.000000       35.500000     0.000000     0.000000            1.250000    394.000000    
50%       0.000000     1.000000       0.000000      1.000000      0.000000    29.000000  ...     0.000000       70.350000     1.000000     0.000000            2.075926   1393.600000    
75%       1.000000     1.000000       0.000000      1.000000      1.000000    55.000000  ...     0.000000       89.850000     1.000000     1.000000            5.946429   3786.100000    
max       1.000000     1.000000       1.000000      1.000000      1.000000    72.000000  ...     1.000000      118.750000     1.000000     1.000000           80.850000   8550.000000    
```

**Comparison**:
- The statistics of the key columns between the original and processed datasets are consistent, indicating that the transformations have been correctly applied and the data integrity has been maintained.

## Conclusion

The data integrity and consistency checks confirm the following:
- There are no missing values in the processed dataset.
- All expected columns are present in the processed dataset.
- The basic statistics of key columns are consistent between the original and processed datasets.

The processed dataset has been validated and is ready for further analysis and modeling.

## Appendix

### Scripts Used

- **validate_data.py**: Script used to perform data validation checks.

### Configuration Files

- **config.json**: Configuration file containing paths to datasets and other settings.

## How to Run the Validation

1. Ensure all required packages are installed.
2. Set up the environment variables from the .env file.
3. Execute the validation script.

```
python validate_data.py
```

