# Feature Engineering Documentation

## Overview

This document provides an overview of the feature engineering steps taken to enhance the dataset. Feature engineering involves creating new features from the existing ones to better capture the underlying patterns in the data, which can improve the performance of machine learning models.

## Objectives

- **Create meaningful new features** that can help in predicting customer churn.
- **Transform existing features** to better represent the underlying data patterns.

## Steps Taken

### 1. Creation of Charges Per Tenure

**Objective**: To create a feature that normalizes the `MonthlyCharges` by the customer's tenure.

**Method**:
- `Charges_Per_Tenure` is calculated as `MonthlyCharges` divided by (`tenure` + 1). The `+1` is used to avoid division by zero.

**Formula**:
```python
df['Charges_Per_Tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
```

### 2. Creation of Total Charges

**Objective**: To create a feature that represents the total charges a customer has incurred.

**Method**:
- `TotalCharges` is calculated as `MonthlyCharges` multiplied by `tenure`.

**Formula**:
```python
df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']
```

### 3. Encoding of Contract Types

**Objective**: To convert the categorical `Contract` feature into numerical values.

**Method**:
- Contract types are mapped to numerical values:
    - Month-to-month → 0
    - One year → 1
    - Two year → 2

**Mapping**:
```python
contract_mapping = {
        'Month-to-month': 0,
        'One year': 1,
        'Two year': 2
}
df['Contract_Type'] = df['Contract'].map(contract_mapping)
```

### 4. Encoding of Payment Methods

**Objective**: To convert the categorical `PaymentMethod` feature into numerical values.

**Method**:
- PaymentMethod types are mapped to numerical values:
    - Electronic check → 0
    - Mailed check → 1
    - Bank transfer (automatic) → 2
    - Credit card (automatic) → 3

**Mapping**:
```python
payment_mapping = {
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
}
df['Payment_Method'] = df['PaymentMethod'].map(payment_mapping)
```

## Implementation

Script: `feature_engineering.py`

The `feature_engineering.py` script is responsible for loading the cleaned dataset, creating new features, and saving the enhanced dataset. Below are the relevant code snippets used in the script:

**Load Data**
```python
import pandas as pd

def load_data(file_path):
        try:
                data = pd.read_csv(file_path)
                print(f"Data loaded successfully from {file_path}")
                return data
        except Exception as e:
                print(f"An error occurred: {e}")
```

**Create New Features**
```python
def create_new_features(df):
        df['Charges_Per_Tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
        df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']
        
        contract_mapping = {
                'Month-to-month': 0,
                'One year': 1,
                'Two year': 2
        }
        df['Contract_Type'] = df['Contract'].map(contract_mapping)

        payment_mapping = {
                'Electronic check': 0,
                'Mailed check': 1,
                'Bank transfer (automatic)': 2,
                'Credit card (automatic)': 3
        }
        df['Payment_Method'] = df['PaymentMethod'].map(payment_mapping)
        
        return df
```

**Save Enhanced Dataset**
```python
if __name__ == "__main__":
        base_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_path, '..', 'data', 'interim', 'cleaned_dataset.csv')
        new_data_path = os.path.join(base_path, '..', 'data', 'processed', 'processed_dataset_with_features.csv')
        
        df = load_data(data_path)

        if df is not None:
                df = create_new_features(df)
                df.to_csv(new_data_path, index=False)
                print("New features added and dataset saved successfully.")
```

## Conclusion

The feature engineering steps have been designed to create meaningful features that capture important aspects of the data, which should help improve model performance. These features have been carefully crafted and validated to ensure they add value to the dataset.

