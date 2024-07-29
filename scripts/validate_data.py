import pandas as pd
import os

def validate_data(df_original, df_processed):
    """
    Validate data integrity and consistency between the original and processed datasets.
    """
    # Check for missing values in the processed dataset
    missing_values = df_processed.isnull().sum().sum()
    print(f"Missing values in the processed dataset: {missing_values}")
    # Returns the total number of missing values in the processed dataset
    
    # Ensure the processed dataset has the expected columns
    expected_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'gender_Female', 'gender_Male', 
                        'Dependents_Yes', 'Dependents_No', 'PhoneService_Yes', 'PhoneService_No', 
                        'MultipleLines_Yes', 'MultipleLines_No', 'InternetService_Fiber optic', 
                        'InternetService_DSL', 'Contract_Month-to-month', 'Contract_One year', 
                        'Contract_Two year', 'Churn_No', 'Churn_Yes', 'Charges_Per_Tenure', 'TotalCharges']
    missing_columns = set(expected_columns) - set(df_processed.columns)
    print(f"Missing columns in the processed dataset: {missing_columns}")
    # Returns a set of columns that are expected but missing in the processed dataset
    
    # Compare some basic statistics between original and processed data
    original_stats = df_original.describe()
    processed_stats = df_processed.describe()
    # Returns statistical summaries of the original and processed datasets
    
    return original_stats, processed_stats

if __name__ == "__main__":
    # Define paths using relative paths
    base_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_path, '..'))
    
    original_data_path = os.path.join(project_root, 'data', 'interim', 'cleaned_dataset.csv')
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'processed_dataset_with_features.csv')
    # Returns the absolute paths for the original and processed datasets

    # Load the original and processed datasets
    df_original = pd.read_csv(original_data_path)
    df_processed = pd.read_csv(processed_data_path)
    # Returns the DataFrames loaded from the respective CSV files
    
    # Validate the data
    original_stats, processed_stats = validate_data(df_original, df_processed)
    # Returns the validation results, including missing values, missing columns, and statistical summaries

    # Print statistical summaries for comparison
    print("Original Data Statistics:")
    print(original_stats)
    # Prints the statistical summary of the original dataset

    print("Processed Data Statistics:")
    print(processed_stats)
    # Prints the statistical summary of the processed dataset
