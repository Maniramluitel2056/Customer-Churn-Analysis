import pandas as pd # type: ignore
import os

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Parameters:
    file_path (str): The path to the CSV file to be loaded.
    
    Returns:
    DataFrame: The data loaded from the CSV file, or None if an error occurs.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"An error occurred: {e}")

def create_new_features(df):
    """
    Create new features for the dataset.
    
    Parameters:
    df (DataFrame): The input DataFrame for which new features will be created.
    
    Returns:
    DataFrame: The DataFrame with the newly created features.
    """
    if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        df['Charges_Per_Tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)

    if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
        df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']

    contract_mapping = {
        'Month-to-month': 0,
        'One year': 1,
        'Two year': 2
    }
    if 'Contract' in df.columns:
        df['Contract_Type'] = df['Contract'].map(contract_mapping)

    payment_mapping = {
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    }
    if 'PaymentMethod' in df.columns:
        df['Payment_Method'] = df['PaymentMethod'].map(payment_mapping)
    
    return df

if __name__ == "__main__":
    # Define the base path and project root directory
    base_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_path, '..'))
    
    # Define paths for interim cleaned data and processed data
    interim_cleaned_data_path = os.path.join(project_root, 'data', 'interim', 'cleaned_dataset.csv')
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'processed_dataset_with_features.csv')

    # Load the cleaned dataset from the interim cleaned data path
    df_cleaned = load_data(interim_cleaned_data_path)

    # If the cleaned data is loaded successfully, create new features
    if df_cleaned is not None:
        df_cleaned = create_new_features(df_cleaned)

        # Save the processed dataset with new features
        df_cleaned.to_csv(processed_data_path, index=False)
        print("New features added and dataset saved successfully.")
