import pandas as pd
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
        # Attempt to read the data from the specified CSV file
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
        # Returns: A DataFrame containing the data from the CSV file
    except Exception as e:
        # Print an error message if the data loading fails
        print(f"An error occurred: {e}")
        # Returns: None if an error occurs while loading the data

def create_new_features(df):
    """
    Create new features for the dataset.
    
    Parameters:
    df (DataFrame): The input DataFrame for which new features will be created.
    
    Returns:
    DataFrame: The DataFrame with the newly created features.
    """
    # Create 'Charges_Per_Tenure' feature if the necessary columns exist
    if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        df['Charges_Per_Tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
        # Adds a new feature 'Charges_Per_Tenure' to the DataFrame
    
    # Create 'TotalCharges' feature if the necessary columns exist
    if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
        df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']
        # Adds a new feature 'TotalCharges' to the DataFrame
    
    # Encode contract types into numerical values
    contract_mapping = {
        'Month-to-month': 0,
        'One year': 1,
        'Two year': 2
    }
    if 'Contract' in df.columns:
        df['Contract_Type'] = df['Contract'].map(contract_mapping)
        # Adds a new feature 'Contract_Type' to the DataFrame with encoded contract types

    # Encode payment methods into numerical values
    payment_mapping = {
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    }
    if 'PaymentMethod' in df.columns:
        df['Payment_Method'] = df['PaymentMethod'].map(payment_mapping)
        # Adds a new feature 'Payment_Method' to the DataFrame with encoded payment methods
    
    return df
    # Returns: The DataFrame with the newly created features

if __name__ == "__main__":
    # Define the base path using the current file's directory
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Define the path to the cleaned dataset using a relative path
    data_path = os.path.join(base_path, '..', 'data', 'interim', 'cleaned_dataset.csv')
    
    # Define the path to save the new dataset with features using a relative path
    new_data_path = os.path.join(base_path, '..', 'data', 'processed', 'processed_dataset_with_features.csv')
    # Returns: Absolute paths for the cleaned dataset and the new dataset with features

    # Load the cleaned data from the specified path
    df = load_data(data_path)
    # Returns: A DataFrame containing the cleaned data, or None if loading fails

    # Check if the data was loaded successfully
    if df is not None:
        # Create new features for the loaded data
        df = create_new_features(df)
        # Returns: A DataFrame with the newly created features added

        # Save the new dataset with features to the specified path
        df.to_csv(new_data_path, index=False)
        print("New features added and dataset saved successfully.")
        # Action: The new dataset with features is saved to new_data_path
