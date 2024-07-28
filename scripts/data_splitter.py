import pandas as pd
from sklearn.model_selection import train_test_split
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
    except FileNotFoundError:
        # Handle file not found error
        print(f"File not found at {file_path}. Please check the file path.")
        # Returns: None if the file is not found
    except pd.errors.EmptyDataError:
        # Handle empty data error
        print(f"No data found at {file_path}. The file is empty.")
        # Returns: None if the file is empty
    except pd.errors.ParserError:
        # Handle parsing error
        print(f"Error parsing data from {file_path}. Please check the file format.")
        # Returns: None if there is a parsing error
    except Exception as e:
        # Handle any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        # Returns: None if an unexpected error occurs

def split_data(df, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    
    Parameters:
    df (DataFrame): The input DataFrame to be split.
    test_size (float): The proportion of the dataset to include in the test split. Default is 0.2 (20% testing, 80% training).
    random_state (int): The seed used by the random number generator.
    
    Returns:
    DataFrame, DataFrame: The training and testing DataFrames.
    """
    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    print(f"Data split into training (size={len(train_df)}) and testing (size={len(test_df)}) sets.")
    # Returns: Two DataFrames - train_df containing the training data and test_df containing the testing data
    return train_df, test_df

if __name__ == "__main__":
    # Define the base path using the current file's directory
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Define the path to the processed dataset with new features using a relative path
    data_path = os.path.join(base_path, '..', 'data', 'processed', 'processed_dataset_with_features.csv')
    
    # Define the paths to save the split datasets using relative paths
    train_path_data = os.path.join(base_path, '..', 'data', 'train', 'train_dataset.csv')
    test_path_data = os.path.join(base_path, '..', 'data', 'test', 'test_dataset.csv')
    train_path_prep = os.path.join(base_path, '..', 'Data_Preparation', 'training_sets', 'train_dataset.csv')
    test_path_prep = os.path.join(base_path, '..', 'Data_Preparation', 'testing_sets', 'test_dataset.csv')
    # Returns: Absolute paths for the processed dataset with new features and the split datasets

    # Load the cleaned dataset from the specified path
    df = load_data(data_path)
    # Returns: A DataFrame containing the cleaned dataset, or None if loading fails

    # Check if the data was loaded successfully
    if df is not None:
        # Split the data into training (80%) and testing (20%) sets
        train_df, test_df = split_data(df)
        
        # Save the training and testing datasets to the specified paths
        train_df.to_csv(train_path_data, index=False)
        test_df.to_csv(test_path_data, index=False)
        train_df.to_csv(train_path_prep, index=False)
        test_df.to_csv(test_path_prep, index=False)
        # Actions: The training and testing datasets are saved to their respective paths

        print("Training and testing datasets created and saved successfully.")
        # Prints a confirmation message indicating successful saving
