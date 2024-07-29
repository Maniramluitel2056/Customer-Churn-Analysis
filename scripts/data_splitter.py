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
    except FileNotFoundError:
        print(f"File not found at {file_path}. Please check the file path.")
    except pd.errors.EmptyDataError:
        print(f"No data found at {file_path}. The file is empty.")
    except pd.errors.ParserError:
        print(f"Error parsing data from {file_path}. Please check the file format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    
    Parameters:
    df (DataFrame): The input DataFrame to be split.
    target_column (str): The target variable column name.
    test_size (float): The proportion of the dataset to include in the test split. Default is 0.2 (20% testing, 80% training).
    random_state (int): The seed used by the random number generator.
    
    Returns:
    DataFrame, DataFrame: The training and testing DataFrames.
    """
    # Separate the features and the target variable
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split the data into training and testing sets
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Combine the features and target variable back into DataFrames
    train_df = pd.concat([train_X, train_y], axis=1)
    test_df = pd.concat([test_X, test_y], axis=1)
    
    print(f"Data split into training (size={len(train_df)}) and testing (size={len(test_df)}) sets.")
    return train_df, test_df

if __name__ == "__main__":
    # Define the base path using the current file's directory
    base_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_path, '..'))
    
    # Define the path to the processed dataset with new features using a relative path
    data_path = os.path.join(project_root, 'data', 'processed', 'processed_dataset_with_features.csv')
    
    # Define the paths to save the split datasets using relative paths
    train_path_data = os.path.join(project_root, 'data', 'train', 'train_dataset.csv')
    test_path_data = os.path.join(project_root, 'data', 'test', 'test_dataset.csv')
    train_path_prep = os.path.join(project_root, 'Data_Preparation', 'training_sets', 'train_dataset.csv')
    test_path_prep = os.path.join(project_root, 'Data_Preparation', 'testing_sets', 'test_dataset.csv')
    
    # Load the processed dataset from the specified path
    df = load_data(data_path)
    
    # Check if the data was loaded successfully
    if df is not None:
        # Split the data into training (80%) and testing (20%) sets
        train_df, test_df = split_data(df, target_column='Churn_Yes')
        
        # Save the training and testing datasets to the specified paths
        train_df.to_csv(train_path_data, index=False)
        test_df.to_csv(test_path_data, index=False)
        train_df.to_csv(train_path_prep, index=False)
        test_df.to_csv(test_path_prep, index=False)
        
        print("Training and testing datasets created and saved successfully.")
