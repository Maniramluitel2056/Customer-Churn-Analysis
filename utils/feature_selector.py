import pandas as pd # type: ignore
import os
import warnings

# Suppress the specific RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: Loaded data as a pandas DataFrame, or None if an error occurs.
    """
    try:
        # Attempt to read the data from the specified CSV file
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        # Handle any unexpected errors
        print(f"An error occurred: {e}")

def select_features(df, target_column):
    """
    Select features for the dataset without dropping any columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The target variable column name.

    Returns:
    pd.DataFrame: DataFrame with the selected features and the target column.
    """
    # Print initial feature statistics
    print("Initial feature statistics:")
    print(df.describe())

    # Separate the target column from the features
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Print target column statistics
    print(f"Target column '{target_column}' statistics:")
    print(y.describe())

    # Ensure the target variable has variance
    if y.nunique() <= 1:
        raise ValueError(f"The target variable '{target_column}' has zero variance.")

    # Ensure no NaN values are present
    if X.isna().any().any():
        print("NaN values found in features. Detailed information:")
        print(X.isna().sum())

    # Print the final DataFrame shape
    df_selected = df.copy()
    print(f"Final DataFrame shape with all features: {df_selected.shape}")
    return df_selected

if __name__ == "__main__":
    # Define paths using relative paths
    base_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_path, '..'))
    interim_cleaned_data_path = os.path.join(project_root, 'data', 'interim', 'cleaned_dataset.csv')

    # Load the interim cleaned data
    df_cleaned = load_data(interim_cleaned_data_path)

    # Check if the data was loaded successfully
    if df_cleaned is not None:
        # Select features based on the target column without dropping any columns
        df_selected = select_features(df_cleaned, target_column='Churn_Yes')

        print("Feature selection completed successfully.")
