import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
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
        # Returns: A DataFrame containing the data from the CSV file
    except Exception as e:
        # Handle any unexpected errors
        print(f"An error occurred: {e}")
        # Returns: None if an error occurs

def select_features(df, target_column, k=10, variance_threshold=0.01):
    """
    Select the top k features based on the target variable.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The target variable column name.
    k (int): The number of top features to select.
    variance_threshold (float): The threshold for variance to filter low variance features.

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

    # Identify and remove low variance features
    print("Applying VarianceThreshold...")
    selector = VarianceThreshold(threshold=variance_threshold)
    X_reduced = selector.fit_transform(X)
    features_kept = X.columns[selector.get_support()]
    X = pd.DataFrame(X_reduced, columns=features_kept)

    # Print feature statistics after removing low variance features
    print("Feature statistics after removing low variance features:")
    print(X.describe())

    # Ensure the target variable has variance
    if y.nunique() <= 1:
        raise ValueError(f"The target variable '{target_column}' has zero variance.")

    # Ensure no NaN values are present
    if X.isna().any().any():
        print("NaN values found in features. Detailed information:")
        print(X.isna().sum())

    # Apply SelectKBest to select top k features
    print("Applying SelectKBest...")
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)

    selected_columns = X.columns[selector.get_support()]
    print(f"Selected columns: {selected_columns}")
    df_selected = X[selected_columns].copy()
    df_selected[target_column] = y.reset_index(drop=True)  # Add the target column back

    print(f"Final selected features DataFrame shape: {df_selected.shape}")
    return df_selected
    # Returns: DataFrame with the selected features and the target column

if __name__ == "__main__":
    # Define paths using relative paths
    base_path = os.path.dirname(os.path.abspath(__file__))
    train_data_path = os.path.join(base_path, '..', 'data', 'train', 'train_dataset_transformed.csv')
    train_path_data = os.path.join(base_path, '..', 'data', 'processed', 'train_dataset_selected.csv')
    train_path_prep = os.path.join(base_path, '..', 'Data_Preparation', 'training_sets', 'train_dataset_selected.csv')
    # Returns: Absolute paths for the transformed training dataset and the paths to save the selected features dataset

    # Load the transformed training data
    df_train = load_data(train_data_path)
    # Returns: A DataFrame containing the transformed training data, or None if loading fails

    # Check if the data was loaded successfully
    if df_train is not None:
        # Select features based on the target column
        df_train_selected = select_features(df_train, target_column='Churn_1', k=10)
        # Returns: DataFrame with the selected features and the target column

        # Save the selected training dataset to the specified paths
        df_train_selected.to_csv(train_path_data, index=False)
        df_train_selected.to_csv(train_path_prep, index=False)
        # Actions: The selected features training dataset is saved to the specified paths

        print("Selected features training dataset saved successfully.")
        # Prints a confirmation message indicating successful saving
