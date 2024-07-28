import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from handle_missing_and_encode import handle_missing_and_encode

print("Executing scaler.py")  # Debugging print statement to indicate the script is running

def apply_standard_scaling(df):
    """
    Apply standard scaling to numeric columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame with numeric columns to be scaled.

    Returns:
    pd.DataFrame: DataFrame with standard scaled numeric columns.
    """
    # Select numeric columns for scaling
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    print(f"Numeric columns for scaling: {numeric_cols}")  # Print numeric columns
    
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # Apply the scaler to the numeric columns
    scaled_df = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
    
    print("Standard scaling applied.")  # Print statement added for confirmation
    # Returns: DataFrame with standard scaled numeric columns
    return scaled_df

def apply_min_max_scaling(df):
    """
    Apply min-max scaling to numeric columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame with numeric columns to be scaled.

    Returns:
    pd.DataFrame: DataFrame with min-max scaled numeric columns.
    """
    # Select numeric columns for scaling
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    print(f"Numeric columns for scaling: {numeric_cols}")  # Print numeric columns
    
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    
    # Apply the scaler to the numeric columns
    scaled_df = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
    
    print("Min-Max scaling applied.")  # Print statement added for confirmation
    # Returns: DataFrame with min-max scaled numeric columns
    return scaled_df

if __name__ == "__main__":
    # Define the path to the raw data
    raw_data_path = 'D:/Customer-Churn-Analysis/data/raw/Dataset (ATS)-1.csv'
    
    # Load the raw data from the specified path
    df_raw = pd.read_csv(raw_data_path)
    print("Data loaded for scaling.")  # Confirmation print statement
    print(f"Raw data:\n{df_raw.head()}")  # Print the first few rows of the raw data

    # Preprocess data before scaling (handle missing values and encode categorical variables)
    df_cleaned = handle_missing_and_encode(df_raw)
    print("Data cleaned for scaling.")  # Confirmation print statement
    print(f"Cleaned data:\n{df_cleaned.head()}")  # Print the first few rows of the cleaned data
    print(f"Data types after cleaning:\n{df_cleaned.dtypes}")  # Print data types to confirm numeric columns

    # Apply standard scaling to numeric columns
    df_standard_scaled = apply_standard_scaling(df_cleaned)
    print("Standard scaled data:\n", df_standard_scaled.head())  # Print the first few rows of the standard scaled data

    # Apply min-max scaling to numeric columns
    df_min_max_scaled = apply_min_max_scaling(df_cleaned)
    print("Min-Max scaled data:\n", df_min_max_scaled.head())  # Print the first few rows of the min-max scaled data
