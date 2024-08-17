import pandas as pd # type: ignore
from sklearn.preprocessing import StandardScaler, MinMaxScaler # type: ignore

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
    scaled_df = df.copy()
    scaled_df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
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
    scaled_df = df.copy()
    scaled_df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    print("Min-Max scaling applied.")  # Print statement added for confirmation
    # Returns: DataFrame with min-max scaled numeric columns
    return scaled_df
