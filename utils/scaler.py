import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def apply_standard_scaling(df):
    """
    Apply Standard Scaling to the dataframe.

    Parameters:
    df (pd.DataFrame): The data to scale.

    Returns:
    pd.DataFrame: Scaled data.
    """
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return scaled_df

def apply_min_max_scaling(df):
    """
    Apply Min-Max Scaling to the dataframe.

    Parameters:
    df (pd.DataFrame): The data to scale.

    Returns:
    pd.DataFrame: Scaled data.
    """
    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return scaled_df
