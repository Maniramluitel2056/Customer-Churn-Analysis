import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

print("Executing handle_missing_and_encode.py")  # Debugging print statement to indicate the script is running

def handle_missing_data(df):
    """
    Handle missing data by imputing mean values for numeric columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame with potential missing values.

    Returns:
    pd.DataFrame: DataFrame with missing values handled.
    """
    # Initialize the SimpleImputer with mean strategy
    imputer = SimpleImputer(strategy='mean')
    
    # Select only numeric columns from the DataFrame
    df_numeric = df.select_dtypes(include=['float64', 'int64'])
    
    # Apply the imputer to the numeric columns
    df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)
    
    # Update the original DataFrame with the imputed values
    df[df_numeric.columns] = df_numeric_imputed
    
    print("Missing data handled by mean imputation.")  # Print statement added for confirmation
    # Returns: DataFrame with missing values imputed for numeric columns
    return df

def encode_categorical_variables(df):
    """
    Encode categorical variables using OneHotEncoder.

    Parameters:
    df (pd.DataFrame): The input DataFrame with potential categorical variables.

    Returns:
    pd.DataFrame: DataFrame with categorical variables encoded.
    """
    # Select only categorical columns from the DataFrame
    df_categorical = df.select_dtypes(include=['object'])
    
    # Check if there are any categorical columns to encode
    if not df_categorical.empty:
        # Initialize the OneHotEncoder
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
        # Apply the encoder to the categorical columns
        encoded_data = pd.DataFrame(
            encoder.fit_transform(df_categorical),
            columns=encoder.get_feature_names_out(df_categorical.columns)
        )
        
        # Drop the original categorical columns from the DataFrame
        df = df.drop(columns=df_categorical.columns)
        
        # Concatenate the encoded data with the original DataFrame
        df = pd.concat([df, encoded_data], axis=1)
        
        print(f"Categorical columns encoded: {list(df_categorical.columns)}")  # Print statement added for confirmation
        # Returns: DataFrame with categorical columns encoded as one-hot vectors
    else:
        print("No categorical columns found to encode.")  # Print statement added for confirmation
        # Returns: The original DataFrame if no categorical columns are found
    return df

def handle_missing_and_encode(df):
    """
    Handle missing values and encode categorical variables in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame with potential missing values and categorical variables.

    Returns:
    pd.DataFrame: DataFrame with missing values handled and categorical variables encoded.
    """
    # Handle missing data by imputing mean values for numeric columns
    df = handle_missing_data(df)
    
    # Encode categorical variables using OneHotEncoder
    df = encode_categorical_variables(df)
    
    print("Data after handling missing values and encoding:\n", df.head())  # Print statement for verification
    # Returns: DataFrame with missing values handled and categorical variables encoded
    return df


