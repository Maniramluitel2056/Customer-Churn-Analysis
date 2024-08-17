import pandas as pd # type: ignore
from sklearn.preprocessing import OneHotEncoder # type: ignore

def clean_data(df):
    """
    Clean the data by handling missing values and encoding categorical variables.
    
    Parameters:
    df (pd.DataFrame): The data to clean.
    
    Returns:
    pd.DataFrame: Cleaned data.
    """
    try:
        # Handle missing values by dropping rows with missing values
        df = df.dropna()
        print("Missing values handled by dropping rows with missing values.")
        # Returns: DataFrame with rows containing missing values removed
        
        # Identify and encode categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        print(f"Categorical columns identified: {categorical_columns}")
        if len(categorical_columns) > 0:
            # Initialize the OneHotEncoder
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            
            # Fit and transform the categorical columns
            encoded_data = pd.DataFrame(
                encoder.fit_transform(df[categorical_columns]),
                columns=encoder.get_feature_names_out(categorical_columns)
            )
            
            # Drop the original categorical columns from the DataFrame
            df = df.drop(columns=categorical_columns)
            
            # Concatenate the encoded data with the original DataFrame
            df = pd.concat([df, encoded_data], axis=1)
            print(f"Categorical columns {list(categorical_columns)} encoded.")
            # Returns: DataFrame with categorical columns encoded as one-hot vectors
        else:
            print("No categorical columns found to encode.")
            # Returns: The original DataFrame if no categorical columns are found
        
        return df
    except Exception as e:
        print(f"An error occurred during data cleaning: {e}")
        # Returns: None if an error occurs during data cleaning
