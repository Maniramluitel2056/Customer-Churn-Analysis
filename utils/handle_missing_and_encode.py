import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def handle_missing_data(df):
    """
    Handle missing data in the dataframe.
    
    Parameters:
    df (pd.DataFrame): The dataframe to process.
    
    Returns:
    pd.DataFrame: The dataframe with missing data handled.
    """
    imputer = SimpleImputer(strategy='mean')
    df_numeric = df.select_dtypes(include=['float64', 'int64'])
    df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)
    
    df[df_numeric.columns] = df_numeric_imputed
    print("Missing data handled by mean imputation.")
    return df

def encode_categorical_variables(df):
    """
    Encode categorical variables in the dataframe.
    
    Parameters:
    df (pd.DataFrame): The dataframe to process.
    
    Returns:
    pd.DataFrame: The dataframe with categorical variables encoded.
    """
    df_categorical = df.select_dtypes(include=['object'])
    if not df_categorical.empty:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_data = pd.DataFrame(encoder.fit_transform(df_categorical), columns=encoder.get_feature_names_out(df_categorical.columns))
        
        df = df.drop(columns=df_categorical.columns)
        df = pd.concat([df, encoded_data], axis=1)
        print(f"Categorical columns encoded: {list(df_categorical.columns)}")
    else:
        print("No categorical columns found to encode.")
    
    return df

def handle_missing_and_encode(df):
    """
    Handle missing data and encode categorical variables.
    
    Parameters:
    df (pd.DataFrame): The dataframe to process.
    
    Returns:
    pd.DataFrame: The processed dataframe.
    """
    df = handle_missing_data(df)
    df = encode_categorical_variables(df)
    return df
