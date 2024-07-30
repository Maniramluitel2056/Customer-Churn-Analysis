# Documentation for data_splitter.py Script
# Script Overview

This script is designed to load a dataset from a CSV file, split it into training and testing sets, and save these sets to specified paths. It includes functions for loading data and splitting the dataset, as well as handling various exceptions during these processes.

## Script Details

### Importing Libraries

```python
import pandas as pd
from sklearn.model_selection import train_test_split
import os
```

- `pandas`: Used for data manipulation and analysis.
- `train_test_split` from `sklearn.model_selection`: Used to split the dataset into training and testing sets.
- `os`: Used for interacting with the operating system, particularly for file path manipulations.

### Load Data Function

```python
def load_data(file_path):
    """
    Load data from a CSV file.
    
    Parameters:
    file_path (str): The path to the CSV file to be loaded.
    
    Returns:
    DataFrame: The data loaded from the CSV file, or None if an error occurs.
    """
    try:
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
```

#### Parameters:
- `file_path` (str): The path to the CSV file to be loaded.

#### Returns:
- `DataFrame`: The data loaded from the CSV file, or None if an error occurs.

#### Exception Handling:
- `FileNotFoundError`: Raised if the file is not found at the specified path.
- `pd.errors.EmptyDataError`: Raised if the file is empty.
- `pd.errors.ParserError`: Raised if there is an error parsing the file.
- `Exception`: Catches any other unexpected errors.

### Split Data Function

```python
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
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    train_df = pd.concat([train_X, train_y], axis=1)
    test_df = pd.concat([test_X, test_y], axis=1)
    
    print(f"Data split into training (size={len(train_df)}) and testing (size={len(test_df)}) sets.")
    return train_df, test_df
```

#### Parameters:
- `df` (DataFrame): The input DataFrame to be split.
- `target_column` (str): The target variable column name.
- `test_size` (float, optional): The proportion of the dataset to include in the test split. Default is 0.2 (20% testing, 80% training).
- `random_state` (int, optional): The seed used by the random number generator. Default is 42.

#### Returns:
- `DataFrame, DataFrame`: The training and testing DataFrames.

### Main Script Execution

```python
if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_path, '..'))
    
    data_path = os.path.join(project_root, 'data', 'processed', 'processed_dataset_with_features.csv')
    
    train_path_data = os.path.join(project_root, 'data', 'train', 'train_dataset.csv')
    test_path_data = os.path.join(project_root, 'data', 'test', 'test_dataset.csv')
    train_path_prep = os.path.join(project_root, 'Data_Preparation', 'training_sets', 'train_dataset.csv')
    test_path_prep = os.path.join(project_root, 'Data_Preparation', 'testing_sets', 'test_dataset.csv')
    
    df = load_data(data_path)
    
    if df is not None:
        train_df, test_df = split_data(df, target_column='Churn_Yes')
        
        train_df.to_csv(train_path_data, index=False)
        test_df.to_csv(test_path_data, index=False)
        train_df.to_csv(train_path_prep, index=False)
        test_df.to_csv(test_path_prep, index=False)
        
        print("Training and testing datasets created and saved successfully.")
```

#### Base Path and Project Root:
- `base_path` and `project_root` are defined to construct relative paths for loading and saving data files.

#### Data Paths:
- `data_path`: Path to the processed dataset.
- `train_path_data` and `test_path_data`: Paths to save the split datasets in the `data/train` and `data/test` directories.
- `train_path_prep` and `test_path_prep`: Paths to save the split datasets in the `Data_Preparation/training_sets` and `Data_Preparation/testing_sets` directories.

#### Load Data:
- `df = load_data(data_path)`: Loads the dataset from the specified path.

#### Split Data:
- If data is loaded successfully, it splits the data into training and testing sets using the `split_data` function.

#### Save Data:
- The split datasets are saved to the specified paths.

### Dataset Sizes and Composition

#### Training Set Size:
- The training set consists of 80% of the original dataset.
- `train_df` size is printed as part of the script execution.

#### Testing Set Size:
- The testing set consists of 20% of the original dataset.
- `test_df` size is printed as part of the script execution.

#### Dataset Composition:
- The dataset is split in a stratified manner based on the `target_column`, ensuring the target variable distribution is consistent across the training and testing sets.

### Conclusion

This script is essential for preparing data for machine learning models by splitting the dataset into training and testing sets, ensuring that the data is correctly formatted and accessible for further analysis and model training. This detailed documentation should help users understand each component of the script and how to use it effectively. The information on dataset sizes and composition ensures clarity on the resulting data splits.