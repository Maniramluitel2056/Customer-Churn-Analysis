# utils/data_loader.py

import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: Loaded data as a pandas DataFrame.
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
