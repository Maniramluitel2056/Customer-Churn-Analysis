import pandas as pd

print("Executing data_loader.py")  # Debugging print statement to indicate the script is running

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
        print(f"Data loaded successfully from {file_path}")  # Print statement added for confirmation
        return data
        # Returns: A DataFrame containing the data from the CSV file
    except FileNotFoundError:
        # Handle file not found error
        print(f"File not found at {file_path}. Please check the file path.")
        # Returns: None if the file is not found
    except pd.errors.EmptyDataError:
        # Handle empty data error
        print(f"No data found at {file_path}. The file is empty.")
        # Returns: None if the file is empty
    except pd.errors.ParserError:
        # Handle parsing error
        print(f"Error parsing data from {file_path}. Please check the file format.")
        # Returns: None if there is a parsing error
    except Exception as e:
        # Handle any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        # Returns: None if an unexpected error occurs


