import json
import pandas as pd
import sys
import os

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.json')
print(f"Config path: {config_path}")
with open(config_path, 'r') as f:
    config = json.load(f)

# Convert relative paths to absolute paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

raw_data_path = os.path.join(project_root, config['raw_data_path'])
interim_cleaned_data_path = os.path.join(project_root, config['interim_cleaned_data_path'])
preprocessed_data_path = os.path.join(project_root, config['preprocessed_data_path'])

print(f"Raw data path (absolute): {raw_data_path}")
print(f"Interim cleaned data path (absolute): {interim_cleaned_data_path}")
print(f"Preprocessed data path (absolute): {preprocessed_data_path}")

# Ensure the utils module can be found
sys.path.append(os.path.join(project_root, 'utils'))

# Import custom modules
from data_loader import load_data
from data_cleaner import clean_data

def main():
    """
    Main function to load, clean, and save the dataset.
    """
    # Print the current working directory
    print(f"Current working directory: {os.getcwd()}")

    # Check if the raw data file exists
    if not os.path.exists(raw_data_path):
        print(f"File does not exist at {raw_data_path}")
        print("Data loading failed. Exiting the script.")
        return

    # Load the raw data
    print(f"Attempting to load raw data from: {raw_data_path}")
    df = load_data(raw_data_path)
    if df is None:
        print(f"File not found at {raw_data_path}. Please check the file path.")
        print("Data loading failed. Exiting the script.")
        return

    # Clean the loaded data
    df_cleaned = clean_data(df)
    if df_cleaned is None:
        print("Data cleaning failed. Exiting the script.")
        return

    # Save the cleaned dataset to interim
    df_cleaned.to_csv(interim_cleaned_data_path, index=False)
    # Save the cleaned dataset to preprocessed_dataset
    df_cleaned.to_csv(preprocessed_data_path, index=False)
    print(f"Cleaned data saved to interim at {interim_cleaned_data_path}")
    print(f"Cleaned data saved to preprocessed_dataset at {preprocessed_data_path}")

if __name__ == "__main__":
    main()
