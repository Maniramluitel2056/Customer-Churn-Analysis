import json
import pandas as pd
import sys
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
standard_scaled_data_path = os.path.join(project_root, 'data_preparation/scaling_techniques/standard_scaled_dataset.csv')
min_max_scaled_data_path = os.path.join(project_root, 'data_preparation/scaling_techniques/min_max_scaled_dataset.csv')

print(f"Raw data path (absolute): {raw_data_path}")
print(f"Interim cleaned data path (absolute): {interim_cleaned_data_path}")
print(f"Preprocessed data path (absolute): {preprocessed_data_path}")
print(f"Standard scaled data path (absolute): {standard_scaled_data_path}")
print(f"Min-Max scaled data path (absolute): {min_max_scaled_data_path}")

# Ensure the utils module can be found
sys.path.append(os.path.join(project_root, 'utils'))

# Import custom modules
from data_loader import load_data
from data_cleaner import clean_data
from handle_missing_and_encode import handle_missing_and_encode
from scaler import apply_standard_scaling, apply_min_max_scaling

def main():
    """
    Main function to load, clean, scale, and save the dataset.
    """
    # Print the current working directory
    print(f"Current working directory: {os.getcwd()}")

    # Check if the raw data file exists
    if not os.path.exists(raw_data_path):
        print(f"File does not exist at {raw_data_path}")
        print("Data loading failed. Exiting the script.")
        return
    # Returns: Exits the script if the raw data file does not exist

    # Load the raw data
    print(f"Attempting to load raw data from: {raw_data_path}")
    df = load_data(raw_data_path)
    if df is None:
        print(f"File not found at {raw_data_path}. Please check the file path.")
        print("Data loading failed. Exiting the script.")
        return
    # Returns: A DataFrame `df` containing the raw data, or exits the script if loading fails

    # Clean the loaded data
    df_cleaned = handle_missing_and_encode(df)
    if df_cleaned is None:
        print("Data cleaning failed. Exiting the script.")
        return
    # Returns: A DataFrame `df_cleaned` containing the cleaned data, or exits the script if cleaning fails

    # Apply scaling techniques
    df_standard_scaled = apply_standard_scaling(df_cleaned)
    df_min_max_scaled = apply_min_max_scaling(df_cleaned)
    # Returns: Two DataFrames - `df_standard_scaled` containing the standard scaled data and `df_min_max_scaled` containing the min-max scaled data

    # Save the cleaned and scaled datasets
    df_cleaned.to_csv(interim_cleaned_data_path, index=False)
    df_cleaned.to_csv(preprocessed_data_path, index=False)
    df_standard_scaled.to_csv(standard_scaled_data_path, index=False)
    df_min_max_scaled.to_csv(min_max_scaled_data_path, index=False)
    # Actions: Saves the cleaned data to interim and preprocessed paths, and the scaled data to their respective paths

    # Print confirmation messages indicating successful saving
    print(f"Cleaned data saved to interim at {interim_cleaned_data_path}")
    print(f"Cleaned data saved to preprocessed at {preprocessed_data_path}")
    print(f"Standard scaled data saved at {standard_scaled_data_path}")
    print(f"Min-Max scaled data saved at {min_max_scaled_data_path}")

if __name__ == "__main__":
    main()
