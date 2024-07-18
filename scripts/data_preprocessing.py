# scripts/data_preprocessing.py

import sys
import os

# Ensure the utils module can be found
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from data_loader import load_data
from data_cleaner import clean_data

def main():
    """
    Main function to load, clean, and save the dataset.
    """
    # Define file paths
    raw_data_path = 'D:/CustomerChurnAnalysis/data/raw/Dataset (ATS)-1.csv'  # Adjust the filename as necessary
    processed_data_path = 'D:/CustomerChurnAnalysis/data/processed/dataset.csv'

    # Load the raw data
    df = load_data(raw_data_path)
    if df is None:
        print("Data loading failed. Exiting the script.")
        return

    # Clean the loaded data
    df = clean_data(df)
    if df is None:
        print("Data cleaning failed. Exiting the script.")
        return

    # Save the cleaned data to the processed data directory
    try:
        df.to_csv(processed_data_path, index=False)
        print(f"Cleaned data saved successfully to {processed_data_path}")
    except Exception as e:
        print(f"Failed to save cleaned data: {e}")

if __name__ == "__main__":
    main()
