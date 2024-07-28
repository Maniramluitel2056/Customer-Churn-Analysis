import json
import pandas as pd
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")


# Add the utils and scripts directories to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

# Load configuration file
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)
    # Returns: A dictionary `config` containing the configuration settings

# Convert relative paths in the configuration file to absolute paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_data_path = os.path.join(project_root, config['raw_data_path'])
interim_cleaned_data_path = os.path.join(project_root, config['interim_cleaned_data_path'])
preprocessed_data_path = os.path.join(project_root, config['preprocessed_data_path'])

# Paths for saving training and testing datasets
train_path = os.path.join(project_root, 'data', 'train', 'train_dataset.csv')
test_path = os.path.join(project_root, 'data', 'test', 'test_dataset.csv')

# Path for saving the processed dataset with new features
processed_data_path = os.path.join(project_root, 'data', 'processed', 'processed_dataset_with_features.csv')

# Import custom modules from utils and scripts directories
from data_loader import load_data
from data_cleaner import clean_data
from data_splitter import split_data
from feature_engineering import create_new_features
from eda_utils import plot_distribution, plot_boxplots, plot_categorical, plot_correlation_matrix, plot_pairplots

def perform_eda(df):
    """
    Perform exploratory data analysis on the dataframe.
    """
    # Plot distributions
    plot_distribution(df, df.columns)

    # Plot boxplots
    plot_boxplots(df, df.columns)

    # Plot categorical variables
    categorical_columns = ['gender_Female', 'gender_Male', 'Dependents_No', 'Dependents_Yes',
                           'PhoneService_No', 'PhoneService_Yes', 'MultipleLines_No', 'MultipleLines_Yes',
                           'InternetService_DSL', 'InternetService_Fiber optic', 'Contract_Month-to-month',
                           'Contract_One year', 'Contract_Two year', 'Churn_No', 'Churn_Yes']
    plot_categorical(df, categorical_columns)

    # Plot correlation matrix
    plot_correlation_matrix(df)

    # Plot pairplots
    plot_pairplots(df, hue='Churn_Yes')

def main():
    """
    Main function to load, clean, create features, and split the dataset.
    """
    # Load the raw data from the specified path
    df = load_data(raw_data_path)
    # Check if data is loaded correctly; if not, print an error message and exit
    if df is None:
        print(f"File not found at {raw_data_path}. Exiting the script.")
        return
    # At this point, df contains the raw data loaded from raw_data_path

    # Clean the loaded data using the clean_data function
    df_cleaned = clean_data(df)
    # Check if data cleaning was successful; if not, print an error message and exit
    if df_cleaned is None:
        print("Data cleaning failed. Exiting the script.")
        return
    # At this point, df_cleaned contains the cleaned data

    # Perform EDA before feature engineering
    perform_eda(df_cleaned)

    # Create new features from the cleaned dataset
    df_features = create_new_features(df_cleaned)
    # Save the dataset with new features to the processed path
    df_features.to_csv(processed_data_path, index=False)
    print(f"Dataset with new features saved to {processed_data_path}")

    # Split the processed data with new features into training and testing datasets
    train_df, test_df = split_data(df_features)
    # Save the training and testing datasets to their respective paths
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Training and testing datasets saved to {train_path} and {test_path}")

if __name__ == "__main__":
    main()
