# Exploratory Data Analysis 
**eda.py Documentation**

This script performs Exploratory Data Analysis (EDA) on the customer churn dataset. It includes functions for loading the dataset, plotting various distributions, and visualizing correlations.

## Table of Contents
1. [Import Libraries and Modules](#import-libraries-and-modules)
2. [Load Configuration and Dataset](#load-configuration-and-dataset)
3. [Functions](#functions)
    - [load_data](#load_data)
    - [plot_distribution](#plot_distribution)
    - [plot_boxplots](#plot_boxplots)
    - [plot_categorical](#plot_categorical)
    - [plot_correlation_matrix](#plot_correlation_matrix)
    - [plot_pairplots](#plot_pairplots)
4. [Main Function](#main-function)

## Import Libraries and Modules

```python
import json
import pandas as pd
import sys
import os

# Add the utils and scripts directories to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
```

This section imports the necessary libraries and adds the utils and scripts directories to the Python path to enable importing custom modules.


## Load Configuration and Dataset

```python
# Load configuration file
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.json')
with open(config_path, 'r') as f:
     config = json.load(f)

# Convert relative paths in the configuration file to absolute paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_data_path = os.path.join(project_root, config['raw_data_path'])

```
This section loads the configuration settings from config.json and sets up the paths for the dataset.

## Functions
 - load_data

```python
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
          print(f"File not found at {file_path}")
     except pd.errors.EmptyDataError:
          print(f"No data found at {file_path}")
     except pd.errors.ParserError:
          print(f"Error parsing data from {file_path}")
     except Exception as e:
          print(f"An error occurred: {e}")

```

This function loads the dataset from a CSV file and handles potential errors that may occur during the loading process. It returns the loaded data as a pandas DataFrame.

 - plot_distribution

 ```python
def plot_distribution(df, columns, bins=30):
     """
     Plot histograms for given columns.
     
     Parameters:
     df (pd.DataFrame): The DataFrame containing the data.
     columns (list): List of column names to plot.
     bins (int): Number of bins for the histogram.
     """
     df[columns].hist(bins=bins, figsize=(20, 15))
     plt.show()

 ```

This function plots histograms for the specified columns in the DataFrame. It is used to visualize the distribution of the data.

 - plot_boxplots

```python
def plot_boxplots(df, columns):
     """
     Plot box plots for given columns.
     
     Parameters:
     df (pd.DataFrame): The DataFrame containing the data.
     columns (list): List of column names to plot.
     """
     plt.figure(figsize=(20, 15))
     sns.boxplot(data=df[columns])
     plt.xticks(rotation=90)
     plt.show()

```
This function creates box plots for the specified columns in the DataFrame. It is used to visualize the spread and identify outliers in the data.


 - plot_categorical
```python
def plot_categorical(df, columns):
     """
     Plot bar plots for categorical columns.
     
     Parameters:
     df (pd.DataFrame): The DataFrame containing the data.
     columns (list): List of categorical column names to plot.
     """
     for column in columns:
          plt.figure(figsize=(10, 5))
          sns.countplot(data=df, x=column)
          plt.xticks(rotation=90)
          plt.show()
```

This function plots bar plots for the specified categorical columns in the DataFrame. It helps visualize the frequency distribution of categorical variables.

 - plot_correlation_matrix

```python

def plot_correlation_matrix(df):
     """
     Plot the correlation matrix.
     
     Parameters:
     df (pd.DataFrame): The DataFrame containing the data.
     """
     corr_matrix = df.corr()
     plt.figure(figsize=(15, 10))
     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
     plt.show()

```

This function plots the correlation matrix of the DataFrame, helping identify relationships between different variables.


 - plot_pairplots

```python
 
def plot_pairplots(df, hue):
     """
     Plot pair plots with a specified hue.
     
     Parameters:
     df (pd.DataFrame): The DataFrame containing the data.
     hue (str): The column name to use for hue.
     """
     sns.pairplot(df, hue=hue)
     plt.show()

```

This function creates pair plots for the DataFrame, using a specified column for hue. It is useful for visualizing the relationships between multiple variables.

## Main Function

The main function orchestrates the EDA process by loading the data, cleaning it, creating new features, and performing various visualizations.

```python
 
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

     # Create new features from the cleaned dataset
     df_features = create_new_features(df_cleaned)
     # Save the dataset with new features to the processed path
     df_features.to_csv(processed_data_path, index=False)
     print(f"Dataset with new features saved to {processed_data_path}")

     # Perform EDA
     perform_eda(df_features)

```

This function loads the dataset, cleans it, creates new features, and saves the processed data. It then calls the `perform_eda` function to perform exploratory data analysis.

```python

if __name__ == "__main__":
     main()
```

