import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_pairplots(df, hue):
    """
    Plot pair plots with a specified hue.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    hue (str): The column name to use for hue.
    """
    sns.pairplot(df, hue=hue)
    plt.show()
