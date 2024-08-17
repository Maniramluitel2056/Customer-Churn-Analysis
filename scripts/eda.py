import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import json
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# Create directory for saving EDA figures inside Data_Preparation
eda_figures_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Data_Preparation', 'eda_visualizations')
os.makedirs(eda_figures_path, exist_ok=True)

# Load the dataset
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', config['raw_data_path']))

# Data Summary
print(df.info())
print(df.head())

# Descriptive Statistics
print(df.describe())
print(df.describe(include=['object']))

# Target Variable Analysis (Churn)
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='Churn')
plt.title('Distribution of Churn')
plt.savefig(os.path.join(eda_figures_path, 'churn_distribution.png'))  # Save the plot as an image
plt.show()

# Distribution Plots
df.hist(bins=30, figsize=(20, 15))
plt.savefig(os.path.join(eda_figures_path, 'numerical_distributions.png'))  # Save the plot as an image
plt.show()

# Box Plots
plt.figure(figsize=(20, 15))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.savefig(os.path.join(eda_figures_path, 'box_plots.png'))  # Save the plot as an image
plt.show()

# Categorical Feature Analysis
for column in df.select_dtypes(include=['object']).columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x=column)
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(eda_figures_path, f'{column}_distribution.png'))  # Save the plot as an image
    plt.show()

# Correlation Matrix (only for numerical columns)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.savefig(os.path.join(eda_figures_path, 'correlation_matrix.png'))  # Save the plot as an image
plt.show()

# Missing Values Analysis
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_data = pd.DataFrame({'Total Missing': missing_values, 'Percentage': missing_percentage})
print(missing_data)

# Outlier Detection
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x=column)
    plt.savefig(os.path.join(eda_figures_path, f'{column}_outliers.png'))  # Save the plot as an image
    plt.show()

# Pair Plots with Churn (only for numerical columns)
sns.pairplot(df, hue='Churn', vars=numeric_cols)
plt.savefig(os.path.join(eda_figures_path, 'pair_plots.png'))  # Save the plot as an image
plt.show()

# Group Analysis
churn_summary = df.groupby('Churn')[numeric_cols].mean()
print(churn_summary)
