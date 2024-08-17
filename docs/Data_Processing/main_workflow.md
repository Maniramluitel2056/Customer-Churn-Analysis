# Main Workflow
## Overview

This document provides an overview of the entire data processing and feature engineering workflow. It details each major step, linking to the relevant scripts and documentation for a comprehensive understanding of the pipeline.

## Workflow Steps

1. **Data Loading**
    - Script: `scripts/data_loader.py`
    - Description: This step involves loading the raw data from a CSV file. The data is read into a pandas DataFrame for further processing.

```python
import pandas as pd

def load_data(file_path):
     try:
          data = pd.read_csv(file_path)
          return data
     except Exception as e:
          print(f"Error loading data: {e}")
```

2. **Data Cleaning**
    - Script: `scripts/data_cleaner.py`
    - Description: This step handles missing values and encodes categorical variables. Missing values are typically handled by mean imputation or dropping rows, while categorical variables are encoded using one-hot encoding.

```python
import pandas as pd
from category_encoders import OneHotEncoder

def clean_data(df):
     df = df.dropna()
     categorical_columns = df.select_dtypes(include=['object']).columns
     if len(categorical_columns) > 0:
          encoder = OneHotEncoder(cols=categorical_columns)
          df = encoder.fit_transform(df)
     return df
```

3. **Handling Missing Data and Encoding**
    - Script: `scripts/handle_missing_and_encode.py`
    - Description: This step involves handling any remaining missing values and encoding categorical variables using more advanced methods.

```python
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def handle_missing_data(df):
     imputer = SimpleImputer(strategy='mean')
     df_numeric = df.select_dtypes(include=['float64', 'int64'])
     df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)
     df[df_numeric.columns] = df_numeric_imputed
     return df

def encode_categorical_variables(df):
     df_categorical = df.select_dtypes(include=['object'])
     if not df_categorical.empty:
          encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
          encoded_data = pd.DataFrame(encoder.fit_transform(df_categorical), columns=encoder.get_feature_names_out(df_categorical.columns))
          df = df.drop(columns=df_categorical.columns)
          df = pd.concat([df, encoded_data], axis=1)
     return df
```

4. **Feature Scaling and Normalization**
    - Script: `scripts/scaler.py`
    - Description: This step applies standard scaling and min-max scaling to the data to ensure that all features are on a comparable scale.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def apply_standard_scaling(df):
     numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
     scaler = StandardScaler()
     scaled_df = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
     return scaled_df

def apply_min_max_scaling(df):
     numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
     scaler = MinMaxScaler()
     scaled_df = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
     return scaled_df
```

5. **Data Splitting**
    - Script: `scripts/data_splitter.py`
    - Description: This step splits the cleaned and scaled data into training and testing sets for model development.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, test_size=0.2, random_state=42):
     train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
     return train_df, test_df
```

6. **Feature Engineering**
    - Script: `scripts/feature_engineering.py`
    - Description: This step involves creating new features to enhance the dataset's predictive power.

```python
import pandas as pd

def create_new_features(df):
     if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
          df['Charges_Per_Tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
     if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
          df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']
     return df
```

7. **Feature Selection**
    - Script: `scripts/feature_selector.py`
    - Description: This step selects the top features based on their importance to the target variable.

```python
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

def select_features(df, target_column, k=10):
     y = df[target_column]
     X = df.drop(columns=[target_column])
     selector = SelectKBest(score_func=f_classif, k=k)
     selector.fit(X, y)
     selected_columns = X.columns[selector.get_support()]
     df_selected = X[selected_columns].copy()
     df_selected[target_column] = y.reset_index(drop=True)
     return df_selected
```

8. **Exploratory Data Analysis (EDA)**
    - Script: `scripts/eda.py`
    - Description: This step performs exploratory data analysis to gain insights into the data and visualize important patterns and relationships.

```python 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
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
    df[columns].hist(bins=bins, figsize=(20, 15))
    plt.show()

def plot_boxplots(df, columns):
    plt.figure(figsize=(20, 15))
    sns.boxplot(data=df[columns])
    plt.xticks(rotation=90)
    plt.show()

def plot_categorical(df, columns):
    for column in columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x=column)
        plt.xticks(rotation=90)
        plt.show()

def plot_correlation_matrix(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()

def plot_pairplots(df, hue):
    sns.pairplot(df, hue=hue)
    plt.show()
```

8. **Main Orchestration**
    - Script: `scripts/main.py`
    - Description: This script orchestrates the entire data processing pipeline, calling each of the steps in sequence.

```python
import os
import sys
import json
import pandas as pd

def load_config(config_path='../config.json'):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def ensure_directories(paths):
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)

if __name__ == "__main__":
    config = load_config()
    data = run_data_loader(config)
    cleaned_data = run_data_cleaner(config)
    train_df, test_df = run_data_splitter(config)
    data_with_features = run_feature_engineering(config)
    selected_data = run_feature_selection(config)
    validate_data_integrity(config)

```

## Summary

This document outlines the detailed steps involved in the data processing, feature engineering, and exploratory data analysis pipeline. Each script and notebook plays a crucial role in transforming raw data into a cleaned, processed, and feature-rich dataset ready for machine learning model development. Ensure that all configurations and environment setups are correctly followed to achieve a smooth workflow execution.

# Clustering Analysis and Visualization: Main Workflow

## Overview
This document provides an overview of the entire clustering analysis process, including data preparation, the application of the K-means algorithm, determination of the optimal number of clusters using the Elbow Method and Silhouette Analysis, and comprehensive visualization of the results. Each step is linked to relevant scripts for a complete understanding of the workflow.

## Environment Setup
Description: Import necessary libraries and configure paths to ensure utility modules are accessible.

```python
import os
import sys
import json
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths for utility modules
notebook_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(notebook_dir, '..'))
utils_path = os.path.join(project_root, 'utils')

if utils_path not in sys.path:
    sys.path.append(utils_path)
```

### Step 3: Load Configuration and Datasets
### Script: scripts/clustering_analysis.py
Description: Load the preprocessed datasets, including both Min-Max scaled and Standard scaled data, for clustering analysis.

```python
# Load configuration and datasets
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

min_max_scaled_data_path = os.path.join(project_root, 'data_preparation/scaling_techniques/min_max_scaled_dataset.csv')
standard_scaled_data_path = os.path.join(project_root, 'data_preparation/scaling_techniques/standard_scaled_dataset.csv')

df_min_max_scaled = pd.read_csv(min_max_scaled_data_path)
df_standard_scaled = pd.read_csv(standard_scaled_data_path)
```

### Step 4: Applying K-means Clustering
### Script: scripts/clustering_analysis.py
Description: Apply K-means clustering to segment customers into groups based on tenure and monthly charges. For this demonstration, we assume 3 clusters. The following code applies the K-means algorithm to the dataset.

```python
def apply_kmeans_and_visualize(df, scaling_label, n_clusters=3):
    features = df[['tenure', 'MonthlyCharges']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(features)
```

### Step 5: Determining the Optimal Number of Clusters
### Script: scripts/clustering_analysis.py
Description: Identify the optimal number of clusters using the Elbow Method and Silhouette Analysis.

```python
# elbow method
def determine_optimal_clusters(df, scaling_label):
    features = df[['tenure', 'MonthlyCharges']]
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title(f'Elbow Method for {scaling_label}')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.savefig(f'elbow_method_{scaling_label.lower().replace(" ", "_")}.png')
    plt.close()

determine_optimal_clusters(df_min_max_scaled, 'Min-Max Scaled')
determine_optimal_clusters(df_standard_scaled, 'Standard Scaled')

# Silhouette Analysis
from sklearn.metrics import silhouette_score

def silhouette_analysis(df, scaling_label):
    features = df[['tenure', 'MonthlyCharges']]
    silhouette_scores = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(features)
        score = silhouette_score(features, kmeans.labels_)
        silhouette_scores.append(score)

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.title(f'Silhouette Analysis for {scaling_label}')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.savefig(f'silhouette_analysis_{scaling_label.lower().replace(" ", "_")}.png')
    plt.close()
```

### Step 6: Training the Final Model
### Script: scripts/clustering_analysis.py
Description: Train the K-means model using the identified optimal number of clusters and analyze the results.

```python
def fit_kmeans_and_analyze(df, scaling_label, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['tenure', 'MonthlyCharges']])
    
    kmeans_model_path = os.path.join(project_root, 'Clustering_Analysis', 'kmeans_model')
    os.makedirs(kmeans_model_path, exist_ok=True)
    
    result_filename = f'{scaling_label.lower().replace(" ", "_")}_4_clusters.csv'
    result_filepath = os.path.join(kmeans_model_path, result_filename)
    df.to_csv(result_filepath, index=False)
    print(f'Saved cluster assignments to: {result_filepath}')
```

### Step 7: Visualization of Clustering Results
### Script: utils/visualizations.py
Description: Description: Generate visualizations using the datasets that include the cluster assignments for 4 clusters. The visualizations are based on the saved CSV files from the previous step

```python
# Load configuration file to get the paths for the clustered data
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
min_max_clusters_path = to_absolute_path(config['min-max_scaled_4_clusters_path'], project_root)
standard_clusters_path = to_absolute_path(config['standard_scaled_4_clusters_path'], project_root)

def plot_cluster_scatter(df, scaling_label):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='tenure', y='MonthlyCharges', hue='Cluster', palette='viridis')
    plt.title(f'{scaling_label} - Clusters')
    plt.savefig(f'{scaling_label.lower().replace(" ", "_")}_clusters_scatter.png')
    plt.close()

    def plot_cluster_boxplots(df, scaling_label):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y='MonthlyCharges', data=df, palette='viridis')
    plt.title(f'{scaling_label} - Monthly Charges by Cluster')
    plt.savefig(f'{scaling_label.lower().replace(" ", "_")}_boxplot_charges.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y='tenure', data=df, palette='viridis')
    plt.title(f'{scaling_label} - Tenure by Cluster')
    plt.savefig(f'{scaling_label.lower().replace(" ", "_")}_boxplot_tenure.png')
    plt.close()

    def plot_cluster_distribution(df, scaling_label):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Cluster', data=df, palette='viridis')
    plt.title(f'{scaling_label} - Cluster Distribution')
    plt.savefig(f'{scaling_label.lower().replace(" ", "_")}_distribution.png')
    plt.close()

    def plot_cluster_heatmap(df, scaling_label):
    cluster_summary = df.groupby('Cluster').agg({
        'tenure': 'mean',
        'MonthlyCharges': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_summary.set_index('Cluster').T, annot=True, cmap='viridis')
    plt.title(f'{scaling_label} - Cluster Heatmap')
    plt.savefig(f'{scaling_label.lower().replace(" ", "_")}_heatmap.png')
    plt.close()
```

### Step 8: Apply Visualizations to Both Min-Max and Standard Scaled Datasets
```python
# Load the clustered datasets with 4 clusters
df_min_max_clusters = pd.read_csv(min_max_clusters_path)
df_standard_clusters = pd.read_csv(standard_clusters_path)

# Define the paths where visualizations will be saved
visualizations_path = os.path.join(project_root, 'Clustering_Analysis', 'visualizations')

# Generate visualizations for Min-Max scaled clusters
plot_cluster_scatter(df_min_max_clusters, 'Min-Max Scaled', visualizations_path, n_clusters=4)
plot_cluster_boxplots(df_min_max_clusters, 'Min-Max Scaled', visualizations_path)
plot_cluster_distribution(df_min_max_clusters, 'Min-Max Scaled', visualizations_path)
plot_cluster_heatmap(df_min_max_clusters, 'Min-Max Scaled', visualizations_path)

# Generate visualizations for Standard scaled clusters
plot_cluster_scatter(df_standard_clusters, 'Standard Scaled', visualizations_path, n_clusters=4)
plot_cluster_boxplots(df_standard_clusters, 'Standard Scaled', visualizations_path)
plot_cluster_distribution(df_standard_clusters, 'Standard Scaled', visualizations_path)
plot_cluster_heatmap(df_standard_clusters, 'Standard Scaled', visualizations_path)
```

### Step 9: Summary
This document outlines the detailed steps involved in the clustering analysis and visualization process, from environment setup, data preparation, applying the clustering algorithm, determining the optimal number of clusters using the Elbow Method and Silhouette Analysis, to training the final model and generating comprehensive visualizations. Each script plays a crucial role in segmenting customers, analyzing the clusters, and providing clear, actionable insights through detailed visual representations.
