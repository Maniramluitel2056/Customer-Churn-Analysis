# Clustering Analysis - Visualization of Results

## Overview
This script generates various visualizations to help interpret the clustering results from K-Means clustering applied to both Min-Max scaled and Standard scaled customer datasets. The visualizations include scatter plots, boxplots, distribution plots, and heatmaps to provide insights into customer segments based on tenure and monthly charges.

## Steps Involved

### 1. Importing Necessary Libraries

 ```python
import os
import json  # Import the json module
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
 ```

 ### 2. Loading Configuration and Data Paths
The script loads the configuration settings from a config.json file and sets up the necessary paths to locate the clustered datasets.

2.1 Load the Configuration:

```python
# Define the path to the config file
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.json')

# Load the configuration
with open(config_path, 'r') as f:
    config = json.load(f)

```

2.2 Path Conversion Utility Function:

```python
# Utility function to convert relative path to absolute path
def to_absolute_path(relative_path, start_path):
    return os.path.abspath(os.path.join(start_path, relative_path))

```

2.3 Define the Project Root and Load Paths:

```python
# Define the project root and load paths from the config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
min_max_clusters_path = to_absolute_path(config['min-max_scaled_4_clusters_path'], project_root)
standard_clusters_path = to_absolute_path(config['standard_scaled_4_clusters_path'], project_root)

```

2.4 Load the Clustered Datasets:

```python
# Load the clustered datasets
df_min_max_clusters = pd.read_csv(min_max_clusters_path)
print(f"Min-Max scaled clusters data loaded successfully from {min_max_clusters_path}")
df_standard_clusters = pd.read_csv(standard_clusters_path)
print(f"Standard scaled clusters data loaded successfully from {standard_clusters_path}")

```

### 3. Creating Visualizations
3.1 Scatter Plot of Clusters with Centroids
This visualization shows the distribution of customers within each cluster, along with the centroids computed by the K-Means algorithm.

3.2 Compute Centroids and Plot Scatter Plot:

```python
# Function to compute centroids
def compute_centroids(df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df[['tenure', 'MonthlyCharges']])
    centroids = kmeans.cluster_centers_
    return centroids

# Function to plot scatter plot of clusters with centroids
def plot_cluster_scatter(df, scaling_label, save_path, n_clusters):
    centroids = compute_centroids(df, n_clusters)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='tenure', y='MonthlyCharges', hue='Cluster', palette='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
    plt.title(f'Customer Segments based on Tenure and Monthly Charges ({scaling_label})')
    plt.xlabel('Tenure')
    plt.ylabel('Monthly Charges')
    plt.legend(title='Cluster')
    plt.tight_layout()
    file_path = os.path.join(save_path, f'cluster_scatter_{scaling_label.lower().replace(" ", "_")}.png')
    plt.savefig(file_path)
    plt.close()

```

3.2 Boxplots of Clusters
This visualization displays the distribution of tenure and MonthlyCharges within each cluster.

3.2.1 Plot Boxplots for Tenure and Monthly Charges:

```python
# Function to plot boxplots of clusters
def plot_cluster_boxplots(df, scaling_label, save_path):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y='tenure', data=df, hue='Cluster', palette='viridis', legend=False)
    plt.title(f'Tenure by Cluster ({scaling_label})')
    plt.xlabel('Cluster')
    plt.ylabel('Tenure')
    plt.tight_layout()
    file_path_tenure = os.path.join(save_path, f'cluster_boxplot_tenure_{scaling_label.lower().replace(" ", "_")}.png')
    plt.savefig(file_path_tenure)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y='MonthlyCharges', data=df, hue='Cluster', palette='viridis', legend=False)
    plt.title(f'Monthly Charges by Cluster ({scaling_label})')
    plt.xlabel('Cluster')
    plt.ylabel('Monthly Charges')
    plt.tight_layout()
    file_path_charges = os.path.join(save_path, f'cluster_boxplot_charges_{scaling_label.lower().replace(" ", "_")}.png')
    plt.savefig(file_path_charges)
    plt.close()

```

3.3 Distribution of Clusters
This plot shows the number of customers in each cluster.

3.3.1 Plot Cluster Distribution:

```python
# Function to plot distribution of clusters
def plot_cluster_distribution(df, scaling_label, save_path):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Cluster', data=df, hue='Cluster', palette='viridis', legend=False)
    plt.title(f'Cluster Distribution ({scaling_label})')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.tight_layout()
    file_path = os.path.join(save_path, f'cluster_distribution_{scaling_label.lower().replace(" ", "_")}.png')
    plt.savefig(file_path)
    plt.close()

```

3.4 Heatmap of Cluster Characteristics
This heatmap visualizes the mean tenure and monthly charges for each cluster.

3.4.1 Plot Heatmap of Cluster Characteristics:

```python
# Function to plot heatmap of cluster characteristics
def plot_cluster_heatmap(df, scaling_label, save_path):
    cluster_summary = df.groupby('Cluster').agg({
        'tenure': 'mean',
        'MonthlyCharges': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_summary.set_index('Cluster').T, annot=True, cmap='viridis')
    plt.title(f'Cluster Heatmap ({scaling_label})')
    plt.tight_layout()
    file_path = os.path.join(save_path, f'cluster_heatmap_{scaling_label.lower().replace(" ", "_")}.png')
    plt.savefig(file_path)
    plt.close()

```

### 4. Running the Visualizations
Finally, the script applies the visualization functions to both the Min-Max scaled and Standard scaled datasets, generating a comprehensive set of visualizations for each.

```python
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

### 5. Summary
This script provides a comprehensive set of visual tools for interpreting the results of K-Means clustering. By generating scatter plots, boxplots, distribution plots, and heatmaps, the script helps uncover patterns and insights within the customer data.

### 6. Next Steps
1. Define the Architecture of the ANN Model
2. Train the Model and Optimize Convergence
3. Predict Customer Churn Based on Critical Attributes
