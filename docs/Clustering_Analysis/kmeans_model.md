# K-means Clustering Analysis Documentation
## Overview

This document provides a step-by-step guide for setting up the environment, loading and preprocessing data, applying K-means clustering, and visualizing the results. The analysis includes clustering customers based on their tenure and monthly charges using Min-Max scaled and Standard scaled data.

## Running the K-means Clustering Analysis

1. Importing Required Libraries
The following code snippet imports the necessary libraries and modules for data processing, clustering, and visualization.

```python
import os
import sys
import json
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
```

2. Ensure Utility Modules are Accessible
Set up paths to ensure that your custom utility modules (data_loader, data_cleaner, etc.) are accessible within the project.

```python
notebook_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(notebook_dir, '..'))
utils_path = os.path.join(project_root, 'utils')

if utils_path not in sys.path:
    sys.path.append(utils_path)

try:
    from data_loader import load_data
    from data_cleaner import clean_data
    from handle_missing_and_encode import handle_missing_and_encode
except ImportError as e:
    print(f"Error importing module: {e}")
    sys.exit(1)

```

3. Load Configuration and Set Up Paths
Load the configuration file `config.json` and convert relative paths to absolute paths. This ensures that the correct data files are accessed during the analysis.

```python
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_data_path = os.path.join(project_root, config['raw_data_path'])
interim_cleaned_data_path = os.path.join(project_root, config['interim_cleaned_data_path'])
preprocessed_data_path = os.path.join(project_root, config['preprocessed_data_path'])
standard_scaled_data_path = os.path.join(project_root, 'data_preparation/scaling_techniques/standard_scaled_dataset.csv')
min_max_scaled_data_path = os.path.join(project_root, 'data_preparation/scaling_techniques/min_max_scaled_dataset.csv')

```

4. Load the Preprocessed Data
Load the preprocessed datasets (both Min-Max scaled and Standard scaled) to prepare them for clustering.

```python
df_min_max_scaled = pd.read_csv(min_max_scaled_data_path)
df_standard_scaled = pd.read_csv(standard_scaled_data_path)

print("Standard Scaled Data:")
print(df_standard_scaled.head())

print("Min-Max Scaled Data:")
print(df_min_max_scaled.head())

```

5. Apply K-means Clustering and Visualize Clusters
Define a function that applies K-means clustering to the dataset and generates visualizations of the clusters. This function will also save the visualizations to a specified directory.

```python
def apply_kmeans_and_visualize(df, scaling_label, n_clusters):
    visualizations_path = os.path.join(project_root, 'Clustering_Analysis', 'visualizations')
    os.makedirs(visualizations_path, exist_ok=True)
    
    features = df[['tenure', 'MonthlyCharges']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    
    df['Cluster'] = kmeans.labels_
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='tenure', y='MonthlyCharges', hue='Cluster', palette='viridis')
    plt.title(f'Customer Segments based on Tenure and Monthly Charges ({scaling_label} - Assumed 3 Clusters)')
    plt.xlabel('Tenure')
    plt.ylabel('Monthly Charges')
    plt.legend(title='Cluster')
    
    visualization_filename = f'{scaling_label.lower().replace(" ", "_")}_3_clusters_assumed.png'
    visualization_filepath = os.path.join(visualizations_path, visualization_filename)
    plt.savefig(visualization_filepath)
    plt.close()
    print(f'Saved cluster visualization: {visualization_filepath}')

```

6. Running K-means Clustering
Apply K-means clustering with an assumed number of clusters (e.g., 3 clusters) to both the Min-Max scaled and Standard scaled datasets. Visualize and save the results.

```python
n_clusters = 3
apply_kmeans_and_visualize(df_min_max_scaled, 'Min-Max Scaled', n_clusters)
apply_kmeans_and_visualize(df_standard_scaled, 'Standard Scaled', n_clusters)

```

## Conclusion
This K-means clustering analysis segments customers based on their tenure and monthly charges. The clusters are visualized and saved, providing insights into different customer segments. The process includes importing necessary libraries, loading configuration files, preprocessing data, and visualizing the clusters.

## Next Steps:
1. Cluster Interpretation: Examine the characteristics of each cluster for business insight
2. Optimal Clustering: Consider using methods like the Elbow Method or Silhouette Score to determine the optimal number of clusters.
3. Advanced Analysis: Explore additional features and clustering techniques to refine the segmentation.
