# Visualization Documentation

## Overview

This document provides detailed information on the visualizations created to interpret the clustering analysis performed as part of the Customer Churn Analysis project. Visualizations help in understanding the distribution of customers within each cluster and the characteristics of each cluster.

## Objectives

- Create visualizations to aid in the interpretation of clustering results.
- Use tools such as Matplotlib or Seaborn to create visual representations of the clusters.

## Scripts Used

### visualizations.py

The `visualizations.py` script performs the following tasks:

1. **Load Data**:
   - Loads the clustered dataset from `data/processed/clustered_dataset.csv`.

2. **Generate Plots**:
   - Creates cluster distribution plots, pair plots, and cluster centroid visualizations.
   - Uses matplotlib and seaborn for visualizations to highlight the differences between clusters.
   - Saves the plots to `visualizations/cluster_plots`.

## Detailed Steps

### Step 1: Data Loading

The clustered dataset is loaded from `data/processed/clustered_dataset.csv`. This dataset contains the original features along with the cluster assignments.

### Step 2: Creating Visualizations

Various visualizations are created to interpret the clustering results:

- **Cluster Distribution Plots**:
  - These plots show how customers are distributed across different clusters.
  
- **Pair Plots**:
  - Pair plots visualize the relationships between different features within each cluster.
  
- **Cluster Centroid Visualizations**:
  - These visualizations show the centroids of the clusters and help in understanding the key characteristics of each cluster.
  
- **Silhouette Analysis**:
  - Silhouette analysis plots show how similar each data point is to its own cluster compared to other clusters.

### Step 3: Saving Visualizations

The generated visualizations are saved to the `visualizations/cluster_plots` directory for easy access and reference.

## Visualization Examples

### Cluster Distribution Plot

```python
def plot_cluster_distribution(data, title, project_root='.'):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Cluster', data=data, palette='viridis')
    plt.title(f'Cluster Distribution - {title}')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    visualization_dir = os.path.join(project_root, 'Clustering_Analysis/visualizations')
    os.makedirs(visualization_dir, exist_ok=True)
    visualization_path = os.path.join(visualization_dir, f'Cluster_Distribution_{title}.png')
    plt.savefig(visualization_path)
    print(f"Visualization saved to: {visualization_path}")
    plt.close()
```

### Pair Plot

```python
def create_visualizations(data, kmeans, title, project_root='.'):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='MonthlyCharges', y='tenure', hue='Cluster', palette='viridis', data=data, legend='full')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
    plt.title(title)
    plt.xlabel('MonthlyCharges')
    plt.ylabel('tenure')
    plt.legend()
    visualization_dir = os.path.join(project_root, 'Clustering_Analysis/visualizations')
    os.makedirs(visualization_dir, exist_ok=True)
    visualization_path = os.path.join(visualization_dir, f'{title}.png')
    plt.savefig(visualization_path)
    print(f"Visualization saved to: {visualization_path}")
    plt.close()
```

### Silhouette Analysis

```python
def plot_silhouette_analysis(data, kmeans, title, project_root='.'):
    X = data.drop('Cluster', axis=1)
    cluster_labels = kmeans.predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    plt.figure(figsize=(10, 6))
    y_lower = 10
    for i in range(kmeans.n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.viridis(float(i) / kmeans.n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.title(f'Silhouette Analysis - {title}')
    plt.xlabel("Silhouette coefficient values")
    plt.ylabel("Cluster")
    visualization_dir = os.path.join(project_root, 'Clustering_Analysis/visualizations')
    os.makedirs(visualization_dir, exist_ok=True)
    visualization_path = os.path.join(visualization_dir, f'Silhouette_Analysis_{title}.png')
    plt.savefig(visualization_path)
    print(f"Visualization saved to: {visualization_path}")
    plt.close()
```

## Conclusion

The visualizations created in this script help in understanding the distribution and characteristics of each cluster. By examining these plots, we can gain insights into the customer segments and tailor strategies accordingly. The detailed steps and scripts provided in this document ensure that the visualization process is transparent and reproducible.

For any questions or further details, refer to the script `visualizations.py` or contact the project team.
