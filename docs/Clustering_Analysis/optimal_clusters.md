# Documentation: Determining the Optimal Number of Clusters
## Overview
This document outlines the process of determining the optimal number of clusters for a K-means clustering analysis using two methods: the Elbow Method and Silhouette Analysis. These methods help in identifying the most appropriate number of clusters to segment the data effectively.

## Prerequisites
Before proceeding with this analysis, ensure that you have already loaded the data and imported the necessary libraries as described in the initial K-means clustering documentation (kmeans_model.md). This includes setting up paths for saving visualizations and accessing the required datasets. The steps in this document are a continuation from where we left off in the kmeans_model.md file.

## Steps
1. Setting Up the Environment
Ensure that your environment is set up correctly, with paths defined for saving visualizations and accessing data files.

```python
# Ensure the correct paths for saving visualizations
optimal_clusters_path = os.path.join(project_root, 'Clustering_Analysis', 'optimal_clusters')
os.makedirs(optimal_clusters_path, exist_ok=True)

```

2. Determining the Optimal Number of Clusters Using the Elbow Method
The Elbow Method helps visualize the Within-Cluster Sum of Square (WCSS) as the number of clusters increases. The "elbow" in the plot indicates the optimal number of clusters.

``` python
def determine_optimal_clusters(df, scaling_label):
    features = df[['tenure', 'MonthlyCharges']]
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(features)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title(f'Elbow Method for Optimal Number of Clusters ({scaling_label})')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    optimal_clusters_filepath = os.path.join(optimal_clusters_path, f'elbow_method_{scaling_label.lower().replace(" ", "_")}.png')
    plt.savefig(optimal_clusters_filepath)
    plt.show()
    print(f'Saved Elbow Method plot: {optimal_clusters_filepath}')

```

3. Determining the Optimal Number of Clusters Using Silhouette Analysis
Silhouette Analysis evaluates how similar each point is within its cluster compared to other clusters. The higher the Silhouette Score, the better the clustering.

```python
def determine_optimal_clusters_with_silhouette(df, scaling_label):
    features = df[['tenure', 'MonthlyCharges']]
    silhouette_scores = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(features)
        silhouette_scores.append(silhouette_score(features, kmeans.labels_))

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.title(f'Silhouette Analysis for Optimal Number of Clusters ({scaling_label})')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    optimal_clusters_filepath = os.path.join(optimal_clusters_path, f'silhouette_analysis_{scaling_label.lower().replace(" ", "_")}.png')
    plt.savefig(optimal_clusters_filepath)
    plt.show()
    print(f'Saved Silhouette Analysis plot: {optimal_clusters_filepath}')

```

4. Applying Both Methods
Apply both the Elbow Method and Silhouette Analysis to the Min-Max scaled and Standard scaled datasets.

```python
determine_optimal_clusters(df_min_max_scaled, 'Min-Max Scaled')
determine_optimal_clusters(df_standard_scaled, 'Standard Scaled')
determine_optimal_clusters_with_silhouette(df_min_max_scaled, 'Min-Max Scaled')
determine_optimal_clusters_with_silhouette(df_standard_scaled, 'Standard Scaled')

```

5. Visualization and Interpretation
After generating the plots, visualize and interpret the results to determine that 4 clusters is the optimal choice.

- Elbow Method: The plots for both Min-Max and Standard scaled data show an "elbow" at 4 clusters, indicating diminishing returns beyond this point.
- Silhouette Analysis: The Silhouette Score peaks at 4 clusters for both datasets, confirming well-defined clusters.

6. Conclusion
- Based on the Elbow Method and Silhouette Analysis, 4 is determined to be the optimal number of clusters for both the Min-Max scaled and Standard scaled datasets.

- Application: This clustering can now be used to further analyze customer segments based on tenure and monthly charges.
Efficiency: The similar results across both scaling methods suggest robustness in the clustering outcome.

7. Next Steps
- Labeling Clusters: Assign labels to the clusters in both datasets.
- Detailed Analysis: Train the clustering model and interpret results
- Comparative Analysis: Create visualizations for the interpretation