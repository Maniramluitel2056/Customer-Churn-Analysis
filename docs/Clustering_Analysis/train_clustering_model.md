# Documentation: Training the Clustering Model and Interpreting Results
## Overview
In the initial phase of our clustering analysis, we applied K-means clustering with an assumed number of clusters (3) to visualize the clusters and gain an understanding of how K-means works, as documented in branch CCA-6. We then determined the optimal number of clusters using the Elbow Method and Silhouette Analysis in branch CCA-7, finding that 4 clusters were optimal for both the Min-Max scaled and Standard scaled datasets.


This document outlines the process of training the K-means model using the optimal number of clusters (4) and interpreting the results by analyzing the cluster characteristics.

## Steps
1. Training the K-means Model
We will train the K-means model with 4 clusters, which was determined as the optimal number from our previous analysis. The model will be trained separately on both the Min-Max scaled and Standard scaled datasets.

```python
# Function to fit KMeans and analyze clusters
def fit_kmeans_and_analyze(df, scaling_label):
    # Fit the KMeans model with 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['tenure', 'MonthlyCharges']])
    
    # Display the first few rows with cluster assignments
    print(f"{scaling_label} Dataset with Cluster Assignments:")
    print(df.head())
    
    # Ensure the correct paths for saving results
    kmeans_model_path = os.path.join(project_root, 'Clustering_Analysis', 'kmeans_model')
    os.makedirs(kmeans_model_path, exist_ok=True)
    
    # Save the DataFrame with cluster assignments
    result_filename = f'{scaling_label.lower().replace(" ", "_")}_4_clusters.csv'
    result_filepath = os.path.join(kmeans_model_path, result_filename)
    df.to_csv(result_filepath, index=False)
    print(f'Saved cluster assignments to: {result_filepath}')
    
    # Update the config file with the new path for cluster assignments
    relative_result_filepath = to_relative_path(result_filepath, project_root)
    config[f'{scaling_label.lower().replace(" ", "_")}_4_clusters_path'] = relative_result_filepath

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Updated config.json with new path for {scaling_label} cluster assignments.")

```

## Explanation:

- K-means Model Training: The K-means model is trained with 4 clusters, which was determined to be optimal. The clusters are assigned to the data points in the dataset.
- Saving Results: The DataFrame with cluster assignments is saved, and the paths are updated in the config.json file for future reference.

2. Analyzing and Interpreting Cluster Characteristics
After training the K-means model, we analyze the characteristics of each cluster to understand the distribution of key variables such as tenure and monthly charges.

```python
def analyze_and_save_cluster_characteristics(df, scaling_label):
    # Ensure the correct paths for saving results
    kmeans_model_path = os.path.join(project_root, 'Clustering_Analysis', 'kmeans_model')
    os.makedirs(kmeans_model_path, exist_ok=True)
    
    # Analyze the cluster characteristics
    cluster_characteristics = df.groupby('Cluster').agg({
        'tenure': ['mean', 'median', 'std'],
        'MonthlyCharges': ['mean', 'median', 'std']
    })
    print(f"\nCluster Characteristics ({scaling_label}):")
    print(cluster_characteristics)
    
    # Save the cluster characteristics
    characteristics_filename = f'{scaling_label.lower().replace(" ", "_")}_cluster_characteristics.csv'
    characteristics_filepath = os.path.join(kmeans_model_path, characteristics_filename)
    cluster_characteristics.to_csv(characteristics_filepath)
    print(f'Saved cluster characteristics to: {characteristics_filepath}')
    
    # Update the config file with the new path for cluster characteristics
    relative_characteristics_filepath = to_relative_path(characteristics_filepath, project_root)
    config[f'{scaling_label.lower().replace(" ", "_")}_cluster_characteristics_path'] = relative_characteristics_filepath

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Updated config.json with new path for {scaling_label} cluster characteristics.")

```

## Explanation:

- Cluster Characteristics: The characteristics of each cluster, including mean, median, and standard deviation of tenure and monthly charges, are calculated and saved.
- Saving and Updating Config: The results are saved to CSV files, and paths are updated in the config.json file for future reference.

3. Applying the Model and Analyzing Characteristics
Finally, the model is applied to both the Min-Max scaled and Standard scaled datasets. The cluster assignments and characteristics are saved and can be used for further analysis.

```python
# Perform clustering analysis on both datasets
fit_kmeans_and_analyze(df_min_max_scaled, 'Min-Max Scaled')
fit_kmeans_and_analyze(df_standard_scaled, 'Standard Scaled')

# Perform characteristics analysis on both datasets
analyze_and_save_cluster_characteristics(df_min_max_scaled, 'Min-Max Scaled')
analyze_and_save_cluster_characteristics(df_standard_scaled, 'Standard Scaled')

```

## Results Obtained
The cluster assignments and characteristics were saved as follows:


Min-Max Scaled Data:

- Cluster Assignments: `Clustering_Analysis/kmeans_model/min-max_scaled_4_clusters.csv`
- Cluster Characteristics: `Clustering_Analysis/kmeans_model/min-max_scaled_cluster_characteristics.csv`


Standard Scaled Data:


- Cluster Assignments: `Clustering_Analysis/kmeans_model/standard_scaled_4_clusters.csv`
- Cluster Characteristics: `Clustering_Analysis/kmeans_model/standard_scaled_cluster_characteristics.csv`


## Overall Insights: Well Presented


### Overall Insights:


| **Cluster**                                   | **Min-Max Scaled** | **Standard Scaled** | **Characteristics**                                                                                                                      |
|-----------------------------------------------|--------------------|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| **High Tenure, High Charges (Premium Customers)** | Cluster 2           | Cluster 1            | Long-term, high-value customers likely subscribed to premium plans. Crucial for retention and upselling opportunities.                     |
| **Low Tenure, Low Charges (New or Basic Customers)** | Cluster 3           | Cluster 0            | New or lower-value customers. Prime targets for engagement strategies aimed at increasing value through promotions and upsell opportunities.|
| **High Tenure, Low Charges (Loyal but Economical Customers)** | Cluster 0           | Cluster 3            | Loyal customers who have opted for more economical plans over time. Potential to increase lifetime value through premium services or rewards.|
| **Moderate Tenure and Charges (Mid-Tier Customers)** | Cluster 1           | Cluster 2            | Mid-tier customers with stable tenure and charges. Potential for growth through targeted marketing and value-added services.               |

#### **Consistency Across Scaling Methods:**
The analysis reveals that the cluster characteristics are consistent across both Min-Max scaled and Standard scaled datasets. This consistency suggests that the customer segmentation is robust and reliable, regardless of the scaling method used. The clusters maintain similar patterns, reinforcing the validity of the segmentation.


## Next steps
- develop the visualiazations to aid in the interpretation of clustering results
- use tools such as Mataplotlib or Seaborn to cretae the visual representation of the clusters