import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

def create_visualizations(data, kmeans, title, project_root='.'):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data['MonthlyCharges'], y=data['tenure'], hue='Cluster', palette='viridis', data=data, legend='full')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
    plt.title(title)
    plt.xlabel('MonthlyCharges')
    plt.ylabel('tenure')
    plt.legend()
    visualization_path = os.path.join(project_root, f'Clustering_Analysis/visualizations/{title}.png')
    plt.savefig(visualization_path)
    print(f"Visualization saved to: {visualization_path}")
    plt.close()

def visualize_clusters(data_dict, project_root='.'):
    for name, (data, kmeans) in data_dict.items():
        create_visualizations(data, kmeans, name, project_root)

def main():
    # Ensure the utils module can be found
    notebook_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(notebook_dir, '..'))

    # Load configuration
    config_path = os.path.join(project_root, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Helper function to get path from config with error handling
    def get_path_from_config(key, config, project_root):
        try:
            return os.path.join(project_root, config[key])
        except KeyError:
            print(f"KeyError: '{key}' not found in config.json")
            sys.exit(1)

    # Convert relative paths to absolute paths
    min_max_segmented_path = get_path_from_config('min_max_segmented_path', config, project_root)
    standard_scaled_segmented_path = get_path_from_config('standard_scaled_segmented_path', config, project_root)

    # Load segmented datasets
    min_max_segmented_df = pd.read_csv(min_max_segmented_path)
    standard_scaled_segmented_df = pd.read_csv(standard_scaled_segmented_path)

    # Apply KMeans using the precomputed clusters
    def load_kmeans(data, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data.drop('Cluster', axis=1))  # fit only on features, excluding 'Cluster'
        return kmeans

    min_max_kmeans = load_kmeans(min_max_segmented_df, min_max_segmented_df['Cluster'].nunique())
    standard_scaled_kmeans = load_kmeans(standard_scaled_segmented_df, standard_scaled_segmented_df['Cluster'].nunique())

    # Prepare data for visualization
    data_dict = {
        'Min-Max Scaled Clusters': (min_max_segmented_df, min_max_kmeans),
        'Standard Scaled Clusters': (standard_scaled_segmented_df, standard_scaled_kmeans)
    }

    # Visualize clusters
    visualize_clusters(data_dict, project_root=project_root)

if __name__ == "__main__":
    main()


