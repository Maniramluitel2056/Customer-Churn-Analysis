import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# 1. Develop visualizations to aid in the interpretation of clustering results.

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

def plot_heatmap_features_by_cluster(data, features, title, project_root='.'):
    cluster_means = data.groupby('Cluster')[features].mean()
    plt.figure(figsize=(10, 8))
    sns.heatmap(cluster_means, annot=True, cmap='coolwarm')
    plt.title(f'Heatmap of Features by Cluster - {title}')
    visualization_dir = os.path.join(project_root, 'Clustering_Analysis/visualizations')
    os.makedirs(visualization_dir, exist_ok=True)
    visualization_path = os.path.join(visualization_dir, f'Heatmap_Features_{title}.png')
    plt.savefig(visualization_path)
    print(f"Visualization saved to: {visualization_path}")
    plt.close()

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

def visualize_clusters(data_dict, project_root='.'):
    for name, (data, kmeans) in data_dict.items():
        create_visualizations(data, kmeans, name, project_root)
        plot_cluster_distribution(data, name, project_root)
        plot_heatmap_features_by_cluster(data, ['MonthlyCharges', 'tenure'], name, project_root)
        plot_silhouette_analysis(data, kmeans, name, project_root)

# 2. Use tools such as Matplotlib or Seaborn to create visual representations of the clusters.

def main():
    notebook_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(notebook_dir, '..'))
    config_path = os.path.join(project_root, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    def get_path_from_config(key, config, project_root):
        try:
            return os.path.join(project_root, config[key])
        except KeyError:
            print(f"KeyError: '{key}' not found in config.json")
            sys.exit(1)

    min_max_segmented_path = get_path_from_config('min_max_segmented_path', config, project_root)
    standard_scaled_segmented_path = get_path_from_config('standard_scaled_segmented_path', config, project_root)

    min_max_segmented_df = pd.read_csv(min_max_segmented_path)
    standard_scaled_segmented_df = pd.read_csv(standard_scaled_segmented_path)

    def load_kmeans(data, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data.drop('Cluster', axis=1))
        return kmeans

    min_max_kmeans = load_kmeans(min_max_segmented_df, min_max_segmented_df['Cluster'].nunique())
    standard_scaled_kmeans = load_kmeans(standard_scaled_segmented_df, standard_scaled_segmented_df['Cluster'].nunique())

    data_dict = {
        'Min-Max Scaled Clusters': (min_max_segmented_df, min_max_kmeans),
        'Standard Scaled Clusters': (standard_scaled_segmented_df, standard_scaled_kmeans)
    }

    visualize_clusters(data_dict, project_root=project_root)

if __name__ == "__main__":
    main()


