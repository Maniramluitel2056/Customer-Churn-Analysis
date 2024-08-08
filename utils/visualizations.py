import os
import json  # Import the json module
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Define the path to the config file
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.json')

# Load the configuration
with open(config_path, 'r') as f:
    config = json.load(f)

# Utility function to convert relative path to absolute path
def to_absolute_path(relative_path, start_path):
    return os.path.abspath(os.path.join(start_path, relative_path))

# Define the project root and load paths from the config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
min_max_clusters_path = to_absolute_path(config['min-max_scaled_4_clusters_path'], project_root)
standard_clusters_path = to_absolute_path(config['standard_scaled_4_clusters_path'], project_root)

# Load the clustered datasets
df_min_max_clusters = pd.read_csv(min_max_clusters_path)
print(f"Min-Max scaled clusters data loaded successfully from {min_max_clusters_path}")
df_standard_clusters = pd.read_csv(standard_clusters_path)
print(f"Standard scaled clusters data loaded successfully from {standard_clusters_path}")

# Define the path for saving visualizations
visualizations_path = os.path.join(project_root, 'Clustering_Analysis', 'visualizations')
os.makedirs(visualizations_path, exist_ok=True)

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
