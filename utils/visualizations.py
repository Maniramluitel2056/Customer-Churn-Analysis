import os
import sys
import json
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the utils module can be found
notebook_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(notebook_dir, '..'))
utils_path = os.path.join(project_root, 'utils')

print(f"Notebook directory: {notebook_dir}")
print(f"Project root: {project_root}")
print(f"Utils path: {utils_path}")

if utils_path not in sys.path:
    sys.path.append(utils_path)

try:
    from data_loader import load_data
    from data_cleaner import clean_data
    from handle_missing_and_encode import handle_missing_and_encode
except ImportError as e:
    print(f"Error importing module: {e}")
    sys.exit(1)

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.json')
print(f"Config path: {config_path}")
with open(config_path, 'r') as f:
    config = json.load(f)

# Convert relative paths to absolute paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
standard_scaled_data_path = os.path.join(project_root, 'data_preparation/scaling_techniques/standard_scaled_dataset.csv')
min_max_scaled_data_path = os.path.join(project_root, 'data_preparation/scaling_techniques/min_max_scaled_dataset.csv')

print(f"Standard scaled data path (absolute): {standard_scaled_data_path}")
print(f"Min-Max scaled data path (absolute): {min_max_scaled_data_path}")

# Load the datasets
df_min_max_scaled = pd.read_csv(min_max_scaled_data_path)
df_standard_scaled = pd.read_csv(standard_scaled_data_path)

# Example of printing the first few rows to verify
print("Standard Scaled Data:")
print(df_standard_scaled.head())

print("Min-Max Scaled Data:")
print(df_min_max_scaled.head())

# Function to apply K-means clustering
def apply_kmeans(df, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['tenure', 'MonthlyCharges']])
    return df

# Apply K-means clustering without saving the results
df_min_max_scaled_4_clusters = apply_kmeans(df_min_max_scaled)
df_standard_scaled_4_clusters = apply_kmeans(df_standard_scaled)

# Visualization functions
# Define the path for saving visualizations
visualizations_path = os.path.join(project_root, 'Clustering_Analysis', 'visualizations')
os.makedirs(visualizations_path, exist_ok=True)

# Function to compute centroids
def compute_centroids(df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
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
plot_cluster_scatter(df_min_max_scaled_4_clusters, 'Min-Max Scaled', visualizations_path, n_clusters=4)
plot_cluster_boxplots(df_min_max_scaled_4_clusters, 'Min-Max Scaled', visualizations_path)
plot_cluster_distribution(df_min_max_scaled_4_clusters, 'Min-Max Scaled', visualizations_path)
plot_cluster_heatmap(df_min_max_scaled_4_clusters, 'Min-Max Scaled', visualizations_path)

# Generate visualizations for Standard scaled clusters
plot_cluster_scatter(df_standard_scaled_4_clusters, 'Standard Scaled', visualizations_path, n_clusters=4)
plot_cluster_boxplots(df_standard_scaled_4_clusters, 'Standard Scaled', visualizations_path)
plot_cluster_distribution(df_standard_scaled_4_clusters, 'Standard Scaled', visualizations_path)
plot_cluster_heatmap(df_standard_scaled_4_clusters, 'Standard Scaled', visualizations_path)

