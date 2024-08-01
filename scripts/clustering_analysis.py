import os
import sys
import json
import pandas as pd
from sklearn.cluster import KMeans

# Ensure the utils module can be found
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
raw_data_path = get_path_from_config('raw_data_path', config, project_root)
interim_cleaned_data_path = get_path_from_config('interim_cleaned_data_path', config, project_root)
preprocessed_data_path = get_path_from_config('preprocessed_data_path', config, project_root)
standard_scaled_data_path = get_path_from_config('standard_scaled_path', config, project_root)
min_max_scaled_data_path = get_path_from_config('min_max_scaled_path', config, project_root)

print(f"Raw data path: {raw_data_path}")
print(f"Interim cleaned data path: {interim_cleaned_data_path}")
print(f"Preprocessed data path: {preprocessed_data_path}")
print(f"Standard scaled data path: {standard_scaled_data_path}")
print(f"Min-Max scaled data path: {min_max_scaled_data_path}")

# Load preprocessed datasets
min_max_scaled_df = pd.read_csv(min_max_scaled_data_path)
standard_scaled_df = pd.read_csv(standard_scaled_data_path)

# Apply KMeans
def apply_kmeans(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    data['Cluster'] = clusters
    return kmeans, data

min_max_kmeans, min_max_segmented = apply_kmeans(min_max_scaled_df)
standard_scaled_kmeans, standard_scaled_segmented = apply_kmeans(standard_scaled_df)

# Save results
min_max_segmented_path = os.path.join(project_root, 'Clustering_Analysis/kmeans_model/min_max_segmented.csv')
standard_scaled_segmented_path = os.path.join(project_root, 'Clustering_Analysis/kmeans_model/standard_scaled_segmented.csv')

min_max_segmented.to_csv(min_max_segmented_path, index=False)
standard_scaled_segmented.to_csv(standard_scaled_segmented_path, index=False)

# Update config.json with the new file paths
config['min_max_segmented_path'] = os.path.relpath(min_max_segmented_path, project_root)
config['standard_scaled_segmented_path'] = os.path.relpath(standard_scaled_segmented_path, project_root)

with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)

print(f"Updated config.json with paths: {config_path}")

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Function to determine the optimal number of clusters
def determine_optimal_clusters(data, max_k=10):
    iters = range(2, max_k + 1, 2)
    sse = []
    silhouette_scores = []

    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].plot(iters, sse, marker='o')
    ax[0].set_xlabel('Cluster Centers')
    ax[0].set_ylabel('SSE')
    ax[0].set_title('SSE by Cluster Center Plot')

    ax[1].plot(iters, silhouette_scores, marker='o')
    ax[1].set_xlabel('Cluster Centers')
    ax[1].set_ylabel('Silhouette Score')
    ax[1].set_title('Silhouette Score by Cluster Center Plot')

    plt.savefig(os.path.join(project_root, 'Clustering_Analysis/optimal_clusters/optimal_clusters_plot.png'))
    plt.show()

# Call the function with both datasets
determine_optimal_clusters(min_max_scaled_df)
determine_optimal_clusters(standard_scaled_df)

