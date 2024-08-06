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
config_path = os.path.join(project_root, 'config.json')
print(f"Config path: {config_path}")

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

# Load the min-max scaled data
df_min_max_scaled = pd.read_csv(min_max_scaled_data_path)
print(f"Min-Max scaled data loaded successfully from {min_max_scaled_data_path}")
df_standard_scaled = pd.read_csv(standard_scaled_data_path)
print(f"Standard scaled data loaded successfully from {standard_scaled_data_path}")

# Example of printing the first few rows to verify
print("Standard Scaled Data:")
print(df_standard_scaled.head())

print("Min-Max Scaled Data:")
print(df_min_max_scaled.head())

# Function to apply K-means clustering and visualize clusters
def apply_kmeans_and_visualize(df, scaling_label, n_clusters):
     # Define path for saving visualizations inside the function
    visualizations_path = os.path.join(project_root, 'Clustering_Analysis', 'visualizations')
    os.makedirs(visualizations_path, exist_ok=True)
    
    # Use only the 'tenure' and 'MonthlyCharges' columns for clustering
    features = df[['tenure', 'MonthlyCharges']]
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    
    # Add cluster labels to the DataFrame
    df['Cluster'] = kmeans.labels_
    
    # Visualize the clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='tenure', y='MonthlyCharges', hue='Cluster', palette='viridis')
    plt.title(f'Customer Segments based on Tenure and Monthly Charges ({scaling_label} - Assumed 3 Clusters)')
    plt.xlabel('Tenure')
    plt.ylabel('Monthly Charges')
    plt.legend(title='Cluster')
    
    # Save the visualization
    visualization_filename = f'{scaling_label.lower().replace(" ", "_")}_3_clusters_assumed.png'
    visualization_filepath = os.path.join(visualizations_path, visualization_filename)
    plt.savefig(visualization_filepath)
    plt.close()
    print(f'Saved cluster visualization: {visualization_filepath}')

# Apply K-means clustering with an assumed number of clusters (3) for both datasets
n_clusters = 3
apply_kmeans_and_visualize(df_min_max_scaled, 'Min-Max Scaled', n_clusters)
apply_kmeans_and_visualize(df_standard_scaled, 'Standard Scaled', n_clusters)
