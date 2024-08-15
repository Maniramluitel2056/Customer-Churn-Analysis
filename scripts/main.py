import json
import pandas as pd
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")

# Add the utils and scripts directories to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

# Load configuration file
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# Convert relative paths in the configuration file to absolute paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_data_path = os.path.join(project_root, config['raw_data_path'])
interim_cleaned_data_path = os.path.join(project_root, config['interim_cleaned_data_path'])
preprocessed_data_path = os.path.join(project_root, config['preprocessed_data_path'])

# Paths for saving training and testing datasets
train_path = os.path.join(project_root, 'data', 'train', 'train_dataset.csv')
test_path = os.path.join(project_root, 'data', 'test', 'test_dataset.csv')

# Path for saving the processed dataset with new features
processed_data_path = os.path.join(project_root, 'data', 'processed', 'processed_dataset_with_features.csv')

# Path for saving EDA figures
eda_figures_path = os.path.join(project_root, 'Data_Preparation', 'eda_visualizations')
os.makedirs(eda_figures_path, exist_ok=True)

# Import custom modules from utils and scripts directories
from data_loader import load_data
from data_cleaner import clean_data
from data_splitter import split_data
from feature_engineering import create_new_features
from scaler import apply_standard_scaling, apply_min_max_scaling
from eda_utils import plot_distribution, plot_boxplots, plot_categorical, plot_correlation_matrix, plot_pairplots

def perform_eda(df):
    """
    Perform exploratory data analysis on the dataframe.
    """
    # Plot distributions
    plot_distribution(df, df.columns)
    
    # Plot boxplots
    plot_boxplots(df, df.columns)
    
    # Plot categorical variables
    categorical_columns = ['gender_Female', 'gender_Male', 'Dependents_No', 'Dependents_Yes',
                           'PhoneService_No', 'PhoneService_Yes', 'MultipleLines_No', 'MultipleLines_Yes',
                           'InternetService_DSL', 'InternetService_Fiber optic', 'Contract_Month-to-month',
                           'Contract_One year', 'Contract_Two year', 'Churn_No', 'Churn_Yes']
    plot_categorical(df, categorical_columns)
    
    # Plot correlation matrix
    plot_correlation_matrix(df)
    
    # Plot pairplots
    plot_pairplots(df, hue='Churn_Yes')

def main():
    """
    Main function to load, clean, create features, scale, and split the dataset.
    """
    # Load the raw data from the specified path
    df = load_data(raw_data_path)
    # Check if data is loaded correctly; if not, print an error message and exit
    if df is None:
        print(f"File not found at {raw_data_path}. Exiting the script.")
        return
    # At this point, df contains the raw data loaded from raw_data_path

    # Clean the loaded data using the clean_data function
    df_cleaned = clean_data(df)
    # Check if data cleaning was successful; if not, print an error message and exit
    if df_cleaned is None:
        print("Data cleaning failed. Exiting the script.")
        return
    # At this point, df_cleaned contains the cleaned data

    # Perform EDA before feature engineering
    perform_eda(df_cleaned)

    # Apply scaling techniques
    df_standard_scaled = apply_standard_scaling(df_cleaned)
    df_min_max_scaled = apply_min_max_scaling(df_cleaned)
    
    # Save the scaled datasets
    standard_scaled_data_path = os.path.join(project_root, 'data_preparation', 'scaling_techniques', 'standard_scaled_dataset.csv')
    min_max_scaled_data_path = os.path.join(project_root, 'data_preparation', 'scaling_techniques', 'min_max_scaled_dataset.csv')
    df_standard_scaled.to_csv(standard_scaled_data_path, index=False)
    df_min_max_scaled.to_csv(min_max_scaled_data_path, index=False)
    print(f"Standard scaled data saved to {standard_scaled_data_path}")
    print(f"Min-Max scaled data saved to {min_max_scaled_data_path}")

    # Create new features from the cleaned dataset
    df_features = create_new_features(df_cleaned)
    # Save the dataset with new features to the processed path
    df_features.to_csv(processed_data_path, index=False)
    print(f"Dataset with new features saved to {processed_data_path}")

    # Split the processed data with new features into training and testing datasets
    train_df, test_df = split_data(df_features, target_column='Churn_Yes')
    # Save the training and testing datasets to their respective paths
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Training and testing datasets saved to {train_path} and {test_path}")

import os
import sys
import json
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

# Ensure the utils module can be found
notebook_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(notebook_dir, '..'))
utils_path = os.path.join(project_root, 'utils')

print(f"Notebook directory: {notebook_dir}")
print(f"Project root: {project_root}")
print(f"Utils path: {utils_path}")

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
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.json')
print(f"Config path: {config_path}")
with open(config_path, 'r') as f:
    config = json.load(f)

# Convert relative paths to absolute paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_data_path = os.path.join(project_root, config['raw_data_path'])
interim_cleaned_data_path = os.path.join(project_root, config['interim_cleaned_data_path'])
preprocessed_data_path = os.path.join(project_root, config['preprocessed_data_path'])
standard_scaled_data_path = os.path.join(project_root, 'data_preparation/scaling_techniques/standard_scaled_dataset.csv')
min_max_scaled_data_path = os.path.join(project_root, 'data_preparation/scaling_techniques/min_max_scaled_dataset.csv')
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_data_path = os.path.join(project_root, config['raw_data_path'])
interim_cleaned_data_path = os.path.join(project_root, config['interim_cleaned_data_path'])
preprocessed_data_path = os.path.join(project_root, config['preprocessed_data_path'])
standard_scaled_data_path = os.path.join(project_root, 'data_preparation/scaling_techniques/standard_scaled_dataset.csv')
min_max_scaled_data_path = os.path.join(project_root, 'data_preparation/scaling_techniques/min_max_scaled_dataset.csv')

print(f"Raw data path (absolute): {raw_data_path}")
print(f"Interim cleaned data path (absolute): {interim_cleaned_data_path}")
print(f"Preprocessed data path (absolute): {preprocessed_data_path}")
print(f"Standard scaled data path (absolute): {standard_scaled_data_path}")
print(f"Min-Max scaled data path (absolute): {min_max_scaled_data_path}")
print(f"Raw data path (absolute): {raw_data_path}")
print(f"Interim cleaned data path (absolute): {interim_cleaned_data_path}")
print(f"Preprocessed data path (absolute): {preprocessed_data_path}")
print(f"Standard scaled data path (absolute): {standard_scaled_data_path}")
print(f"Min-Max scaled data path (absolute): {min_max_scaled_data_path}")

# Utility function to convert absolute path to relative path
def to_relative_path(absolute_path, start_path):
    return os.path.relpath(absolute_path, start=start_path).replace('\\', '/')

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
# Utility function to convert absolute path to relative path
def to_relative_path(absolute_path, start_path):
    return os.path.relpath(absolute_path, start=start_path).replace('\\', '/')

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

# Ensure the correct paths for saving visualizations
optimal_clusters_path = os.path.join(project_root, 'Clustering_Analysis', 'optimal_clusters')
os.makedirs(optimal_clusters_path, exist_ok=True)

# Function to determine optimal number of clusters using the Elbow Method
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

# Ensure the correct paths for saving visualizations
optimal_clusters_path = os.path.join(project_root, 'Clustering_Analysis', 'optimal_clusters')
os.makedirs(optimal_clusters_path, exist_ok=True)

# Function to determine optimal number of clusters using the Elbow Method
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

# Function to determine optimal number of clusters using Silhouette Analysis
def determine_optimal_clusters_with_silhouette(df, scaling_label):
    features = df[['tenure', 'MonthlyCharges']]
# Function to determine optimal number of clusters using Silhouette Analysis
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

# Apply the Elbow Method and Silhouette Analysis to determine the optimal number of clusters for both datasets
determine_optimal_clusters(df_min_max_scaled, 'Min-Max Scaled')
determine_optimal_clusters(df_standard_scaled, 'Standard Scaled')
determine_optimal_clusters_with_silhouette(df_min_max_scaled, 'Min-Max Scaled')
determine_optimal_clusters_with_silhouette(df_standard_scaled, 'Standard Scaled')

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

# Perform clustering analysis on both datasets
fit_kmeans_and_analyze(df_min_max_scaled, 'Min-Max Scaled')
fit_kmeans_and_analyze(df_standard_scaled, 'Standard Scaled')

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

# Perform characteristics analysis on both datasets
analyze_and_save_cluster_characteristics(df_min_max_scaled, 'Min-Max Scaled')
analyze_and_save_cluster_characteristics(df_standard_scaled, 'Standard Scaled')

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
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
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
plot_cluster_scatter(df_min_max_scaled_4_clusters, 'Min-Max Scaled', visualizations_path, n_clusters=4)
plot_cluster_boxplots(df_min_max_scaled_4_clusters, 'Min-Max Scaled', visualizations_path)
plot_cluster_distribution(df_min_max_scaled_4_clusters, 'Min-Max Scaled', visualizations_path)
plot_cluster_heatmap(df_min_max_scaled_4_clusters, 'Min-Max Scaled', visualizations_path)

# Generate visualizations for Standard scaled clusters
plot_cluster_scatter(df_standard_scaled_4_clusters, 'Standard Scaled', visualizations_path, n_clusters=4)
plot_cluster_boxplots(df_standard_scaled_4_clusters, 'Standard Scaled', visualizations_path)
plot_cluster_distribution(df_standard_scaled_4_clusters, 'Standard Scaled', visualizations_path)
plot_cluster_heatmap(df_standard_scaled_4_clusters, 'Standard Scaled', visualizations_path)



if __name__ == "__main__":
    main()