# Determining the Optimal Number of Clusters

## Objective
The goal is to determine the optimal number of clusters for K-means clustering on preprocessed datasets (min-max scaled and standard scaled) using the Elbow Method and Silhouette Analysis.

## Configuration

**Reason:** To dynamically load the file paths from `config.json`, ensuring that the paths used in the script are up-to-date and easily changeable.

**Description:** Load configuration settings from `config.json` and define a helper function to get paths from the config with error handling.

**Snippet:**
```python
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

# Load data paths
min_max_scaled_data_path = get_path_from_config('min_max_scaled_path', config, project_root)
standard_scaled_data_path = get_path_from_config('standard_scaled_path', config, project_root)

Steps Taken
Importing Libraries and Modules
Reason: Essential libraries and custom modules are imported for data processing and clustering tasks.

Description: Import libraries such as pandas, os, sys, json, KMeans, matplotlib, seaborn, and silhouette_score. Also, ensure the utils module can be found and import custom modules like data_loader, data_cleaner, and handle_missing_and_encode.

Loading the Preprocessed Data
Reason: Load the preprocessed datasets (min-max scaled and standard scaled) to apply clustering algorithms.

Description: Read the CSV files containing the min-max scaled and standard scaled datasets into pandas DataFrames.

# Load the preprocessed data
df_min_max_scaled = pd.read_csv(min_max_scaled_data_path)
df_standard_scaled = pd.read_csv(standard_scaled_data_path)

Determining Optimal Clusters
Using the Elbow Method
Reason: The Elbow Method helps in identifying the optimal number of clusters by fitting the model with a range of values for the number of clusters and plotting the Within-Cluster Sum of Squares (WCSS).

Description: Implement a function to determine the optimal number of clusters using the Elbow Method. This function calculates WCSS for different numbers of clusters and plots the results.

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

Applying the Methods to Both Datasets
Reason: Apply the Elbow Method and Silhouette Analysis to determine the optimal number of clusters for both min-max scaled and standard scaled datasets.

Description: Call the functions determine_optimal_clusters and determine_optimal_clusters_with_silhouette for both datasets.

# Apply the Elbow Method and Silhouette Analysis to determine the optimal number of clusters for both datasets
determine_optimal_clusters(df_min_max_scaled, 'Min-Max Scaled')
determine_optimal_clusters(df_standard_scaled, 'Standard Scaled')
determine_optimal_clusters_with_silhouette(df_min_max_scaled, 'Min-Max Scaled')
determine_optimal_clusters_with_silhouette(df_standard_scaled, 'Standard Scaled')

Results
Elbow Method
Plots were generated for both min-max scaled and standard scaled datasets, showing the WCSS for a range of cluster numbers.
Silhouette Analysis
Plots were generated for both min-max scaled and standard scaled datasets, showing the silhouette scores for a range of cluster numbers.
Saved Plots
The Elbow Method and Silhouette Analysis plots were saved in the Clustering_Analysis/optimal_clusters directory.

Summary
In this task, I successfully:

Imported necessary libraries and custom modules.
Loaded the configuration file to manage file paths.

Loaded the preprocessed datasets.
Applied the Elbow Method and Silhouette Analysis to determine the optimal number of clusters.
Generated and saved the plots for further analysis.
I identified the optimal number of clusters to be 4.

Next Steps
Interpret Results: Analyze the Elbow Method and Silhouette Analysis plots to determine the optimal number of clusters.
Apply Optimal Clustering: Use the determined optimal number of clusters to apply K-means clustering and analyze the results.
Report Findings: Compile the results and visualizations into a comprehensive report.
