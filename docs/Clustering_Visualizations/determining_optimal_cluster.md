Clustering Analysis Process and Results
Steps Taken:

1. Importing Libraries and Modules:

Essential libraries and custom modules are imported for data processing and clustering tasks.
Libraries include: pandas, os, sys, json, KMeans, matplotlib, seaborn, and silhouette_score.
Custom modules include: data_loader, data_cleaner, and handle_missing_and_encode.

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

2. Loading the Preprocessed Data:

Load the preprocessed datasets (min-max scaled and standard scaled) to apply clustering algorithms.
Read the CSV files containing the min-max scaled and standard scaled datasets into pandas DataFrames.

# Load data paths
min_max_scaled_data_path = get_path_from_config('min_max_scaled_path', config, project_root)
standard_scaled_data_path = get_path_from_config('standard_scaled_path', config, project_root)

# Load the preprocessed data
df_min_max_scaled = pd.read_csv(min_max_scaled_data_path)
df_standard_scaled = pd.read_csv(standard_scaled_data_path)

3. Determining Optimal Clusters Using the Elbow Method:

The Elbow Method helps in identifying the optimal number of clusters by fitting the model with a range of values for the number of clusters and plotting the Within-Cluster Sum of Squares (WCSS).
Implement a function to determine the optimal number of clusters.

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


The reason the descriptions are appearing in code form is likely due to the way the text is formatted or copied into the document. Let's reformat it to clearly separate the code from the descriptions.

Here is the corrected and clear format:

Clustering Analysis Process and Results
Steps Taken:

1. Importing Libraries and Modules:

Essential libraries and custom modules are imported for data processing and clustering tasks.
Libraries include: pandas, os, sys, json, KMeans, matplotlib, seaborn, and silhouette_score.
Custom modules include: data_loader, data_cleaner, and handle_missing_and_encode.

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

2. Loading the Preprocessed Data:

Load the preprocessed datasets (min-max scaled and standard scaled) to apply clustering algorithms.
Read the CSV files containing the min-max scaled and standard scaled datasets into pandas DataFrames.

# Load data paths
min_max_scaled_data_path = get_path_from_config('min_max_scaled_path', config, project_root)
standard_scaled_data_path = get_path_from_config('standard_scaled_path', config, project_root)

# Load the preprocessed data
df_min_max_scaled = pd.read_csv(min_max_scaled_data_path)
df_standard_scaled = pd.read_csv(standard_scaled_data_path)
3. Determining Optimal Clusters Using the Elbow Method:

The Elbow Method helps in identifying the optimal number of clusters by fitting the model with a range of values for the number of clusters and plotting the Within-Cluster Sum of Squares (WCSS).
Implement a function to determine the optimal number of clusters.
python
Copy code
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

4. Applying the Methods to Both Datasets:

Apply the Elbow Method and Silhouette Analysis to determine the optimal number of clusters for both min-max scaled and standard scaled datasets.

# Apply the Elbow Method and Silhouette Analysis to determine the optimal number of clusters for both datasets
determine_optimal_clusters(df_min_max_scaled, 'Min-Max Scaled')
determine_optimal_clusters(df_standard_scaled, 'Standard Scaled')
determine_optimal_clusters_with_silhouette(df_min_max_scaled, 'Min-Max Scaled')
determine_optimal_clusters_with_silhouette(df_standard_scaled, 'Standard Scaled')

Results:

Elbow Method:

Plots were generated for both min-max scaled and standard scaled datasets, showing the WCSS for a range of cluster numbers.
Silhouette Analysis:

Plots were generated for both min-max scaled and standard scaled datasets, showing the silhouette scores for a range of cluster numbers.
Saved Plots:

The Elbow Method and Silhouette Analysis plots were saved in the Clustering_Analysis/optimal_clusters directory.
Summary:

Imported necessary libraries and custom modules.
Loaded the configuration file to manage file paths.
Loaded the preprocessed datasets.
Applied the Elbow Method and Silhouette Analysis to determine the optimal number of clusters.
Generated and saved the plots for further analysis.
Identified the optimal number of clusters to be 4.
Next Steps:

Interpret Results: Analyze the Elbow Method and Silhouette Analysis plots to determine the optimal number of clusters.
Apply Optimal Clustering: Use the determined optimal number of clusters to apply K-means clustering and analyze the results.
Report Findings: Compile the results and visualizations into a comprehensive report.

