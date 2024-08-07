Clustering Analysis
Objective
The goal of this task is to apply K-means clustering on the preprocessed datasets (both min-max scaled and standard scaled), visualize the clusters, and save the visualizations for further analysis.

Steps Taken and Explanations
Importing Libraries and Modules
Reason: Essential libraries (pandas, os, sys, json, KMeans, matplotlib, seaborn) and custom modules (data_loader, data_cleaner, handle_missing_and_encode) are imported to handle data processing and clustering tasks.

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

Loading Configuration
Reason: To dynamically load the file paths from config.json, ensuring that the paths used in the script are up-to-date and easily changeable.

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

Loading the Preprocessed Data
Reason: Load the preprocessed datasets (min-max scaled and standard scaled) to apply clustering algorithms.
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


Clustering Analysis
Objective
The goal of this task is to apply K-means clustering on the preprocessed datasets (both min-max scaled and standard scaled), visualize the clusters, and save the visualizations for further analysis.

Steps Taken and Explanations
Importing Libraries and Modules
Reason: Essential libraries (pandas, os, sys, json, KMeans, matplotlib, seaborn) and custom modules (data_loader, data_cleaner, handle_missing_and_encode) are imported to handle data processing and clustering tasks.

python
Copy code
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
Loading Configuration
Reason: To dynamically load the file paths from config.json, ensuring that the paths used in the script are up-to-date and easily changeable.

Snippet:

python
Copy code
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
Loading the Preprocessed Data
Reason: Load the preprocessed datasets (min-max scaled and standard scaled) to apply clustering algorithms.

Snippet:

python
Copy code
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
Applying K-means Clustering and Visualizing Clusters
Reason: Apply K-means clustering on the preprocessed datasets to segment customers based on their tenure and monthly charges, and visualize the clusters for better interpretation. For this task, we assumed the number of clusters to be 3.

Results Obtained
Loaded Preprocessed Data:

Min-Max scaled data loaded successfully from data_preparation/scaling_techniques/min_max_scaled_dataset.csv.
Standard scaled data loaded successfully from data_preparation/scaling_techniques/standard_scaled_dataset.csv.
Applied K-means Clustering and Visualized Clusters:

Successfully applied K-means clustering on both min-max scaled and standard scaled datasets with an assumed number of clusters (3).
Visualizations of clusters based on tenure and monthly charges were created and saved.

Summary
In this task, I successfully imported the necessary libraries and custom modules, loaded the configuration file to manage file paths, loaded the preprocessed datasets, applied K-means clustering on the datasets with an assumed number of clusters (3), and visualized the clusters. The visualizations provide a clear representation of customer segments based on tenure and monthly charges.

Next Steps
Optimize the Number of Clusters: Use methods like the elbow method or silhouette analysis to determine the optimal number of clusters.
Interpret Clustering Results: Analyze the characteristics of each cluster to gain insights into customer segments.
Report Findings: Compile the results and visualizations into a comprehensive report.