import pandas as pd
from sklearn.cluster import KMeans
import os

# Ensure results directory exists
if not os.path.exists('C:/Users/kusha/OneDrive/Documents/Customer-Churn-Analysis-main/clustering_analysis_v2/results'):
    os.makedirs('C:/Users/kusha/OneDrive/Documents/Customer-Churn-Analysis-main/clustering_analysis_v2/results')
print("Results directory ensured.")

# Load datasets
try:
    min_max_data = pd.read_csv('C:/Users/kusha/OneDrive/Documents/Customer-Churn-Analysis-main/Data_Preparation/scaling_techniques/min_max_scaled_dataset.csv')
    standard_data = pd.read_csv('C:/Users/kusha/OneDrive/Documents/Customer-Churn-Analysis-main/Data_Preparation/scaling_techniques/standard_scaled_dataset.csv')
    print("Datasets loaded successfully.")
    print("Min-Max Data Sample:")
    print(min_max_data.head())
    print("Standard Scaled Data Sample:")
    print(standard_data.head())
except FileNotFoundError as e:
    print(f"Error loading datasets: {e}")
    exit(1)

if min_max_data.empty:
    print("Min-Max dataset is empty.")
else:
    print("Min-Max dataset is not empty.")

if standard_data.empty:
    print("Standard dataset is empty.")
else:
    print("Standard dataset is not empty.")

# Assuming the optimal number of clusters is 3 (replace with your determined number)
optimal_clusters = 3
print(f"Optimal number of clusters determined: {optimal_clusters}")

# Apply K-Means clustering
def apply_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    print(f"Clusters for n_clusters={n_clusters} computed.")
    return clusters

# Cluster and save results for min-max scaled data
min_max_clusters = apply_kmeans(min_max_data, optimal_clusters)
min_max_data['Cluster'] = min_max_clusters
print("Min-Max Cluster Data Sample:")
print(min_max_data.head())
min_max_data.to_csv('C:/Users/kusha/OneDrive/Documents/Customer-Churn-Analysis-main/clustering_analysis_v2/results/min_max_clustered_data.csv', index=False)
print("Min-max scaled data clustered and saved.")

# Cluster and save results for standard scaled data
standard_clusters = apply_kmeans(standard_data, optimal_clusters)
standard_data['Cluster'] = standard_clusters
print("Standard Cluster Data Sample:")
print(standard_data.head())
standard_data.to_csv('C:/Users/kusha/OneDrive/Documents/Customer-Churn-Analysis-main/clustering_analysis_v2/results/standard_clustered_data.csv', index=False)
print("Standard scaled data clustered and saved.")

print("Clustering completed and results saved.")
