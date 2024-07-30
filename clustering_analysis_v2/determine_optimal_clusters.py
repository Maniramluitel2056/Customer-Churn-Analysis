import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

# Ensure results directory exists
results_dir = 'C:/Users/kusha/OneDrive/Documents/Customer-Churn-Analysis-main/clustering_analysis_v2/results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
print("Results directory ensured.")

# Load datasets
try:
    min_max_data_path = 'C:/Users/kusha/OneDrive/Documents/Customer-Churn-Analysis-main/Data_Preparation/scaling_techniques/min_max_scaled_dataset.csv'
    standard_data_path = 'C:/Users/kusha/OneDrive/Documents/Customer-Churn-Analysis-main/Data_Preparation/scaling_techniques/standard_scaled_dataset.csv'

    min_max_data = pd.read_csv(min_max_data_path)
    standard_data = pd.read_csv(standard_data_path)
    print("Datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading datasets: {e}")
    exit(1)

# Elbow method to determine the optimal number of clusters
def elbow_method(data, title, filename):
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title(title)
    plt.savefig(f'{results_dir}/{filename}.png')
    plt.show()

# Determine optimal clusters for min-max scaled data
elbow_method(min_max_data, 'Elbow Method for Min-Max Scaled Data', 'elbow_min_max')

# Determine optimal clusters for standard scaled data
elbow_method(standard_data, 'Elbow Method for Standard Scaled Data', 'elbow_standard')

print("Elbow method completed and results saved.")



