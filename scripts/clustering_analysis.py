from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Define absolute paths
min_max_path = 'C:/Users/kusha/OneDrive/Documents/Customer-Churn-Analysis-main/Data_Preparation/scaling_techniques/min_max_scaled_dataset.csv'
standard_scaled_path = 'C:/Users/kusha/OneDrive/Documents/Customer-Churn-Analysis-main/Data_Preparation/scaling_techniques/standard_scaled_dataset.csv'

# Load datasets
min_max_df = pd.read_csv(min_max_path)
standard_scaled_df = pd.read_csv(standard_scaled_path)

# Task CCA-6: Utilize Clustering Algorithms to Segment Customers
def segment_customers(df, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df)
    df['Cluster'] = clusters
    return df, kmeans

min_max_segmented, min_max_kmeans = segment_customers(min_max_df)
standard_scaled_segmented, standard_scaled_kmeans = segment_customers(standard_scaled_df)
# Save outputs for Task CCA-6
min_max_segmented.to_csv('C:/Users/kusha/OneDrive/Documents/Customer-Churn-Analysis-main/Clustering_Analysis/kmeans_model/min_max_segmented.csv', index=False)
standard_scaled_segmented.to_csv('C:/Users/kusha/OneDrive/Documents/Customer-Churn-Analysis-main/Clustering_Analysis/kmeans_model/standard_scaled_segmented.csv', index=False)

# Task CCA-7: Determine the Optimal Number of Clusters
def determine_optimal_clusters(df, filename):
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), sse, marker='o')
    plt.title('Elbow Method for ' + filename)
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.savefig(f'C:/Users/kusha/OneDrive/Documents/Customer-Churn-Analysis-main/Clustering_Analysis/optimal_clusters/{filename}.png')
    plt.close()  # Close the figure to ensure it does not overlap

# Determine optimal clusters for both datasets
determine_optimal_clusters(min_max_df, 'elbow_method_min_max')
determine_optimal_clusters(standard_scaled_df, 'elbow_method_standard_scaled')
# Task CCA-8: Train the Clustering Model and Interpret Results
def interpret_clusters(df, kmeans):
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=df.columns[:-1])
    return cluster_centers

# Interpret clusters for both datasets
min_max_cluster_centers = interpret_clusters(min_max_segmented, min_max_kmeans)
standard_scaled_cluster_centers = interpret_clusters(standard_scaled_segmented, standard_scaled_kmeans)

# Save outputs for Task CCA-8
min_max_cluster_centers.to_csv('C:/Users/kusha/OneDrive/Documents/Customer-Churn-Analysis-main/Clustering_Analysis/optimal_clusters/min_max_cluster_centers.csv', index=False)
standard_scaled_cluster_centers.to_csv('C:/Users/kusha/OneDrive/Documents/Customer-Churn-Analysis-main/Clustering_Analysis/optimal_clusters/standard_scaled_cluster_centers.csv', index=False)
import seaborn as sns

# Task CCA-9: Create Visualizations for Interpretation
def create_visualizations(df, kmeans, title):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=df['Cluster'], palette='viridis')
    plt.title(title)
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.legend(title='Cluster')
    plt.savefig(f'C:/Users/kusha/OneDrive/Documents/Customer-Churn-Analysis-main/Clustering_Analysis/visualizations/{title}.png')
    plt.close()  # Close the figure to ensure it does not overlap

# Create visualizations for both datasets
create_visualizations(min_max_segmented, min_max_kmeans, 'Min-Max Scaled Clusters')
create_visualizations(standard_scaled_segmented, standard_scaled_kmeans, 'Standard Scaled Clusters')


