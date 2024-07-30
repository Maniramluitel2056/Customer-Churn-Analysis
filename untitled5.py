import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Load the cleaned dataset
cleaned_data = pd.read_csv('path/to/cleaned_dataset.csv')

# Example preprocessing (if needed)
# data_for_clustering = preprocess(cleaned_data)  # Adjust as necessary

# Apply K-means clustering
kmeans = KMeans(n_clusters=3)  # Number of clusters can be adjusted
kmeans.fit(cleaned_data)
cleaned_data['Cluster'] = kmeans.labels_

# Evaluate clustering model
silhouette_avg = silhouette_score(cleaned_data, kmeans.labels_)
print(f'Silhouette Score: {silhouette_avg}')

# Visualize the clusters (example with 2 features)
plt.scatter(cleaned_data['Feature1'], cleaned_data['Feature2'], c=cleaned_data['Cluster'])
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.title('K-means Clustering Results')
plt.show()
