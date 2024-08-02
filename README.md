# Customer Churn Analysis Project - Clustering Analysis

## Project Overview

This project aims to analyze customer churn data for a telecommunications company, identify key factors contributing to churn, and develop strategies to improve customer retention. The focus here is on clustering analysis to segment customers and create visualizations for interpretation. The insights gained from this analysis will help in understanding different customer segments and tailoring retention strategies accordingly.

## Project Structure

- **clustering_analysis.py**: Script to perform clustering analysis on the customer dataset. This includes loading the dataset, applying clustering algorithms, and saving the clustered results.
- **visualizations.py**: Script to create visualizations for the clustering results. It generates plots to help interpret the clusters and their characteristics.

## Tasks Completed

### Task 1: Clustering Analysis

1. **Objective**: Segment customers using clustering algorithms to identify distinct groups based on their characteristics.
2. **Scripts Used**:
    - `clustering_analysis.py`
3. **Steps**:
    - Loaded the preprocessed dataset from `data/processed/processed_dataset_with_features.csv`.
    - Applied K-Means clustering to segment customers into distinct groups.
    - Determined the optimal number of clusters using the elbow method and silhouette analysis.
    - Analyzed the characteristics of each cluster to understand the customer segments.
4. **Output**:
    - Clustering model saved to `models/kmeans_model.pkl`.
    - Clustered dataset saved to `data/processed/clustered_dataset.csv`.

### Task 2: Visualizations for Clustering Results

1. **Objective**: Create visualizations to interpret the clustering results and provide insights into customer segments.
2. **Scripts Used**:
    - `visualizations.py`
3. **Steps**:
    - Created visualizations such as cluster distribution plots, pair plots, and cluster centroids.
    - Used matplotlib and seaborn for visualizations to highlight the differences between clusters.
    - Generated plots to visualize the optimal number of clusters and the distribution of data points within each cluster.
4. **Output**:
    - Visualization figures saved to `visualizations/cluster_plots`.

## Configuration

### settings.json

This configuration file sets up the Python environment and paths for the project.

```json
{
    "python.pythonPath": "C:\Users\kusha\anaconda3\envs\churn_analysis\python.exe",
    "python.envFile": "${workspaceFolder}/.env",
    "python.autoComplete.extraPaths": [
        "${workspaceFolder}/utils",
        "${workspaceFolder}/scripts"
    ],
    "python.analysis.extraPaths": [
        "${workspaceFolder}/utils",
        "${workspaceFolder}/scripts"
    ],
    "terminal.integrated.env.windows": {
        "PYTHONPATH": "${workspaceFolder}"
    }
}
```

### config.json

This configuration file includes the paths to the various datasets and output files used in the project.

```json
{
    "raw_data_path": "data/raw/Dataset (ATS)-1.csv",
    "interim_cleaned_data_path": "data/interim/cleaned_dataset.csv",
    "preprocessed_data_path": "Data_Preparation/preprocessed_dataset/cleaned_dataset.csv",
    "processed_data_path": "data/processed/processed_dataset_with_features.csv",
    "train_data_path": "data/train/train_dataset.csv",
    "test_data_path": "data/test/test_dataset.csv",
    "min_max_scaled_path": "Data_Preparation/scaling_techniques/min_max_scaled_dataset.csv",
    "standard_scaled_path": "Data_Preparation/scaling_techniques/standard_scaled_dataset.csv",
    "training_set_path": "Data_Preparation/training_sets/train_dataset.csv",
    "testing_set_path": "Data_Preparation/testing_sets/test_dataset.csv",
    "min_max_segmented_path": "Clustering_Analysis/kmeans_model/min_max_segmented.csv",
    "standard_scaled_segmented_path": "Clustering_Analysis/kmeans_model/standard_scaled_segmented.csv"
}
```

## How to Run

1. **Set Up Environment**:
    - Ensure all required packages are installed. This can be done by setting up a virtual environment and installing dependencies listed in `requirements.txt`.
    - Set up the environment variables from the `.env` file.

2. **Run Clustering Analysis**:
    - Execute the clustering analysis script to segment customers using clustering algorithms.
    ```bash
    python clustering_analysis.py
    ```

3. **Generate Visualizations**:
    - Execute the visualizations script to create and save plots for the clustering results.
    ```bash
    python visualizations.py
    ```

### Next Steps

1. **Review Clustering Results**:
    - Analyze the clustering results to confirm the optimal number of clusters.
    - Understand the characteristics of each cluster to derive meaningful insights.

2. **Model Training**:
    - Proceed with training machine learning models using the processed and clustered dataset.
    - Evaluate model performance on the segmented customer data.

3. **Documentation**:
    - Document each step and summarize the results for clarity and reproducibility.
    - Update the project documentation to reflect any changes or additional findings.

**Note**: For detailed documentation on the clustering analysis and visualizations, please refer to `Notebooks` or the following files:
   - `clustering_analysis_documentation.md` for clustering analysis documentation.
   - `visualizations_documentation.md` for visualizations documentation.

