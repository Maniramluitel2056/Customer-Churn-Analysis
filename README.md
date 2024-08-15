# Customer Churn Analysis Project

## Project Overview

This project aims to analyze customer churn data for a telecommunications company, identify key factors contributing to churn, and develop strategies to improve customer retention.

## Project Structure

- **data_loader.py**: Script to load data from a CSV file.
- **data_cleaner.py**: Script to clean data by handling missing values and encoding categorical variables.
- **handle_missing_and_encode.py**: Advanced script for handling missing data and encoding categorical variables.
- **scaler.py**: Script to apply standard scaling and min-max scaling to the data.
- **data_preprocessing.py**: Main script to load, clean, scale, and save the dataset.
- **feature_engineering.py**: Script to create new features for the dataset.
- **feature_selector.py**: Script to select the top features based on their relevance to the target variable.
- **validate_data.py**: Script to validate data integrity and consistency.
- **settings.json**: Configuration file with paths and settings for the project.
- **.env**: Environment configuration file.

## Tasks Completed

### Task 1: Data Loading and Initial Preprocessing (CCA2)

1. **Objective**: Load and preprocess the dataset.
2. **Scripts Used**: 
    - `data_loader.py`
    - `data_cleaner.py`
3. **Steps**:
    - Loaded the raw dataset from `data/raw/Dataset (ATS)-1.csv`.
    - Handled missing values by dropping rows with missing data.
    - Encoded categorical variables using one-hot encoding.
4. **Output**:
    - Cleaned dataset saved to `data/interim/cleaned_dataset.csv`.

### Task 2: Advanced Handling of Missing Data and Encoding (CCA3)

1. **Objective**: Perform advanced handling of missing data and encoding of categorical variables.
2. **Scripts Used**:
    - `handle_missing_and_encode.py`
3. **Steps**:
    - Handled missing data using mean imputation for numeric columns.
    - Encoded categorical variables using `OneHotEncoder`.
4. **Output**:
    - Processed dataset saved to `data/interim/cleaned_dataset.csv`.

### Task 3: Feature Scaling and Normalization (CCA4)

1. **Objective**: Apply feature scaling and normalization to the dataset.
2. **Scripts Used**:
    - `scaler.py`
3. **Steps**:
    - Applied standard scaling to the cleaned dataset.
    - Applied min-max scaling to the cleaned dataset.
4. **Output**:
    - Standard scaled dataset saved to `data_preparation/scaling_techniques/standard_scaled_dataset.csv`.
    - Min-max scaled dataset saved to `data_preparation/scaling_techniques/min_max_scaled_dataset.csv`.

### Task 4: Feature Engineering

1. **Objective**: Create new features for the dataset.
2. **Scripts Used**:
    - `feature_engineering.py`
3. **Steps**:
    - Created new features such as `Charges_Per_Tenure` and `TotalCharges`.
    - Encoded contract types and payment methods.
4. **Output**:
    - Dataset with new features saved to `data/processed/processed_dataset_with_features.csv`.

### Task 5: Feature Selection

1. **Objective**: Select the top features based on their relevance to the target variable.
2. **Scripts Used**:
    - `feature_selector.py`
3. **Steps**:
    - Applied variance thresholding and SelectKBest to select the top features.
4. **Output**:
    - Selected features dataset saved to `Data_Preparation/training_sets/train_dataset.csv`.

### Task 6: Exploratory Data Analysis (EDA)

1. **Objective**: Conduct exploratory data analysis to understand data distribution and relationships.
2. **Scripts Used**:
    - `eda.py`
3. **Steps**:
    - Plotted distributions, boxplots, and categorical variables.
    - Plotted correlation matrix and pair plots.
4. **Output**:
    - EDA figures saved to `Data_Preparation/eda_visualizations`.

## Configuration

### settings.json

```json
{
     "python.pythonPath": "C:\\Users\\iambh\\anaconda3\\envs\\churn_analysis\\python.exe",
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
     "testing_set_path": "Data_Preparation/testing_sets/test_dataset.csv"
}
```

### How to Run

1. Ensure all required packages are installed.
2. Set up the environment variables from the .env file.
3. Execute main.py to run the entire data processing workflow.

```bash
python main.py
```

### Next Steps

1. Review the validation results to confirm data integrity.
2. Proceed with training machine learning models using the processed dataset.
3. Document each step and summarize the results for clarity and reproducibility.

**Note** For detailed documentation on the preprocessing and validation steps, please refer to `Notebooks` or the following files:
   - `config_and_setup.md` for configuration and setup information.
   - `data_loading_preprocessing.md` for data loading and preprocessing documentation.
   - `data_splitter_documentation.md` for data splitting documentation.
   - `EDA_documentation.md` for exploratory data analysis documentation.
   - `feature_engineering_.md` for feature engineering documentation.
   - `main_workflow.md` for the main workflow documentation.

# Customer Churn Analysis Project - Clustering Analysis

## Project Overview

This project aims to segment customers based on their tenure and monthly charges using clustering analysis. The goal is to identify key customer segments and develop strategies to improve customer retention.

## Project Structure

- **clustering_analysis.py**: Script to perform K-means clustering on the preprocessed datasets.
- **visualizations.py**: Script to generate visualizations for interpreting the clustering results.
- **settings.json**: Configuration file with paths and settings for the project.
- **.env**: Environment configuration file.

## Tasks Completed
### Task 1: Clustering Analysis Setup and Configuration (CCA-22)

**Objective**: Set up the environment and configure necessary files for clustering analysis.

**Scripts Used**:
- `clustering_analysis.py`
- `visualizations.py`

**Steps**:
- Set up the project environment using `conda`.
- Configured paths and settings in `config.json` and `.env` files.

**Output**:
- Environment ready for clustering analysis.

### Task 2: Applying K-Means Clustering (CCA-22)

**Objective**: Apply K-means clustering to segment customers based on their tenure and monthly charges assuming number of cluster to be 3 just to show how to apply the K-Means

**Scripts Used**:
- `clustering_analysis.py`

**Steps**:
- Loaded preprocessed datasets (Min-Max scaled and Standard scaled).
- Applied K-means clustering to both datasets.

**Output**:
- file path: Clustering_Analysis/Visualizations
`min_max_scaled_3_clusters_assumed.png`
`standard_scaled_3_clusters_assumed.png`

### Task 3: Determining the Optimal Number of Clusters (CCA-22)

**Objective**: Determine the optimal number of clusters using the Elbow Method and Silhouette Analysis.

**Scripts Used**:
- `clustering_analysis.py`

**Steps**:
- Applied the Elbow Method to evaluate the within-cluster sum of squares (WCSS).
- Applied Silhouette Analysis to assess the quality of clustering.
- Visualized and interpreted the results to confirm the optimal number of clusters.

**Output**:
- Elbow Method and Silhouette Analysis plots saved to `Clustering_Analysis/optimal_clusters`.


### Task 4: Training the Clustering Model and Interpreting Results (CCA-22)

**Objective**: Train the clustering model with the optimal number of clusters and analyze the characteristics of each cluster.

**Scripts Used**:
- `clustering_analysis.py`

**Steps**:
- Trained the K-means model using 4 clusters for both Min-Max scaled and Standard scaled datasets.
- Analyzed and saved the cluster characteristics.
- Updated `config.json` with paths for the cluster assignments and characteristics.

**Output**:
- Cluster characteristics saved to `Clustering_Analysis/kmeans_model/min-max_scaled_cluster_characteristics.csv` and `Clustering_Analysis/kmeans_model/standard_scaled_cluster_characteristics.csv`.
- Cluster assignments saved to `Clustering_Analysis/kmeans_model/min-max_scaled_4_clusters.csv` and `Clustering_Analysis/kmeans_model/standard_scaled_4_clusters.csv`.

### Task 5: Visualizing Clustering Results (CCA-22)

**Objective**: Generate visualizations to aid in the interpretation of clustering results.

**Scripts Used**:
- `visualizations.py`

**Steps**:
- Created scatter plots, boxplots, distribution plots, and heatmaps for both Min-Max scaled and Standard scaled datasets.
- Saved visualizations to the specified paths.

**Output**:
- Visualizations saved to `Clustering_Analysis/visualizations`.

## Configuration
### settings.json

```python
{
    "python.pythonPath": "C:\\Users\\kusha\\OneDrive\\Pictures\\ANOCONDA\\python.exe",
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


#### config.json
```python
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
    "min-max_scaled_4_clusters_path": "Clustering_Analysis/kmeans_model/min-max_scaled_4_clusters.csv",
    "standard_scaled_4_clusters_path": "Clustering_Analysis/kmeans_model/standard_scaled_4_clusters.csv",
    "min-max_scaled_cluster_characteristics_path": "Clustering_Analysis/kmeans_model/min-max_scaled_cluster_characteristics.csv",
    "standard_scaled_cluster_characteristics_path": "Clustering_Analysis/kmeans_model/standard_scaled_cluster_characteristics.csv"
}
```


### Step 5: How to Run
## How to Run

1. Ensure all required packages are installed.
2. Set up the environment variables from the `.env` file.
3. Execute `clustering_analysis.py` to run the clustering analysis workflow.

`python clustering_analysis.py`

**For further documentation** on the clustering analysis process, refer to the following files in the `docs` folder under `clustering_analysis`:

- **clustering_setup_and_config.md**: Documentation for setting up the environment and configuring the clustering analysis pipeline.
- **kmeans_model.md**: Detailed documentation on the K-means clustering process, including the steps to segment customers.
- **optimal_clusters.md**: Documentation on determining the optimal number of clusters using the Elbow Method and Silhouette Analysis.
- **train_clustering_model.md**: Guide on training the clustering model using the optimal number of clusters and interpreting the results.
- **visualizations.md**: Documentation on creating visualizations to interpret clustering results.


## Next Steps

1. Review the clustering results and visualizations to confirm the validity of the clusters.
2. Use the clustered data to develop targeted strategies for customer retention.
3. Implement and train an Artificial Neural Network (ANN) model to predict customer churn based on the clustering results and other key features.
4. Document each step and summarize the results for clarity and reproducibility.
