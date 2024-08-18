# Customer Churn Analysis Project

## Overview

This project aims to analyze customer churn data for a telecommunications company. The goal is to identify key factors contributing to customer churn and develop strategies to improve customer retention. The project is structured into several key components, including data preprocessing, feature engineering, clustering analysis, and predictive modeling.

## Directory Structure

Here's a brief overview of the project directory structure to help you navigate the files:

```
📁 Customer-Churn-Analysis/
├── 📁 .vscode/
│   ├── 📄 settings.json             # Configuration settings for VS Code
├── 📁 Clustering_Analysis/
│   ├── 📁 optimal_clusters/         # Results of optimal cluster analysis
│   ├── 📁 kmeans_model/             # Trained K-means model files
│   ├── 📁 visualizations/           # Cluster analysis visualizations
├── 📁 data/
│   ├── 📁 interim/                  # Interim cleaned datasets
│   │   ├── 📄 cleaned_dataset.xlsx
│   ├── 📁 processed/                # Processed datasets with engineered features
│   │   ├── 📄 processed_dataset_with_features.xlsx
│   ├── 📁 raw/                      # Raw datasets
│   │   ├── 📄 Dataset (ATS)-1.xlsx
│   ├── 📁 test/                     # Test datasets
│   │   ├── 📄 test_dataset.xlsx
│   ├── 📁 train/                    # Training datasets
│   │   ├── 📄 train_dataset.xlsx
├── 📁 Data_Preparation/
│   ├── 📁 preprocessed_dataset/     # Preprocessed datasets
│   │   ├── 📄 cleaned_dataset.xlsx
│   ├── 📁 scaling_techniques/       # Datasets after scaling
│   │   ├── 📄 min_max_scaled_dataset.xlsx
│   │   ├── 📄 standard_scaled_dataset.xlsx
│   ├── 📁 testing_sets/             # Testing datasets after preprocessing
│   ├── 📁 training_sets/            # Training datasets after preprocessing
│   ├── 📁 visualization/            # EDA visualizations
│   │   ├── 📁 EDA_figures/
├── 📁 docs/
│   ├── 📁 Data_Processing/          # Documentation for data processing and feature engineering
│   ├── 📁 stakeholder_documentation/ # Documentation for stakeholders
├── 📁 notebooks/
│   ├── 📁 clustering/               # Jupyter Notebooks for clustering analysis
│   ├── 📁 data_preparation/         # Jupyter Notebooks for data preparation
│   ├── 📁 data_preprocessing/       # Jupyter Notebooks for data preprocessing
│   ├── 📁 feature_engineering/      # Jupyter Notebooks for feature engineering
│   ├── 📁 predictive_modeling/      # Jupyter Notebooks for predictive modeling
│   ├── 📁 reporting/                # Jupyter Notebooks for reporting
│   ├── 📄 main.ipynb                # Main Notebook integrating all steps
├── 📁 Predictive_Modeling/
│   ├── 📁 ann_architecture/         # ANN model architecture files
│   ├── 📁 trained_model/            # Trained predictive models
│   ├── 📁 results/                  # Results of predictive modeling
├── 📁 Reports/                      # Reports
├── 📁 scripts/
│   ├── 📄 clustering_analysis.py    # Script for clustering analysis
│   ├── 📄 data_preprocessing.py     # Main script for data preprocessing
│   ├── 📄 data_splitter.py          # Script for splitting data into train and test sets
│   ├── 📄 feature_engineering.py    # Script for feature engineering
│   ├── 📄 main.py                   # Main script to run the entire project
│   ├── 📄 predictive_modeling.py    # Script for running predictive models
│   ├── 📄 reporting.py              # Script for generating reports
│   ├── 📄 validate_data.py          # Script to validate data integrity and consistency
├── 📁 utils/
│   ├── 📄 data_cleaner.py           # Utility script for cleaning data
│   ├── 📄 data_loader.py            # Utility script for loading data
│   ├── 📄 feature_selector.py       # Utility script for selecting features
│   ├── 📄 handle_missing_and_encode.py # Script for handling missing data and encoding
│   ├── 📄 scaler.py                 # Utility script for scaling data
│   ├── 📄 visualizations.py         # Utility script for creating visualizations
├── 📁 Video_Demonstration/          # Folder for video demonstration files
├── 📄 .env                          # Environment configuration file
├── 📄 .gitignore                    # Git ignore file
├── 📄 config.json                   # Main configuration file for paths and settings
├── 📄 environment.yml               # Environment setup file for dependencies
├── 📄 LICENSE                       # License file
├── 📄 README.md                     # Project README file (this file)

```


## How to Run the Project


## Setting Up the Environment

### 1. Create and Activate a Virtual Environment

To create a virtual environment and install all dependencies, follow these steps:

1. **Create the Virtual Environment**:
    - Using `conda`:
      ```bash
      conda env create -f environment.yml
      conda activate churn_analysis
      ```
    - Using `virtualenv`:
      ```bash
      python -m venv churn_analysis
      source churn_analysis/bin/activate  # On Windows, use `churn_analysis\Scripts\activate`
      ```

2. **Install Dependencies**:
    - If you’re using `conda`, dependencies will be installed automatically from the `environment.yml` file.


### 2. Configure Settings

#### **config.json**

The `config.json` file contains paths and other configuration settings used throughout the project. Here is an example configuration:

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
    "min-max_scaled_4_clusters_path": "Clustering_Analysis/kmeans_model/min-max_scaled_4_clusters.csv",
    "standard_scaled_4_clusters_path": "Clustering_Analysis/kmeans_model/standard_scaled_4_clusters.csv",
    "min-max_scaled_cluster_characteristics_path": "Clustering_Analysis/kmeans_model/min-max_scaled_cluster_characteristics.csv",
    "standard_scaled_cluster_characteristics_path": "Clustering_Analysis/kmeans_model/standard_scaled_cluster_characteristics.csv"
}
```

Ensure all the paths are correctly set according to your project directory structure.

#### **settings.json**

The `settings.json` file is used for configuring settings within your IDE, such as Visual Studio Code. Here is an example:

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

This configuration ensures that your IDE knows where to find your Python environment, additional scripts, and modules.

### 3. Set Up Environment Variables

Ensure that the `.env` file is properly configured with any environment variables needed for your project. Example content might include:

```env
PYTHONPATH=${PYTHONPATH}:${workspaceFolder}
```

This setup allows your scripts to reference the necessary paths and configurations throughout the project.

## Running the Scripts

### a. Run the Data Preprocessing Script
This script handles data loading, cleaning, and initial preprocessing steps.
```bash
python scripts/data_preprocessing.py
```
- **Outputs**:
    - Cleaned and preprocessed data will be saved to the `data/interim` and `Data_Preparation` directories.

### b. Run the Feature Engineering Script
This script is used to create new features that will be used for modeling.
```bash
python scripts/feature_engineering.py
```
- **Outputs**:
    - The engineered dataset with new features will be saved to `data/processed`.

### c. Run the Data Splitting Script
This script splits the data into training and testing sets.
```bash
python scripts/data_splitter.py
```
- **Outputs**:
    - Training and testing datasets will be saved to `data/train` and `data/test` directories.

### d. Run the Exploratory Data Analysis (EDA) Script
Conduct exploratory data analysis to gain insights into the data distribution and relationships.
```bash
python scripts/eda.py
```
- **Outputs**:
    - EDA visualizations will be saved in the `Data_Preparation/visualization/EDA_figures` directory.

### e. Validate Data Integrity and Consistency
Ensure the dataset’s integrity and consistency across different stages.
```bash
python scripts/validate_data.py
```
- **Outputs**:
    - Validation reports will be generated to confirm data consistency.

### f. Run the Clustering Analysis Script
This script applies K-means clustering to segment customers and generates related visualizations.
```bash
python scripts/clustering_analysis.py
```
- **Outputs**:
    - Clustering results and visualizations will be saved in the `Clustering_Analysis` directory.

### g. Run the Main Script
This script orchestrates the entire data processing workflow, executing the required steps in sequence.
```bash
python scripts/main.py
```
- **Outputs**:
    - The complete data processing pipeline will run, generating all necessary outputs for analysis.

**Note**: Ensure that each script is executed in the order specified above to maintain the integrity and flow of data processing.


## Tasks Completed

### Data Engineer Contributions

#### Task 1: Data Loading and Initial Preprocessing (CCA2)
- **Objective**: Load and preprocess the dataset.
- **Scripts Used**: 
  - `data_loader.py`
  - `data_cleaner.py`
- **Steps**:
  - Loaded the raw dataset from `data/raw/Dataset (ATS)-1.csv`.
  - Handled missing values by dropping rows with missing data.
  - Encoded categorical variables using one-hot encoding.
- **Output**:
  - Cleaned dataset saved to `data/interim/cleaned_dataset.csv`.

#### Task 2: Advanced Handling of Missing Data (CCA3)
- **Objective**: Perform advanced handling of missing data.
- **Scripts Used**: 
  - (Script not currently available)
- **Steps**:
  - (Steps to be determined when the script is available)
- **Output**:
  - (Output details to be provided)

#### Task 3: Feature Scaling, Normalization, and Exploratory Data Analysis (EDA) (CCA4)
- **Objective**: Apply feature scaling, normalization, and conduct EDA.
- **Scripts Used**:
  - `scaler.py`
  - `eda.py`
- **Steps**:
  - Applied standard scaling to the cleaned dataset.
  - Applied min-max scaling to the cleaned dataset.
  - Conducted EDA, including plotting distributions, boxplots, and correlation matrices.
- **Output**:
  - Standard scaled dataset saved to `Data_Preparation/scaling_techniques/standard_scaled_dataset.xlsx`.
  - Min-max scaled dataset saved to `Data_Preparation/scaling_techniques/min_max_scaled_dataset.xlsx`.
  - EDA figures saved to `Data_Preparation/visualization/EDA_figures`.

#### Task 4: Ensuring Data Integrity and Consistency, Feature Engineering, and Data Splitting (CCA5)
- **Objective**: Ensure data integrity and consistency, perform feature engineering, and split the data into training and testing sets.
- **Scripts Used**:
  - `validate_data.py`
  - `feature_engineering.py`
  - `data_splitter.py`
- **Steps**:
  - Validated data integrity and consistency across different stages.
  - Created new features such as `Charges_Per_Tenure` and `TotalCharges`.
  - Encoded contract types and payment methods.
  - Split the data into training and testing sets.
- **Output**:
  - Validation reports generated to confirm data consistency.
  - Dataset with new features saved to `data/processed/processed_dataset_with_features.xlsx`.
  - Training and testing datasets saved to `data/train` and `data/test` directories.

### Data Analyst Contributions (Clustering Analysis)

#### Task 5: Clustering Analysis Setup and Configuration (CCA22)
- **Objective**: Set up the environment and configure necessary files for clustering analysis.
- **Scripts Used**:
  - `clustering_analysis.py`
  - `visualizations.py`
- **Steps**:
  - Set up the project environment using `conda`.
  - Configured paths and settings in `settings.json` and `.env` files.
- **Output**:
  - Environment ready for clustering analysis.

#### Task 6: Applying K-Means Clustering (CCA22)
- **Objective**: Apply K-means clustering to segment customers based on their tenure and monthly charges.
- **Scripts Used**:
  - `clustering_analysis.py`
- **Steps**:
  - Loaded preprocessed datasets (Min-Max scaled and Standard scaled).
  - Applied K-means clustering to both datasets.
- **Output**:
  - Visualizations saved to `Clustering_Analysis/Visualizations`:
    - `min_max_scaled_3_clusters_assumed.png`
    - `standard_scaled_3_clusters_assumed.png`

#### Task 7: Determining the Optimal Number of Clusters (CCA22)
- **Objective**: Determine the optimal number of clusters using the Elbow Method and Silhouette Analysis.
- **Scripts Used**:
  - `clustering_analysis.py`
- **Steps**:
  - Applied the Elbow Method to evaluate the within-cluster sum of squares (WCSS).
  - Applied Silhouette Analysis to assess the quality of clustering.
  - Visualized and interpreted the results to confirm the optimal number of clusters.
- **Output**:
  - Elbow Method and Silhouette Analysis plots saved to `Clustering_Analysis/optimal_clusters`.

#### Task 8: Training the Clustering Model and Interpreting Results (CCA22)
- **Objective**: Train the clustering model with the optimal number of clusters and analyze the characteristics of each cluster.
- **Scripts Used**:
  - `clustering_analysis.py`
- **Steps**:
  - Trained the K-means model using 4 clusters for both Min-Max scaled and Standard scaled datasets.
  - Analyzed and saved the cluster characteristics.
  - Updated `settings.json` with paths for the cluster assignments and characteristics.
- **Output**:
  - Cluster characteristics saved to:
    - `Clustering_Analysis/kmeans_model/min-max_scaled_cluster_characteristics.csv`
    - `Clustering_Analysis/kmeans_model/standard_scaled_cluster_characteristics.csv`
  - Cluster assignments saved to:
    - `Clustering_Analysis/kmeans_model/min-max_scaled_4_clusters.csv`
    - `Clustering_Analysis/kmeans_model/standard_scaled_4_clusters.csv`

#### Task 9: Visualizing Clustering Results (CCA22)
- **Objective**: Generate visualizations to aid in the interpretation of clustering results.
- **Scripts Used**:
  - `visualizations.py`
- **Steps**:
  - Created scatter plots, boxplots, distribution plots, and heatmaps for both Min-Max scaled and Standard scaled datasets.
  - Saved visualizations to the specified paths.
- **Output**:
  - Visualizations saved to `Clustering_Analysis/visualizations`.


## Next Steps

1. **Review Clustering Results**:
   - Analyze the clustering results and visualizations to confirm the validity and relevance of the clusters.
   - Determine if any adjustments are needed in the clustering process or feature engineering based on the results.

2. **Develop Targeted Strategies**:
   - Use the insights gained from clustering to develop targeted strategies for customer retention.
   - Identify key customer segments that require specific interventions to reduce churn.

3. **Implement and Train Predictive Models**:
   - Develop and train an Artificial Neural Network (ANN) model or other machine learning models to predict customer churn based on the clustering results and other key features.
   - Evaluate the model's performance and make necessary adjustments to improve accuracy.

4. **Document and Present Findings**:
   - Document each step of the process, including data preprocessing, clustering analysis, and model training.
   - Summarize the findings and prepare a presentation for stakeholders, highlighting key insights and actionable recommendations.

5. **Deploy and Monitor the Solution**:
   - Deploy the predictive model into a production environment, integrating it with the company's existing systems.
   - Monitor the model's performance over time, making adjustments as needed to maintain accuracy and relevance.

## Overall Summary

This project has successfully analyzed customer churn data for a telecommunications company, with the goal of identifying factors contributing to churn and developing strategies to improve customer retention. Key accomplishments include:

- **Data Preprocessing**: Successfully loaded, cleaned, and preprocessed the dataset, including handling missing data, feature scaling, and normalization.
- **Exploratory Data Analysis (EDA)**: Conducted comprehensive EDA to understand data distribution and relationships, providing valuable insights for feature engineering and model development.
- **Feature Engineering and Selection**: Created new features and selected the most relevant ones, improving the dataset's quality for predictive modeling.
- **Clustering Analysis**: Applied K-means clustering to segment customers, identifying key customer segments and visualizing the results to inform retention strategies.
- **Model Development**: Laid the groundwork for developing and training predictive models, with plans to implement and evaluate an ANN model for churn prediction.

Moving forward, the project will focus on refining the predictive models, developing targeted strategies for customer retention, and deploying the solution to help the company reduce churn and improve customer satisfaction.
