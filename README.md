# Customer Churn Analysis Project

## Overview

This project analyzes customer churn data for a telecommunications company to identify key factors contributing to churn and develop strategies to improve retention. The project includes data preprocessing, feature engineering, clustering analysis, and predictive modeling.

## Directory Structure

```
📁 Customer-Churn-Analysis/
├── 📁 .vscode/
│   ├── 📄 settings.json
├── 📁 Clustering_Analysis/
│   ├── 📁 optimal_clusters/
│   ├── 📁 kmeans_model/
│   ├── 📁 visualizations/
├── 📁 data/
│   ├── 📁 interim/
│   │   ├── 📄 cleaned_dataset.xlsx
│   ├── 📁 processed/
│   │   ├── 📄 processed_dataset_with_features.xlsx
│   ├── 📁 raw/
│   │   ├── 📄 Dataset (ATS)-1.xlsx
│   ├── 📁 test/
│   │   ├── 📄 test_dataset.xlsx
│   ├── 📁 train/
│   │   ├── 📄 train_dataset.xlsx
├── 📁 Data_Preparation/
│   ├── 📁 preprocessed_dataset/
│   │   ├── 📄 cleaned_dataset.xlsx
│   ├── 📁 scaling_techniques/
│   │   ├── 📄 min_max_scaled_dataset.xlsx
│   │   ├── 📄 standard_scaled_dataset.xlsx
│   ├── 📁 testing_sets/
│   ├── 📁 training_sets/
│   ├── 📁 visualization/
│   │   ├── 📁 EDA_figures/
├── 📁 docs/
│   ├── 📁 Data_Processing/
│   ├── 📁 stakeholder_documentation/
├── 📁 notebooks/
│   ├── 📁 clustering/
│   ├── 📁 data_preparation/
│   ├── 📁 data_preprocessing/
│   ├── 📁 feature_engineering/
│   ├── 📁 predictive_modeling/
│   ├── 📁 reporting/
│   ├── 📄 main.ipynb
├── 📁 Predictive_Modeling/
│   ├── 📁 ann_architecture/
│   ├── 📁 trained_model/
│   ├── 📁 results/
├── 📁 Reports/
├── 📁 scripts/
│   ├── 📄 clustering_analysis.py
│   ├── 📄 data_preprocessing.py
│   ├── 📄 data_splitter.py
│   ├── 📄 feature_engineering.py
│   ├── 📄 main.py
│   ├── 📄 predictive_modeling.py
│   ├── 📄 reporting.py
│   ├── 📄 validate_data.py
├── 📁 utils/
│   ├── 📄 data_cleaner.py
│   ├── 📄 data_loader.py
│   ├── 📄 feature_selector.py
│   ├── 📄 handle_missing_and_encode.py
│   ├── 📄 scaler.py
│   ├── 📄 visualizations.py
├── 📁 Video_Demonstration/
├── 📄 .env
├── 📄 .gitignore
├── 📄 config.json
├── 📄 environment.yml
├── 📄 LICENSE
├── 📄 README.md
```

## How to Run the Project

### Setting Up the Environment

1. **Create and Activate a Virtual Environment**:
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
  - If using `conda`, dependencies will be installed automatically from the `environment.yml` file.

### Configure Settings

#### **config.json**

Ensure paths are correctly set according to your project directory structure.

#### **settings.json**

Configure settings within your IDE, such as Visual Studio Code.

### Set Up Environment Variables

Ensure the `.env` file is properly configured with any environment variables needed for your project.

## Running the Scripts

1. **Data Preprocessing**:
  ```bash
  python scripts/data_preprocessing.py
  ```

2. **Feature Engineering**:
  ```bash
  python scripts/feature_engineering.py
  ```

3. **Data Splitting**:
  ```bash
  python scripts/data_splitter.py
  ```

4. **Exploratory Data Analysis (EDA)**:
  ```bash
  python scripts/eda.py
  ```

5. **Validate Data Integrity**:
  ```bash
  python scripts/validate_data.py
  ```

6. **Clustering Analysis**:
  ```bash
  python scripts/clustering_analysis.py
  ```

7. **Run the Main Script**:
  ```bash
  python scripts/main.py
  ```

## Tasks Completed

### Data Engineer Contributions

 1. **Data Loading and Initial Preprocessing**:
     - Cleaned dataset saved to `data/interim/cleaned_dataset.csv`.

 2. **Advanced Handling of Missing Data**

 3. **Feature Scaling, Normalization, and EDA**:
      - Scaled datasets saved to `Data_Preparation/scaling_techniques/`.
     - EDA figures saved to `Data_Preparation/visualization/EDA_figures`.

 4. **Ensuring Data Integrity, Feature Engineering, and Data Splitting**:
     - Validation reports generated.
     - Dataset with new features saved to `data/processed/processed_dataset_with_features.xlsx`.
     - Training and testing datasets saved to `data/train` and `data/test`.

### Data Analyst Contributions (Clustering Analysis)

 1. **Clustering Analysis Setup and Configuration**:
     - Environment ready for clustering analysis.

 2. **Applying K-Means Clustering**:
     - Visualizations saved to `Clustering_Analysis/Visualizations`.

 3. **Determining the Optimal Number of Clusters**:
     - Elbow Method and Silhouette Analysis plots saved to `Clustering_Analysis/optimal_clusters`.

 4. **Training the Clustering Model and Interpreting Results**:
     - Cluster characteristics saved to `Clustering_Analysis/kmeans_model/`.

 5. **Visualizing Clustering Results**:
     - Visualizations saved to `Clustering_Analysis/visualizations`.

### Data Analyst Contributions (Predictive Modeling)

 1. **Defining the ANN Model**:
     - Model architecture saved to `Predictive_Modeling/ann_architecture/ann_architecture.json`.

 2. **Training the Model**:
     - Trained model saved to `Predictive_Modeling/trained_model/best_model.h5`.
     - Training results saved to `Predictive_Modeling/results/training_results.csv`.

 3. **Predicting Churn**:
     - Predictions saved to `Predictive_Modeling/results/predictions.csv`.

 4. **Evaluating the Model**:
     - Evaluation results saved to `Predictive_Modeling/results/evaluation_predictions.csv`.

## Overall Summary

The project successfully identified key factors contributing to customer churn and developed predictive models to anticipate churn behavior. Clustering analysis highlighted distinct customer segments, allowing for targeted retention strategies. The predictive models can be integrated into business processes to reduce churn rates and improve customer satisfaction.

