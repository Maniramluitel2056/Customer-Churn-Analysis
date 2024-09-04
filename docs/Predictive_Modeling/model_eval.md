# 📊 EvaluateModelPerformance.py

## 📋 Objective

The purpose of `EvaluateModelPerformance.py` is to evaluate the performance of our trained Artificial Neural Network (ANN) model for predicting customer churn. This script calculates key metrics such as accuracy, precision, recall, and F1 score to assess the model's effectiveness. It also generates various visualizations to help analyze the model's predictions.

## 🛠️ Steps

1. **Load Configurations and Data**:  
   - Load the necessary configurations from the `config.json` file.
   - Load the test dataset, which includes both features and the target variable (`Churn_Yes` and `Churn_No`).

2. **Load Trained Model**:  
   - Load the trained ANN model from the specified path using the model's saved architecture and weights.

3. **Make Predictions**:  
   - Use the loaded model to make predictions on the test data.
   - Generate predicted probabilities and binary classifications for customer churn.

4. **Evaluate Performance Metrics**:  
   - Calculate key metrics such as accuracy, precision, recall, and F1 score.
   - Generate a comprehensive classification report.

5. **Generate Visualizations**:  
   - Create and save the following visualizations:
     - **Confusion Matrix**: To visualize the performance of the model in distinguishing between different classes.
     - **ROC Curve**: To analyze the trade-off between true positive and false positive rates.
     - **Precision-Recall Curve**: To understand the balance between precision and recall for the model.

6. **Save Evaluation Results**:  
   - Save the classification report and generated visualizations to the `evaluation` folder inside the `visualization` directory for further analysis.

## 📄 Output Files

- **Classification Report** (`classification_report_evaluation.txt`): A text file containing detailed performance metrics of the model.
- **Evaluation Matrix** (`evaluation_matrix.png`): A visual representation of the model's predictions versus actual values.
- **ROC Curve Evaluation** (`roc_curve_evaluation.png`): A plot illustrating the model's true positive rate against its false positive rate.
- **Precision-Recall Curve Evaluation** (`precision_recall_curve_evaluation.png`): A plot showing the relationship between precision and recall.

## 📊 Results Obtained

- **Confusion Matrix:**
  - 🟦 **True Negatives (TN):** 978 - Customers correctly identified as non-churn.
  - 🔵 **False Positives (FP):** 57 - Non-churn customers incorrectly predicted as churn.
  - 🔴 **False Negatives (FN):** 253 - Churn customers not identified by the model.
  - 🟩 **True Positives (TP):** 121 - Customers correctly identified as churn.

- **Evaluation Metrics:**
  - **Accuracy:** 78%
  - **Precision:** 68%
  - **Recall:** 32%
  - **F1 Score:** 44%
  - **AUC (Area Under ROC Curve):** 0.79

- **ROC Curve and AUC Score:**
  - 📈 The ROC curve illustrates the trade-off between true positive rate and false positive rate.
  - ⭐ An AUC score of 0.79 indicates a good discriminatory ability of the model, suggesting it effectively distinguishes between customers who will churn and those who won't.

- **Precision-Recall Curve:**
  - ⚖️ The precision-recall curve provides a balance between precision (accuracy of positive predictions) and recall (ability to find all positive instances). The curve shows the model's effectiveness in identifying true churns while minimizing false positives.

## 🔍 Why Are the Prediction and Evaluation Metrics Reports the Same?

Both the `predict_churn.py` and `EvaluateModelPerformance.py` scripts are designed to work with the same trained ANN model and test dataset. As a result, they produce the same metrics and visualizations. The **classification report**, **confusion matrix**, **ROC curve**, and **precision-recall curve** generated in both scripts are identical because they are based on the same predictions and ground truth data.

This consistency ensures that the model's performance evaluation is reliable and that the metrics reported are accurate reflections of its predictive capabilities.

## 📊 Summary

The `EvaluateModelPerformance.py` script is essential for assessing the accuracy and reliability of the ANN model in predicting customer churn. By analyzing the output metrics and visualizations, stakeholders can better understand the model's strengths and areas for improvement. The alignment between prediction and evaluation reports further reinforces the validity of the model's performance.

---
