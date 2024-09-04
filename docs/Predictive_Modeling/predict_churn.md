# 📝 Predicting Churn Documentation

## 🎯 Objective

The objective of this script, `predict_churn.py`, is to utilize a pre-trained Artificial Neural Network (ANN) model to predict customer churn based on a test dataset. The goal is to analyze the critical attributes influencing the prediction and assess the model's effectiveness in identifying customers likely to churn.

## 🛠️ Steps Taken

1. **Loading Configurations and Data:**
   - 📂 The script begins by loading necessary configurations from a JSON file to ensure all paths and parameters are correctly set.
   - 📊 It then loads the test dataset, which includes customer attributes and the actual churn outcomes for evaluation.

2. **Model Loading:**
   - 🧠 The pre-trained ANN model is loaded from the specified directory. This model has been trained on historical customer data to predict churn.

3. **Making Predictions:**
   - 🔍 The script uses the loaded model to make predictions on the test dataset, generating probabilities of churn for each customer.
   - 📊 These probabilities are converted into binary predictions (churn or no churn) based on a threshold of 0.5.

4. **Results Saving and Visualization:**
   - 📝 The predictions, along with their probabilities, are saved in a CSV file for further analysis.
   - 📊 Various evaluation metrics, including a confusion matrix, ROC curve, and Precision-Recall curve, are generated and saved as visualizations to assess the model's performance.

## 📊 Results Obtained

- **Confusion Matrix:**
  - 🟦 **True Negatives (TN):** 978 - Customers correctly identified as non-churn.
  - 🔵 **False Positives (FP):** 57 - Non-churn customers incorrectly predicted as churn.
  - 🔴 **False Negatives (FN):** 253 - Churn customers not identified by the model.
  - 🟩 **True Positives (TP):** 121 - Customers correctly identified as churn.

- **Precision-Recall Curve:**
  - ⚖️ The precision-recall curve provides a balance between precision (accuracy of positive predictions) and recall (ability to find all positive instances). The curve shows the model's effectiveness in identifying true churns while minimizing false positives.

- **ROC Curve and AUC Score:**
  - 📈 The ROC curve illustrates the trade-off between true positive rate and false positive rate.
  - ⭐ An AUC score of 0.79 indicates a good discriminatory ability of the model, suggesting it effectively distinguishes between customers who will churn and those who won't.

## 📝 Overall Summary

The `predict_churn.py` script is a critical component of our churn prediction pipeline, leveraging a trained ANN model to provide actionable insights on customer churn. The results indicate that the model performs well in identifying non-churn customers but has room for improvement in detecting churners. The robust performance is visualized through various metrics, offering a comprehensive view of the model's strengths and areas for potential enhancement. The script is designed for easy integration into a larger data analysis workflow, providing valuable predictions that can be used to inform business strategies and customer retention efforts.
