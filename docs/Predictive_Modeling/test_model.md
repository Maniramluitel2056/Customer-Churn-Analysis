# Testing the Model

In this section, we evaluate the performance of our trained Artificial Neural Network (ANN) model using various metrics, including accuracy, precision, recall, F1 score, and Area Under the Curve (AUC) of the Receiver Operating Characteristic (ROC) curve. These metrics provide a comprehensive understanding of how well our model predicts customer churn.

## Confusion Matrix

The confusion matrix summarizes the model's performance by showing the number of correct and incorrect predictions made by the model compared to the actual outcomes. The confusion matrix for our model is as follows:

| Actual \ Predicted | No Churn (0) | Churn (1) |
|--------------------|--------------|-----------|
| **No Churn (0)**   | 978          | 57        |
| **Churn (1)**      | 253          | 121       |

- **True Negatives (TN):** 978 (correctly predicted as no churn)
- **False Positives (FP):** 57 (incorrectly predicted as churn)
- **False Negatives (FN):** 253 (incorrectly predicted as no churn)
- **True Positives (TP):** 121 (correctly predicted as churn)

## Evaluation Metrics

The model was evaluated using several key metrics:

- **Accuracy:** 0.78
- **Precision:** 0.68
- **Recall:** 0.32
- **F1 Score:** 0.44
- **AUC (Area Under ROC Curve):** 0.79

These metrics indicate that while the model is reasonably good at identifying customers who will not churn (high precision for 'No Churn'), it has a lower recall for predicting actual churn events, suggesting some difficulty in capturing all potential churn cases.

## ROC Curve

The ROC curve is a graphical representation of the model's ability to distinguish between classes (churn vs. no churn). The AUC score of 0.79 indicates a good level of separability between the two classes. Here’s the ROC curve:

![ROC Curve](Customer-Churn-Analysis\Predictive_Modeling\results\test_results\roc_curve.png)

## Precision-Recall Curve

The precision-recall curve helps visualize the trade-off between precision and recall for different thresholds. The model’s precision-recall curve is shown below, highlighting its ability to maintain higher precision at varying levels of recall:

![Precision-Recall Curve](Customer-Churn-Analysis\Predictive_Modeling\visualization\precision_recall_curve.png)

## Conclusion

The model shows a strong overall performance with an accuracy of 78% and an AUC of 0.79. However, there is room for improvement in recall, particularly in accurately predicting churn events. Future work could focus on improving feature selection, adjusting class weights, or exploring different model architectures to enhance recall and ensure a more balanced model performance.
