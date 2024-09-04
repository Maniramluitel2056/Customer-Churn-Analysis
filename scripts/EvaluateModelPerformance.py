import os
import json
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# Define paths using absolute paths from the configuration file
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the base directory
test_data_path = os.path.join(base_dir, config['test_data_path'])
trained_model_path = os.path.join(base_dir, config['trained_model_path'])
results_path = os.path.join(base_dir, config['results_path'])
visualization_path = os.path.join(base_dir, config['visualization_path'])

# Create 'evaluation' folder inside 'visualization' directory if it doesn't exist
evaluation_path = os.path.join(visualization_path, 'evaluation')
os.makedirs(evaluation_path, exist_ok=True)

# Load the test data
test_data = pd.read_csv(test_data_path)

# Separate features (X) and target (y)
X_test = test_data.drop(columns=['Churn_No', 'Churn_Yes'])
y_test = test_data['Churn_Yes']

# Load the trained model
model = load_model(os.path.join(trained_model_path, 'best_model.h5'))

# Evaluate the model
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Save the evaluation results
evaluation_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred.flatten(),
    'Prediction_Probability': y_pred_prob.flatten()
})
evaluation_df.to_csv(os.path.join(results_path, 'evaluation_predictions.csv'), index=False)

print(f"Evaluation predictions saved to {os.path.join(results_path, 'evaluation_predictions.csv')}")

# Generate and save confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Evaluation')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(evaluation_path, 'confusion_matrix_evaluation.png'))  # Updated path
plt.close()

# Classification report
class_report = classification_report(y_test, y_pred)
with open(os.path.join(results_path, 'classification_report_evaluation.txt'), 'w') as f:
    f.write(class_report)

print(f"Classification report saved to {os.path.join(results_path, 'classification_report_evaluation.txt')}")

# Generate and save ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Evaluation')
plt.legend(loc='lower right')
plt.savefig(os.path.join(evaluation_path, 'roc_curve_evaluation.png'))  # Updated path
plt.close()

print(f"ROC curve saved to {os.path.join(evaluation_path, 'roc_curve_evaluation.png')}")

# Generate and save Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(recall, precision, color='green', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Evaluation')
plt.savefig(os.path.join(evaluation_path, 'precision_recall_curve_evaluation.png'))  # Updated path
plt.close()

print(f"Precision-Recall curve saved to {os.path.join(evaluation_path, 'precision_recall_curve_evaluation.png')}")
