import os
import json
import sys
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Set up paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config.json')

# Load the configuration file from the main directory
try:
    with open(config_path) as config_file:
        config = json.load(config_file)
    print("Loaded configuration:", config)
except FileNotFoundError:
    print(f"Configuration file not found at {config_path}")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON configuration file: {e}")
    sys.exit(1)

# Define paths using absolute paths from the configuration file
base_dir = os.path.dirname(os.path.abspath(''))  # Get the base directory
test_data_path = os.path.join(base_dir, config['test_data_path'])  # Path to the test data
trained_model_path = os.path.join(base_dir, config['trained_model_path'], 'best_model.h5')  # Path to the saved model
results_dir = os.path.join(base_dir, config['results_path'], 'test_results')  # Path to save test results

# Ensure the results directory exists
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Load the test dataset
print(f"Loading test data from {test_data_path}")
test_data = pd.read_csv(test_data_path)

# Separate features and target variable
X_test = test_data.drop(columns=['Churn_No', 'Churn_Yes']).values
y_test = test_data['Churn_Yes'].values

# Load the trained model
print(f"Loading trained model from {trained_model_path}")
model = load_model(trained_model_path)

# Evaluate the model performance on the test data
print("Evaluating the model on the test data...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Make predictions on the test data
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

# Calculate additional metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Save evaluation metrics to a text file
metrics_file_path = os.path.join(results_dir, 'evaluation_metrics.txt')
with open(metrics_file_path, 'w') as file:
    file.write(f"Test Loss: {loss:.4f}\n")
    file.write(f"Test Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1 Score: {f1:.4f}\n")

print(f"Evaluation metrics saved to {metrics_file_path}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Save confusion matrix to a text file
cm_file_path = os.path.join(results_dir, 'confusion_matrix.txt')
with open(cm_file_path, 'w') as file:
    file.write("Confusion Matrix:\n")
    file.write(str(cm))

print(f"Confusion matrix saved to {cm_file_path}")

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
auc = roc_auc_score(y_test, y_pred_prob)
print(f"\nAUC: {auc:.4f}")

# Save ROC AUC to the text file
with open(metrics_file_path, 'a') as file:
    file.write(f"AUC: {auc:.4f}\n")

# Plot ROC Curve and save the plot
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
roc_curve_path = os.path.join(results_dir, 'roc_curve.png')
plt.savefig(roc_curve_path)
plt.show()

print(f"ROC curve saved to {roc_curve_path}")