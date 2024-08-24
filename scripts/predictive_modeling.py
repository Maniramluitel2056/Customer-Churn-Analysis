import os
import sys
import json
import numpy as np
import tensorflow as tf
import random
import pandas as pd
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
import matplotlib.pyplot as plt

# Set the seed for reproducibility
seed = 42  # Chosen seed for balance and robustness
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Ensure that the 'utils' directory is correctly added to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(utils_path)

# Import the utility function for plotting training metrics
from training_metrics import plot_training_metrics # type: ignore

# Import the function for tallying predictions
from PredictionTally import run_prediction_tally

# Determine the base directory (the root of your project)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the configuration file from the main directory
config_path = os.path.join(base_dir, 'config.json')
with open(config_path) as config_file:
    config = json.load(config_file)

# Define paths using absolute paths
train_data_path = os.path.join(base_dir, config['train_data_path'])
test_data_path = os.path.join(base_dir, config['test_data_path'])
trained_model_path = os.path.join(base_dir, config['trained_model_path'])
ann_architecture_path = os.path.join(base_dir, config['ann_architecture_path'])
results_path = os.path.join(base_dir, config['results_path'])

# Load the training and testing data
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Separate features (all columns except the target) and target (the target column) for both training and testing data
X_train = train_data.drop(columns=['Churn_No', 'Churn_Yes'])  # Drop the target columns to get features
y_train = train_data['Churn_Yes']  # Select the target column

X_test = test_data.drop(columns=['Churn_No', 'Churn_Yes'])
y_test = test_data['Churn_Yes']  # Select the target column

# Define the architecture of the ANN model
model = Sequential()

# Input layer
model.add(Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)))

# Hidden layers
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Define the model checkpoint to save the best model
checkpoint = ModelCheckpoint(os.path.join(trained_model_path, 'best_model.h5'), 
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min')

# Train the model for 50 epochs
history = model.fit(X_train, y_train, 
                    epochs=50, 
                    batch_size=32, 
                    validation_data=(X_test, y_test), 
                    callbacks=[checkpoint])

# Save the model architecture
with open(os.path.join(ann_architecture_path, 'ann_architecture.json'), 'w') as f:
    f.write(model.to_json())

# Save training results
with open(os.path.join(results_path, 'training_results.txt'), 'w') as f:
    f.write(str(history.history))

# Generate predictions on the test data
predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions (0 or 1)

# Create a DataFrame to store actual vs predicted values
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': predictions.flatten()
})

# Save the predictions to a CSV file in the results directory
results_df.to_csv(os.path.join(results_path, 'predictions.csv'), index=False)

print(f"Predictions saved to {os.path.join(results_path, 'predictions.csv')}")
print("Training complete. Model and results saved.")

# Run the prediction tally
run_prediction_tally(base_dir)


# Task 2: Train the model and optimize convergence

# Compile the model 
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Set up callbacks for early stopping and model checkpoint
model_checkpoint = ModelCheckpoint(os.path.join(trained_model_path, 'best_model.h5'), save_best_only=True, monitor='val_loss')

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[model_checkpoint]) # type: ignore

# Save training results
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(results_path, 'training_results.csv'), index=False)

# Call the plot_training_metrics function to generate and save plots
plot_training_metrics(training_results_path=os.path.join(results_path, 'training_results.csv'), save_dir=results_path)

# Save the model if it has improved
model.save(os.path.join(trained_model_path, 'best_model.h5'))

# Generate predictions on the test data
predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions (0 or 1)

# Create a DataFrame to store actual vs predicted values
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': predictions.flatten()
})

# Save the predictions to a CSV file in the results directory
results_df.to_csv(os.path.join(results_path, 'predictions.csv'), index=False)

print(f"Predictions saved to {os.path.join(results_path, 'predictions.csv')}")
print("Training complete. Model and results saved.")
