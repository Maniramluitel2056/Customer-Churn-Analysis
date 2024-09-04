import os
import sys

import json
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Add the parent directory to the sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

# Now we will import the function
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

# Train the model
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
