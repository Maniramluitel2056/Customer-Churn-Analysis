import os
import sys
import json
import random
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

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

# Check if all necessary keys are in the configuration
required_keys = ['ann_architecture_path', 'train_data_path']
for key in required_keys:
    if key not in config:
        print(f"Missing key '{key}' in configuration file.")
        sys.exit(1)

ann_architecture_path = os.path.join(base_dir, config['ann_architecture_path'])
train_data_path = os.path.join(base_dir, config['train_data_path'])

# Load the training data to determine input dimensions
train_data = pd.read_csv(train_data_path)

# Determine input dimensions based on provided dataset (excluding target columns)
feature_columns = train_data.drop(columns=['Churn_No', 'Churn_Yes']).shape[1]
input_dim = feature_columns

# Task 1: Define the ANN model architecture

# Define ANN model architecture
model = Sequential()
model.add(Dense(units=64, activation='relu', input_shape=(input_dim,)))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Define the file path for saving the model architecture
architecture_file_path = os.path.join(ann_architecture_path, 'ann_architecture.json')

# Check if the architecture file already exists
if os.path.exists(architecture_file_path):
    print(f"Model architecture file already exists at {architecture_file_path}. Skipping save.")
else:
    # Save the model architecture
    with open(architecture_file_path, 'w') as f:
        f.write(model.to_json())
    print(f"Model architecture defined and saved to {architecture_file_path}")



# Task 2: Train the ANN model

# Load the training dataset
train_data = pd.read_csv(train_data_path)

# Prepare training data by separating features and target
X_train = train_data.drop(columns=['Churn_No', 'Churn_Yes']).values
y_train = train_data['Churn_Yes'].values

# Define paths for saving the trained model, results, and visualizations
trained_model_path = os.path.join(base_dir, config['trained_model_path'], 'best_model.h5')
results_path_csv = os.path.join(base_dir, config['results_path'], 'training_results.csv')
results_path_txt = os.path.join(base_dir, config['results_path'], 'training_results.txt')
visualization_path = os.path.join(base_dir, 'Predictive_Modeling', 'visualization')

# Ensure the visualization directory exists
os.makedirs(visualization_path, exist_ok=True)

# Set up ModelCheckpoint and EarlyStopping
checkpoint = ModelCheckpoint(trained_model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Define the number of epochs and batch size based on your project requirements
epochs = 50  # Reasonable starting point
batch_size = 32  # Typical batch size

# Train the model with checkpoint and early stopping
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,  # Use 20% of the training data for validation
    callbacks=[checkpoint, early_stopping]
)

# Save training history to CSV
history_df = pd.DataFrame(history.history)
history_df.to_csv(results_path_csv, index=False)
print(f"Training history saved to {results_path_csv}")

# Save training results to TXT
with open(results_path_txt, 'w') as f:
    for key, values in history.history.items():
        f.write(f'{key}: {values}\n')
print(f"Training results saved to {results_path_txt}")

# Plot training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()

# Save the plot to the visualization folder
plt.savefig(os.path.join(visualization_path, 'training_visualization.png'))
plt.show()

print(f"Visualizations saved to {visualization_path}")