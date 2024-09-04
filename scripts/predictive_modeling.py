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