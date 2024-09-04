# 🧠 **Task 1 & Task 2: Defining and Training the ANN Model**

## 🎯 **Objective**

The primary objective of these tasks was to build a robust Artificial Neural Network (ANN) model for predicting customer churn. This involved two key phases:

1. **Task 1: Defining the ANN Model Architecture** - Establishing the structure of the neural network, including input, hidden, and output layers, and selecting the appropriate activation functions and optimizer.
2. **Task 2: Training the ANN Model** - Training the model using the defined architecture on the training dataset, optimizing for convergence, and preventing overfitting.

---

## 🧩 **Task 1: Defining the ANN Model Architecture**

### 🛠️ **Steps Taken**

1. **Model Initialization:**
   - **Input Layer**: Configured to match the number of features in the dataset, ensuring all relevant customer attributes are processed by the model.  
     - 🔢 *Input Shape*: The input layer was set to accommodate all input features of the dataset.

2. **Designing Hidden Layers:**
   - **Hidden Layers**: Added two dense layers with 64 and 32 neurons, respectively.  
     - 🏗 *Activation Function*: ReLU was used for the hidden layers to introduce non-linearity, allowing the model to learn complex patterns.

3. **Output Layer:**
   - **Output Layer**: Configured with a single neuron with a sigmoid activation function for binary classification (churn or no churn).  
     - ⚙️ *Output*: Provides a probability score indicating the likelihood of customer churn.

4. **Model Compilation:**
   - **Optimizer**: Adam optimizer was selected for its efficiency and effectiveness in handling large datasets.  
   - **Loss Function**: Binary cross-entropy was used as the loss function to measure the difference between predicted probabilities and actual binary outcomes.  
   - 🏃‍♂️ *Metrics*: Accuracy was chosen to monitor the model’s performance during training.

### 📊 **Results Obtained**

- The model architecture was successfully defined with an appropriate balance of complexity and computational efficiency.
- A clear plan was laid out to move forward with training, ensuring the model was well-prepared for handling the provided dataset.

---

## 🚀 **Task 2: Training the ANN Model**

### 🛠️ **Steps Taken**

1. **Data Preparation:**
   - **Training Data**: Loaded and preprocessed to remove any anomalies and standardize feature values.  
     - 📦 *Data Split*: An 80-20 split was utilized for training and validation to monitor performance on unseen data.

2. **Training Process:**
   - **Epochs**: The model was trained over a fixed number of epochs (50) to allow sufficient learning without overfitting.  
   - **Batch Size**: Set to 32, balancing memory efficiency and training speed.  
   - **Early Stopping**: Implemented to halt training if the model stopped improving on the validation set, preventing overfitting.

3. **Model Evaluation:**
   - **Training & Validation Loss/Accuracy**: Plotted to visualize learning progress and detect overfitting.  
   - **Checkpointing**: The best model based on validation loss was saved for future prediction tasks.  
   - 📈 **Metrics**: Final metrics indicated a good fit, with a training accuracy of ~77% and validation accuracy of ~78%.

### 📊 **Results Obtained (Updated)**

- **Training Visualizations**: Provided insights into the model's learning process, highlighting stable convergence and good generalization capabilities.  
- **Best Model Saved**: The optimal model configuration was saved, ready for deployment in prediction tasks.


---

## 📋 **Overall Summary**

The completion of Task 1 and Task 2 sets a solid foundation for the project’s next phases. The defined and trained ANN model demonstrates the capability to predict customer churn effectively, balancing complexity and performance. With the model now ready, we can proceed to utilize it for real-world predictions, further refining our approach based on business needs and model performance.

📊 **Key Highlights**:
- Robust model architecture tailored for binary classification.
- Efficient training process with early stopping to avoid overfitting.
- Strong model performance with good generalization to unseen data.

With these foundational steps completed, the project is well-positioned to deliver valuable insights into customer churn dynamics, guiding strategic decisions.

---

