import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_training_metrics(training_results_path, save_dir):
    print(f"Training results path: {training_results_path}")
    print(f"Save directory: {save_dir}")

    # Create the 'visualization' directory if it doesn't exist
    visualization_dir = os.path.join(save_dir, 'visualization')
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)
        print(f"Created directory: {visualization_dir}")
    else:
        print(f"Directory already exists: {visualization_dir}")

    # Create the 'training_metrics' directory inside 'visualization'
    training_metrics_dir = os.path.join(visualization_dir, 'training_metrics')
    if not os.path.exists(training_metrics_dir):
        os.makedirs(training_metrics_dir)
        print(f"Created directory: {training_metrics_dir}")
    else:
        print(f"Directory already exists: {training_metrics_dir}")

    # Load the training results
    history_df = pd.read_csv(training_results_path)
    print(f"Loaded training results: {history_df.shape}")

    # Plot training & validation accuracy values
    plt.figure(figsize=(10, 6))
    plt.plot(history_df['accuracy'], label='Train Accuracy')
    plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.grid(True)

    # Save the accuracy plot
    accuracy_plot_path = os.path.join(training_metrics_dir, 'model_accuracy.png')
    plt.savefig(accuracy_plot_path)
    plt.show()  # Ensure it displays in Jupyter Notebook
    plt.close()
    print(f"Saved accuracy plot to: {accuracy_plot_path}")

    # Plot training & validation loss values
    plt.figure(figsize=(10, 6))
    plt.plot(history_df['loss'], label='Train Loss')
    plt.plot(history_df['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.grid(True)

    # Save the loss plot
    loss_plot_path = os.path.join(training_metrics_dir, 'model_loss.png')
    plt.savefig(loss_plot_path)
    plt.show()  # Ensure it displays in Jupyter Notebook
    plt.close()
    print(f"Saved loss plot to: {loss_plot_path}")

    print(f"Plots saved to {training_metrics_dir}")
