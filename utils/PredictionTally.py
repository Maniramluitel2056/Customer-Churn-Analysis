import os
import json
import pandas as pd

def run_prediction_tally(base_dir):
    # Load the configuration file from the main directory
    config_path = os.path.join(base_dir, 'config.json')
    with open(config_path) as config_file:
        config = json.load(config_file)

    # Define paths using absolute paths
    predictions_path = os.path.join(base_dir, config['predictions_path'])
    test_data_path = os.path.join(base_dir, config['test_data_path'])

    # Load the predictions.csv
    predictions_df = pd.read_csv(predictions_path)

    # Load the test_data.csv
    test_data_df = pd.read_csv(test_data_path)

    # Tally the predictions with the actual values
    comparison_df = pd.DataFrame({
        'Actual': predictions_df['Actual'],
        'Predicted': predictions_df['Predicted'],
        'Churn_Yes (Test Data)': test_data_df['Churn_Yes']
    })

    # Check if Actual matches Churn_Yes in the test data
    comparison_df['Match'] = comparison_df['Actual'] == comparison_df['Churn_Yes (Test Data)']

    # Save the tally results to a new CSV file
    tally_results_path = os.path.join(base_dir, config['results_path'], 'tally_results.csv')
    comparison_df.to_csv(tally_results_path, index=False)

    # Output the result
    print(f"Tally results saved to {tally_results_path}")
    if len(comparison_df[comparison_df['Match'] == False]) > 0:
        print(f"Total mismatches: {len(comparison_df[comparison_df['Match'] == False])}")
        print("Mismatches found:")
        print(comparison_df[comparison_df['Match'] == False])
    else:
        print("All predictions match the actual values in test data.")
