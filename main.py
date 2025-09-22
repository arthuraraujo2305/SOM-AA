import argparse
import os
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler

import arff
import pandas as pd

from functions import get_parameter_values
from functions import macro_precision_recall_fmeasure_windows
from kohonen import kohonen_offline_global, kohonen_online_bayes_nd


def load_arff_data(file_path):
    """
    Loads an ARFF file and separates it into a DataFrame, feature indices, and label indices.
    """
    with open(file_path, 'r') as f:
        arff_data = arff.load(f)

    # Get attribute names and types
    attributes = arff_data['attributes']

    # Create a pandas DataFrame
    data = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in attributes])

    # Find feature and label indices based on their type in the ARFF file
    # Typically, features are 'NUMERIC' and labels are 'NOMINAL' {0, 1}
    feature_indices = [i for i, attr in enumerate(attributes) if attr[1] == 'NUMERIC']
    label_indices = [i for i, attr in enumerate(attributes) if isinstance(attr[1], list)]

    # Convert all columns to numeric, handling potential errors
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    return data, feature_indices, label_indices

def main():
    """
    The main script to run the MLSC Kohonen map experiment.
    """

    # Configure the argument parser to accept a configuration file from the command line
    parser = argparse.ArgumentParser(description="Run the MLSC Kohonen Map experiment.")
    parser.add_argument('param_file', type=str,
                        help='Path to the parameter configuration file.')
    args = parser.parse_args()

    # Load parameters from the specified file using our translated function
    print("Loading parameters...")
    parameters = get_parameter_values(args.param_file)

    # Print the loaded parameters to confirm they were read correctly
    print("Parameters loaded successfully:")
    for key, value in parameters.items():
        print(f"- {key}: {value}")

    print("\nLoading data...")

    # Load training data
    train_file_path = parameters['train_data']
    train_data, train_feature_indices, train_label_indices = load_arff_data(train_file_path)

    # Load test data
    test_file_path = parameters['test_data']
    test_data, test_feature_indices, test_label_indices = load_arff_data(test_file_path)

    print("Data loaded successfully.")

    # Separate features and labels for the offline (training) phase
    offline_dataset = train_data.iloc[:, train_feature_indices].values
    offline_classes = train_data.iloc[:, train_label_indices]

    # Separate features and labels for the online (testing) phase
    online_dataset = test_data.iloc[:, test_feature_indices].values
    online_classes = test_data.iloc[:, test_label_indices]

    # Handle the 'novel_classes' parameter to exclude them from initial training
    if 'novel_classes' in parameters and parameters['novel_classes'] != 0:
        # Ensure novel_classes is a list of integers
        novel_class_indices = [int(i) for i in parameters['novel_classes']]

        # Get the original column names before dropping
        original_label_names = offline_classes.columns.tolist()

        # Get the names of the columns to drop
        cols_to_drop = [original_label_names[i] for i in novel_class_indices]

        # Drop the novel classes from the offline training labels
        offline_classes = offline_classes.drop(columns=cols_to_drop)

        print(f"Novelty classes at indices {novel_class_indices} removed for offline training.")

    print(f"Offline dataset shape: {offline_dataset.shape}")
    print(f"Offline classes shape: {offline_classes.shape}")
    print(f"Online dataset shape: {online_dataset.shape}")
    print(f"Online classes shape: {online_classes.shape}")

    # --- NEW: Data Standardization Step ---
    print("\nStandardizing data...")
    scaler = StandardScaler()

    # Fit the scaler ONLY on the training data to learn the mean and std deviation
    scaler.fit(offline_dataset)

    # Apply the learned transformation to both training and test data
    offline_dataset_scaled = scaler.transform(offline_dataset)
    online_dataset_scaled = scaler.transform(online_dataset)

    print("Data standardized successfully.")

    print("\n--- Starting Offline Phase ---")

    # Extract the necessaries parameters of the dictionary to pass to the function
    # This turns the function call cleaner and readable
    num_iterations = int(parameters['num_iterations'])
    init_n = parameters['n0']
    final_n = parameters['n1']
    grid_d = int(parameters['grid_dimension'])
    tr_mode = parameters['train_mode']
    min_ex = int(parameters['min_examples_cluster'])

    mapping = kohonen_offline_global(
        offline_dataset=offline_dataset_scaled,
        offline_classes=offline_classes,
        num_it=num_iterations,
        init_n=init_n,
        final_n=final_n,
        grid_d=grid_d,
        tr_mode=tr_mode,
        min_ex=min_ex
    )

    print("--- Offline Phase Completed ---")
    print("Model mapping created successfully.")

    print("\n--- Starting Online Phase ---")

    #Extract the necessaries parameters to the online function
    init_n = parameters['n0']
    novel_classes = parameters['novel_classes']
    update_model_info = bool(parameters['update_model_info'])
    num_offline_instances = len(offline_dataset)

    if not isinstance(novel_classes, list):
        novel_classes = [novel_classes]

    #Call the online function that we translated in kohonen.py
    online_results = kohonen_online_bayes_nd(
        mapping=mapping,
        online_dataset=online_dataset_scaled,
        init_n=init_n,
        novel_classes=novel_classes,
        update_model_info=update_model_info,
        num_offline_instances=num_offline_instances
    )

    print("--- Online Phase Completed ---")
    if 'indexes_explained' in online_results:
        print(f"{len(online_results['indexes_explained'])} instances were explained by the model.")

    #Evaluate and Save Results
    print("\n--- Evaluating Results ---")

    # Get the predictions and the true labels just for the instances that the model explained
    predictions = online_results['predictions']
    explained_indices = online_results['indexes_explained']

    true_classes_explained = online_classes.iloc[explained_indices]
    predicted_classes_explained = predictions.iloc[explained_indices]

    #Call our translated metric function
    num_windows = int(parameters['num_evaluation_windows'])
    evaluation_metrics = macro_precision_recall_fmeasure_windows(
        true_labels=true_classes_explained.values,
        predicted_labels=predicted_classes_explained.values,
        num_evaluation_windows=num_windows
    )

    #Add the metrics to our results dictionary
    online_results['evaluation_metrics'] = evaluation_metrics

    print("Evaluation completed:")
    print(f"  - Macro F-Measure: {evaluation_metrics['ma_fmeasure']:.4f}")
    print(f"  - Macro Precision: {evaluation_metrics['ma_precision']:.4f}")
    print(f"  - Macro Recall: {evaluation_metrics['ma_recall']:.4f}")


    #Saving results to files
    #Create a folder 'Results' if it doesn't exist
    if not os.path.exists('Results'):
        os.makedirs('Results')

    dataset_name = os.path.basename(parameters['test_data']).split('.')[0]
    timestamp = datetime.now().strftime("%H.%M.%S")
    grid_dim = int(parameters['grid_dimension'])

    #Save the parameters in a txt file
    params_filename = f"Results/{dataset_name}-{timestamp}-parameters-{grid_dim}.txt"
    with open(params_filename, 'w') as f:
        f.write(str(parameters))
    print(f"\nParameters saved to {params_filename}")

    #Save the completed results object (including model and predictions)
    model_filename = f"Results/{dataset_name}-{timestamp}-model-{grid_dim}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(online_results, f)
    print(f"Full results object saved to {model_filename}")

if __name__ == "__main__":
    main()
