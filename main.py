import argparse
import os
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import arff
import pandas as pd
from functions import get_parameter_values, macro_precision_recall_fmeasure_windows
from kohonen import kohonen_offline_global, kohonen_online_bayes_nd

def load_arff_data(file_path):
    with open(file_path, 'r') as f:
        arff_data = arff.load(f)
    attributes = arff_data['attributes']
    data = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in attributes])
    feature_indices = [i for i, attr in enumerate(attributes) if attr[1] == 'NUMERIC']
    label_indices = [i for i, attr in enumerate(attributes) if isinstance(attr[1], list)]
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data, feature_indices, label_indices

def main():
    parser = argparse.ArgumentParser(description="Run the MLSC Kohonen Map experiment.")
    parser.add_argument('param_file', type=str, help='Path to the parameter configuration file.')
    args = parser.parse_args()

    print("Loading parameters")
    parameters = get_parameter_values(args.param_file)
    print("Parameters loaded successfully:")
    for key, value in parameters.items():
        print(f"- {key}: {value}")

    print("\nLoading data")
    train_data, train_feature_indices, train_label_indices = load_arff_data(parameters['train_data'])
    test_data, test_feature_indices, test_label_indices = load_arff_data(parameters['test_data'])
    print("Data loaded successfully.")

    offline_dataset = train_data.iloc[:, train_feature_indices].values
    offline_classes = train_data.iloc[:, train_label_indices]
    online_dataset = test_data.iloc[:, test_feature_indices].values
    online_classes = test_data.iloc[:, test_label_indices]

    print(f"Offline dataset shape: {offline_dataset.shape}")
    print(f"Offline classes shape: {offline_classes.shape}")
    print(f"Online dataset shape: {online_dataset.shape}")
    print(f"Online classes shape: {online_classes.shape}")

    print("\nStandardizing data...")
    scaler = StandardScaler()
    scaler.fit(offline_dataset)
    offline_dataset_scaled = scaler.transform(offline_dataset)
    online_dataset_scaled = scaler.transform(online_dataset)
    print("Data standardized successfully.")

    print("\nStarting Offline Phase")

    num_epochs = int(parameters['num_iterations'])
    num_samples = len(offline_dataset_scaled)
    num_iterations_total = num_epochs * num_samples
    print(f"Training Info: {num_epochs} epochs * {num_samples} samples = {num_iterations_total} total iterations.")

    mapping = kohonen_offline_global(
        offline_dataset=offline_dataset_scaled,
        offline_classes=offline_classes,
        num_it=num_iterations_total,
        init_n=parameters['n0'],
        final_n=parameters['n1'],
        grid_d=int(parameters['grid_dimension']),
        tr_mode=parameters['train_mode'],
        min_ex=int(parameters['min_examples_cluster'])
    )
    print("Offline Phase Completed")
    print("Model mapping created successfully.")

    print("\n--- Starting Online Phase ---")
    init_n = parameters['n0']
    novel_classes = parameters['novel_classes']
    if not isinstance(novel_classes, list):
        novel_classes = [novel_classes]
    update_model_info = bool(parameters['update_model_info'])
    num_offline_instances = len(offline_dataset)

    online_results = kohonen_online_bayes_nd(
        mapping=mapping,
        online_dataset=online_dataset_scaled,
        init_n=init_n,
        novel_classes=novel_classes,
        update_model_info=update_model_info,
        num_offline_instances=num_offline_instances
    )

    print("Online Phase Completed")
    if 'indexes_explained' in online_results:
        print(f"{len(online_results['indexes_explained'])} instances were explained by the model.")

    print("\nEvaluating Results")
    predictions = online_results['predictions']
    explained_indices = online_results['indexes_explained']
    true_classes_explained = online_classes.iloc[explained_indices]
    predicted_classes_explained = predictions.iloc[explained_indices]

    num_windows = int(parameters['num_evaluation_windows'])
    evaluation_metrics = macro_precision_recall_fmeasure_windows(
        true_labels=true_classes_explained.values,
        predicted_labels=predicted_classes_explained.values,
        num_evaluation_windows=num_windows
    )
    online_results['evaluation_metrics'] = evaluation_metrics

    print("Evaluation completed:")
    print(f"  - Macro F-Measure: {evaluation_metrics['ma_fmeasure']:.4f}")
    print(f"  - Macro Precision: {evaluation_metrics['ma_precision']:.4f}")
    print(f"  - Macro Recall: {evaluation_metrics['ma_recall']:.4f}")

    if not os.path.exists('Results'):
        os.makedirs('Results')
    dataset_name = os.path.basename(parameters['test_data']).split('.')[0]
    timestamp = datetime.now().strftime("%H.%M.%S")
    grid_dim = int(parameters['grid_dimension'])
    params_filename = f"Results/{dataset_name}-{timestamp}-parameters-{grid_dim}.txt"
    with open(params_filename, 'w') as f:
        f.write(str(parameters))
    print(f"\nParameters saved to {params_filename}")
    model_filename = f"Results/{dataset_name}-{timestamp}-model-{grid_dim}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(online_results, f)
    print(f"Full results object saved to {model_filename}")

if __name__ == "__main__":
    main()