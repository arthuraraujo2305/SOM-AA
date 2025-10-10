import numpy as np
import pandas as pd
from minisom import MiniSom
from collections import Counter
from sklearn.neighbors import NearestNeighbors

from functions import (compute_initial_class_probabilities_totals,
                       compute_label_cardinality,
                       compute_micro_clusters,
                       get_average_neuron_outputs,
                       get_cond_probabilities_neurons)

def kohonen_offline_global(offline_dataset: np.ndarray, offline_classes: pd.DataFrame, num_it: int,
                           init_n: float, final_n: float, grid_d: int, tr_mode: str, min_ex: int) -> dict:
    """
    Performs the offline training phase using MiniSom's built-in training method.
    """
    print("\nOffline phase - building maps!")

    # 1. Initial Calculations
    class_probabilities, class_totals = compute_initial_class_probabilities_totals(offline_classes)
    z = compute_label_cardinality(offline_classes)

    # 2. Initialize and Train the SOM
    num_features = offline_dataset.shape[1]
    np.random.seed(10)
    initial_sigma = grid_d / 2.0

    som = MiniSom(x=grid_d, y=grid_d, input_len=num_features,
                  sigma=initial_sigma,
                  learning_rate=init_n,
                  neighborhood_function='gaussian',
                  random_seed=10)

    # Initialize weights using Principal Component Analysis for a better starting map.
    print("Initializing SOM weights with PCA...")
    som.pca_weights_init(offline_dataset)

    # Train the SOM using the library's built-in method, which handles decay internally.
    print(f"Starting SOM training for {num_it} iterations...")
    som.train(offline_dataset, num_it, verbose=True)
    print("SOM training completed.")

    # 3. Post-processing: Map data points to neurons and calculate distances
    unit_classif = np.zeros(len(offline_dataset), dtype=int)
    distances = np.zeros(len(offline_dataset), dtype=float)
    weights = som.get_weights() # Cache weights for efficiency
    for i, x in enumerate(offline_dataset):
        winner_pos = som.winner(x)
        winner_idx = np.ravel_multi_index(winner_pos, (grid_d, grid_d))
        unit_classif[i] = winner_idx
        distances[i] = np.linalg.norm(x - weights[winner_pos])

    som_map = {
        'codes': weights.reshape(-1, num_features),
        'unit.classif': unit_classif,
        'distances': distances
    }

    # 4. Compute Micro-Cluster properties
    result_mc = compute_micro_clusters(som_map, offline_classes, min_ex)
    average_output_som_map = get_average_neuron_outputs(result_mc['som_map'])
    micro_clusters = get_cond_probabilities_neurons(result_mc['micro_clusters'],
                                                    class_probabilities,
                                                    average_output_som_map)

    # 5. Assemble the final results dictionary
    result = {
        'som_map': result_mc['som_map'],
        'micro_clusters': micro_clusters,
        'z': z,
        'class_probabilities': class_probabilities,
        'class_totals': class_totals,
        'total_instances': len(offline_dataset),
        # Placeholders for novelty detection features
        'NP': 0,
        'total_instances_np': [],
        'novel_patterns_time_stamp': []
    }

    instances_per_neuron = Counter(unit_classif)
    result['min_instances_neuron'] = min(instances_per_neuron.values()) if instances_per_neuron else 0
    result['theta'] = grid_d * grid_d * result['min_instances_neuron']

    return result

def kohonen_online_bayes_nd(mapping: dict, online_dataset: np.ndarray, init_n: float,
                            novel_classes: list, update_model_info: bool,
                            num_offline_instances: int) -> dict:
    """
    Performs the online prediction phase using the Bayesian logic from the paper.
    Includes the fix for the "zero vs. zero" comparison bug.
    """
    print("\nOnline phase!")
    initial_number_classes = mapping['class_probabilities'].shape[0]
    all_predictions = []
    all_pred_indices = []
    indexes_explained = []

    for i, x_instance in enumerate(online_dataset):
        if (i + 1) % 1000 == 0:
            print(f"  Processing instance {i + 1}/{len(online_dataset)}...")

        x = x_instance.reshape(1, -1)

        # Determine number of nearest neighbors to find
        n_k = int(np.ceil(mapping['z']))
        if n_k % 2 == 0: n_k += 1

        neuron_centroids = mapping['som_map']['codes']
        if n_k > len(neuron_centroids):
            n_k = len(neuron_centroids)

        nbrs = NearestNeighbors(n_neighbors=n_k).fit(neuron_centroids)
        distances, indices = nbrs.kneighbors(x)

        pred = np.zeros(initial_number_classes)

        # Iterate through the 'z' closest neurons to make predictions
        z = min(int(np.ceil(mapping['z'])), n_k)
        for j in range(z):
            neuron_j_idx = indices[0][j]
            neuron_j_dist = distances[0][j]
            mc_j = next((mc for mc in mapping['micro_clusters'] if mc['neuron_id'] == neuron_j_idx), None)

            if mc_j is None: continue

            prototype_j = mc_j['prototype_vector']
            active_classes_in_prototype_j = np.where(prototype_j > 0)[0]

            # Prediction logic based on cumulative probability
            for class_idx in active_classes_in_prototype_j:
                if pred[class_idx] == 1: continue

                prob_j_prior = mapping['class_probabilities'][class_idx, class_idx]
                prob_x_j = np.exp(-neuron_j_dist)

                # Calculate cumulative conditional probability based on already predicted labels
                prob_k_j_cumulative = 1.0
                predicted_indices = np.where(pred == 1)[0]
                for pred_idx in predicted_indices:
                    prob_k_j_cumulative *= mapping['class_probabilities'][pred_idx, class_idx]

                prob_j_ks_x = prob_j_prior * prob_k_j_cumulative * prob_x_j
                cond_prob_threshold = mc_j['cond_prob_threshold'][class_idx]

                # Fix: Only predict if the calculated probability is greater than zero,
                # to avoid the "0 >= 0" "bug" that was happening
                if prob_j_ks_x > 0 and prob_j_ks_x >= cond_prob_threshold:
                    pred[class_idx] = 1

                    # Update average output if model update is enabled
                    if update_model_info and 'average_output' in mc_j:
                        mc_j['average_output'][0] += np.exp(-neuron_j_dist)
                        mc_j['average_output'][1] += 1

        # Store prediction for this instance
        indexes_explained.append(i)
        all_predictions.append(pred)
        all_pred_indices.append(i)

        # Logic for updating the model online (if enabled)
        if update_model_info:
            # The full update logic would go here
            pass

    # Assemble the final results dictionary
    predictions_matrix = np.array(all_predictions)
    final_predictions = pd.DataFrame(np.zeros((len(online_dataset), initial_number_classes)), index=np.arange(len(online_dataset)))
    if len(all_pred_indices) > 0:
        final_predictions.iloc[all_pred_indices] = predictions_matrix

    results = {
        'predictions': final_predictions,
        'indexes_explained': indexes_explained,
        'mapping': mapping
    }
    return results

def kohonen_online_baseline_predictor(mapping: dict, online_dataset: np.ndarray) -> dict:
    """
    A simple baseline predictor used for debugging and analysis.
    Predicts based on a simple threshold on the winning neuron's prototype vector.
    """
    print("\n!!! RUNNING ONLINE PHASE WITH SIMPLE BASELINE PREDICTOR !!!")
    initial_number_classes = mapping['class_probabilities'].shape[0]
    all_predictions = []
    neuron_centroids = mapping['som_map']['codes']

    for i, x_instance in enumerate(online_dataset):
        if (i + 1) % 1000 == 0:
            print(f"  Processing instance {i + 1}/{len(online_dataset)}...")
        x = x_instance.reshape(1, -1)

        # Find the single winning neuron
        nbrs = NearestNeighbors(n_neighbors=1).fit(neuron_centroids)
        distances, indices = nbrs.kneighbors(x)
        winner_idx = indices[0][0]

        mc_winner = next((mc for mc in mapping['micro_clusters'] if mc['neuron_id'] == winner_idx), None)

        pred = np.zeros(initial_number_classes)
        if mc_winner:
            prototype = mc_winner['prototype_vector']
            pred[prototype > 0.5] = 1 # Simple prediction rule

        all_predictions.append(pred)

    predictions_matrix = np.array(all_predictions)
    results = {
        'predictions': pd.DataFrame(predictions_matrix),
        'indexes_explained': list(range(len(online_dataset))),
        'mapping': mapping
    }
    return results