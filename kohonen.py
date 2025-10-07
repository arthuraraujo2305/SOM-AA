import numpy as np
import pandas as pd
from minisom import MiniSom
from collections import Counter
from sklearn.neighbors import NearestNeighbors

from functions import (compute_initial_class_probabilities_totals,
                       compute_label_cardinality,
                       compute_micro_clusters,
                       get_average_neuron_outputs,
                       get_cond_probabilities_neurons,
                       update_model_information,
                       update_cond_probabilities_neurons,
                       update_class_totals_probabilities,
                       )

def kohonen_offline_global(offline_dataset: np.ndarray, offline_classes: pd.DataFrame, num_it: int,
                           init_n: float, final_n: float, grid_d: int, tr_mode: str, min_ex: int) -> dict:
    """
    Performs the offline training phase of the Kohonen map with manual linear decay.
    """
    print("\nOffline phase - building maps!")

    # 1. Initial Calculations (using our functions.py module)
    prob_results = compute_initial_class_probabilities_totals(offline_classes)
    class_totals = prob_results[1]
    class_probabilities = prob_results[0]
    z = compute_label_cardinality(offline_classes)

    # 2. Initialize and Train the SOM using MiniSom
    num_features = offline_dataset.shape[1]

    # Set a seed for reproducibility
    np.random.seed(10)

    # MiniSom initialization - Note: sigma starts higher now.
    initial_sigma = grid_d / 2.0
    som = MiniSom(x=grid_d, y=grid_d, input_len=num_features,
                  sigma=initial_sigma,
                  learning_rate=init_n,
                  neighborhood_function='gaussian',
                  random_seed=10)

    # Initialize weights with random samples from the dataset
    som.random_weights_init(offline_dataset)

    # --- NEW: Manual Training Loop for Linear Decay ---
    # This block replaces the simple som.train_random() to mimic R's linear decay.
    print("Starting SOM training with manual linear decay...")
    for t in range(num_it):
        # Linearly decay learning rate from init_n to final_n
        learning_rate_t = init_n + (final_n - init_n) * (t / num_it)

        # Linearly decay sigma from initial_sigma to 1.0
        sigma_t = initial_sigma + (1.0 - initial_sigma) * (t / num_it)

        # Update MiniSom's internal parameters for this iteration
        som._learning_rate = learning_rate_t
        som._sigma = sigma_t

        # Pick a random data sample and update the map
        rand_i = np.random.randint(len(offline_dataset))
        sample = offline_dataset[rand_i]
        som.update(sample, som.winner(sample), t, num_it)
    print("SOM training completed.")
    # --- END of Manual Training Loop ---

    # --- 3. Post-processing to match R's 'som.map' structure ---
    # (O resto da função continua igual)
    unit_classif = np.zeros(len(offline_dataset), dtype=int)
    distances = np.zeros(len(offline_dataset), dtype=float)

    for i, x in enumerate(offline_dataset):
        winner_pos = som.winner(x)
        winner_idx = np.ravel_multi_index(winner_pos, (grid_d, grid_d))
        unit_classif[i] = winner_idx
        distances[i] = np.linalg.norm(x - som.get_weights()[winner_pos])

    som_map = {
        'codes': som.get_weights().reshape(-1, num_features),
        'unit.classif': unit_classif,
        'distances': distances
    }

    # --- 4. Compute Micro-Cluster properties ---
    result_mc = compute_micro_clusters(som_map, offline_classes, min_ex)

    average_output_som_map = get_average_neuron_outputs(result_mc['som_map'], len(result_mc['micro_clusters']))

    micro_clusters = get_cond_probabilities_neurons(result_mc['micro_clusters'],
                                                    class_probabilities,
                                                    average_output_som_map)

    # --- 5. Assemble the final results dictionary ---
    result = {
        'som_map': result_mc['som_map'],
        'micro_clusters': micro_clusters,
        'z': z,
        'class_probabilities': class_probabilities,
        'class_totals': class_totals,
        'total_instances': len(offline_dataset),
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
    Performs the online phase of the algorithm, processing new data instance by instance.
    This is the complete and faithful translation of the original R code.
    """
    print("\nOnline phase!")
    time_stamp = 0
    n0 = init_n

    #Initialization
    initial_number_classes = mapping['class_probabilities'].shape[0]
    all_predictions = []
    all_pred_indices = []

    indexes_explained = []
    indexes_unknown_classified = []
    indexes_unknown_unclassified = []
    all_unclassified_indexes = []
    time_stamp_stm_removed = []

    # Main Loop: Process each instance
    for i, x_instance in enumerate(online_dataset):
        time_stamp += 1
        x = x_instance.reshape(1, -1) # Reshape for k-NN

        # Find Nearest Neurons (k-NN)
        n_k = int(np.ceil(mapping['z']))
        if n_k % 2 == 0: n_k += 1

        neuron_centroids = mapping['som_map']['codes']
        if n_k > len(neuron_centroids):
            n_k = len(neuron_centroids)

        nbrs = NearestNeighbors(n_neighbors=n_k).fit(neuron_centroids)
        distances, indices = nbrs.kneighbors(x)

        print(f"\n--- DEBUGGING INSTANCE {i} ---")
        print(f"Top {n_k} winning neuron IDs: {indices[0]}")
        print(f"Distances to winners: {[f'{d:.4f}' for d in distances[0]]}")

        winner_dist = distances[0][0]
        winner_idx = indices[0][0]

        # Check if the pattern is "known" (Novelty Detection)
        mc_winner = next((mc for mc in mapping['micro_clusters'] if mc['neuron_id'] == winner_idx), None)

        if mc_winner is None: continue

        radius_factor_1 = mc_winner.get('radius_factor_1', float('inf'))
        radius_factor_2 = mc_winner.get('radius_factor_2', float('inf'))

        pred = np.zeros(initial_number_classes)
        explained = False

        if novel_classes[0] == 0 or winner_dist <= radius_factor_1:
            explained = True

            # START OF FULL PREDICTION LOGIC
            z = min(int(np.ceil(mapping['z'])), n_k)

            for j in range(z): # Loop through the z-nearest neurons
                neuron_j_idx = indices[0][j]
                neuron_j_dist = distances[0][j]

                mc_j = next((mc for mc in mapping['micro_clusters'] if mc['neuron_id'] == neuron_j_idx), None)
                if mc_j is None: continue

                prototype_j = mc_j['prototype_vector']

                print(f"\n  --- Analyzing Neuron J={j}, ID={neuron_j_idx} ---")
                print(f"  Prototype Vector (first 5 values): {[f'{p:.2f}' for p in prototype_j[:5]]}")

                if j == 0: #First winning neuron
                    mc_j['average_output'][0] += np.exp(-neuron_j_dist)
                    mc_j['average_output'][1] += 1

                    id_max = np.argmax(prototype_j)
                    pred[id_max] = 1

                    sorted_indices = np.argsort(prototype_j)[::-1]
                    active_classes_k = sorted_indices[prototype_j[sorted_indices] > 0]

                    for class_idx in active_classes_k:
                        if class_idx == id_max: continue

                        prob_j = mapping['class_probabilities'][class_idx, class_idx]
                        prob_x_j = np.exp(-neuron_j_dist)
                        prob_k_j = 1.0

                        for k_idx in active_classes_k:
                            if pred[k_idx] == 1 and k_idx != class_idx:
                                prob_k_j *= mapping['class_probabilities'][k_idx, class_idx]

                        prob_j_ks_x = prob_j * prob_k_j * prob_x_j

                        cond_prob_threshold = mc_j['cond_prob_threshold'][class_idx]

                        print(f"    - Checking Class {class_idx}:")
                        print(f"      - Calculated Prob P(y_j|...): {prob_j_ks_x:.6f}")
                        print(f"      - Threshold to beat:         {cond_prob_threshold:.6f}")

                        if prob_j_ks_x >= cond_prob_threshold:
                            pred[class_idx] = 1
                            print("      - DECISION: PREDICTED (1)")

                        else:
                            print("      - DECISION: NOT PREDICTED (0)")

                else: # For the next closest neurons (j > 1)
                    active_classes_j = np.where(prototype_j > 0)[0]
                    for class_idx in active_classes_j:
                        if pred[class_idx] == 0:
                            prob_j = mapping['class_probabilities'][class_idx, class_idx]
                            prob_x_j = np.exp(-neuron_j_dist)
                            prob_k_j = 1.0

                            for k in range(j): # Check previously considered neurons
                                prev_neuron_idx = indices[0][k]
                                mc_k = next((mc for mc in mapping['micro_clusters'] if mc['neuron_id'] == prev_neuron_idx), None)
                                if mc_k is None: continue

                                prototype_k = mc_k['prototype_vector']
                                active_classes_k = np.where(prototype_k > 0)[0]

                                for k_idx in active_classes_k:
                                    if pred[k_idx] == 1 and k_idx != class_idx:
                                        prob_k_j *= mapping['class_probabilities'][k_idx, class_idx]

                            prob_j_ks_x = prob_j * prob_k_j * prob_x_j
                            cond_prob_threshold = mc_j['cond_prob_threshold'][class_idx]

                            print(f"    - Checking Class {class_idx} (as new):")
                            print(f"      - Calculated Prob P(y_j|...): {prob_j_ks_x:.6f}")
                            print(f"      - Threshold to beat:         {cond_prob_threshold:.6f}")

                            if prob_j_ks_x >= cond_prob_threshold:
                                mc_j['average_output'][0] += np.exp(-neuron_j_dist)
                                mc_j['average_output'][1] += 1
                                pred[class_idx] = 1
                                print("      - DECISION: PREDICTED (1)")

                            else:
                                print("      - DECISION: NOT PREDICTED (0)")
            print(f"\n  --- FINAL PREDICTION VECTOR (first 15 values): {pred[:15].astype(int)} ---")

            if winner_dist > radius_factor_2:
                indexes_unknown_classified.append(i)

        if explained:
            indexes_explained.append(i)
            all_predictions.append(pred)
            all_pred_indices.append(i)

            if update_model_info:
                winner_obj = {'nn_index': indices, 'nn_dist': distances}

                mapping = update_model_information(mapping, x.flatten(), time_stamp, n0, winner_obj, 0)
                mapping = update_class_totals_probabilities(mapping, pred.reshape(1, -1), 1,
                                                            initial_number_classes, 0,
                                                            num_offline_instances)
                mapping['micro_clusters'] = update_cond_probabilities_neurons(mapping['micro_clusters'], mapping['class_probabilities'])

            mapping['z'] = ((mapping['total_instances'] - 1) * mapping['z'] + np.sum(pred)) / mapping['total_instances']

    # Assemble final results
    predictions_matrix = np.array(all_predictions)
    # Re-index the predictions matrix to match original instance indices
    final_predictions = pd.DataFrame(np.zeros((len(online_dataset), initial_number_classes)),
                                     index=np.arange(len(online_dataset)))
    if len(all_pred_indices) > 0:
        final_predictions.iloc[all_pred_indices] = predictions_matrix

    results = {
        'predictions': final_predictions,
        'indexes_explained': indexes_explained,
        'indexes_unknown_classified': indexes_unknown_classified,
        'indexes_unknown_unclassified': indexes_unknown_unclassified,
        'all_unclassified_indexes': all_unclassified_indexes,
        'time_stamp_stm_removed': time_stamp_stm_removed,
        'mapping': mapping
    }

    return results