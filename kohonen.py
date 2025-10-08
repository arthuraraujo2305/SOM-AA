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

# Em kohonen.py, substitua a função inteira por esta versão CORRIGIDA E SIMPLIFICADA:

def kohonen_offline_global(offline_dataset: np.ndarray, offline_classes: pd.DataFrame, num_it: int,
                           init_n: float, final_n: float, grid_d: int, tr_mode: str, min_ex: int) -> dict:
    """
    Performs the offline training phase.
    THIS IS THE CORRECTED VERSION using MiniSom's built-in training method.
    """
    print("\nOffline phase - building maps!")

    # 1. Initial Calculations
    class_probabilities, class_totals = compute_initial_class_probabilities_totals(offline_classes)
    z = compute_label_cardinality(offline_classes)

    # 2. Initialize and Train the SOM using MiniSom
    num_features = offline_dataset.shape[1]
    np.random.seed(10)

    # Sigma inicial é um parâmetro importante. O MiniSom o decai para perto de 0.
    initial_sigma = grid_d / 2.0

    som = MiniSom(x=grid_d, y=grid_d, input_len=num_features,
                  sigma=initial_sigma,
                  learning_rate=init_n, # MiniSom usa isso como a taxa de aprendizado inicial
                  neighborhood_function='gaussian',
                  random_seed=10)

    # Inicializa os pesos usando PCA, que é nossa melhor abordagem até agora.
    print("Initializing SOM weights with PCA...")
    som.pca_weights_init(offline_dataset)

    # --- AQUI ESTÁ A MUDANÇA FUNDAMENTAL ---
    # Substituímos nosso loop de treinamento manual pelo método de treino da própria biblioteca.
    # Ele gerencia o decaimento do sigma e da taxa de aprendizado internamente.
    print(f"Starting SOM training for {num_it} iterations...")
    som.train(offline_dataset, num_it, verbose=True) # verbose=True mostrará o progresso
    print("SOM training completed.")
    # ------------------------------------

    # 3. Post-processing (O resto da função permanece o mesmo)
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
    Performs the online phase. This version is a more faithful translation
    of the R code's iterative probability calculation logic.
    """
    print("\nOnline phase!")
    time_stamp = 0
    n0 = init_n
    initial_number_classes = mapping['class_probabilities'].shape[0]
    all_predictions = []
    all_pred_indices = []
    indexes_explained = []

    for i, x_instance in enumerate(online_dataset):
        if (i + 1) % 1000 == 0:
            print(f"  Processing instance {i + 1}/{len(online_dataset)}...")

        time_stamp += 1
        x = x_instance.reshape(1, -1)
        n_k = int(np.ceil(mapping['z']))
        if n_k % 2 == 0: n_k += 1

        neuron_centroids = mapping['som_map']['codes']
        if n_k > len(neuron_centroids):
            n_k = len(neuron_centroids)

        nbrs = NearestNeighbors(n_neighbors=n_k).fit(neuron_centroids)
        distances, indices = nbrs.kneighbors(x)

        pred = np.zeros(initial_number_classes)
        explained = True

        z = min(int(np.ceil(mapping['z'])), n_k)
        for j in range(z):
            neuron_j_idx = indices[0][j]
            neuron_j_dist = distances[0][j]
            mc_j = next((mc for mc in mapping['micro_clusters'] if mc['neuron_id'] == neuron_j_idx), None)
            if mc_j is None: continue

            prototype_j = mc_j['prototype_vector']
            active_classes_in_prototype_j = np.where(prototype_j > 0)[0]

            for class_idx in active_classes_in_prototype_j:
                if pred[class_idx] == 1: continue

                prob_j_prior = mapping['class_probabilities'][class_idx, class_idx]
                prob_x_j = np.exp(-neuron_j_dist)

                prob_k_j_cumulative = 1.0
                predicted_indices = np.where(pred == 1)[0]
                for pred_idx in predicted_indices:
                    prob_k_j_cumulative *= mapping['class_probabilities'][pred_idx, class_idx]

                prob_j_ks_x = prob_j_prior * prob_k_j_cumulative * prob_x_j
                cond_prob_threshold = mc_j['cond_prob_threshold'][class_idx]

                # --- AQUI ESTÁ A CORREÇÃO CRUCIAL ---
                if prob_j_ks_x > 0 and prob_j_ks_x >= cond_prob_threshold:
                    # ------------------------------------
                    pred[class_idx] = 1

                    if 'average_output' in mc_j:
                        mc_j['average_output'][0] += np.exp(-neuron_j_dist)
                        mc_j['average_output'][1] += 1

        if explained:
            indexes_explained.append(i)
            all_predictions.append(pred)
            all_pred_indices.append(i)

    predictions_matrix = np.array(all_predictions)
    final_predictions = pd.DataFrame(np.zeros((len(online_dataset), initial_number_classes)), index=np.arange(len(online_dataset)))
    if len(all_pred_indices) > 0:
        final_predictions.iloc[all_pred_indices] = predictions_matrix
    results = {'predictions': final_predictions, 'indexes_explained': indexes_explained, 'mapping': mapping} # simplificado
    return results

def kohonen_online_baseline_predictor(mapping: dict, online_dataset: np.ndarray) -> dict:
    """
    A very simple baseline predictor to test the quality of the SOM map itself.
    It bypasses the complex Bayesian logic and predicts based on a simple threshold
    on the winning neuron's prototype vector.
    """
    print("\n!!! RUNNING ONLINE PHASE WITH SIMPLE BASELINE PREDICTOR !!!")

    initial_number_classes = mapping['class_probabilities'].shape[0]
    all_predictions = []

    neuron_centroids = mapping['som_map']['codes']

    # Process each instance
    for i, x_instance in enumerate(online_dataset):
        if (i + 1) % 1000 == 0:
            print(f"  Processing instance {i + 1}/{len(online_dataset)}...")

        x = x_instance.reshape(1, -1)

        # Find the single winning neuron
        nbrs = NearestNeighbors(n_neighbors=1).fit(neuron_centroids)
        distances, indices = nbrs.kneighbors(x)
        winner_idx = indices[0][0]

        # Find the corresponding micro-cluster
        mc_winner = next((mc for mc in mapping['micro_clusters'] if mc['neuron_id'] == winner_idx), None)

        pred = np.zeros(initial_number_classes)
        if mc_winner:
            prototype = mc_winner['prototype_vector']
            # Simple prediction rule: predict if prototype value is > 0.5
            pred[prototype > 0.5] = 1

        all_predictions.append(pred)

    # Assemble final results (simplified version)
    predictions_matrix = np.array(all_predictions)

    results = {
        'predictions': pd.DataFrame(predictions_matrix),
        'indexes_explained': list(range(len(online_dataset))),
        'mapping': mapping
    }
    return results