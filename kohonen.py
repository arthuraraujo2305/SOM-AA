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
                       update_class_totals_probabilities,
                       update_cond_probabilities_neurons,
                       update_model_information,
                       compute_radius_factor_mc)

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
    #print("Initializing SOM weights with PCA...")
    #som.pca_weights_init(offline_dataset)

    print("Initializing SOM weights randomly (sampling from data)...")
    som.random_weights_init(offline_dataset)

    # Train the SOM using the library's built-in method, which handles decay internally.
    print(f"Starting SOM training (BATCH MODE)) for {num_it} epochs...")
    som.train_batch(offline_dataset, num_it, verbose=True)
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

    # Compute radius factor for each neuron
    micro_clusters = compute_radius_factor_mc(micro_clusters, result_mc['som_map'], offline_dataset)                                               

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
    
    print("\nOnline phase (Dynamic Distances + Radius Check)!")
    initial_number_classes = mapping['class_probabilities'].shape[0]
    all_predictions = []
    all_pred_indices = []
    indexes_explained = []

    # Passo 1: Extrair apenas os micro-clusters válidos
    valid_mcs = mapping['micro_clusters']
    
    if not valid_mcs:
        print("Erro Crítico: Nenhum micro-cluster válido encontrado.")
        return {'predictions': pd.DataFrame(), 'indexes_explained': [], 'mapping': mapping}

    # Pré-alocação para performance
    # Vamos manter uma lista de IDs reais para mapear o índice do array de volta para o ID do neurônio
    valid_neuron_ids = [mc['neuron_id'] for mc in valid_mcs]

    for i, x_instance in enumerate(online_dataset):
        if (i + 1) % 1000 == 0:
            print(f"  Processing instance {i + 1}/{len(online_dataset)}...")

        x = x_instance.reshape(1, -1)
        
        # DISTÂNCIAS DINÂMICAS
        # 1. Pegamos os centróides ATUAIS (eles mudam a cada update!)
        #    Usamos list comprehension pois é rápido o suficiente para ~100 neurônios
        current_centroids = np.array([mc['centroid'] for mc in valid_mcs])
        
        # 2. Calculamos a distância Euclidiana do exemplo x para TODOS os centróides válidos
        #    axis=1 faz o cálculo por linha (neurônio)
        dists = np.linalg.norm(current_centroids - x, axis=1)
        
        # 3. Ordenamos pelos menores distâncias (indices do array local)
        sorted_idxs = np.argsort(dists)
        
        # O Vencedor é o primeiro da lista ordenada
        winner_idx_local = sorted_idxs[0]
        winner_dist = dists[winner_idx_local]
        mc_winner = valid_mcs[winner_idx_local]
        
        # --- VERIFICAÇÃO DO RAIO ---
        r_factor_1 = mc_winner.get('radius_factor_1', float('inf'))
        
        # Se novel_classes == 0, raio é infinito
        if isinstance(novel_classes, list) and len(novel_classes) > 0 and novel_classes[0] == 0:
            r_factor_1 = float('inf')
        elif isinstance(novel_classes, (int, float)) and novel_classes == 0:
            r_factor_1 = float('inf')

        is_explained = False
        
        if winner_dist <= r_factor_1:
            is_explained = True
            
            # --- BLOCO DE PREDIÇÃO
            pred = np.zeros(initial_number_classes)
            z_current = mapping['z']
            # Garante que z não é maior que o número de micro-clusters existentes
            z = min(int(np.ceil(z_current)), len(valid_mcs))

            for rank_idx in range(z):
                # Pega o índice do array local baseado no rank
                mc_idx_local = sorted_idxs[rank_idx]
                neuron_j_dist = dists[mc_idx_local]
                mc_j = valid_mcs[mc_idx_local]
                
                prototype_j = mc_j['prototype_vector']
                active_indices = np.where(prototype_j > 0)[0]
                
                if len(active_indices) == 0: continue
                
                active_weights = prototype_j[active_indices]
                sorted_order = np.argsort(active_weights)[::-1]
                active_classes_sorted = active_indices[sorted_order]

                # Lógica do Vencedor (Rank 0)
                if rank_idx == 0:
                    id_max = active_classes_sorted[0]
                    pred[id_max] = 1
                    if 'average_output' in mc_j:
                        mc_j['average_output'][0] += np.exp(-neuron_j_dist)
                        mc_j['average_output'][1] += 1
                
                # Lógica dos Vizinhos (Bayes)
                for class_idx in active_classes_sorted:
                    if pred[class_idx] == 1: continue

                    prob_j_prior = mapping['class_probabilities'][class_idx, class_idx]
                    prob_x_j = np.exp(-neuron_j_dist)
                    
                    prob_k_j_cumulative = 1.0
                    for k_idx in active_classes_sorted:
                        if pred[k_idx] == 1 and k_idx != class_idx:
                            prob_k_j_cumulative *= mapping['class_probabilities'][k_idx, class_idx]

                    prob_j_ks_x = prob_j_prior * prob_k_j_cumulative * prob_x_j
                    cond_prob_threshold = mc_j['cond_prob_threshold'][class_idx]

                    if prob_j_ks_x > 0 and prob_j_ks_x >= cond_prob_threshold:
                        pred[class_idx] = 1
                        if 'average_output' in mc_j:
                            mc_j['average_output'][0] += np.exp(-neuron_j_dist)
                            mc_j['average_output'][1] += 1

            indexes_explained.append(i)
            all_predictions.append(pred)
            all_pred_indices.append(i)

            # --- ATUALIZAÇÃO DO MODELO (ONLINE LEARNING) ---
            if update_model_info:
                neighbor_real_ids = [valid_mcs[idx]['neuron_id'] for idx in sorted_idxs[:z]]
                neighbor_dists = [dists[idx] for idx in sorted_idxs[:z]]

                winner_dict = {
                    'nn_index': [neighbor_real_ids],
                    'nn_dist': [neighbor_dists]
                }
                
                mapping = update_model_information(mapping, x, i, init_n, winner_dict, 0)

            # Atualização de Estatísticas Globais
            pred_row = pred.reshape(1, -1)
            mapping = update_class_totals_probabilities(mapping, pred_row, 1, initial_number_classes, 0, num_offline_instances)
            
            N = mapping['total_instances']
            z_old = mapping['z']
            cardinality_current = np.sum(pred)
            mapping['z'] = ((N - 1) * z_old + cardinality_current) / N
            mapping['micro_clusters'] = update_cond_probabilities_neurons(mapping['micro_clusters'], mapping['class_probabilities'])

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