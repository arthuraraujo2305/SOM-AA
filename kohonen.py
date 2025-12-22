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
                       update_model_information)

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
    
    print("\nOnline phase!")
    initial_number_classes = mapping['class_probabilities'].shape[0]
    all_predictions = []
    all_pred_indices = []
    indexes_explained = []

    # --- CORREÇÃO FUNDAMENTAL AQUI ---
    # Passo 1: Extrair apenas os micro-clusters válidos (filtrados na fase offline)
    valid_mcs = mapping['micro_clusters']
    
    # Se não houver micro-clusters válidos (algo deu muito errado no treino), retornamos vazio
    if not valid_mcs:
        print("Erro Crítico: Nenhum micro-cluster válido encontrado.")
        return {'predictions': pd.DataFrame(), 'indexes_explained': [], 'mapping': mapping}

    # Passo 2: Criar uma matriz apenas com os centróides desses MCs válidos
    valid_centroids = np.array([mc['centroid'] for mc in valid_mcs])
    
    # Passo 3: Configurar o KNN para olhar APENAS para esses centróides
    # O R usa 'n.k <- ceiling(mapping$z)'. Se for par, soma 1.
    n_k_search = int(np.ceil(mapping['z']))
    if n_k_search % 2 == 0: 
        n_k_search += 1
    
    # Garante que não pedimos mais vizinhos do que existem de centróides válidos
    n_k_search = min(n_k_search, len(valid_centroids))
    
    # Treina o KNN apenas com os válidos
    nbrs = NearestNeighbors(n_neighbors=n_k_search).fit(valid_centroids)

    for i, x_instance in enumerate(online_dataset):
        if (i + 1) % 1000 == 0:
            print(f"  Processing instance {i + 1}/{len(online_dataset)}...")

        x = x_instance.reshape(1, -1)
        
        # Busca os vizinhos. 
        # ATENÇÃO: 'indices' aqui retorna a posição na lista 'valid_centroids', 
        # não o ID original do neurônio (neuron_id).
        distances, indices = nbrs.kneighbors(x)

        pred = np.zeros(initial_number_classes)
        
        # Define quantos neurônios vamos consultar (z)
        z_current = mapping['z']
        # No R: z <- min(z, length(winner$nn.index))
        # Aqui garantimos que z não seja maior que o número de vizinhos encontrados
        z = min(int(np.ceil(z_current)), len(indices[0]))

        # Loop pelos vizinhos encontrados (que já são garantidamente válidos)
        for rank_idx in range(z):
            # O índice retornado pelo KNN refere-se à lista valid_mcs
            mc_list_index = indices[0][rank_idx]
            neuron_j_dist = distances[0][rank_idx]
            
            # Recupera o micro-cluster diretamente da lista filtrada
            mc_j = valid_mcs[mc_list_index]
            
            # Como treinamos só com válidos, não precisamos verificar if mc_j is None
            
            prototype_j = mc_j['prototype_vector']
            active_indices = np.where(prototype_j > 0)[0]
            
            # Se o protótipo estiver vazio (sem classes), pulamos (embora MCs validos costumem ter classes)
            if len(active_indices) == 0: 
                continue
            
            active_weights = prototype_j[active_indices]
            # Ordena classes pela relevância no protótipo (decrescente)
            sorted_order = np.argsort(active_weights)[::-1]
            active_classes_sorted = active_indices[sorted_order]

            # --- LÓGICA DO VENCEDOR (rank_idx == 0) ---
            if rank_idx == 0:
                id_max = active_classes_sorted[0]
                pred[id_max] = 1
                
                # Atualização de estatística (sempre ocorre, conforme R)
                if 'average_output' in mc_j:
                     mc_j['average_output'][0] += np.exp(-neuron_j_dist)
                     mc_j['average_output'][1] += 1
            
            # --- LÓGICA DOS VIZINHOS E DEMAIS CLASSES ---
            for class_idx in active_classes_sorted:
                # Se já predito como 1, pula
                if pred[class_idx] == 1: continue

                prob_j_prior = mapping['class_probabilities'][class_idx, class_idx]
                prob_x_j = np.exp(-neuron_j_dist)

                # Regra de Bayes considerando o que JÁ foi predito (pred == 1)
                prob_k_j_cumulative = 1.0  
                
                for k_idx in active_classes_sorted:
                    if pred[k_idx] == 1 and k_idx != class_idx:
                        prob_k_j_cumulative *= mapping['class_probabilities'][k_idx, class_idx]

                prob_j_ks_x = prob_j_prior * prob_k_j_cumulative * prob_x_j
                cond_prob_threshold = mc_j['cond_prob_threshold'][class_idx]

                if prob_j_ks_x > 0 and prob_j_ks_x >= cond_prob_threshold:
                    pred[class_idx] = 1

                    # Se ativou classe extra no vencedor ou é vizinho, atualiza estatística
                    if 'average_output' in mc_j:
                        mc_j['average_output'][0] += np.exp(-neuron_j_dist)
                        mc_j['average_output'][1] += 1

        indexes_explained.append(i)
        all_predictions.append(pred)
        all_pred_indices.append(i)

        # 1. ATUALIZAÇÃO DE PESOS (Centróides) - Online Learning
        if update_model_info:
             # Precisamos passar o 'winner_dict' para a função de update.
             # Como mudamos a lógica do KNN, o 'indices' agora aponta para valid_mcs.
             # A função update_model_information precisa saber o neuron_id REAL para atualizar o mapa global.
             
             # Converte indices da lista filtrada para neuron_ids reais da grid original
             real_neuron_ids = [valid_mcs[idx]['neuron_id'] for idx in indices[0]]
             
             winner_dict = {
                 'nn_index': [real_neuron_ids], # Lista de listas, pois o código original espera estrutura assim para batch
                 'nn_dist': distances # Já é lista de listas (1, k)
             }
             
             # Passamos inst_l=0 pois estamos processando instância a instância (batch de 1)
             mapping = update_model_information(mapping, x, i, init_n, winner_dict, 0)

        # 2. ATUALIZAÇÃO DE ESTATÍSTICAS GLOBAIS (Cardealidade, Totais)
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