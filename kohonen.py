import numpy as np
import pandas as pd
from minisom import MiniSom
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from growing_som import GrowingSOM

from functions import (
    compute_initial_class_probabilities_totals,
    compute_label_cardinality,
    compute_micro_clusters,
    get_average_neuron_outputs,
    get_cond_probabilities_neurons,
    update_class_totals_probabilities,
    update_cond_probabilities_neurons,
    update_model_information,
    compute_radius_factor_mc,
    run_novelty_detection,
    forget_obsolete_information,
)

def kohonen_offline_global(offline_dataset: np.ndarray, offline_classes: pd.DataFrame, num_it: int,
                           init_n: float, sf: float, min_ex: int) -> dict:
    print("\nOffline phase - building global G-SOM!")

    class_probabilities, class_totals = compute_initial_class_probabilities_totals(offline_classes)
    z = compute_label_cardinality(offline_classes)

    num_features = offline_dataset.shape[1]
    
    # Cálculo do Growth Threshold baseado no Spread Factor de Alahakoon
    # sf deve estar entre 0 e 1 (ex: 0.5)
    sf_safe = max(min(sf, 0.999), 0.001) 
    growth_threshold = -num_features * np.log(sf_safe)

    print(f"Inicializando Global G-SOM (SF={sf_safe}, GT={growth_threshold:.4f})...")
    
    global_gsom = GrowingSOM(
        input_dim=num_features,
        growth_threshold=growth_threshold,
        spread_factor=0.9,
        learning_rate=init_n,
        max_nodes=80, # Limite generoso para o mapa global
        random_seed=10
    )

    print(f"Starting G-SOM training for {num_it} epochs...")
    global_gsom.train(offline_dataset, num_epochs=num_it)
    print("G-SOM training completed.")

    # Reconstruindo o som_map no formato que suas outras funções esperam
    nodes_dict = global_gsom.get_node_weights()
    node_positions = list(nodes_dict.keys())
    codes = np.array(list(nodes_dict.values()))
    
    unit_classif = np.zeros(len(offline_dataset), dtype=int)
    distances = np.zeros(len(offline_dataset), dtype=float)

    for i, x in enumerate(offline_dataset):
        bmu_pos, bmu_dist = global_gsom._find_bmu(x)
        bmu_idx = node_positions.index(bmu_pos)
        unit_classif[i] = bmu_idx
        distances[i] = bmu_dist

    som_map = {
        'codes': codes,
        'unit.classif': unit_classif,
        'distances': distances,
        'node_positions': node_positions # Salvamos as posições reais (x,y)
    }

    result_mc = compute_micro_clusters(som_map, offline_classes, min_ex)
    average_output_som_map = get_average_neuron_outputs(result_mc['som_map'])
    micro_clusters = get_cond_probabilities_neurons(
        result_mc['micro_clusters'],
        class_probabilities,
        average_output_som_map
    )

    micro_clusters = compute_radius_factor_mc(micro_clusters, result_mc['som_map'], offline_dataset)

    result = {
        'gsom_model': global_gsom, # SALVANDO A INSTÂNCIA DO G-SOM GLOBAL AQUI
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
    result['theta'] = len(nodes_dict) * result['min_instances_neuron']

    return result

def kohonen_online_bayes_nd(mapping: dict, online_dataset: np.ndarray, init_n: float,
                            novel_classes: list, update_model_info: bool,
                            num_offline_instances: int,
                            theta: float, min_ex: int, window_stm_check: int,
                            sf_online: float) -> dict:

    print("\n[DEBUG INÉRCIA] Verificando totais antes da Fase Online:")
    print(f"Total Instances (Denominador Global): {mapping.get('total_instances', 'NÃO ENCONTRADO')}")
    if 'class_totals' in mapping:
        print(f"Soma da Matriz class_totals: {np.sum(mapping['class_totals'])}")
        print(f"Exemplo class_totals[0,0]: {mapping['class_totals'][0,0]}")
    else:
        print("Matriz class_totals NÃO ENCONTRADA no mapping!")

    print("\nOnline phase (Dynamic Distances + Radius Check)!")

    #Atualiza o spread factor
    if 'gsom_model' in mapping:
        global_gsom = mapping['gsom_model']
        print(f"\n[TRANSITION] Atualizando G-SOM para Fase Online com SF={sf_online}")
        global_gsom.update_spread_factor(sf_online)
        
        #zerando os erros acumulados da fase offline
        for pos in global_gsom.errors.keys():
            global_gsom.errors[pos] = 0.0
        print("[TRANSITION] Erros residuais do treino offline zerados.")
    else:
        print("[AVISO] gsom_model não encontrado no mapping. SF não atualizado.")

    base_number_classes = mapping['class_probabilities'].shape[0]
    current_number_classes = base_number_classes
    all_predictions = []
    all_pred_indices = []
    indexes_explained = []
    num_extensions_detected = 0
    num_nps_detected = 0
    window_predictions = []

    valid_mcs = mapping['micro_clusters']
    if not valid_mcs:
        print("Erro Crítico: Nenhum micro-cluster válido encontrado.")
        return {'predictions': pd.DataFrame(), 'indexes_explained': [], 'mapping': mapping}

    short_term_memory = []
    stm_indices = []

    for i, x_instance in enumerate(online_dataset):
        if (i + 1) % 1000 == 0:
            print(f"  Processing instance {i + 1}/{len(online_dataset)}...")

        valid_mcs = mapping['micro_clusters']
        if not valid_mcs:
            print("Erro Crítico: Nenhum micro-cluster válido encontrado durante a fase online.")
            break

        x = x_instance.reshape(1, -1)
        current_centroids = np.array([mc['centroid'] for mc in valid_mcs])

        dists = np.linalg.norm(current_centroids - x, axis=1)
        sorted_idxs = np.argsort(dists)

        winner_idx_local = sorted_idxs[0]
        winner_dist = dists[winner_idx_local]
        mc_winner = valid_mcs[winner_idx_local]

        r_factor_1 = mc_winner.get('radius_factor_1', float('inf'))

        if winner_dist <= r_factor_1:
            pred = np.zeros(current_number_classes)
            z_current = mapping['z']
            z = min(int(np.ceil(z_current)), len(valid_mcs))

            for rank_idx in range(z):
                mc_idx_local = sorted_idxs[rank_idx]
                neuron_j_dist = dists[mc_idx_local]
                mc_j = valid_mcs[mc_idx_local]

                prototype_j = mc_j['prototype_vector']
                active_indices = np.where(prototype_j > 0)[0]

                if len(active_indices) == 0:
                    continue

                active_weights = prototype_j[active_indices]
                sorted_order = np.argsort(active_weights)[::-1]
                active_classes_sorted = active_indices[sorted_order]

                if rank_idx == 0:
                    id_max = active_classes_sorted[0]
                    pred[id_max] = 1
                    if 'average_output' in mc_j:
                        mc_j['average_output'][0] += np.exp(-neuron_j_dist)
                        mc_j['average_output'][1] += 1
                    mc_j['last_timestamp'] = i

                debug_mode = ('NP' in str(mc_j.get('neuron_id', ''))) and (prob_j_ks_x < cond_prob_threshold)
                for class_idx in active_classes_sorted:
                    if pred[class_idx] == 1:
                        continue

                    prob_j_prior = mapping['class_probabilities'][class_idx, class_idx]
                    prob_x_j = np.exp(-neuron_j_dist)

                    prob_k_j_cumulative = 1.0
                    for k_idx in active_classes_sorted:
                        if pred[k_idx] == 1 and k_idx != class_idx:
                            prob_k_j_cumulative *= mapping['class_probabilities'][k_idx, class_idx]

                    prob_j_ks_x = prob_j_prior * prob_k_j_cumulative * prob_x_j
                    cond_prob_threshold = mc_j['cond_prob_threshold'][class_idx]

                    #if debug_mode:
                        #print(f"\n[VETO BAYESIANO] Instância {i} | Neurônio: {mc_j['neuron_id']} | Classe: {class_idx}")
                        #print(f"Eq 6 (Calculado): {prob_j_ks_x:.10f}")
                        #print(f"Eq 7 (Threshold): {cond_prob_threshold:.10f}")
                        #print(f"Distância Geométrica: {neuron_j_dist:.4f} (Raio Máximo: {r_factor_1:.4f})")
                        #print(f"Status: Geometria APROVOU, Bayes VETOU -> Foi para Unknown")

                    # Se for NP, confia na Geometria. Se for classe velha, usa Bayes.
                    is_np = 'NP' in str(mc_j.get('neuron_id', ''))
                    if is_np or (prob_j_ks_x > 0 and prob_j_ks_x >= cond_prob_threshold):
                        pred[class_idx] = 1
                        if 'average_output' in mc_j:
                            mc_j['average_output'][0] += np.exp(-neuron_j_dist)
                            mc_j['average_output'][1] += 1
                        mc_j['last_timestamp'] = i

            indexes_explained.append(i)
            all_predictions.append(pred)
            all_pred_indices.append(i)
            window_predictions.append(pred.copy())

            if update_model_info:
                neighbor_real_ids = [valid_mcs[idx]['neuron_id'] for idx in sorted_idxs[:z]]
                neighbor_dists = [dists[idx] for idx in sorted_idxs[:z]]

                winner_dict = {
                    'nn_index': [neighbor_real_ids],
                    'nn_dist': [neighbor_dists]
                }

                mapping = update_model_information(mapping, x, i, init_n, winner_dict, 0)

            pred_row = pred.reshape(1, -1)
            mapping = update_class_totals_probabilities(mapping, pred_row, 1, base_number_classes, 0, num_offline_instances)
            mapping['micro_clusters'] = update_cond_probabilities_neurons(mapping['micro_clusters'], mapping['class_probabilities'])

        else:
            short_term_memory.append(x_instance)
            stm_indices.append(i)

            pred = np.zeros(current_number_classes)
            all_predictions.append(pred)
            all_pred_indices.append(i)

            if (i + 1) % window_stm_check == 0 and len(short_term_memory) >= min_ex:
                mapping, short_term_memory, stm_indices, detected_events = run_novelty_detection(
                    mapping, short_term_memory, stm_indices, min_ex, i
                )

                for event in detected_events:
                    if event['type'] == 'extension':
                        num_extensions_detected += 1
                        print(f"   [EXT] Grupo da STM absorvido como extensão de {event['target_mc']}")

                        closest_mc = next(mc for mc in mapping['micro_clusters'] if mc['neuron_id'] == event['target_mc'])
                        pred_ext = np.zeros(current_number_classes)

                        active_labels = np.where(closest_mc['prototype_vector'] > 0.5)[0]
                        pred_ext[active_labels] = 1.0

                        for original_idx in event['indices']:
                            try:
                                list_idx = all_pred_indices.index(original_idx)
                                all_predictions[list_idx] = pred_ext.copy()

                                if original_idx not in indexes_explained:
                                    indexes_explained.append(original_idx)

                                pred_row = pred_ext.reshape(1, -1)
                                mapping = update_class_totals_probabilities(
                                    mapping, pred_row, 1, base_number_classes, 0, num_offline_instances
                                )
                                window_predictions.append(pred_ext.copy())

                            except ValueError:
                                pass

                        mapping['micro_clusters'] = update_cond_probabilities_neurons(
                            mapping['micro_clusters'], mapping['class_probabilities']
                        )

                    elif event['type'] == 'NP':
                        num_nps_detected += 1
                        np_id = event['np_id']
                        np_class_idx = event['np_class_idx']
                        centroid_np = event['centroid']

                        print(f"   [+] Transformando em Novelty Pattern: NP_{np_id} e EXPANDINDO REDE GLOBAL")

                        global_gsom = mapping['gsom_model']
                        
                        # 1. Encontra o BMU na rede global para a novidade
                        bmu_pos, _ = global_gsom._find_bmu(centroid_np)
                        
                        # 2. Força o erro desse BMU para disparar o crescimento
                        global_gsom.errors[bmu_pos] += global_gsom.growth_threshold + 1.0
                        
                        # 3. Dispara o crescimento ao redor do BMU
                        nodes_before = len(global_gsom.nodes)
                        global_gsom._grow_around_bmu(bmu_pos)
                        nodes_after = len(global_gsom.nodes)
                        
                        if nodes_after > nodes_before:
                            print(f"       -> Mapa global expandiu! Novos neurônios criados ao redor de {bmu_pos}.")
                        
                        # Atualiza os códigos e posições do som_map global
                        mapping['som_map']['codes'] = np.array(list(global_gsom.nodes.values()))
                        mapping['som_map']['node_positions'] = list(global_gsom.nodes.keys())

                        for idx_pred in range(len(all_predictions)):
                            all_predictions[idx_pred] = np.append(all_predictions[idx_pred], 0.0)

                        for idx_pred in range(len(window_predictions)):
                            window_predictions[idx_pred] = np.append(window_predictions[idx_pred], 0.0)

                        current_number_classes += 1

                        for original_idx in event['indices']:
                            try:
                                list_idx = all_pred_indices.index(original_idx)

                                pred_np = np.zeros(current_number_classes)
                                pred_np[np_class_idx] = 1.0

                                if event.get('extensions'):
                                    for ext_id in event['extensions']:
                                        ext_mc = next((mc for mc in mapping['micro_clusters'] if mc['neuron_id'] == ext_id), None)
                                        if ext_mc is not None:
                                            known_labels = np.where(ext_mc['prototype_vector'] > 0.5)[0]
                                            pred_np[known_labels] = 1.0

                                all_predictions[list_idx] = pred_np

                                pred_row = pred_np.reshape(1, -1)
                                mapping = update_class_totals_probabilities(
                                    mapping, pred_row, 1, base_number_classes, 1, num_offline_instances
                                )
                                window_predictions.append(pred_np.copy())

                            except ValueError:
                                pass

                        mapping['micro_clusters'] = update_cond_probabilities_neurons(
                            mapping['micro_clusters'], mapping['class_probabilities']
                        )

        if (i + 1) % window_stm_check == 0:
            if len(window_predictions) > 0:
                padded_window_predictions = []

                for pred_vec in window_predictions:
                    pred_vec = np.asarray(pred_vec, dtype=float)

                    if len(pred_vec) < current_number_classes:
                        pad_size = current_number_classes - len(pred_vec)
                        pred_vec = np.append(pred_vec, np.zeros(pad_size))
                    elif len(pred_vec) > current_number_classes:
                        pred_vec = pred_vec[:current_number_classes]

                    padded_window_predictions.append(pred_vec)

                window_pred_arr = np.vstack(padded_window_predictions)
                mapping['z'] = np.mean(np.sum(window_pred_arr, axis=1))
                window_predictions = []

            mapping, short_term_memory, stm_indices = forget_obsolete_information(
                mapping, short_term_memory, stm_indices, i, window_stm_check
            )

    predictions_matrix = np.array(all_predictions)
    final_predictions = pd.DataFrame(
        np.zeros((len(online_dataset), current_number_classes)),
        index=np.arange(len(online_dataset))
    )

    if len(all_pred_indices) > 0:
        final_predictions.iloc[all_pred_indices] = predictions_matrix

    print("\n[RESUMO ND]")
    print(f"Extensões detectadas: {num_extensions_detected}")
    print(f"NPs detectados: {num_nps_detected}")
    print(f"Número final de classes no modelo: {current_number_classes}")

    results = {
        'predictions': final_predictions,
        'indexes_explained': indexes_explained,
        'mapping': mapping
    }
    return results

def kohonen_online_baseline_predictor(mapping: dict, online_dataset: np.ndarray) -> dict:
    print("\n!!! RUNNING ONLINE PHASE WITH SIMPLE BASELINE PREDICTOR !!!")
    initial_number_classes = mapping['class_probabilities'].shape[0]
    all_predictions = []
    neuron_centroids = mapping['som_map']['codes']

    for i, x_instance in enumerate(online_dataset):
        if (i + 1) % 1000 == 0:
            print(f"  Processing instance {i + 1}/{len(online_dataset)}...")

        x = x_instance.reshape(1, -1)

        nbrs = NearestNeighbors(n_neighbors=1).fit(neuron_centroids)
        distances, indices = nbrs.kneighbors(x)
        winner_idx = indices[0][0]

        mc_winner = next((mc for mc in mapping['micro_clusters'] if mc['neuron_id'] == winner_idx), None)

        pred = np.zeros(initial_number_classes)
        if mc_winner:
            prototype = mc_winner['prototype_vector']
            pred[prototype > 0.5] = 1

        all_predictions.append(pred)

    predictions_matrix = np.array(all_predictions)
    results = {
        'predictions': pd.DataFrame(predictions_matrix),
        'indexes_explained': list(range(len(online_dataset))),
        'mapping': mapping
    }
    return results