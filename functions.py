import numpy as np
import pandas as pd
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from minisom import MiniSom
import os

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates the Euclidean distance between two numpy vectors."""
    return np.sqrt(np.sum((x - y)**2))

def compute_label_cardinality(offline_classes: pd.DataFrame) -> float:
    """Calculates the label cardinality (average number of labels per instance)."""
    total_labels = offline_classes.values.sum()
    num_instances = offline_classes.shape[0]
    return total_labels / num_instances if num_instances > 0 else 0

def get_probabilities(classes: pd.DataFrame, i: int, j: int) -> dict:
    if i == j:  # Calculate prior probability P(i)
        total_i = classes.iloc[:, i].sum()
        total_instances = len(classes)
        probability = total_i / total_instances if total_instances > 0 else 0
        result = {'prob': probability, 'total': int(total_i)}
    else:  # Calculate conditional probability P(i|j)
        total_j = classes.iloc[:, j].sum()
        if total_j == 0:
            probability = 0
            intersection_total = 0
        else:
            intersection_total = ((classes.iloc[:, i] == 1) & (classes.iloc[:, j] == 1)).sum()
            probability = intersection_total / total_j
        result = {'prob': probability, 'total': int(intersection_total)}
    return result

def compute_initial_class_probabilities_totals(offline_classes: pd.DataFrame) -> tuple:
    num_classes = offline_classes.shape[1]
    class_probabilities = np.zeros((num_classes, num_classes))
    class_totals = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            class_prob_totals = get_probabilities(offline_classes, i, j)
            class_probabilities[i, j] = class_prob_totals['prob']
            class_totals[i, j] = class_prob_totals['total']
    return class_probabilities, class_totals

def get_parameter_values(param_file: str) -> dict:
    parameters = {}
    with open(param_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.strip().startswith('#'):
                key, value = line.split('=', 1)
                key = key.strip().replace('.', '_')
                value = value.strip()
                try:
                    if ',' in value:
                        parts = [float(part) for part in value.split(',')]
                        parameters[key] = parts if len(parts) > 1 else parts[0]
                    else:
                        parameters[key] = float(value)
                except ValueError:
                    parameters[key] = value
    return parameters

def compute_micro_clusters(som_map: dict, offline_classes: pd.DataFrame, min_ex: int) -> dict:
    neuron_counts = Counter(som_map['unit.classif'])
    valid_neurons = sorted([neuron for neuron, count in neuron_counts.items() if count >= min_ex])
    micro_clusters = []

    for neuron_id in valid_neurons:
        indexes = np.where(som_map['unit.classif'] == neuron_id)[0]
        prototype_vector = offline_classes.iloc[indexes].mean(axis=0).values

        micro_cluster_dict = {
            'neuron_id': neuron_id,
            'centroid': som_map['codes'][neuron_id].copy(), 
            'num_instances': len(indexes),
            'prototype_vector': prototype_vector,
            'cond_prob_threshold': np.zeros(offline_classes.shape[1]),
            'average_output': [0, 0] 
        }
        micro_clusters.append(micro_cluster_dict)

    results = {'som_map': som_map, 'micro_clusters': micro_clusters}
    return results

def get_average_neuron_outputs(som_map: dict) -> dict:
    average_outputs = {}
    unit_classif = np.array(som_map['unit.classif'])
    distances = np.array(som_map['distances'])
    unique_neuron_ids = np.unique(unit_classif)

    for neuron_id in unique_neuron_ids:
        neuron_distances = distances[unit_classif == neuron_id]
        if len(neuron_distances) > 0:
            outputs = np.exp(-neuron_distances)
            average_outputs[neuron_id] = [outputs.sum(), len(neuron_distances)]
        else:
            average_outputs[neuron_id] = [0, 0]
    return average_outputs

def get_cond_probabilities_neurons(micro_clusters: list, class_probabilities: np.ndarray,
                                   average_neuron_outputs: dict) -> list:
    debug_printed = False 

    for mc in micro_clusters:
        prototype_vector = mc['prototype_vector']
        active_classes_indices = np.where(prototype_vector > 0)[0]
        neuron_id = mc['neuron_id']

        sum_outputs, count_outputs = average_neuron_outputs.get(neuron_id, [0, 0])
        avg_output = sum_outputs / count_outputs if count_outputs > 0 else 0

        mc['average_output'] = [sum_outputs, count_outputs]

        if avg_output == 0:
            continue
            
        for class_idx in active_classes_indices:
            prob_j = class_probabilities[class_idx, class_idx]
            prob_k_j = 1.0

            multiplied_values = []

            for idx_k in active_classes_indices:
                if idx_k != class_idx:
                    val = class_probabilities[idx_k, class_idx]
                    if val > 0:
                        prob_k_j *= val
                        multiplied_values.append(val)
            
            weight_factor = prototype_vector[class_idx]
            if prob_j < 1e-9: prob_j = 1e-9
            
            prob_j_ks_x = prob_j * prob_k_j * avg_output
            exponential_term = np.exp(-(1 - weight_factor))
            threshold = prob_j_ks_x * exponential_term

            mc['cond_prob_threshold'][class_idx] = threshold

            # --- DEBUG (Primeiro caso válido) ---
            if not debug_printed and len(multiplied_values) > 0:
                print("\n" + "="*50)
                print(f"[DEBUG] Analisando Neurônio {neuron_id}, Classe {class_idx}")
                print(f"1. Prob Prior (p(cj)): {prob_j:.10f}")
                print(f"2. Multiplicação Condicional (prod p(cl|ck)): {prob_k_j:.20f}")
                print(f"   -> Valores multiplicados: {multiplied_values}")
                print(f"3. Avg Output do Neurônio (p(Xb|ck)): {avg_output:.10f}")
                print(f"4. Termo Exponencial (exp(-(1-v))): {exponential_term:.10f} (Peso: {weight_factor:.4f})")
                print("-" * 30)
                print(f"RESULTADO FINAL (Lado Direito): {threshold:.20f}")
                debug_printed = True
            # ------------------------------------
            
    return micro_clusters

def update_cond_probabilities_neurons(micro_clusters: list, class_probabilities: np.ndarray) -> list:
    for mc in micro_clusters:
        prototype_vector = mc['prototype_vector']
        active_classes_indices = np.where(prototype_vector > 0)[0]

        avg_output = mc['average_output'][0] / mc['average_output'][1] if mc['average_output'][1] > 0 else 0
        if avg_output == 0:
            continue

        for class_idx in active_classes_indices:
            prob_j = class_probabilities[class_idx, class_idx]
            prob_k_j = 1.0

            for idx_k in active_classes_indices:
                if idx_k != class_idx and class_probabilities[idx_k, class_idx] > 0:
                    prob_k_j *= class_probabilities[idx_k, class_idx]

            weight_factor = prototype_vector[class_idx]
            if prob_j < 1e-9: prob_j = 1e-9
            
            prob_j_ks_x = prob_j * prob_k_j * avg_output
            threshold = prob_j_ks_x * np.exp(-(1 - weight_factor))
            mc['cond_prob_threshold'][class_idx] = threshold

    return micro_clusters

def update_class_totals_probabilities(mapping: dict, pred: np.ndarray, num_pred: int,
                                      initial_number_classes: int, is_novelty: int,
                                      num_offline_instances: int) -> dict:
    """Updates class total counts and probability matrices based on new predictions."""
    mapping['total_instances'] += num_pred

    if is_novelty == 0 and 'total_instances_np' in mapping and isinstance(mapping['total_instances_np'], list):
        pass 

    if pred.shape[0] > 0:
        for r in range(pred.shape[0]):
            predicted_indices = np.where(pred[r, :] > 0)[0]
            if len(predicted_indices) > 0:
                for idx_i in predicted_indices:
                    for idx_j in predicted_indices:
                        mapping['class_totals'][idx_i, idx_j] += 1
    
    # --- PROBABILIDADES REATIVADAS ---
    num_total_classes = mapping['class_totals'].shape[0]
    for idx_i in range(num_total_classes):
        for idx_j in range(num_total_classes):
            total_j = mapping['class_totals'][idx_j, idx_j]
            
            if idx_i == idx_j: # P(i)
                 mapping['class_probabilities'][idx_i, idx_j] = mapping['class_totals'][idx_i, idx_j] / mapping['total_instances'] if mapping['total_instances'] > 0 else 0
            else: # P(i|j)
                mapping['class_probabilities'][idx_i, idx_j] = mapping['class_totals'][idx_i, idx_j] / total_j if total_j > 0 else 0
    # ---------------------------------

    return mapping


def update_model_information(mapping: dict, x: np.ndarray, time_stamp: int, n0: float,
                             winner: dict, inst_l: int) -> dict:
    
    neuron_indices = winner['nn_index'][inst_l]
    distances = winner['nn_dist'][inst_l]
    x = x.flatten() 

    for i, neuron_idx in enumerate(neuron_indices):
        micro_cluster = next((mc for mc in mapping['micro_clusters'] if mc['neuron_id'] == neuron_idx), None)
        if micro_cluster is None:
            continue

        distance = distances[i]
        micro_cluster['num_instances'] += 1
        
        learning_rate = n0 
        
        delta = learning_rate * (x - micro_cluster['centroid']) * np.exp(-distance)

        # Debug para confirmar a estabilidade
        if time_stamp == 20000: 
           print(f"\n--- [DEBUG FINAL] Atualização de Peso ---")
           print(f"Learning Rate (Fixo): {learning_rate}")
           print(f"Distância Normalizada: {distance:.4f}")
           print(f"Exp(-dist) [Amortecimento Natural]: {np.exp(-distance):.4f}")
           print(f"Magnitude do Delta: {np.linalg.norm(delta):.10f}")
        
        micro_cluster['centroid'] += delta

        if isinstance(neuron_idx, (int, np.integer)):
            mapping['som_map']['codes'][neuron_idx] += delta

    return mapping

def macro_precision_recall_fmeasure_windows(true_labels: np.ndarray, predicted_labels: np.ndarray,
                                            num_evaluation_windows: int, dataset_name="debug") -> dict:
    debug_dir = f"debug_windows_{dataset_name}"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    num_labels = true_labels.shape[1]
    num_examples = true_labels.shape[0]
    results = {}
    ma_precision_window, ma_recall_window, ma_fmeasure_window = [], [], []

    num_examples_window = num_examples // num_evaluation_windows
    evaluation_windows = np.full(num_evaluation_windows, num_examples_window)
    rest = num_examples - (num_examples_window * num_evaluation_windows)
    if rest > 0:
        evaluation_windows[:rest] += 1

    start_idx = 0
    beta = 1.0

    print(f"\n[DEBUG] Calculando métricas de forma ACUMULATIVA (igual ao R)...")

    tp_cum = np.zeros(num_labels)
    fp_cum = np.zeros(num_labels)
    fn_cum = np.zeros(num_labels)

    for w_idx, window_size in enumerate(evaluation_windows):
        end_idx = start_idx + window_size
        
        true_window = true_labels[start_idx:end_idx]
        predicted_window = predicted_labels[start_idx:end_idx]

        total_prec_window, total_recall_window, total_fmeasure_window = 0, 0, 0

        for j in range(num_labels):
            tp_cum[j] += np.sum((true_window[:, j] == 1) & (predicted_window[:, j] == 1))
            fp_cum[j] += np.sum((true_window[:, j] == 0) & (predicted_window[:, j] == 1))
            fn_cum[j] += np.sum((true_window[:, j] == 1) & (predicted_window[:, j] == 0))
            
            tp = tp_cum[j]
            fp = fp_cum[j]
            fn = fn_cum[j]

            if tp + fp + fn == 0:
                prec = 1.0; recall = 1.0; fmeasure = 1.0 
            elif tp + fp == 0:
                prec = 0.0; recall = tp/(tp+fn) 
                fmeasure = 0.0
            elif tp + fn == 0:
                prec = tp/(tp+fp); recall = 0.0
                fmeasure = 0.0
            else:
                prec = tp / (tp + fp)
                recall = tp / (tp + fn)
                
                if prec + recall == 0:
                    fmeasure = 0.0
                else:
                    beta2 = beta * beta
                    fmeasure = ((beta2 + 1) * prec * recall) / (beta2 * prec + recall)

            total_prec_window += prec
            total_recall_window += recall
            total_fmeasure_window += fmeasure

        ma_precision_window.append(total_prec_window / num_labels if num_labels > 0 else 0)
        ma_recall_window.append(total_recall_window / num_labels if num_labels > 0 else 0)
        ma_fmeasure_window.append(total_fmeasure_window / num_labels if num_labels > 0 else 0)

        start_idx = end_idx
 
    results['ma_precision'] = ma_precision_window[-1]
    results['ma_recall'] = ma_recall_window[-1]
    results['ma_fmeasure'] = ma_fmeasure_window[-1]
    
    results['ma_precision_window'] = ma_precision_window
    results['ma_recall_window'] = ma_recall_window
    results['ma_fmeasure_window'] = ma_fmeasure_window

    return results

def compute_radius_factor_mc(micro_clusters: list, som_map: dict, data: np.ndarray) -> list:
    """
    Calcula o raio de cada micro-cluster de forma fiel ao CF-Vector / CluStream:
        R = sqrt( (1/N) * sum ||x_i - c||^2 )
    e define a fronteira máxima como:
        radius_factor_1 = 2 * R

    Para clusters com apenas 1 ponto, usa 0 temporariamente.
    """
    unit_classif = som_map['unit.classif']

    for mc in micro_clusters:
        neuron_id = mc['neuron_id']
        centroid = mc['centroid']
        indexes_mapped = np.where(unit_classif == neuron_id)[0]
        data_mapped = data[indexes_mapped]

        if len(data_mapped) <= 1:
            mc['radius'] = 0.0
            mc['radius_factor_1'] = 0.0
            continue

        # ||x_i - c||^2 para cada ponto do micro-cluster
        sq_dists = np.sum((data_mapped - centroid) ** 2, axis=1)

        # RMS deviation ao centróide
        radius = np.sqrt(np.mean(sq_dists))

        mc['radius'] = radius
        mc['radius_factor_1'] = 2.0 * radius

    return micro_clusters


def build_candidate_mc_from_stm(stm_data: np.ndarray, winner_indices: list, centroid: np.ndarray) -> dict:
    """
    Cria um micro-cluster candidato a partir de um grupo encontrado na STM.
    """
    group_data = stm_data[winner_indices]

    if len(group_data) == 0:
        return None

    sq_dists = np.sum((group_data - centroid) ** 2, axis=1)
    radius = np.sqrt(np.mean(sq_dists)) if len(group_data) > 0 else 0.0

    candidate = {
        'centroid': centroid.copy(),
        'num_instances': len(group_data),
        'radius': radius,
        'radius_factor_1': 2.0 * radius,
        'group_data': group_data,
        'prototype_vector': None,
        'cond_prob_threshold': None,
        'average_output': [0.0, 0],
    }
    return candidate

def decide_extension_or_novelty(candidate_mc: dict, mapping: dict) -> dict:
    valid_mcs = mapping['micro_clusters']
    if not valid_mcs:
        return {'type': 'NP', 'closest_mc': None, 'extensions': []}

    distances = []
    for mc in valid_mcs:
        d = np.linalg.norm(candidate_mc['centroid'] - mc['centroid'])
        distances.append((d, mc))

    distances.sort(key=lambda x: x[0])
    closest_dist, closest_mc = distances[0]

    extensions = []
    for d, mc in distances:
        boundary = mc.get('radius', 0.0)
        if d <= boundary:
            extensions.append(mc)

    if len(extensions) > 0:
        return {
            'type': 'extension',
            'closest_mc': closest_mc,
            'extensions': extensions,
            'distance': closest_dist
        }

    return {
        'type': 'NP',
        'closest_mc': closest_mc,
        'extensions': [],
        'distance': closest_dist
    }

def expand_model_for_new_class(mapping: dict) -> tuple:
    old_n = mapping['class_probabilities'].shape[0]
    new_n = old_n + 1

    new_probs = np.zeros((new_n, new_n))
    new_totals = np.zeros((new_n, new_n))

    new_probs[:old_n, :old_n] = mapping['class_probabilities']
    new_totals[:old_n, :old_n] = mapping['class_totals']

    mapping['class_probabilities'] = new_probs
    mapping['class_totals'] = new_totals

    for mc in mapping['micro_clusters']:
        mc['prototype_vector'] = np.append(mc['prototype_vector'], 0.0)
        mc['cond_prob_threshold'] = np.append(mc['cond_prob_threshold'], 0.0)

    return mapping, old_n

def add_novel_micro_cluster(mapping: dict, candidate_mc: dict, np_class_idx: int, np_count: int) -> dict:
    """
    Cria um novo micro-cluster no modelo representando um NP.
    """
    num_classes = mapping['class_probabilities'].shape[0]

    prototype_vector = np.zeros(num_classes)
    prototype_vector[np_class_idx] = 1.0

    new_mc = {
        'neuron_id': f'NP_{np_count}',
        'centroid': candidate_mc['centroid'].copy(),
        'num_instances': candidate_mc['num_instances'],
        'prototype_vector': prototype_vector,
        'cond_prob_threshold': np.zeros(num_classes),
        'average_output': [0.0, 0],
        'radius': candidate_mc['radius'],
        'radius_factor_1': candidate_mc['radius_factor_1'],
    }

    mapping['micro_clusters'].append(new_mc)
    return mapping

def absorb_candidate_as_extension(mapping: dict, candidate_mc: dict, target_mc: dict) -> dict:
    """
    Atualiza um micro-cluster existente com o candidato vindo da STM.
    """
    n_old = target_mc['num_instances']
    n_new = candidate_mc['num_instances']
    total_n = n_old + n_new

    if total_n == 0:
        return mapping

    # média ponderada dos centróides
    target_mc['centroid'] = ((n_old * target_mc['centroid']) + (n_new * candidate_mc['centroid'])) / total_n
    target_mc['num_instances'] = total_n

    # atualiza raio de forma conservadora
    target_mc['radius'] = max(target_mc.get('radius', 0.0), candidate_mc.get('radius', 0.0))
    target_mc['radius_factor_1'] = 2.0 * target_mc['radius']

    return mapping

def run_novelty_detection(mapping: dict, short_term_memory: list, stm_indices: list, min_ex: int) -> tuple:
    """
    Roda um SOM na STM, gera grupos candidatos e decide:
      - extensão
      - ou Novelty Pattern (NP)

    Retorna:
      mapping atualizado,
      STM restante,
      índices restantes da STM,
      e lista de eventos detectados.
    """
    if len(short_term_memory) < min_ex:
        return mapping, short_term_memory, stm_indices, []

    stm_data = np.array(short_term_memory)
    grid_size = 3

    tmp_som = MiniSom(grid_size, grid_size, stm_data.shape[1], sigma=1.0, learning_rate=0.5, random_seed=10)
    tmp_som.random_weights_init(stm_data)
    tmp_som.train_batch(stm_data, 200, verbose=False)

    weights = tmp_som.get_weights().reshape(-1, stm_data.shape[1])

    detected_events = []
    used_local_indices = set()

    if 'NP_count' not in mapping:
        mapping['NP_count'] = 0

    for neuron_idx in range(len(weights)):
        winners = [
            i for i, x in enumerate(stm_data)
            if np.ravel_multi_index(tmp_som.winner(x), (grid_size, grid_size)) == neuron_idx
        ]

        if len(winners) < min_ex:
            continue

        centroid = weights[neuron_idx]
        candidate_mc = build_candidate_mc_from_stm(stm_data, winners, centroid)
        if candidate_mc is None:
            continue

        decision = decide_extension_or_novelty(candidate_mc, mapping)

        original_indices = [stm_indices[i] for i in winners]

        if decision['type'] == 'extension':
            target_mc = decision['closest_mc']
            mapping = absorb_candidate_as_extension(mapping, candidate_mc, target_mc)

            detected_events.append({
                'type': 'extension',
                'indices': original_indices,
                'target_mc': target_mc['neuron_id'],
                'centroid': centroid.copy()
            })

        else:
            mapping, new_class_idx = expand_model_for_new_class(mapping)
            mapping['NP_count'] += 1
            np_id = mapping['NP_count']

            mapping = add_novel_micro_cluster(mapping, candidate_mc, new_class_idx, np_id)

            detected_events.append({
                'type': 'NP',
                'indices': original_indices,
                'np_class_idx': new_class_idx,
                'np_id': np_id,
                'centroid': centroid.copy(),
                'extensions': [mc['neuron_id'] for mc in decision['extensions']]
            })

        used_local_indices.update(winners)

    new_stm = [x for i, x in enumerate(short_term_memory) if i not in used_local_indices]
    new_stm_indices = [idx for i, idx in enumerate(stm_indices) if i not in used_local_indices]

    return mapping, new_stm, new_stm_indices, detected_events