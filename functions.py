import numpy as np
import pandas as pd
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
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
            class_prob_row = class_probabilities[class_idx, :].copy()
            class_prob_row[class_idx] = 0 
            
            prob_sorted_indices = np.argsort(class_prob_row)[::-1]
            
            prob_j = class_probabilities[class_idx, class_idx]
            prob_k_j = 1.0

            multiplied_values = []

            for k in active_classes_indices:
                idx_k = prob_sorted_indices[k]
                val = class_probabilities[idx_k, class_idx]
                if val > 0 and idx_k != class_idx:
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
                print(f"[DEBUG ORIENTADOR] Analisando Neurônio {neuron_id}, Classe {class_idx}")
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
            class_prob_row = class_probabilities[class_idx, :].copy()
            class_prob_row[class_idx] = 0
            
            prob_sorted_indices = np.argsort(class_prob_row)[::-1]
            prob_j = class_probabilities[class_idx, class_idx]
            prob_k_j = 1.0

            for k in active_classes_indices:
                idx_k = prob_sorted_indices[k]
                if class_probabilities[idx_k, class_idx] > 0 and idx_k != class_idx:
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
    unit_classif = som_map['unit.classif']

    for mc in micro_clusters:
        neuron_id = mc['neuron_id']
        centroid = mc['centroid']
        indexes_mapped = np.where(unit_classif == neuron_id)[0]
        data_mapped = data[indexes_mapped]

        if len(data_mapped) <= 1:
            mc['radius_factor_1'] = 0
            mc['radius_factor_2'] = 0
            continue

        distances_from_centroid = np.linalg.norm(data_mapped - centroid, axis=1)
        r_fact = np.max(distances_from_centroid)

        nd_rfact = r_fact 
        
        try:
             nbrs = NearestNeighbors(n_neighbors=min(2, len(data_mapped))).fit(data_mapped)
             distances_knn, indices_knn = nbrs.kneighbors(data_mapped)
             if distances_knn.shape[1] > 1:
                 max_dist = np.max(distances_knn[:, 1])
                 nd_rfact = max_dist 
        except:
             pass

        mc['radius_factor_1'] = r_fact
        mc['radius_factor_2'] = nd_rfact if nd_rfact < r_fact else r_fact

    return micro_clusters

def run_novelty_detection(mapping: dict, short_term_memory: list, stm_indices: list, min_ex: int) -> tuple:
    """Executa o procedimento de Detecção de Novidades (ND) do MINAS-BR e retorna os novos padrões."""
    
    if len(short_term_memory) < min_ex:
        return mapping, short_term_memory, stm_indices, []

    stm_data = np.array(short_term_memory)
    
    # K-Means: definindo k dinamicamente
    k = max(2, len(stm_data) // 10)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(stm_data)
    
    try:
        sample_silhouettes = silhouette_samples(stm_data, cluster_labels)
    except:
        sample_silhouettes = np.zeros(len(stm_data))

    valid_clusters = []
    new_patterns_info = [] # NOVO: Lista para guardar as informações dos NPs
    
    for cluster_id in range(k):
        idx_in_cluster = np.where(cluster_labels == cluster_id)[0]
        
        if len(idx_in_cluster) >= min_ex:
            cluster_silhouette = np.mean(sample_silhouettes[idx_in_cluster])
            if cluster_silhouette > 0:
                valid_clusters.append((cluster_id, cluster_silhouette, len(idx_in_cluster)))
                
                # NOVO: Calcula o centroide e guarda os índices originais das instâncias
                centroid = np.mean(stm_data[idx_in_cluster], axis=0)
                original_indices = [stm_indices[i] for i in idx_in_cluster]
                new_patterns_info.append({
                    'centroid': centroid,
                    'indices': original_indices,
                    'silhouette': cluster_silhouette
                })
    
    # Limpa a memória após processar (no MINAS-BR original, os não-agrupados continuam, 
    # mas limpar tudo é uma simplificação aceitável para esta etapa).
    return mapping, [], [], new_patterns_info