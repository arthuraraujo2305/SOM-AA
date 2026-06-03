import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import silhouette_samples
from growing_som import GrowingSOM
import os

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(np.sum((x - y) ** 2))

def compute_label_cardinality(offline_classes: pd.DataFrame) -> float:
    total_labels = offline_classes.values.sum()
    num_instances = offline_classes.shape[0]
    return total_labels / num_instances if num_instances > 0 else 0.0

def get_probabilities(classes: pd.DataFrame, i: int, j: int) -> dict:
    if i == j:
        total_i = classes.iloc[:, i].sum()
        total_instances = len(classes)
        probability = total_i / total_instances if total_instances > 0 else 0.0
        result = {'prob': probability, 'total': int(total_i)}
    else:
        total_j = classes.iloc[:, j].sum()
        if total_j == 0:
            probability = 0.0
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


def _ensure_total_instances_np(mapping: dict, num_novel_classes: int) -> list:
    total_instances_np = mapping.setdefault('total_instances_np', [])

    if len(total_instances_np) < num_novel_classes:
        total_instances_np.extend([0] * (num_novel_classes - len(total_instances_np)))

    return total_instances_np

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
    valid_neuron_indices = sorted([idx for idx, count in neuron_counts.items() if count >= min_ex])
    micro_clusters = []

    node_positions = som_map.get('node_positions', [])

    for idx in valid_neuron_indices:
        indexes = np.where(som_map['unit.classif'] == idx)[0]
        prototype_vector = offline_classes.iloc[indexes].mean(axis=0).values

        # Usa a posição topológica (x,y) como ID se disponível, senão usa o índice
        neuron_id = node_positions[idx] if node_positions else idx

        micro_cluster_dict = {
            'neuron_id': neuron_id,
            'centroid': som_map['codes'][idx].copy(),
            'num_instances': len(indexes),
            'prototype_vector': prototype_vector,
            'cond_prob_threshold': np.zeros(offline_classes.shape[1]),
            'average_output': [0.0, 0],
            'last_timestamp': 0,
            'radius': 0.0,
            'radius_factor_1': 0.0,
            'std_dev': 0.0,
            'gsom_idx': idx # Referência para o array 'codes'
        }
        micro_clusters.append(micro_cluster_dict)

    return {'som_map': som_map, 'micro_clusters': micro_clusters}

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
            average_outputs[neuron_id] = [0.0, 0]
    return average_outputs

def get_cond_probabilities_neurons(micro_clusters: list, class_probabilities: np.ndarray,
                                   average_neuron_outputs: dict) -> list:
    debug_printed = False

    for mc in micro_clusters:
        prototype_vector = mc['prototype_vector']
        active_classes_indices = np.where(prototype_vector > 0)[0]
        neuron_id = mc['neuron_id']
        
        # AQUI ESTÁ A CORREÇÃO: Pega o índice inteiro para buscar no dicionário de médias
        idx_busca = mc.get('gsom_idx', neuron_id)

        sum_outputs, count_outputs = average_neuron_outputs.get(idx_busca, [0.0, 0])
        avg_output = sum_outputs / count_outputs if count_outputs > 0 else 0.0

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
            if prob_j < 1e-9:
                prob_j = 1e-9

            prob_j_ks_x = prob_j * prob_k_j * avg_output
            exponential_term = np.exp(-(1 - weight_factor))
            threshold = prob_j_ks_x * exponential_term

            mc['cond_prob_threshold'][class_idx] = threshold

            if not debug_printed and len(multiplied_values) > 0:
                print("\n" + "=" * 50)
                print(f"[DEBUG] Analisando Neurônio {neuron_id}, Classe {class_idx}")
                print(f"1. Prob Prior (p(cj)): {prob_j:.10f}")
                print(f"2. Multiplicação Condicional (prod p(cl|ck)): {prob_k_j:.20f}")
                print(f"   -> Valores multiplicados: {multiplied_values}")
                print(f"3. Avg Output do Neurônio (p(Xb|ck)): {avg_output:.10f}")
                print(f"4. Termo Exponencial (exp(-(1-v))): {exponential_term:.10f} (Peso: {weight_factor:.4f})")
                print("-" * 30)
                print(f"RESULTADO FINAL (Lado Direito): {threshold:.20f}")
                debug_printed = True

    return micro_clusters

def update_cond_probabilities_neurons(micro_clusters: list, class_probabilities: np.ndarray) -> list:
    for mc in micro_clusters:
        prototype_vector = mc['prototype_vector']
        active_classes_indices = np.where(prototype_vector > 0)[0]

        avg_output = mc['average_output'][0] / mc['average_output'][1] if mc['average_output'][1] > 0 else 0.0
        if avg_output == 0:
            continue

        for class_idx in active_classes_indices:
            prob_j = class_probabilities[class_idx, class_idx]
            prob_k_j = 1.0

            for idx_k in active_classes_indices:
                if idx_k != class_idx and class_probabilities[idx_k, class_idx] > 0:
                    prob_k_j *= class_probabilities[idx_k, class_idx]

            weight_factor = prototype_vector[class_idx]
            if prob_j < 1e-9:
                prob_j = 1e-9

            prob_j_ks_x = prob_j * prob_k_j * avg_output
            threshold = prob_j_ks_x * np.exp(-(1 - weight_factor))
            mc['cond_prob_threshold'][class_idx] = threshold

    return micro_clusters

def update_class_totals_probabilities(mapping: dict, pred: np.ndarray, num_pred: int,
                                      initial_number_classes: int, is_novelty: int,
                                      num_offline_instances: int) -> dict:
    mapping['total_instances'] += num_pred

    num_total_classes = mapping['class_totals'].shape[0]
    num_novel_classes = max(0, num_total_classes - initial_number_classes)
    total_instances_np = _ensure_total_instances_np(mapping, num_novel_classes)

    if is_novelty == 0 and total_instances_np:
        for i in range(len(total_instances_np)):
            total_instances_np[i] += num_pred

    if pred.shape[0] > 0:
        for r in range(pred.shape[0]):
            predicted_indices = np.where(pred[r, :] > 0)[0]
            if len(predicted_indices) > 0:
                for idx_i in predicted_indices:
                    for idx_j in predicted_indices:
                        mapping['class_totals'][idx_i, idx_j] += 1

                if is_novelty == 1:
                    for idx_i in predicted_indices:
                        if idx_i >= initial_number_classes:
                            np_idx = idx_i - initial_number_classes
                            if np_idx < len(total_instances_np):
                                total_instances_np[np_idx] += num_pred

    for idx_i in range(num_total_classes):
        for idx_j in range(num_total_classes):
            total_j = mapping['class_totals'][idx_j, idx_j]

            if idx_i == idx_j:
                if idx_i >= initial_number_classes:
                    np_idx = idx_j - initial_number_classes
                    denominator = total_instances_np[np_idx] if np_idx < len(total_instances_np) else 0
                    if denominator <= 0:
                        denominator = mapping['total_instances']
                else:
                    denominator = mapping['total_instances']

                mapping['class_probabilities'][idx_i, idx_j] = (
                    mapping['class_totals'][idx_i, idx_j] / denominator
                    if denominator > 0 else 0.0
                )
            else:
                mapping['class_probabilities'][idx_i, idx_j] = (
                    mapping['class_totals'][idx_i, idx_j] / total_j
                    if total_j > 0 else 0.0
                )

    return mapping

def update_model_information(mapping: dict, x: np.ndarray, time_stamp: int, n0: float,
                             winner: dict, inst_l: int) -> dict:
    neuron_ids = winner['nn_index'][inst_l]
    distances = winner['nn_dist'][inst_l]
    x = x.flatten()

    for i, neuron_id in enumerate(neuron_ids):
        micro_cluster = next((mc for mc in mapping['micro_clusters'] if mc['neuron_id'] == neuron_id), None)
        if micro_cluster is None:
            continue

        distance = distances[i]
        micro_cluster['num_instances'] += 1
        micro_cluster['last_timestamp'] = time_stamp

        learning_rate = n0
        delta = learning_rate * (x - micro_cluster['centroid']) * np.exp(-distance)

        micro_cluster['centroid'] += delta

        # Atualiza a matriz de códigos global baseada no índice armazenado
        if 'gsom_idx' in micro_cluster:
            idx = micro_cluster['gsom_idx']
            mapping['som_map']['codes'][idx] += delta
            
            # Reflete a mudança no modelo global G-SOM
            if 'gsom_model' in mapping:
                node_pos = mapping['som_map']['node_positions'][idx]
                mapping['gsom_model'].nodes[node_pos] += delta

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

        # --- NOVA MÁSCARA MINAS-BR ---
        # Ignora instâncias Unknown (soma das predições == 0) igual ao 'if(!Z.contains("unk"))' do Java
        known_mask = np.sum(predicted_window, axis=1) > 0
        
        true_w_filt = true_window[known_mask]
        pred_w_filt = predicted_window[known_mask]

        total_prec_window, total_recall_window, total_fmeasure_window = 0.0, 0.0, 0.0
        active_classes = 0

        for j in range(num_labels):
            # Conta TP, FP e FN usando apenas as instâncias que não foram dadas como Unknown
            tp_cum[j] += np.sum((true_w_filt[:, j] == 1) & (pred_w_filt[:, j] == 1))
            fp_cum[j] += np.sum((true_w_filt[:, j] == 0) & (pred_w_filt[:, j] == 1))
            fn_cum[j] += np.sum((true_w_filt[:, j] == 1) & (pred_w_filt[:, j] == 0))

            tp = tp_cum[j]
            fp = fp_cum[j]
            fn = fn_cum[j]

            # Se a classe nunca apareceu e nunca foi predita, ela é um fantasma. Ignora!
            if tp + fp + fn == 0:
                continue
            
            # A classe existe, então entra no denominador
            active_classes += 1

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if prec + recall == 0:
                fmeasure = 0.0
            else:
                beta2 = beta * beta
                fmeasure = ((beta2 + 1) * prec * recall) / (beta2 * prec + recall)

            total_prec_window += prec
            total_recall_window += recall
            total_fmeasure_window += fmeasure

        # Divide estritamente pelo número de classes ATIVAS (como no Java)
        ma_precision_window.append(total_prec_window / active_classes if active_classes > 0 else 0.0)
        ma_recall_window.append(total_recall_window / active_classes if active_classes > 0 else 0.0)
        ma_fmeasure_window.append(total_fmeasure_window / active_classes if active_classes > 0 else 0.0)

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
        idx_busca = mc.get('gsom_idx', mc['neuron_id'])
        centroid = mc['centroid']
        
        indexes_mapped = np.where(unit_classif == idx_busca)[0]
        data_mapped = data[indexes_mapped]

        if len(data_mapped) <= 1:
            mc['radius'] = 0.0
            mc['radius_factor_1'] = 0.0
            mc['std_dev'] = 0.0
            continue

        sq_dists = np.sum((data_mapped - centroid) ** 2, axis=1)
        dists = np.sqrt(sq_dists)

        radius = np.sqrt(np.mean(sq_dists))
        std_dev = np.std(dists, ddof=0)

        mc['radius'] = radius
        mc['radius_factor_1'] = 3.0 * radius  # <-- AUMENTADO PARA 3.0
        mc['std_dev'] = std_dev

    return micro_clusters

def build_candidate_mc_from_stm(stm_data: np.ndarray, winner_indices: list, centroid: np.ndarray, mapping: dict = None) -> dict:
    group_data = stm_data[winner_indices]

    if len(group_data) == 0:
        return None

    sq_dists = np.sum((group_data - centroid) ** 2, axis=1)
    dists = np.sqrt(sq_dists)

    radius = np.sqrt(np.mean(sq_dists)) if len(group_data) > 0 else 0.0

    # --- TRAVA DE PROTEÇÃO DO RAIO ---
    if mapping and 'micro_clusters' in mapping:
        valid_radii = [mc.get('radius', 0.0) for mc in mapping['micro_clusters'] if mc.get('radius', 0.0) > 0]
        if valid_radii:
            global_mean_radius = np.mean(valid_radii)
            radius = max(radius, global_mean_radius)
    # ---------------------------------

    std_dev = np.std(dists, ddof=0) if len(dists) > 1 else 0.0

    candidate = {
        'centroid': centroid.copy(),
        'num_instances': len(group_data),
        'radius': radius,
        'radius_factor_1': 3.0 * radius,  # <-- AUMENTADO PARA 3.0
        'std_dev': std_dev,
        'group_data': group_data,
        'prototype_vector': None,
        'cond_prob_threshold': None,
        'average_output': [0.0, 0],
    }
    return candidate

def validate_candidate_cluster_from_indices(stm_data: np.ndarray, cluster_labels: np.ndarray,
                                            winners: list, min_ex: int) -> bool:
    """
    Aproxima a validação do MINAS-BR:
    - tamanho mínimo
    - coesão positiva via silhouette > 0
    """
    if len(winners) < min_ex:
        return False

    unique_labels = np.unique(cluster_labels)
    if len(unique_labels) < 2:
        return True

    try:
        sil_values = silhouette_samples(stm_data, cluster_labels)
        group_sil = sil_values[winners]
        return np.mean(group_sil) > 0
    except Exception:
        return True

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
        r = mc.get('radius', 0.0)
        
        if r > 0 and d <= 2.0 * r: 
            extensions.append(mc)

    z_model = max(1, int(np.ceil(mapping['z'])))

    if len(extensions) >= z_model:
        return {
            'type': 'extension',
            'closest_mc': closest_mc,
            'extensions': extensions,
            'distance': closest_dist
        }

    return {
        'type': 'NP',
        'closest_mc': closest_mc,
        'extensions': extensions,
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
    mapping.setdefault('total_instances_np', []).append(0)

    for mc in mapping['micro_clusters']:
        mc['prototype_vector'] = np.append(mc['prototype_vector'], 0.0)
        mc['cond_prob_threshold'] = np.append(mc['cond_prob_threshold'], 0.0)

    return mapping, old_n

def add_novel_micro_cluster(mapping: dict, candidate_mc: dict, np_class_idx: int, np_count: int) -> dict:
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
        'std_dev': candidate_mc.get('std_dev', 0.0),
        'last_timestamp': 0,
    }

    mapping['micro_clusters'].append(new_mc)
    return mapping

def absorb_candidate_as_extension(mapping: dict, candidate_mc: dict, target_mc: dict, current_t: int) -> dict:
    n_old = target_mc['num_instances']
    n_new = candidate_mc['num_instances']
    total_n = n_old + n_new

    if total_n == 0:
        return mapping

    target_mc['centroid'] = ((n_old * target_mc['centroid']) + (n_new * candidate_mc['centroid'])) / total_n
    target_mc['num_instances'] = total_n
    target_mc['radius'] = max(target_mc.get('radius', 0.0), candidate_mc.get('radius', 0.0))
    target_mc['radius_factor_1'] = 3.0 * target_mc['radius'] # <-- AUMENTADO PARA 3.0
    target_mc['std_dev'] = max(target_mc.get('std_dev', 0.0), candidate_mc.get('std_dev', 0.0))
    target_mc['last_timestamp'] = current_t

    return mapping

def forget_obsolete_information(mapping: dict, short_term_memory: list, stm_indices: list,
                                current_t: int, omega: int):
    """
    Esquece exemplos antigos da STM e micro-clusters sem atividade recente.
    Aproxima a política de esquecimento do MINAS-BR.
    """
    new_stm = []
    new_indices = []
    for x, idx in zip(short_term_memory, stm_indices):
        if current_t - idx <= omega:
            new_stm.append(x)
            new_indices.append(idx)

    filtered_mcs = []
    for mc in mapping['micro_clusters']:
        last_t = mc.get('last_timestamp', 0)
        # preserva micro-clusters originais do SOM e também os ativos recentemente
        if isinstance(mc['neuron_id'], (int, np.integer)) or current_t - last_t <= omega:
            filtered_mcs.append(mc)

    mapping['micro_clusters'] = filtered_mcs
    return mapping, new_stm, new_indices

def run_novelty_detection(mapping: dict, short_term_memory: list, stm_indices: list,
                          min_ex: int, current_t: int) -> tuple:
    """
    ND com Growing SOM na STM:
    - treina um G-SOM sobre os exemplos rejeitados
    - obtém os neurônios ativos
    - usa neurônios densos como grupos candidatos
    - valida e decide extensão vs NP
    """
    if len(short_term_memory) < min_ex:
        return mapping, short_term_memory, stm_indices, []

    stm_data = np.array(short_term_memory)

    # ---- G-SOM na STM ----
    gsom = GrowingSOM(
        input_dim=stm_data.shape[1],
        growth_threshold=5.0,
        spread_factor=0.9,
        learning_rate=0.3,
        sigma=1.0,
        max_nodes=25,
        random_seed=10,
    )
    gsom.train(stm_data, num_epochs=10)

    cluster_labels = gsom.get_cluster_labels(stm_data)
    node_weights = gsom.get_node_weights()
    active_nodes = gsom.get_active_nodes()

    detected_events = []
    used_local_indices = set()

    if 'NP_count' not in mapping:
        mapping['NP_count'] = 0

    # Converte cada label para string estável, ex.: "(0, 1)"
    cluster_labels_str = np.array([str(tuple(lbl)) for lbl in cluster_labels], dtype=object)

    for node_pos in active_nodes:
        node_pos_str = str(tuple(node_pos))

        winners = np.where(cluster_labels_str == node_pos_str)[0].tolist()

        if not validate_candidate_cluster_from_indices(
            stm_data,
            cluster_labels_str,
            winners,
            min_ex
        ):
            continue

        centroid = node_weights[node_pos]
        candidate_mc = build_candidate_mc_from_stm(stm_data, winners, centroid, mapping)
        if candidate_mc is None:
            continue

        decision = decide_extension_or_novelty(candidate_mc, mapping)
        original_indices = [stm_indices[i] for i in winners]

        if decision['type'] == 'extension':
            target_mc = decision['closest_mc']
            mapping = absorb_candidate_as_extension(mapping, candidate_mc, target_mc, current_t)

            detected_events.append({
                'type': 'extension',
                'indices': original_indices,
                'target_mc': target_mc['neuron_id'],
                'centroid': centroid.copy(),
                'gsom_node': node_pos,
            })

        else:
            mapping, new_class_idx = expand_model_for_new_class(mapping)
            mapping['NP_count'] += 1
            np_id = mapping['NP_count']

            mapping = add_novel_micro_cluster(mapping, candidate_mc, new_class_idx, np_id)
            mapping['micro_clusters'][-1]['last_timestamp'] = current_t

            detected_events.append({
                'type': 'NP',
                'indices': original_indices,
                'np_class_idx': new_class_idx,
                'np_id': np_id,
                'centroid': centroid.copy(),
                'extensions': [mc['neuron_id'] for mc in decision['extensions']],
                'gsom_node': node_pos,
            })

        used_local_indices.update(winners)

    new_stm = [x for i, x in enumerate(short_term_memory) if i not in used_local_indices]
    new_stm_indices = [idx for i, idx in enumerate(stm_indices) if i not in used_local_indices]

    return mapping, new_stm, new_stm_indices, detected_events