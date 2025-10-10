import numpy as np
import pandas as pd
from collections import Counter
from sklearn.neighbors import NearestNeighbors


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates the Euclidean distance between two numpy vectors."""
    return np.sqrt(np.sum((x - y)**2))

def compute_label_cardinality(offline_classes: pd.DataFrame) -> float:
    """Calculates the label cardinality (average number of labels per instance)."""
    total_labels = offline_classes.values.sum()
    num_instances = offline_classes.shape[0]
    return total_labels / num_instances if num_instances > 0 else 0

def get_probabilities(classes: pd.DataFrame, i: int, j: int) -> dict:
    """
    Calculates P(i) if i==j, or P(i|j) if i!=j.
    Also returns the total count used for the numerator.
    """
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
    """Computes the prior/conditional probability matrix and the counts matrix from the offline data."""
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
    """Reads a 'key = value' configuration file and returns a dictionary of parameters."""
    parameters = {}
    with open(param_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Ignore empty lines or comments
            if line.strip() and not line.strip().startswith('#'):
                key, value = line.split('=', 1)
                key = key.strip().replace('.', '_')
                value = value.strip()
                try:
                    # Handle comma-separated values for lists
                    if ',' in value:
                        parts = [float(part) for part in value.split(',')]
                        parameters[key] = parts if len(parts) > 1 else parts[0]
                    else:
                        parameters[key] = float(value)
                except ValueError:
                    parameters[key] = value  # Keep as string if conversion fails
    return parameters

def compute_micro_clusters(som_map: dict, offline_classes: pd.DataFrame, min_ex: int) -> dict:
    """Computes the properties of micro-clusters from a trained SOM map."""
    neuron_counts = Counter(som_map['unit.classif'])
    valid_neurons = sorted([neuron for neuron, count in neuron_counts.items() if count >= min_ex])
    micro_clusters = []

    for neuron_id in valid_neurons:
        indexes = np.where(som_map['unit.classif'] == neuron_id)[0]
        prototype_vector = offline_classes.iloc[indexes].mean(axis=0).values

        micro_cluster_dict = {
            'neuron_id': neuron_id,
            'centroid': som_map['codes'][neuron_id],
            'num_instances': len(indexes),
            'prototype_vector': prototype_vector,
            'cond_prob_threshold': np.zeros(offline_classes.shape[1])
        }
        micro_clusters.append(micro_cluster_dict)

    results = {'som_map': som_map, 'micro_clusters': micro_clusters}
    return results

def get_average_neuron_outputs(som_map: dict) -> dict:
    """Computes the sum of outputs and count for each neuron that has associated data points."""
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
    """Calculates the conditional probability thresholds for each class within each neuron (Offline Phase)."""
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
            for k_idx in active_classes_indices:
                if class_idx != k_idx:
                    prob_k_j *= class_probabilities[k_idx, class_idx]

            weight_factor = prototype_vector[class_idx]
            prob_j_ks_x = prob_j * prob_k_j * avg_output
            threshold = prob_j_ks_x * np.exp(-(1 - weight_factor))

            mc['cond_prob_threshold'][class_idx] = threshold
    return micro_clusters

def update_cond_probabilities_neurons(micro_clusters: list, class_probabilities: np.ndarray) -> list:
    """Updates conditional probability thresholds during the online phase."""
    for mc in micro_clusters:
        prototype_vector = mc['prototype_vector']
        active_classes_indices = np.where(prototype_vector > 0)[0]

        avg_output = mc['average_output'][0] / mc['average_output'][1] if mc['average_output'][1] > 0 else 0
        if avg_output == 0:
            continue

        for class_idx in active_classes_indices:
            prob_j = class_probabilities[class_idx, class_idx]

            prob_k_j = 1.0
            for k_idx in active_classes_indices:
                if class_idx != k_idx:
                    prob_k_j *= class_probabilities[k_idx, class_idx]

            weight_factor = prototype_vector[class_idx]
            prob_j_ks_x = prob_j * prob_k_j * avg_output
            threshold = prob_j_ks_x * np.exp(-(1 - weight_factor))
            mc['cond_prob_threshold'][class_idx] = threshold

    return micro_clusters

def update_class_totals_probabilities(mapping: dict, pred: np.ndarray, num_pred: int,
                                      initial_number_classes: int, is_novelty: int,
                                      num_offline_instances: int) -> dict:
    """Updates class total counts and probability matrices based on new predictions."""
    mapping['total_instances'] += num_pred

    if is_novelty == 0 and 'total_instances_np' in mapping:
        for i in range(len(mapping['total_instances_np'])):
            mapping['total_instances_np'][i] += num_pred

    if pred.shape[0] > 0:
        for r in range(pred.shape[0]):
            predicted_indices = np.where(pred[r, :] > 0)[0]
            if len(predicted_indices) > 0:
                for idx_i in predicted_indices:
                    for idx_j in predicted_indices:
                        mapping['class_totals'][idx_i, idx_j] += 1

    # Recalculate the entire class_probabilities matrix based on new totals
    num_total_classes = mapping['class_totals'].shape[0]
    for idx_i in range(num_total_classes):
        for idx_j in range(num_total_classes):
            total_j = mapping['class_totals'][idx_j, idx_j]
            # Handles all cases: original, novelty, and mixed relations
            if idx_i == idx_j: # Prior probability P(i)
                if idx_i >= initial_number_classes: # Novelty class prior
                    total_np_instances = mapping['total_instances_np'][idx_j - initial_number_classes]
                    mapping['class_probabilities'][idx_i, idx_j] = mapping['class_totals'][idx_i, idx_j] / total_np_instances if total_np_instances > 0 else 0
                else: # Original class prior
                    mapping['class_probabilities'][idx_i, idx_j] = mapping['class_totals'][idx_i, idx_j] / mapping['total_instances'] if mapping['total_instances'] > 0 else 0
            else: # Conditional probability P(i|j)
                mapping['class_probabilities'][idx_i, idx_j] = mapping['class_totals'][idx_i, idx_j] / total_j if total_j > 0 else 0

    return mapping

def update_model_information(mapping: dict, x: np.ndarray, time_stamp: int, n0: float,
                             winner: dict, inst_l: int) -> dict:
    """Updates the winning neurons' weights based on a new data sample (online learning)."""
    neuron_indices = winner['nn_index'][inst_l]
    distances = winner['nn_dist'][inst_l]

    for i, neuron_idx in enumerate(neuron_indices):
        micro_cluster = next((mc for mc in mapping['micro_clusters'] if mc['neuron_id'] == neuron_idx), None)
        if micro_cluster is None:
            continue

        distance = distances[i]
        micro_cluster['num_instances'] += 1
        micro_cluster['timestamp_last_update'] = time_stamp

        delta = n0 * (x - micro_cluster['centroid']) * np.exp(-distance)

        # Apply the delta to the centroid in both the micro_cluster and the main SOM map
        micro_cluster['centroid'] += delta
        mapping['som_map']['codes'][neuron_idx] += delta

    return mapping

def macro_precision_recall_fmeasure_windows(true_labels: np.ndarray, predicted_labels: np.ndarray,
                                            num_evaluation_windows: int) -> dict:
    """Calculates macro-averaged precision, recall, and F-measure across evaluation windows."""
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

    for window_size in evaluation_windows:
        end_idx = start_idx + window_size
        true_window = true_labels[start_idx:end_idx]
        predicted_window = predicted_labels[start_idx:end_idx]

        total_prec_window, total_recall_window, total_fmeasure_window = 0, 0, 0

        for j in range(num_labels):
            tp = np.sum((true_window[:, j] == 1) & (predicted_window[:, j] == 1))
            fp = np.sum((true_window[:, j] == 0) & (predicted_window[:, j] == 1))
            fn = np.sum((true_window[:, j] == 1) & (predicted_window[:, j] == 0))

            # Mulan-inspired edge case handling for precision and recall
            if tp + fp + fn == 0:
                prec = 1.0; recall = 1.0
            else:
                prec = tp / (tp + fp) if tp + fp > 0 else 0.0
                recall = tp / (tp + fn) if tp + fn > 0 else 0.0

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

    # Final score is the mean of the scores from each window
    results['ma_precision'] = np.mean(ma_precision_window)
    results['ma_recall'] = np.mean(ma_recall_window)
    results['ma_fmeasure'] = np.mean(ma_fmeasure_window)

    results['ma_precision_window'] = ma_precision_window
    results['ma_recall_window'] = ma_recall_window
    results['ma_fmeasure_window'] = ma_fmeasure_window

    return results

def compute_radius_factor_mc(micro_clusters: list, som_map: dict, data: np.ndarray) -> list:
    """Computes radius factors for each micro-cluster for novelty detection."""
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

        # Radius 1: Max distance from the centroid to any point in the cluster.
        distances_from_centroid = np.linalg.norm(data_mapped - centroid, axis=1)
        r_fact = np.max(distances_from_centroid)

        # Radius 2: A more complex novelty radius based on internal cluster density.
        # Find the most isolated point (largest distance to its nearest neighbor).
        # We ask for 2 neighbors because the first neighbor of any point is itself (distance 0).
        nbrs = NearestNeighbors(n_neighbors=2).fit(data_mapped)
        distances_knn, indices_knn = nbrs.kneighbors(data_mapped)

        nearest_neighbor_distances = distances_knn[:, 1]
        max_dist = np.max(nearest_neighbor_distances)
        isolated_indices = np.where(nearest_neighbor_distances == max_dist)[0]

        # Tie-breaking: if multiple points are equally isolated, pick the one farthest from the centroid.
        if len(isolated_indices) > 1:
            isolated_distances_from_centroid = np.linalg.norm(data_mapped[isolated_indices] - centroid, axis=1)
            isolated_point_idx = isolated_indices[np.argmax(isolated_distances_from_centroid)]
        else:
            isolated_point_idx = isolated_indices[0]

        # This loop walks from the isolated point inwards via nearest neighbors
        # to find a boundary for novelty detection.
        nd_rfact = float('inf')
        current_pos = isolated_point_idx
        previous_pos = -1
        while nd_rfact >= r_fact and current_pos != previous_pos:
            neighbor_idx = indices_knn[current_pos, 1]
            nd_rfact = np.linalg.norm(data_mapped[neighbor_idx] - centroid)
            previous_pos = current_pos
            current_pos = neighbor_idx

        mc['radius_factor_1'] = r_fact
        mc['radius_factor_2'] = nd_rfact if nd_rfact < r_fact else r_fact

    return micro_clusters