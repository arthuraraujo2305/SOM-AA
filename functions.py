import numpy as np
import pandas as pd
from collections import Counter
from sklearn.neighbors import NearestNeighbors


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:

    distance = np.sqrt(np.sum((x - y)**2))

    return distance

def compute_label_cardinality(offline_classes: pd.DataFrame) -> float:

    total_labels = offline_classes.values.sum()
    num_instances = offline_classes.shape[0]

    return total_labels / num_instances

def get_probabilities(classes: pd.DataFrame, i: int, j: int) -> dict:

    if i == j:
        total_i = classes.iloc[:, i].sum()
        total_instances = len(classes)

        if total_instances == 0:
            probability = 0
        else:
            probability = total_i / total_instances
        result = {'prob': probability, 'total': int(total_i)}

    else:
        total_j = classes.iloc[:, j].sum()

        if total_j == 0:
            probability = 0
            intersection_total = 0

        else:
            intersection_total = ((classes.iloc[:, i] == 1) & (classes.iloc[:, j] == 1)).sum()
            probability = intersection_total / total_j

        result = {'prob': probability, 'total': int(intersection_total)}

    return result

def compute_initial_class_probabilities_totals(offline_classes: pd.DataFrame)-> tuple:
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
    """
    Lê um arquivo de configuração e retorna os parâmetros em um dicionário.

    O arquivo deve ter o formato 'parametro = valor' em cada linha.
    """
    parameters = {}
    with open(param_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Ignora linhas vazias ou comentários
            if line.strip() and not line.strip().startswith('#'):
                key, value = line.split('=', 1)

                # Limpa espaços em branco da chave e do valor
                key = key.strip().replace('.', '_')
                value = value.strip()

                # Tenta converter o valor para um número (float)
                # Se não conseguir, mantém como string
                try:
                    # Caso especial: múltiplos valores separados por vírgula
                    if ',' in value:
                        # Converte cada parte para float, se possível
                        parts = [float(part) for part in value.split(',')]
                        parameters[key] = parts if len(parts) > 1 else parts[0]
                    else:
                        parameters[key] = float(value)
                except ValueError:
                    # Se não for um número, salva como texto
                    parameters[key] = value


    return parameters

def compute_micro_clusters(som_map: dict, offline_classes: pd.DataFrame, min_ex: int) -> dict:
    """
    Computes the properties of micro-clusters from a trained SOM map.

    This function filters neurons based on a minimum number of examples,
    calculates various properties for each valid neuron (micro-cluster), and
    returns a data structure containing this information
    """

    # Count the occurrences of each neuron (equivalent to R's table())
    neuron_counts = Counter(som_map['unit.classif'])

    # Filter to keep only neurons with a minimum number of examples
    # .sort() is added to ensure a consistent order
    valid_neurons = sorted([neuron for neuron, count in neuron_counts.items() if count >= min_ex])

    micro_clusters = []

    # Loop through the valid neurons to calculate their properties
    for neuron_id in valid_neurons:
        indexes = np.where(som_map['unit.classif'] == neuron_id)[0]

        prototype_vector = offline_classes.iloc[indexes].mean(axis=0).values

        # Create a dictionary for the current micro-cluster. This is much more
        # readable than using numeric indices like in the R code (e.g., micro.clusters[[pos]][[5]]).
        micro_cluster_dict = {
            'neuron_id': neuron_id,
            'centroid': som_map['codes'][neuron_id],  # Neuron's weight vector
            'num_instances': len(indexes),
            'prototype_vector': prototype_vector,
            'timestamp_creation': 0,
            'class_position': 0,
            'cond_prob_threshold': np.full(offline_classes.shape[1], 9.0)  # Creates an array filled with 9s
        }

        # Add the micro-cluster dictionary to our list
        micro_clusters.append(micro_cluster_dict)

    results = {
        'som_map': som_map,
        'micro_clusters': micro_clusters
    }

    return results

def get_average_neuron_outputs(som_map: dict, num_micro_clusters: int) -> list:
    """
    Computes the average output for each neuron.

    The output for a single sample is calculated as exp(-distance). This function
    calculates the sum of these outputs and the total count for each neuron.
    """

    average_outputs = []

    # Convert to NumPy arrays for efficient boolean indexing
    unit_classif = np.array(som_map['unit.classif'])
    distances = np.array(som_map['distances'])

    for i in range(1, num_micro_clusters + 1):
        # Find all distances for samples mapped to the current neuron 'i'
        neuron_distances = distances[unit_classif == i]

        if len(neuron_distances) > 0:
            # Calculate exp(-distance) for all found distances at once
            outputs = np.exp(-neuron_distances)

            # Append the sum of outputs and the count of samples
            average_outputs.append([outputs.sum(), len(neuron_distances)])
        else:
            # If a neuron has no samples, append [0, 0]
            average_outputs.append([0, 0])

    return average_outputs

def get_cond_probabilities_neurons(micro_clusters: list, class_probabilities: np.ndarray,
                                   average_neuron_outputs: list) -> list:
    """
    Calculates the conditional probability thresholds for each class within each neuron.

    This is used in the offline phase to set initial thresholds. The threshold is based
    on the prior probability of the class, the conditional probability of other
    co-occurring classes, and the neuron's average output.
    """

    # Using enumerate to get both the index and the item, which is cleaner
    for i, mc in enumerate(micro_clusters):
        prototype_vector = mc['prototype_vector']
        # Find indices of classes that are active in this neuron's prototype
        active_classes_indices = np.where(prototype_vector > 0)[0]

        for class_idx in active_classes_indices:
            # P(y_j), the prior probability of the current class
            prob_j = class_probabilities[class_idx, class_idx]

            # This term represents the product of P(y_k|y_j) for all other
            # co-occurring active classes k.
            prob_k_j = 1.0
            for k_idx in active_classes_indices:
                if class_idx != k_idx:
                    # In class_probabilities, P(k|j) is stored at [k, j]
                    prob_k_j *= class_probabilities[k_idx, class_idx]

            # The weight factor is the value from the prototype vector itself
            weight_factor = prototype_vector[class_idx]

            # p(x|y_j), the average output of the neuron, calculated from the sum and count
            sum_outputs = average_neuron_outputs[i][0]
            count_outputs = average_neuron_outputs[i][1]

            # Add a check to prevent division by zero
            if count_outputs > 0:
                avg_output = sum_outputs / count_outputs
            else:
                avg_output = 0 # If no samples, the average output is zero

            # Final probability p(y_j | y_k, x) calculation
            prob_j_ks_x = prob_j * prob_k_j * avg_output

            # The final threshold is weighted by the prototype vector value
            threshold = prob_j_ks_x * np.exp(-(1 - weight_factor))

            # Update the micro-cluster's threshold for this specific class
            mc['cond_prob_threshold'][class_idx] = threshold

            mc['average_output'] = average_neuron_outputs[i]

    return micro_clusters

def update_cond_probabilities_neurons(micro_clusters: list, class_probabilities: np.ndarray) -> list:
    """
    Updates the conditional probability thresholds for each class within each neuron.

    This is used in the online phase to refresh the thresholds as the model
    and the micro-clusters' properties evolve with new data.
    """

    for mc in micro_clusters:
        prototype_vector = mc['prototype_vector']
        active_classes_indices = np.where(prototype_vector > 0)[0]

        for class_idx in active_classes_indices:
            prob_j = class_probabilities[class_idx, class_idx]

            prob_k_j = 1.0
            for k_idx in active_classes_indices:
                if class_idx != k_idx:
                    prob_k_j *= class_probabilities[k_idx, class_idx]

            weight_factor = prototype_vector[class_idx]

            if mc['average_output'][1] > 0:
                avg_output = mc['average_output'][0] / mc['average_output'][1]
            else:
                avg_output = 0

            prob_j_ks_x = prob_j * prob_k_j * avg_output

            threshold = prob_j_ks_x * np.exp(-(1 - weight_factor))

            mc['cond_prob_threshold'][class_idx] = threshold

    return micro_clusters

def update_class_totals_probabilities(mapping: dict, pred: np.ndarray, num_pred: int,
                                      initial_number_classes: int, is_novelty: int,
                                      num_offline_instances: int) -> dict:
    """
    Updates the class totals and class probability matrices based on new predictions.

    """
    mapping['total_instances'] += num_pred

    if is_novelty == 0 and 'total_instances_np' in mapping:
        mapping['total_instances_np'] = [count + num_pred for count in mapping['total_instances_np']]

    if pred.shape[0] > 0:
        for r in range(pred.shape[0]):
            # Find the indices of the predicted classes for this instance
            # Equivalent to R's which(pred[r,] > 0)
            predicted_indices = np.where(pred[r, :] > 0)[0]

            if len(predicted_indices) > 0:
                # Update the co-occurrence counts in the class_totals matrix
                for idx_i in predicted_indices:
                    for idx_j in predicted_indices:
                        mapping['class_totals'][idx_i, idx_j] += 1

    # Recalculate the entire class_probabilities matrix based on new totals
    num_total_classes = mapping['class_totals'].shape[0]
    for idx_i in range(num_total_classes):
        for idx_j in range(num_total_classes):

            # Case 1: A Novelty Pattern (NP) class's relation to an original class
            if idx_i >= initial_number_classes and idx_j < initial_number_classes:
                if mapping['class_totals'][idx_j, idx_j] > 0:
                    mapping['class_probabilities'][idx_i, idx_j] = mapping['class_totals'][idx_i, idx_j] / mapping['class_totals'][idx_j, idx_j]

            # Case 2: Relation between two NP classes
            elif idx_i >= initial_number_classes and idx_j >= initial_number_classes:
                if idx_i == idx_j: # Prior probability of an NP
                    total_np_instances = mapping['total_instances_np'][idx_j - initial_number_classes]
                    if total_np_instances > 0:
                        mapping['class_probabilities'][idx_i, idx_j] = mapping['class_totals'][idx_i, idx_j] / total_np_instances
                else: # Conditional probability between two NPs
                    if mapping['class_totals'][idx_j, idx_j] > 0:
                        mapping['class_probabilities'][idx_i, idx_j] = mapping['class_totals'][idx_i, idx_j] / mapping['class_totals'][idx_j, idx_j]

            # Case 3: An original class's relation to an NP class
            elif idx_i < initial_number_classes and idx_j >= initial_number_classes:
                if mapping['class_totals'][idx_j, idx_j] > 0:
                    mapping['class_probabilities'][idx_i, idx_j] = mapping['class_totals'][idx_i, idx_j] / mapping['class_totals'][idx_j, idx_j]

            # Case 4: Relation between two original classes
            else:
                if idx_i == idx_j: # Prior probability P(i)
                    if mapping['total_instances'] > 0:
                        mapping['class_probabilities'][idx_i, idx_j] = mapping['class_totals'][idx_i, idx_j] / mapping['total_instances']
                else: # Conditional probability P(i|j)
                    if mapping['class_totals'][idx_j, idx_j] > 0:
                        mapping['class_probabilities'][idx_i, idx_j] = mapping['class_totals'][idx_i, idx_j] / mapping['class_totals'][idx_j, idx_j]

    return mapping

def update_model_information(mapping: dict, x: np.ndarray, time_stamp: int, n0: float,
                             winner: dict, inst_l: int) -> dict:
    """
    Updates the winning neurons' properties and weights based on a new data sample.

    This function implements the SOM learning rule for the online phase.
    """

    # Get the nearest neurons and their distances from the winner object
    neuron_indices = winner['nn_index'][inst_l]
    distances = winner['nn_dist'][inst_l]

    # Loop through each winning neuron
    for i, neuron_idx in enumerate(neuron_indices):
        # The neuron_idx in the R code is 1-based and seems to be the index
        # for the micro_clusters list. We'll assume 0-based index here.

        # Get the specific micro-cluster and distance for this iteration
        micro_cluster = next((mc for mc in mapping['micro_clusters'] if mc['neuron_id'] == neuron_idx), None)

        if micro_cluster is None:
            continue
        distance = distances[i]

        # Update the micro-cluster's metadata
        micro_cluster['num_instances'] += 1
        # Let's add a new key for the timestamp for clarity
        micro_cluster['timestamp_last_update'] = time_stamp

        # Calculate the delta (weight change) using the SOM learning rule
        # This is a vector operation thanks to NumPy
        delta = n0 * (x - micro_cluster['centroid']) * np.exp(-distance)

        # Apply the delta to update the neuron's centroid (weight vector)
        micro_cluster['centroid'] += delta
        # Also update the centroid in the main SOM map structure for consistency
        mapping['som_map']['codes'][neuron_idx] += delta

    return mapping

def macro_precision_recall_fmeasure_windows(true_labels: np.ndarray, predicted_labels: np.ndarray,
                                            num_evaluation_windows: int) -> dict:
    """
    Calculates macro-averaged precision, recall, and F-measure across evaluation windows.

    The logic is a direct translation of the R version, but uses NumPy vectorization
    for efficient calculation of TP, FP, and FN, avoiding the innermost loop.
    """

    num_labels = true_labels.shape[1]
    num_examples = true_labels.shape[0]
    results = {}
    ma_precision_window = []
    ma_recall_window = []
    ma_fmeasure_window = []

    # --- Logic to define window sizes (same as R) ---
    num_examples_window = num_examples // num_evaluation_windows
    evaluation_windows = np.full(num_evaluation_windows, num_examples_window)
    rest = num_examples - (num_examples_window * num_evaluation_windows)
    if rest > 0:
        evaluation_windows[:rest] += 1

    start_idx = 0
    beta = 1  # For F-measure calculation

    for window_size in evaluation_windows:
        end_idx = start_idx + window_size

        # Get the data for the current window
        true_window = true_labels[start_idx:end_idx]
        predicted_window = predicted_labels[start_idx:end_idx]

        total_prec = 0
        total_recall = 0
        total_fmeasure = 0

        for j in range(num_labels):  # Loop through each label
            # --- Vectorized TP, FP, FN calculation ---
            # This replaces the innermost loop from the R code
            tp = np.sum((true_window[:, j] == 1) & (predicted_window[:, j] == 1))
            fp = np.sum((true_window[:, j] == 0) & (predicted_window[:, j] == 1))
            fn = np.sum((true_window[:, j] == 1) & (predicted_window[:, j] == 0))

            # --- Precision, Recall, F-measure logic (direct translation from R) ---
            # Handle edge cases just like the original Mulan-inspired R code
            if tp + fp + fn == 0:
                prec = 1.0
            elif tp + fp == 0:
                prec = 0.0
            else:
                prec = tp / (tp + fp)

            if tp + fn == 0: # Note: R code had tp+fp+fn==0 here, but tp+fn is more standard
                recall = 1.0 if tp + fp + fn == 0 else 0.0
            else:
                recall = tp / (tp + fn)

            if prec + recall == 0:
                fmeasure = 0.0
            else:
                beta2 = beta * beta
                fmeasure = ((beta2 + 1) * prec * recall) / (beta2 * prec + recall)
                # Fallback for the case where F-measure is NaN
                if np.isnan(fmeasure):
                    fmeasure = 0.0

            total_prec += prec
            total_recall += recall
            total_fmeasure += fmeasure

        ma_precision_window.append(total_prec / num_labels)
        ma_recall_window.append(total_recall / num_labels)
        ma_fmeasure_window.append(total_fmeasure / num_labels)

        start_idx = end_idx

    results['ma_precision'] = np.mean(ma_precision_window)
    results['ma_recall'] = np.mean(ma_recall_window)
    results['ma_fmeasure'] = np.mean(ma_fmeasure_window)

    results['ma_precision_window'] = ma_precision_window
    results['ma_recall_window'] = ma_recall_window
    results['ma_fmeasure_window'] = ma_fmeasure_window

    return results

def compute_radius_factor_mc(micro_clusters: list, som_map: dict, data: np.ndarray) -> list:
    """
    Computes radius factors for each micro-cluster for novelty detection.
    """

    unit_classif = som_map['unit.classif']

    for mc in micro_clusters:
        neuron_id = mc['neuron_id']
        centroid = mc['centroid']

        # Find indices and data points mapped to the current neuron
        indexes_mapped = np.where(unit_classif == neuron_id)[0]
        data_mapped = data[indexes_mapped]

        # If there's only one or zero points, we can't calculate these radii
        if len(data_mapped) <= 1:
            mc['radius_factor_1'] = 0
            mc['radius_factor_2'] = 0
            continue

        # --- Calculate Radius 1 (rFact): Max distance from centroid to any point ---
        distances_from_centroid = np.linalg.norm(data_mapped - centroid, axis=1)
        r_fact = np.max(distances_from_centroid)

        # --- Calculate Radius 2 (nd.rFact): The complex part ---
        # Find the most isolated point (largest distance to its nearest neighbor)
        # We ask for 2 neighbors because the first neighbor of any point is itself
        nbrs = NearestNeighbors(n_neighbors=2).fit(data_mapped)
        distances_knn, indices_knn = nbrs.kneighbors(data_mapped)

        # The distance to the actual nearest neighbor is in the second column
        nearest_neighbor_distances = distances_knn[:, 1]
        max_dist = np.max(nearest_neighbor_distances)

        # Find all points that are the most isolated
        isolated_indices = np.where(nearest_neighbor_distances == max_dist)[0]

        # Tie-breaking rule from R: if more than one, pick the one farthest from the centroid
        if len(isolated_indices) > 1:
            isolated_distances_from_centroid = np.linalg.norm(data_mapped[isolated_indices] - centroid, axis=1)
            isolated_point_idx = isolated_indices[np.argmax(isolated_distances_from_centroid)]
        else:
            isolated_point_idx = isolated_indices[0]

        # --- While loop logic to find the novelty radius ---
        nd_rfact = float('inf')
        current_pos = isolated_point_idx
        previous_pos = -1

        # This loop walks from the isolated point inwards via nearest neighbors
        # until the neighbor's distance to the centroid is less than the max radius (r_fact)
        while nd_rfact >= r_fact and current_pos != previous_pos:
            # Get the nearest neighbor of the current point
            neighbor_idx = indices_knn[current_pos, 1]

            # Calculate this neighbor's distance to the main centroid
            nd_rfact = np.linalg.norm(data_mapped[neighbor_idx] - centroid)

            # Move to the next point for the next iteration
            previous_pos = current_pos
            current_pos = neighbor_idx

        mc['radius_factor_1'] = r_fact
        mc['radius_factor_2'] = nd_rfact if nd_rfact < r_fact else r_fact

    return micro_clusters