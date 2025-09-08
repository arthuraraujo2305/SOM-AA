import numpy as np
import pandas as pd
from pandas.core.computation.expr import intersection
from collections import Counter


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

    Args:
        param_file: O caminho para o arquivo de configuração.

    Returns:
        Um dicionário com os nomes dos parâmetros como chaves e seus
        respectivos valores.
    """
    parameters = {}
    with open(param_file, 'r') as f:
        for line in f:
            # Ignora linhas vazias ou comentários
            if line.strip() and not line.strip().startswith('#'):
                key, value = line.split('=', 1)

                # Limpa espaços em branco da chave e do valor
                key = key.strip()
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
            avg_output = average_neuron_outputs[i][0] / average_neuron_outputs[i][1]

            # Final probability p(y_j | y_k, x) calculation
            prob_j_ks_x = prob_j * prob_k_j * avg_output

            # The final threshold is weighted by the prototype vector value
            threshold = prob_j_ks_x * np.exp(-(1 - weight_factor))

            # Update the micro-cluster's threshold for this specific class
            mc['cond_prob_threshold'][class_idx] = threshold

            mc['average_output'] = average_neuron_outputs[i]

    return micro_clusters
