import numpy as np
import pandas as pd
from pandas.core.computation.expr import intersection


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:

    distance = np.sqrt(np.sum((x - y)**2))

    return distance

def compute_label_cadinality(offline_classes: pd.DataFrame) -> float:

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
