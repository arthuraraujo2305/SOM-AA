from functions import macro_precision_recall_fmeasure_windows
import numpy as np
import os
import time

def evaluate_novelty_detection_jaccard(true_labels: np.ndarray,
                                       predicted_labels: np.ndarray,
                                       num_known_classes: int) -> dict:
    """
    Avalia NPs conforme a metodologia do MINAS-BR:
      1) calcula Jaccard entre cada NP e cada classe NOVA real;
      2) associa NP à classe nova com maior Jaccard;
      3) transfere as predições dos NPs para a matriz final;
      4) recalcula Macro F-Measure;
      5) calcula UnkRM macro.
    """
    num_instances = true_labels.shape[0]
    num_real_classes = true_labels.shape[1]
    num_pred_classes = predicted_labels.shape[1]

    # As primeiras colunas da predição correspondem ao espaço original de classes
    # (mesmo que algumas classes não tenham aparecido no treino).
    np_start_col = num_real_classes

    num_novel_real_classes = num_real_classes - num_known_classes
    num_nps = num_pred_classes - np_start_col

    print(f"[JACCARD] Instâncias: {num_instances}")
    print(f"[JACCARD] Classes reais: {num_real_classes}")
    print(f"[JACCARD] Classes conhecidas no treino: {num_known_classes}")
    print(f"[JACCARD] Classes novas reais: {num_novel_real_classes}")
    print(f"[JACCARD] Colunas originais no modelo: {num_real_classes}")
    print(f"[JACCARD] NPs gerados: {num_nps}")

    if num_nps <= 0:
        print("[JACCARD] Nenhum NP foi gerado.")
        final_predictions = predicted_labels[:, :num_real_classes].copy()

        f1, prec, rec = compute_macro_metrics(true_labels, final_predictions)
        unkrm, unk_count = compute_macro_unkrm(true_labels, predicted_labels)

        return {
            'F-Measure': f1,
            'Precision': prec,
            'Recall': rec,
            'UnkRM_Percent': unkrm,
            'Desconhecidos_Total': unk_count,
            'Associacoes': {},
            'final_predictions': final_predictions
        }

    # Matriz Jaccard: linhas = NPs, colunas = classes novas reais
    matriz_jaccard = np.zeros((num_nps, num_novel_real_classes))

    print("\n[JACCARD] Calculando matriz de Jaccard...")
    start_time = time.time()

    for np_idx in range(num_nps):
        col_pred_idx = np_start_col + np_idx

        for offset in range(num_novel_real_classes):
            real_class_idx = num_known_classes + offset

            a = np.sum((predicted_labels[:, col_pred_idx] == 1) & (true_labels[:, real_class_idx] == 1))
            b = np.sum((predicted_labels[:, col_pred_idx] == 0) & (true_labels[:, real_class_idx] == 1))
            c = np.sum((predicted_labels[:, col_pred_idx] == 1) & (true_labels[:, real_class_idx] == 0))

            denominator = a + b + c
            matriz_jaccard[np_idx, offset] = (a / denominator) if denominator > 0 else 0.0

    print(f"[JACCARD] Matriz construída em {time.time() - start_time:.2f}s")

    final_predictions = np.zeros((num_instances, num_real_classes), dtype=int)

    # Copia todas as colunas originais do espaço de classes
    final_predictions[:, :num_real_classes] = predicted_labels[:, :num_real_classes]

    associacoes = {}

    # Associa cada NP à classe nova real com maior Jaccard
    for np_idx in range(num_nps):
        col_pred_idx = np_start_col + np_idx

        if num_novel_real_classes == 0:
            continue

        best_offset = np.argmax(matriz_jaccard[np_idx, :])
        best_real_class = num_known_classes + best_offset
        best_jaccard_value = matriz_jaccard[np_idx, best_offset]

        if best_jaccard_value > 0:
            associacoes[f"NP_{np_idx+1}"] = {
                'classe_real_associada': int(best_real_class),
                'jaccard': float(best_jaccard_value)
            }

            final_predictions[:, best_real_class] = np.logical_or(
                final_predictions[:, best_real_class],
                predicted_labels[:, col_pred_idx]
            ).astype(int)

    macro_fmeasure, macro_precision, macro_recall = compute_macro_metrics(true_labels, final_predictions)
    unkrm, unk_count = compute_macro_unkrm(true_labels, predicted_labels)

    return {
        'F-Measure': macro_fmeasure,
        'Precision': macro_precision,
        'Recall': macro_recall,
        'UnkRM_Percent': unkrm,
        'Desconhecidos_Total': unk_count,
        'Associacoes': associacoes,
        'final_predictions': final_predictions
    }


def compute_macro_metrics(true_labels: np.ndarray, predicted_labels: np.ndarray):
    num_classes = true_labels.shape[1]

    total_prec, total_recall, total_fmeasure = 0.0, 0.0, 0.0
    beta = 1.0

    for j in range(num_classes):
        tp = np.sum((true_labels[:, j] == 1) & (predicted_labels[:, j] == 1))
        fp = np.sum((true_labels[:, j] == 0) & (predicted_labels[:, j] == 1))
        fn = np.sum((true_labels[:, j] == 1) & (predicted_labels[:, j] == 0))

        if tp + fp + fn == 0:
            prec = 1.0
            recall = 1.0
            fmeasure = 1.0
        elif tp + fp == 0:
            prec = 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fmeasure = 0.0
        elif tp + fn == 0:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = 0.0
            fmeasure = 0.0
        else:
            prec = tp / (tp + fp)
            recall = tp / (tp + fn)

            if prec + recall == 0:
                fmeasure = 0.0
            else:
                beta2 = beta * beta
                fmeasure = ((beta2 + 1) * prec * recall) / (beta2 * prec + recall)

        total_prec += prec
        total_recall += recall
        total_fmeasure += fmeasure

    macro_precision = total_prec / num_classes
    macro_recall = total_recall / num_classes
    macro_fmeasure = total_fmeasure / num_classes

    return macro_fmeasure, macro_precision, macro_recall


def compute_macro_unkrm(true_labels: np.ndarray, predicted_labels: np.ndarray):
    """
    UnkRM macro:
      média, por classe real, da fração de exemplos dessa classe
      que ficaram totalmente desconhecidos.
    """
    num_instances, num_real_classes = true_labels.shape

    linhas_vazias = np.sum(predicted_labels, axis=1) == 0
    unk_count = int(np.sum(linhas_vazias))

    soma = 0.0
    for j in range(num_real_classes):
        Nj = np.sum(true_labels[:, j] == 1)
        UNKj = np.sum((true_labels[:, j] == 1) & linhas_vazias)

        if Nj > 0:
            soma += (UNKj / Nj)

    unkrm = 100.0 * (soma / num_real_classes) if num_real_classes > 0 else 0.0
    return unkrm, unk_count


def main():
    dataset_name = "MOA-5C-7C-2D-test"

    real_path = f"Results/Debug_Matrices/{dataset_name}_real.txt"
    pred_path = f"Results/Debug_Matrices/{dataset_name}_pred.txt"

    if not os.path.exists(real_path) or not os.path.exists(pred_path):
        print(f"Erro: Arquivos não encontrados em {real_path} ou {pred_path}")
        return

    print(f"Carregando matrizes de {dataset_name}...")
    true_labels = np.loadtxt(real_path, dtype=int)
    predicted_labels = np.loadtxt(pred_path, dtype=int)

    num_known_classes = 5

    results = evaluate_novelty_detection_jaccard(true_labels, predicted_labels, num_known_classes)

    print("\nCalculando as métricas por janela (Pós-Jaccard) para o gráfico...")
    metrics_windows = macro_precision_recall_fmeasure_windows(
        true_labels=true_labels,
        predicted_labels=results['final_predictions'], # Usando a matriz fundida
        num_evaluation_windows=50,
        dataset_name=dataset_name
    )
    
    fmeasure_window_values = metrics_windows['ma_fmeasure_window']
    values_str = ",".join([f"{x:.8f}" for x in fmeasure_window_values])
    
    results_txt_path = f"Results/{dataset_name}_Jaccard.txt"
    with open(results_txt_path, "w", encoding='utf-8') as f:
        f.write(f"SOM-AA-GSOM-0.85,{values_str}\n")
    print(f"Resultados por janela salvos em: {results_txt_path}")
    print("\n" + "=" * 60)
    print("RESULTADOS FINAIS - AVALIAÇÃO JACCARD / MINAS-BR")
    print("=" * 60)
    print(f"Macro F-Measure : {results['F-Measure']:.4f}")
    print(f"Macro Precision : {results['Precision']:.4f}")
    print(f"Macro Recall    : {results['Recall']:.4f}")
    print(f"UnkRM (%)       : {results['UnkRM_Percent']:.2f}%")
    print(f"Desconhecidos   : {results['Desconhecidos_Total']}")
    print(f"Associações     : {results['Associacoes']}")
    print("=" * 60)


if __name__ == "__main__":
    main()