import numpy as np
import os
import time

def evaluate_novelty_detection_jaccard(true_labels: np.ndarray, predicted_labels: np.ndarray, num_known_classes: int) -> dict:
    """
    Avalia os Novelty Patterns gerados pelo modelo usando o Índice de Jaccard,
    conforme descrito na Seção 5 do artigo MINAS-BR.
    """
    num_instances = true_labels.shape[0]
    num_real_classes = true_labels.shape[1]
    num_pred_classes = predicted_labels.shape[1]
    
    print(f"[JACCARD] Instâncias: {num_instances}")
    print(f"[JACCARD] Classes Reais Totales: {num_real_classes} (Conhecidas: {num_known_classes}, Novas: {num_real_classes - num_known_classes})")
    print(f"[JACCARD] Colunas Preditas: {num_pred_classes} ({num_pred_classes - num_known_classes} NPs gerados)")

    # Passo 1: Construir a Matriz de Mapeamento (NPs x Classes Reais)
    # matriz_jaccard[i, j] = JI entre NP_i e a classe real j
    # Os NPs começam depois do índice 'num_known_classes'
    
    num_nps = num_pred_classes - num_known_classes
    matriz_jaccard = np.zeros((num_nps, num_real_classes))
    
    print("\n[JACCARD] Calculando Jaccard Index para os NPs...")
    start_time = time.time()
    
    for np_idx in range(num_nps):
        col_pred_idx = num_known_classes + np_idx
        
        for real_class_idx in range(num_real_classes):
            # A: Onde a predição é 1 E o valor real é 1
            a = np.sum((predicted_labels[:, col_pred_idx] == 1) & (true_labels[:, real_class_idx] == 1))
            
            # B: Onde a predição é 0 E o valor real é 1
            b = np.sum((predicted_labels[:, col_pred_idx] == 0) & (true_labels[:, real_class_idx] == 1))
            
            # C: Onde a predição é 1 E o valor real é 0
            c = np.sum((predicted_labels[:, col_pred_idx] == 1) & (true_labels[:, real_class_idx] == 0))
            
            denominator = a + b + c
            if denominator > 0:
                matriz_jaccard[np_idx, real_class_idx] = a / denominator
            else:
                matriz_jaccard[np_idx, real_class_idx] = 0.0

    print(f"[JACCARD] Matriz Jaccard construída em {time.time() - start_time:.2f}s")

    # Passo 2: Associar cada NP à classe real com maior Jaccard
    print("\n[JACCARD] Associando NPs às Classes Reais...")
    
    # Nova matriz de predições com o tamanho correto (igual às classes reais)
    final_predictions = np.zeros((num_instances, num_real_classes))
    
    # 2.1 Copia as predições das classes conhecidas (as primeiras colunas)
    for k in range(num_known_classes):
        final_predictions[:, k] = predicted_labels[:, k]
        
    # 2.2 Transfere as predições dos NPs para a coluna da classe associada
    associacoes = {}
    for np_idx in range(num_nps):
        col_pred_idx = num_known_classes + np_idx
        best_real_class = np.argmax(matriz_jaccard[np_idx, :])
        best_jaccard_value = matriz_jaccard[np_idx, best_real_class]
        
        if best_jaccard_value > 0:
            associacoes[f"NP_{np_idx+1}"] = f"Classe_{best_real_class}"
            # Onde o NP foi previsto, marcamos 1 na classe real associada
            final_predictions[:, best_real_class] = np.logical_or(
                final_predictions[:, best_real_class], 
                predicted_labels[:, col_pred_idx]
            ).astype(int)
    
    print(f"[JACCARD] {len(associacoes)} NPs foram associados com sucesso (JI > 0).")

    # Passo 3: Recalcular as métricas Finais (F-Measure Macro) com as predições corrigidas
    print("\n[JACCARD] Calculando Métricas Finais...")
    tp_cum = np.zeros(num_real_classes)
    fp_cum = np.zeros(num_real_classes)
    fn_cum = np.zeros(num_real_classes)
    
    for j in range(num_real_classes):
        tp_cum[j] = np.sum((true_labels[:, j] == 1) & (final_predictions[:, j] == 1))
        fp_cum[j] = np.sum((true_labels[:, j] == 0) & (final_predictions[:, j] == 1))
        fn_cum[j] = np.sum((true_labels[:, j] == 1) & (final_predictions[:, j] == 0))

    total_prec, total_recall, total_fmeasure = 0.0, 0.0, 0.0
    beta = 1.0
    
    for j in range(num_real_classes):
        tp, fp, fn = tp_cum[j], fp_cum[j], fn_cum[j]
        
        if tp + fp + fn == 0:
            prec = 1.0; recall = 1.0; fmeasure = 1.0 
        elif tp + fp == 0:
            prec = 0.0; recall = tp/(tp+fn) if (tp+fn) > 0 else 0.0
            fmeasure = 0.0
        elif tp + fn == 0:
            prec = tp/(tp+fp) if (tp+fp) > 0 else 0.0; recall = 0.0
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
        
    macro_precision = total_prec / num_real_classes
    macro_recall = total_recall / num_real_classes
    macro_fmeasure = total_fmeasure / num_real_classes

    # Passo 4: Calcular UnkRM (Unknown Rate Model)
    # Instâncias não explicadas são aquelas onde todas as colunas da predição original são 0
    # No seu log: 57255 foram explicadas. Total 90000. Desconhecidas = 90000 - 57255 = 32745.
    linhas_vazias = np.sum(predicted_labels, axis=1) == 0
    unk_count = np.sum(linhas_vazias)
    unkrm = (unk_count / num_instances) * 100 if num_instances > 0 else 0.0

    return {
        'F-Measure': macro_fmeasure,
        'Precision': macro_precision,
        'Recall': macro_recall,
        'UnkRM_Percent': unkrm,
        'Desconhecidos_Total': unk_count
    }

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
    
    # Número de classes que o modelo conhecia no treino (Neste dataset, eram 5)
    num_known_classes = 5 
    
    results = evaluate_novelty_detection_jaccard(true_labels, predicted_labels, num_known_classes)
    
    print("\n" + "="*50)
    print("   RESULTADOS FINAIS MINAS-BR (Avaliação Jaccard)")
    print("="*50)
    print(f" Macro F-Measure : {results['F-Measure']:.4f}")
    print(f" Macro Precision : {results['Precision']:.4f}")
    print(f" Macro Recall    : {results['Recall']:.4f}")
    print(f" UnkRM (%)       : {results['UnkRM_Percent']:.2f}% (Instâncias rejeitadas sem formar cluster: {results['Desconhecidos_Total']})")
    print("="*50)

if __name__ == "__main__":
    main()