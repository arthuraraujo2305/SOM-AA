import argparse
import os
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import arff
import pandas as pd
from functions import get_parameter_values, macro_precision_recall_fmeasure_windows
from kohonen import kohonen_offline_global, kohonen_online_bayes_nd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_arff_data(file_path):
    with open(file_path, 'r') as f:
        arff_data = arff.load(f)
    attributes = arff_data['attributes']
    data = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in attributes])
    feature_indices = [i for i, attr in enumerate(attributes) if attr[1] == 'NUMERIC']
    label_indices = [i for i, attr in enumerate(attributes) if isinstance(attr[1], list)]
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data, feature_indices, label_indices

def plot_som_hits(mapping, grid_dimension, dataset_name, base_results_dir="Results"):
    """
    Gera um Heatmap da contagem de exemplos por neurônio e salva em Results/NeuronViewer.
    """
    # 1. Preparar Diretório
    viewer_dir = os.path.join(base_results_dir, "NeuronViewer")
    if not os.path.exists(viewer_dir):
        os.makedirs(viewer_dir)
        print(f"[Plot] Diretório criado: {viewer_dir}")

    # 2. Recuperar dados do SOM
    # No seu código Python, unit_classif está dentro de mapping['som_map']
    unit_classif = mapping['som_map']['unit.classif']
    
    # 3. Construir a matriz de contagem
    hits_map = np.zeros((grid_dimension, grid_dimension))
    
    for neuron_idx in unit_classif:
        # Conversão de índice linear para (linha, coluna)
        # Nota: Ajuste se a orientação ficar transposta em relação ao R
        x = neuron_idx // grid_dimension
        y = neuron_idx % grid_dimension
        
        if x < grid_dimension and y < grid_dimension:
            hits_map[x, y] += 1

    # 4. Configurar e Salvar o Gráfico
    plt.figure(figsize=(10, 8))
    
    # Heatmap com números inteiros
    sns.heatmap(hits_map, annot=True, fmt='g', cmap="viridis", 
                cbar_kws={'label': 'Quantidade de Exemplos'})
    
    plt.title(f"SOM Counts - {dataset_name} ({grid_dimension}x{grid_dimension})")
    plt.xlabel("Dimensão Y")
    plt.ylabel("Dimensão X")
    
    # Nome do arquivo padronizado
    filename = f"{dataset_name}_counts.png"
    filepath = os.path.join(viewer_dir, filename)
    
    plt.savefig(filepath)
    plt.close() # Fecha a figura para liberar memória
    
    print(f"[Plot] Gráfico de contagem salvo em: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Run the MLSC Kohonen Map experiment.")
    parser.add_argument('param_file', type=str, help='Path to the parameter configuration file.')
    args = parser.parse_args()

    print("Loading parameters")
    parameters = get_parameter_values(args.param_file)
    print("Parameters loaded successfully:")
    for key, value in parameters.items():
        print(f"- {key}: {value}")

    print("\nLoading data")
    train_data, train_feature_indices, train_label_indices = load_arff_data(parameters['train_data'])
    test_data, test_feature_indices, test_label_indices = load_arff_data(parameters['test_data'])
    print("Data loaded successfully.")

    offline_dataset = train_data.iloc[:, train_feature_indices].values
    offline_classes = train_data.iloc[:, train_label_indices]
    online_dataset = test_data.iloc[:, test_feature_indices].values
    online_classes = test_data.iloc[:, test_label_indices]

    print(f"Offline dataset shape: {offline_dataset.shape}")
    print(f"Offline classes shape: {offline_classes.shape}")
    print(f"Online dataset shape: {online_dataset.shape}")
    print(f"Online classes shape: {online_classes.shape}")

    print("\nStandardizing data...")
    scaler = StandardScaler()
    scaler.fit(offline_dataset)
    offline_dataset_scaled = scaler.transform(offline_dataset)
    online_dataset_scaled = scaler.transform(online_dataset)
    print("Data standardized successfully.")

    #offline_dataset_scaled = offline_dataset
    #online_dataset_scaled = online_dataset

    print("\nStarting Offline Phase")

    num_epochs = int(parameters['num_iterations'])
    num_samples = len(offline_dataset_scaled)
    num_iterations_total = num_epochs * num_samples
    #print(f"Training Info: {num_epochs} epochs * {num_samples} samples = {num_iterations_total} total iterations.")
    print(f"Training Info: Batch Mode selected. Running for {num_epochs} epochs.")

    mapping = kohonen_offline_global(
        offline_dataset=offline_dataset_scaled,
        offline_classes=offline_classes,
        num_it=num_epochs,
        init_n=parameters['n0'],
        final_n=parameters['n1'],
        grid_d=int(parameters['grid_dimension']),
        tr_mode=parameters['train_mode'],
        min_ex=int(parameters['min_examples_cluster'])
    )
    print("Offline Phase Completed")
    print("Model mapping created successfully.")

    dataset_name = os.path.basename(parameters['test_data']).split('.')[0]
    grid_dim = int(parameters['grid_dimension'])
    
    # Chama a função de plotagem (lembre de importar plot_som_hits no topo do main.py)
    plot_som_hits(mapping, grid_dim, dataset_name)

    print("\n--- Starting Online Phase ---")
    init_n = parameters['n0']
    novel_classes = parameters['novel_classes']
    if not isinstance(novel_classes, list):
        novel_classes = [novel_classes]
    update_model_info = bool(parameters['update_model_info'])
    num_offline_instances = len(offline_dataset)

    online_results = kohonen_online_bayes_nd(
        mapping=mapping,
        online_dataset=online_dataset_scaled,
        init_n=init_n,
        novel_classes=novel_classes,
        update_model_info=update_model_info,
        num_offline_instances=num_offline_instances,
        theta=parameters['theta'],
        min_ex=parameters['min_examples_cluster']
    )

    print("Online Phase Completed")

    # Verificando matrizes
    print("\n[DEBUG] Exportando matrizes preditas e reais para conferência...")
    
    # Salvando em Results e depois em Debug_Matrices
    debug_dir = os.path.join("Results", "Debug_Matrices")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    dataset_name = os.path.basename(parameters['test_data']).split('.')[0]
    
    # prreparando a Matriz Real
    real_matrix_df = online_classes.astype(int)
    
    #preparando a Matriz Predita
    # online_results['predictions'] já vem do kohonen.py
    pred_matrix_df = online_results['predictions'].astype(int)
    
    #salvando os arquivos
    real_path = os.path.join(debug_dir, f"{dataset_name}_real.txt")
    pred_path = os.path.join(debug_dir, f"{dataset_name}_pred.txt")
    
    np.savetxt(real_path, real_matrix_df.values, fmt='%d', delimiter=' ')
    np.savetxt(pred_path, pred_matrix_df.values, fmt='%d', delimiter=' ')
    
    print(f"Matriz REAL salva em: {real_path}")
    print(f"Matriz PREDITA salva em: {pred_path}")


    if 'indexes_explained' in online_results:
        print(f"{len(online_results['indexes_explained'])} instances were explained by the model.")

    print("\nEvaluating Results")
    predictions = online_results['predictions']
    explained_indices = online_results['indexes_explained']
    true_classes_explained = online_classes.iloc[explained_indices]
    predicted_classes_explained = predictions.iloc[explained_indices]

    num_windows = int(parameters['num_evaluation_windows'])

    dataset_name = os.path.basename(parameters['test_data']).split('.')[0]
    
    evaluation_metrics = macro_precision_recall_fmeasure_windows(
        true_labels=true_classes_explained.values,
        predicted_labels=predicted_classes_explained.values,
        num_evaluation_windows=num_windows,
        dataset_name=dataset_name
    )
    online_results['evaluation_metrics'] = evaluation_metrics

    print("Evaluation completed:")
    print(f"  - Macro F-Measure: {evaluation_metrics['ma_fmeasure']:.4f}")
    print(f"  - Macro Precision: {evaluation_metrics['ma_precision']:.4f}")
    print(f"  - Macro Recall: {evaluation_metrics['ma_recall']:.4f}")
    
    # Definindo o caminho do arquivo de resultados
    results_txt_path = f"Results/{dataset_name}.txt"
    
    # Cria o rótulo do algoritmo
    grid_dim = int(parameters['grid_dimension'])
    algo_label = f"SOM-AA-{grid_dim}"
    
    # Pega a lista de F-Measure por janela que já foi calculada
    fmeasure_window_values = evaluation_metrics['ma_fmeasure_window']
    
    # Formata como string separada por vírgulas
    values_str = ",".join([f"{x:.8f}" for x in fmeasure_window_values])
    line_to_write = f"{algo_label},{values_str}\n"
    
    with open(results_txt_path, "a", encoding='utf-8') as f:
        f.write(line_to_write)
        
    print(f"Resultados por janela salvos em: {results_txt_path}")

    if not os.path.exists('Results'):
        os.makedirs('Results')
    dataset_name = os.path.basename(parameters['test_data']).split('.')[0]
    timestamp = datetime.now().strftime("%H.%M.%S")
    grid_dim = int(parameters['grid_dimension'])
    params_filename = f"Results/{dataset_name}-{timestamp}-parameters-{grid_dim}.txt"
    with open(params_filename, 'w') as f:
        f.write(str(parameters))
    print(f"\nParameters saved to {params_filename}")
    model_filename = f"Results/{dataset_name}-{timestamp}-model-{grid_dim}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(online_results, f)
    print(f"Full results object saved to {model_filename}")

if __name__ == "__main__":
    main()