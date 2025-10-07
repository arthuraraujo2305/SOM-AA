import pandas as pd
import numpy as np

# 1. Importar a função do arquivo
from functions import compute_micro_clusters


# 2. Criação dos Dados de Teste (Dummy Data)
# Usaremos os mesmos rótulos do Passo 1.
dummy_classes = pd.DataFrame([
    # Instância | c1 | c2 | c3 | c4
    # ---------------------------------
    [1, 1, 0, 0],  # Instância 0
    [0, 1, 1, 0],  # Instância 1
    [1, 0, 0, 0],  # Instância 2
    [1, 1, 1, 0],  # Instância 3
    [0, 0, 1, 1],  # Instância 4
], columns=['c1', 'c2', 'c3', 'c4'])

# Criamos também um dataset de atributos. Os valores não importam muito,
# apenas a estrutura.
dummy_dataset = np.array([
    [0.1, 0.9],
    [0.8, 0.2],
    [0.2, 0.8],
    [0.3, 0.7],
    [0.9, 0.1]
])

# 3. Simulação da Saída de um SOM Treinado
# Em vez de treinar um SOM, vamos criar um dicionário 'som_map' falso
# que simula o resultado de um treinamento. O mais importante é 'unit.classif'.
# 'unit.classif' diz qual instância foi mapeada para qual neurônio (ID do neurônio).
simulated_som_map = {
    'codes': np.random.rand(10, 2),  # Pesos dos neurônios (não são usados neste teste)
    'unit.classif': np.array([
        5, # Instância 0 -> Neurônio 5
        2, # Instância 1 -> Neurônio 2
        5, # Instância 2 -> Neurônio 5
        5, # Instância 3 -> Neurônio 5
        2  # Instância 4 -> Neurônio 2
    ]),
    'distances': np.random.rand(5) # Distâncias (não são usadas neste teste)
}
min_examples_for_cluster = 1 # Mínimo de exemplos para um neurônio ser válido

print("--- Simulação ---")
print("Mapeamento simulado (Instância -> Neurônio):")
for i, neuron_id in enumerate(simulated_som_map['unit.classif']):
    print(f"  - Instância {i} foi mapeada para o Neurônio {neuron_id}")
print("\n")


# 4. Execução da Função a Ser Testada
# Chamamos sua função com os dados simulados.
result_mc = compute_micro_clusters(
    som_map=simulated_som_map,
    offline_classes=dummy_classes,
    min_ex=min_examples_for_cluster
)
# A função retorna uma lista de dicionários, um para cada micro-cluster (neurônio válido).
python_micro_clusters = result_mc['micro_clusters']


# 5. Cálculo Manual para Verificação (Ground Truth)
# Vamos calcular o vetor protótipo esperado para o Neurônio 5.
# Instâncias mapeadas para o Neurônio 5: 0, 2, 3.
# Pegamos as linhas 0, 2 e 3 de dummy_classes.
rows_for_neuron_5 = dummy_classes.iloc[[0, 2, 3]]
# O vetor protótipo é a média dessas linhas (Equação 1 do artigo).
manual_prototype_vector_n5 = rows_for_neuron_5.mean(axis=0).values

print("--- INICIANDO VERIFICAÇÃO (para o Neurônio 5) ---")
print("Instâncias que deveriam compor o Neurônio 5:")
print(rows_for_neuron_5)

# 6. Comparação e Impressão dos Resultados
# Vamos encontrar o micro-cluster correspondente ao Neurônio 5 na saída da sua função.
# Nota: O ID do neurônio está em 'neuron_id' no dicionário do micro-cluster.
python_prototype_vector_n5 = None
for mc in python_micro_clusters:
    if mc['neuron_id'] == 5:
        python_prototype_vector_n5 = mc['prototype_vector']
        break

print("\n[VERIFICANDO O VETOR PROTÓTIPO do Neurônio 5]")
if python_prototype_vector_n5 is not None:
    print(f"  - Vetor da sua função Python: {np.round(python_prototype_vector_n5, 4)}")
    print(f"  - Vetor esperado (manual):    {np.round(manual_prototype_vector_n5, 4)}")

    # Comparamos os dois vetores
    if np.allclose(python_prototype_vector_n5, manual_prototype_vector_n5):
        print("  => SUCESSO: O vetor protótipo foi calculado corretamente!")
    else:
        print("  => FALHA: O vetor protótipo está incorreto!")
else:
    print("  => FALHA: Não foi encontrado o micro-cluster para o Neurônio 5 na saída da função.")

print("\n--- VERIFICAÇÃO CONCLUÍDA ---")