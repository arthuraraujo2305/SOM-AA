import numpy as np

# 1. Importar as funções do projeto
from functions import get_cond_probabilities_neurons, get_average_neuron_outputs


# 2. Configuração dos Dados de Teste (Inputs para a função)
# Estes são os resultados validados dos passos anteriores.

# Matriz de probabilidades validada no Passo 1
class_probabilities = np.array([
    [0.6   , 0.6667, 0.3333, 0.    ],
    [0.6667, 0.6   , 0.6667, 0.    ],
    [0.3333, 0.6667, 0.6   , 1.    ],
    [0.    , 0.    , 0.3333, 0.2   ]
])

# Lista de micro-clusters (baseada na simulação do Passo 2)
# Vamos focar apenas no Neurônio 5 para nosso teste.
micro_clusters = [{
    'neuron_id': 5,
    'centroid': np.random.rand(2), # Não usado neste teste
    'num_instances': 3,
    'prototype_vector': np.array([1.0, 0.66666667, 0.33333333, 0.0]), # Validado no Passo 2
    'timestamp_creation': 0,
    'class_position': 0,
    'cond_prob_threshold': np.full(4, 9.0) # Valor inicial que será substituído
}]

# Saída simulada do SOM (distâncias são cruciais aqui)
simulated_som_map = {
    'unit.classif': np.array([5, 2, 5, 5, 2]), # Instâncias 0, 2, 3 -> Neurônio 5
    'distances': np.array([
        0.1, # Distância da instância 0 ao seu vencedor (Neurônio 5)
        0.5, # ... inst 1
        0.2, # ... inst 2 (Neurônio 5)
        0.15,# ... inst 3 (Neurônio 5)
        0.6  # ... inst 4
    ])
}

# 3. Execução das Funções do seu Código
# 3.1. Primeiro, calculamos o output médio dos neurônios.
python_avg_outputs = get_average_neuron_outputs(simulated_som_map) # 6 para cobrir o ID 5
# 3.2. Agora, calculamos os thresholds. A função modifica a lista 'micro_clusters' internamente.
python_mc_with_thresholds = get_cond_probabilities_neurons(
    micro_clusters, class_probabilities, python_avg_outputs
)


# 4. Cálculo Manual Detalhado para Verificação (Ground Truth)
# Foco: Neurônio 5, para a classe c2 (índice 1).

print("--- INICIANDO VERIFICAÇÃO (Neurônio 5, Classe c2) ---")

# 4.1. Passo A: Calcular o output médio p(Xb|ck) para o Neurônio 5 - Equação (8)
# Instâncias mapeadas para o neurônio 5 são 0, 2, 3.
# Distâncias correspondentes: 0.1, 0.2, 0.15
dists_n5 = np.array([0.1, 0.2, 0.15])
outputs_n5 = np.exp(-dists_n5) # [exp(-0.1), exp(-0.2), exp(-0.15)]
manual_avg_output_n5 = outputs_n5.mean()
print(f"  [Cálculo Manual A] Output Médio p(Xb|ck) para Neurônio 5: {manual_avg_output_n5:.6f}")

# 4.2. Passo B: Calcular a probabilidade P(ck|C,Xb) - Equação (7)
# Para a classe c2 (k=1). As classes ativas no protótipo do neurônio 5 são c1, c2, c3.
# P(c2|{c1,c2,c3},Xb) = p(c2) * p(c1|c2) * p(c3|c2) * p(Xb|c2)
# (Usamos p(Xb|c2) como sendo o output médio do neurônio)
p_c2 = class_probabilities[1, 1]          # 0.6
p_c1_dado_c2 = class_probabilities[0, 1]  # 0.6667
p_c3_dado_c2 = class_probabilities[2, 1]  # 0.6667
manual_prob_j_ks_x = p_c2 * p_c1_dado_c2 * p_c3_dado_c2 * manual_avg_output_n5
print(f"  [Cálculo Manual B] Probabilidade P(ck|C,Xb): {manual_prob_j_ks_x:.6f}")

# 4.3. Passo C: Aplicar o fator de peso - Regra da linha 141 do artigo
# threshold = P(ck|C,Xb) * exp(-(1 - v_{n,k}))
# Para a classe c2 (k=1), o valor no vetor protótipo é v_{5,2}.
weight_factor = micro_clusters[0]['prototype_vector'][1] # 0.666667
manual_threshold = manual_prob_j_ks_x * np.exp(-(1 - weight_factor))
print(f"  [Cálculo Manual C] Fator de peso v_nk para c2: {weight_factor:.6f}")


# 5. Comparação e Impressão dos Resultados
print("\n--- COMPARAÇÃO FINAL ---")
print(f"  - THRESHOLD ESPERADO (MANUAL) PARA CLASSE c2: {manual_threshold:.6f}")

# Extrair o threshold calculado pela sua função para a classe c2 (índice 1)
python_threshold_c2 = python_mc_with_thresholds[0]['cond_prob_threshold'][1]
print(f"  - THRESHOLD CALCULADO (PYTHON) PARA CLASSE c2: {python_threshold_c2:.6f}")

if np.allclose(manual_threshold, python_threshold_c2):
    print("\n  => SUCESSO: O threshold foi calculado corretamente!")
else:
    print("\n  => FALHA: Os valores do threshold são diferentes!")

print("\n--- VERIFICAÇÃO CONCLUÍDA ---")