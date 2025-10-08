import numpy as np
import pandas as pd
from kohonen import kohonen_online_bayes_nd

print("Preparando o ambiente para o Passo 4")

# 1. Construir o objeto 'mapping' com nossos dados validados
# Usamos os resultados dos passos 1, 2 e 3.

# Adicionamos o Neurônio 2 para ter mais de uma opção no k-NN
micro_clusters = [
    { # Este é o nosso Neurônio 5, validado anteriormente
        'neuron_id': 5,
        'centroid': np.array([0.2, 0.8]), # Centroide simulado
        'num_instances': 3,
        'prototype_vector': np.array([1.0, 0.6667, 0.3333, 0.0]),
        'cond_prob_threshold': np.array([0.0, 0.164613, 0.0549, 0.0]), # Thresholds (c2 validado)
        'average_output': [0, 0] # Chave adicionada que estava faltando!
    },
    { # Um segundo neurônio para o teste
        'neuron_id': 2,
        'centroid': np.array([0.9, 0.1]), # Centroide simulado
        'num_instances': 2,
        'prototype_vector': np.array([0.0, 0.5, 1.0, 0.5]),
        'cond_prob_threshold': np.zeros(4), # Thresholds zerados para simplificar
        'average_output': [0, 0] # Chave adicionada que estava faltando!
    }
]

# Objeto 'mapping' completo
mapping = {
    'som_map': {
        # Os centroides aqui precisam ser os mesmos definidos nos micro_clusters
        'codes': np.array([
            [0,0], [0,0],               # Placeholders para neurônios 0, 1
            micro_clusters[1]['centroid'], # Neurônio 2
            [0,0], [0,0],               # Placeholders
            micro_clusters[0]['centroid']  # Neurônio 5
        ])
    },
    'micro_clusters': micro_clusters,
    'class_probabilities': np.array([ # Validado no Passo 1
        [0.6, 0.6667, 0.3333, 0.],
        [0.6667, 0.6, 0.6667, 0.],
        [0.3333, 0.6667, 0.6, 1.],
        [0., 0., 0.3333, 0.2]
    ]),
    'z': 2.0, # Validado no Passo 1
    'total_instances': 5,
    'class_totals': np.zeros((4,4)) # Não usado na predição
}

# 2. Criar uma única instância de teste online
# Esta instância é muito parecida com o centroide do Neurônio 5,
# então ele deve ser o vencedor.
online_instance = np.array([[0.21, 0.79]])

print("Ambiente pronto. 'mapping' criado e instância de teste definida.")
print("Iniciando a execução de 'kohonen_online_bayes_nd'...\n")


# 3. Executar a função (que já tem os prints de depuração)
results = kohonen_online_bayes_nd(
    mapping=mapping,
    online_dataset=online_instance,
    init_n=0.05,
    novel_classes=[0],
    update_model_info=False,
    num_offline_instances=5
)

# 4. Imprimir o resultado final
final_prediction = results['predictions'].iloc[0].values
print("\n--- Resultado Final da Predição ---")
print(f"Vetor de predição final para a instância: {final_prediction.astype(int)}")

# Verificação manual rápida:
# Espera-se que c1 e c2 sejam preditos (1), e c3 e c4 não (0).
if final_prediction[0] == 1 and final_prediction[1] == 1 and final_prediction[2] == 0 and final_prediction[3] == 0:
    print("\nResultado consistente com o esperado! ✅")
else:
    print("\nResultado inconsistente com o esperado. ❌")