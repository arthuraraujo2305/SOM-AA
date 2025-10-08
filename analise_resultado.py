import pickle
import numpy as np

# Mude o nome do arquivo para o que foi gerado na sua pasta Results/
NOME_DO_ARQUIVO_DE_RESULTADOS = "Results/mediamill-test-11.11.47-model-10.pkl"

try:
    with open(NOME_DO_ARQUIVO_DE_RESULTADOS, 'rb') as f:
        results = pickle.load(f)

    print(f"Analisando o arquivo: {NOME_DO_ARQUIVO_DE_RESULTADOS}")

    predictions = results['predictions'].values

    # Calcula quantos rótulos foram previstos para cada instância
    labels_per_instance = np.sum(predictions, axis=1)

    # Calcula a cardinalidade média da predição
    average_cardinality = np.mean(labels_per_instance)

    print("\n--- Análise de Cardinalidade da Predição ---")
    print(f"Cardinalidade original do dataset (do artigo): 2.96")
    print(f"Cardinalidade MÉDIA das suas predições: {average_cardinality:.4f}")

    if average_cardinality > 4: # Um limiar arbitrário para agressividade
        print("\nDiagnóstico: O modelo está prevendo um número muito alto de rótulos por instância.")
        print("Isso confirma a hipótese do Recall alto e Precisão baixa.")
    else:
        print("\nDiagnóstico: A cardinalidade da predição parece razoável.")

except FileNotFoundError:
    print(f"Erro: Arquivo '{NOME_DO_ARQUIVO_DE_RESULTADOS}' não encontrado. Verifique o nome.")