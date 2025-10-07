import pandas as pd
import numpy as np

# 1. Importar as
from functions import compute_label_cardinality, compute_initial_class_probabilities_totals

# 2. Criação dos Dados de Teste (Dummy Data)
# Um DataFrame pequeno e simples para o qual podemos calcular os resultados na mão.
dummy_classes = pd.DataFrame([
    # Instância | c1 | c2 | c3 | c4
    # ---------------------------------
    [1, 1, 0, 0],  # Instância 0
    [0, 1, 1, 0],  # Instância 1
    [1, 0, 0, 0],  # Instância 2
    [1, 1, 1, 0],  # Instância 3
    [0, 0, 1, 1],  # Instância 4
], columns=['c1', 'c2', 'c3', 'c4'])

print("--- Dados de Teste (dummy_classes) ---")
print(dummy_classes)
print("\n")


# 3. Execução das Funções a Serem Testadas
# Chamamos as suas funções com os dados de teste.
python_z = compute_label_cardinality(dummy_classes)
python_probs, python_totals = compute_initial_class_probabilities_totals(dummy_classes)


# 4. Cálculos Manuais para Verificação (Ground Truth)
# Aqui, calculamos os valores que esperamos que as funções retornem, baseados no artigo.

# 4.1. Cardinalidade (z) - Equação (2) do artigo
# z = (Soma de todos os rótulos) / (Número de instâncias)
manual_z = (1+1+0+0 + 0+1+1+0 + 1+0+0+0 + 1+1+1+0 + 0+0+1+1) / 5.0  # 10 / 5 = 2.0

# 4.2. Totais de Classe (matriz de coocorrência)
# A diagonal f(cj) é a contagem de cada classe.
# Fora da diagonal f(ck, cj) é a contagem de instâncias que têm ck E cj.
manual_totals = np.array([
    #     c1, c2, c3, c4
    [3.0, 2.0, 1.0, 0.0],  # c1
    [2.0, 3.0, 2.0, 0.0],  # c2
    [1.0, 2.0, 3.0, 1.0],  # c3
    [0.0, 0.0, 1.0, 1.0],  # c4
])

# 4.3. Probabilidades de Classe - Equação (3) do artigo
# A diagonal p(cj) = f(cj) / N_instancias
# Fora da diagonal p(ck|cj) = f(ck, cj) / f(cj)
manual_probs = np.zeros((4, 4))
num_instances = 5.0
# Diagonais (probabilidades a priori)
manual_probs[0,0] = manual_totals[0,0] / num_instances # 3/5
manual_probs[1,1] = manual_totals[1,1] / num_instances # 3/5
manual_probs[2,2] = manual_totals[2,2] / num_instances # 3/5
manual_probs[3,3] = manual_totals[3,3] / num_instances # 1/5
# Fora da diagonal (probabilidades condicionais)
# p(ck|cj) -> linha k, coluna j
manual_probs[0,1] = manual_totals[0,1] / manual_totals[1,1] # p(c1|c2) = 2/3
manual_probs[0,2] = manual_totals[0,2] / manual_totals[2,2] # p(c1|c3) = 1/3
manual_probs[1,0] = manual_totals[1,0] / manual_totals[0,0] # p(c2|c1) = 2/3
manual_probs[2,1] = manual_totals[2,1] / manual_totals[1,1] # p(c3|c2) = 2/3
# ...e assim por diante. Vamos preencher o resto para uma comparação completa.
manual_probs[1,2] = manual_totals[1,2] / manual_totals[2,2] # p(c2|c3) = 2/3
manual_probs[2,0] = manual_totals[2,0] / manual_totals[0,0] # p(c3|c1) = 1/3
manual_probs[3,2] = manual_totals[3,2] / manual_totals[2,2] # p(c4|c3) = 1/3
manual_probs[2,3] = manual_totals[2,3] / manual_totals[3,3] # p(c3|c4) = 1/1
# O resto é zero, pois as coocorrências são zero


# 5. Comparação e Impressão dos Resultados
print("--- INICIANDO VERIFICAÇÃO ---")

# 5.1. Comparando a Cardinalidade (z)
print("\n[VERIFICANDO A CARDINALIDADE 'z']")
print(f"  - Valor da sua função Python: {python_z}")
print(f"  - Valor esperado (manual):    {manual_z}")
if abs(python_z - manual_z) < 1e-9:
    print("  => SUCESSO: Os valores de 'z' são iguais!")
else:
    print("  => FALHA: Os valores de 'z' são diferentes!")

# 5.2. Comparando a Matriz de Totais
print("\n[VERIFICANDO A MATRIZ DE TOTAIS]")
print("  - Matriz da sua função Python:")
print(python_totals)
print("\n  - Matriz esperada (manual):")
print(manual_totals)
if np.allclose(python_totals, manual_totals):
    print("  => SUCESSO: As matrizes de totais são iguais!")
else:
    print("  => FALHA: As matrizes de totais são diferentes!")

# 5.3. Comparando a Matriz de Probabilidades
print("\n[VERIFICANDO A MATRIZ DE PROBABILIDADES]")
print("  - Matriz da sua função Python:")
print(np.round(python_probs, 4)) # Arredondar para facilitar a leitura
print("\n  - Matriz esperada (manual):")
print(np.round(manual_probs, 4)) # Arredondar para facilitar a leitura
if np.allclose(python_probs, manual_probs):
    print("  => SUCESSO: As matrizes de probabilidades são iguais!")
else:
    print("  => FALHA: As matrizes de probabilidades são diferentes!")

print("\n--- VERIFICAÇÃO CONCLUÍDA ---")