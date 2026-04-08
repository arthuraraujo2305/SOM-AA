import numpy as np

dataset_name = "MOA-5C-7C-2D-test"
pred_path = f"Results/Debug_Matrices/{dataset_name}_pred.txt"

pred = np.loadtxt(pred_path, dtype=int)

print(f"Shape da matriz predita: {pred.shape}")
print()

for j in range(pred.shape[1]):
    total_ones = int(pred[:, j].sum())
    print(f"Coluna {j}: {total_ones} ativações")