import numpy as np

dataset_name = "MOA-5C-7C-2D-test"
pred_path = f"Results/Debug_Matrices/{dataset_name}_pred.txt"

pred = np.loadtxt(pred_path, dtype=int)

linhas_vazias = np.sum(pred, axis=1) == 0
unk_count = int(np.sum(linhas_vazias))

print(f"Desconhecidos totais: {unk_count}")
print(f"Percentual: {100 * unk_count / pred.shape[0]:.6f}%")