import numpy as np

dataset_name = "MOA-5C-7C-2D-test"
real_path = f"Results/Debug_Matrices/{dataset_name}_real.txt"
pred_path = f"Results/Debug_Matrices/{dataset_name}_pred.txt"

true_labels = np.loadtxt(real_path, dtype=int)
pred = np.loadtxt(pred_path, dtype=int)

num_known = 5

print(f"Shape true_labels: {true_labels.shape}")
print(f"Shape pred: {pred.shape}")
print()

for np_col in range(num_known, pred.shape[1]):
    print(f"===== NP coluna {np_col} =====")
    for real_col in range(num_known, true_labels.shape[1]):
        a = np.sum((pred[:, np_col] == 1) & (true_labels[:, real_col] == 1))
        b = np.sum((pred[:, np_col] == 0) & (true_labels[:, real_col] == 1))
        c = np.sum((pred[:, np_col] == 1) & (true_labels[:, real_col] == 0))

        ji = a / (a + b + c) if (a + b + c) > 0 else 0.0

        print(
            f"Classe real {real_col}: "
            f"a={a}, b={b}, c={c}, JI={ji:.6f}"
        )
    print()