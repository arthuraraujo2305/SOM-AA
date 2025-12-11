import matplotlib.pyplot as plt
import numpy as np  
import sys
import os

def plot_from_txt(file_path):
    if not os.path.exists(file_path):
        print(f"Arquivo não encontrado: {file_path}")
        return
    dataset_name = os.path.basename(file_path).split('.')[0]
    
    # Configuração do gráfico (tamanho e estilo)
    plt.figure(figsize=(10, 6))
    plt.title(f"Macro F-Measure over Evaluation Windows - {dataset_name}")
    plt.xlabel("Evaluation Windows")
    plt.ylabel("Macro F-Measure")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.6)

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
                
            label = parts[0]
            
            try:
                values = [float(x) for x in parts[1:]]
            except ValueError:
                print(f"Erro ao ler linha: {label}")
                continue
            
            # Eixo X (número das janelas)
            windows = range(1, len(values) + 1)
            
            # Escolher estilo de linha baseado no algoritmo (opcional, para diferenciar)
            linestyle = '-'
            linewidth = 2
            if "SOM-AA" in label:
                linewidth = 2.5 # Destacar o seu método
            elif "SOM-PT" in label:
                linestyle = '--'
            
            plt.plot(windows, values, label=label, linestyle=linestyle, linewidth=linewidth)

    # Adicionar legenda
    plt.legend(loc='lower right', fontsize='small', framealpha=0.9)
    
    # Salvar a figura
    output_img = file_path.replace('.txt', '.png')
    plt.savefig(output_img, dpi=300)
    print(f"Gráfico salvo em: {output_img}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python plot_results.py <caminho_do_arquivo_txt>")
    else:
        plot_from_txt(sys.argv[1])