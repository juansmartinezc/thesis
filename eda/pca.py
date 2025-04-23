import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def run_pca(X, n_components=2, save_path=None, show_plot=True):
    print("🔄 Escalando variables numéricas...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"📊 Ejecutando PCA con {n_components} componentes principales...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_ * 100
    print(f"📈 Varianza explicada por componente: {explained_var.round(2)}")

    df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])

    if show_plot and n_components == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(df_pca['PC1'], df_pca['PC2'], alpha=0.6)
        plt.title(f'PCA - Varianza explicada: PC1={explained_var[0]:.2f}%, PC2={explained_var[1]:.2f}%')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"💾 Gráfico de PCA guardado en: {save_path}")
        else:
            plt.show()

    return df_pca, pca, explained_var
