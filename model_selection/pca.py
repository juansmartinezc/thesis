import os
import numpy as np
import pandas as pd
from dotenv import  load_dotenv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

def preprocess_data(numerical_cols, categorical_cols):
    preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
])
    return preprocessor


def run_pca(df, n_components=2, save_path=None, show_plot=True):
    categorical_cols = ['state_name', 'county_name']
    numerical_cols = [col for col in df.columns if col not in categorical_cols + ['Value']]
    X = df.drop(columns='Value')
    y = df['Value']
    print("🔄 Escalando variables numéricas...")
    preprocessor = preprocess_data(numerical_cols, categorical_cols)
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


def main():
    print("📦 Cargando dataset...")
    load_dotenv()
    source_data_directory = os.environ.get("SOURCE_DATA_DIRECTORY")
    results_path = 'results'
    experiment = 'PCA'
    models_path = 'models_results'
    models_results_path = os.path.join(results_path, models_path, experiment)
    best_models_path = os.path.join(models_results_path, 'best_models')
    summary_path = os.path.join(models_results_path, 'summary_best_models.csv')
    df = pd.read_csv(os.path.join(source_data_directory, 'historical_monthly_climate_data_apr_sept_by_scan_stations_and_nasa_final.csv'))

    print("🔍 Preparando datos para AutoML...")
    categorical_cols = ['state_name', 'county_name']
    numerical_cols = [col for col in df.columns if col not in categorical_cols + ['Value']]
    
    X = df.drop(columns='Value')
    y = df['Value']

    # Crear carpetas necesarias
    for path in [results_path, models_results_path, best_models_path]:
        os.makedirs(path, exist_ok=True)
        print(f"📁 Carpeta creada: {path}")
    
    # Ejecutar AutoML y guardar resumen
    run_pca(X)

# =========================
# EJECUCIÓN PRINCIPAL
# =========================
if __name__ == '__main__':
    main()
