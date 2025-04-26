import os
import pandas as pd
from dotenv import load_dotenv
from pycaret.regression import setup, compare_models, save_model, pull

# =========================
# CONFIGURACIÓN DE RUTAS
# =========================
results_path = 'results'
experiment = 'auto_ml'
models_path = 'models_results'
source_data_directory = os.environ.get("SOURCE_DATA_DIRECTORY")

models_results_path = os.path.join(results_path, models_path, experiment)
best_models_path = os.path.join(models_results_path, 'best_models')
summary_path = os.path.join(models_results_path, 'summary_best_models.csv')


# =========================
# FUNCIONES
# =========================


def run_automl(X, y, save_path):
    df = X.copy()
    df['target'] = y
    
    print("🔧 Configurando entorno AutoML con PyCaret...")
    setup(data=df, target='target', session_id=42, experiment_name="automl_experiment")

    print("🤖 Comparando modelos con AutoML...")
    best_model = compare_models()

    print("💾 Guardando mejor modelo encontrado por AutoML...")
    save_model(best_model, os.path.join(best_models_path, 'pycaret_best_model'))

    print("📝 Extrayendo resumen de métricas del mejor modelo...")
    leaderboard = pull()  # Extrae el leaderboard después de compare_models()
    # Calcular NRMSE para todos los modelos
    y_mean = y.mean()
    leaderboard['NRMSE_%'] = leaderboard['RMSE'] / y_mean * 100

    # Guardar todos los modelos evaluados con sus métricas
    leaderboard.to_csv(os.path.join(save_path, 'pycaret_leaderboard.csv'), index=False)
    print(f"✅ Leaderboard completo guardado en: {os.path.join(save_path, 'pycaret_leaderboard.csv')}")
   
    # Guardar resumen del mejor modelo
    best_row = leaderboard.iloc[0]
    summary = best_row[['Model', 'MAE', 'MSE', 'RMSE', 'R2', 'MAPE', 'NRMSE_%']].to_frame().T
    summary.to_csv(summary_path, index=False)
    print(f"✅ Resumen del mejor modelo guardado en: {summary_path}")


def main():
    print("📦 Cargando dataset...")
    load_dotenv()
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
    run_automl(X, y, models_results_path)


# =========================
# EJECUCIÓN PRINCIPAL
# =========================
if __name__ == '__main__':
    main()
