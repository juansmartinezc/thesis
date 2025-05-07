import os
import mlflow
import logging
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from pycaret.regression import setup, load_model, tune_model, pull, save_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    load_dotenv()
    data_dir = os.getenv("SOURCE_DATA_DIRECTORY")
    if not data_dir:
        logger.error("SOURCE_DATA_DIRECTORY no está definida en el .env")
        return

    # Preparación del nombre del experimento y rutas
    config = {
        'log_experiment': True,
        'outliers': False,
        'normalize': True,
        'multicol': False,
        'pca': False
    }
    experiment = "auto_ml_wo_time_wo_location_top_3__" + "__".join([f"{k[:3]}={v}" for k, v in config.items()])
    print(f"experiment: {experiment}")
    base_dir = Path("results") / "models_results" / experiment
    print(f"base dir: {base_dir}")
    best_models_dir = base_dir / "best_models"
    print(f"best_models_dir: {best_models_dir}")
    model_path = best_models_dir / "pycaret_best_model"
    print(f"model_path: {model_path}")
    
    # Cargar datos y preparar
    data_path = Path(__file__).resolve().parent.parent / data_dir / "historical_monthly_climate_data_apr_sept_by_scan_stations_and_nasa_final.csv"
    df = pd.read_csv(data_path)
    drop_cols = [
        'Unnamed: 0', 'state_name', 'county_name', 'state_alpha', 'county_code', 'year', 'month', 'stationTriplet',
        'stationId', 'name', 'lat_centroid', 'lon_centroid', 'latitude', 'longitude', 'unit_desc',
    ]
    states = ['IOWA', 'INDIANA', 'ILLINOIS']
    df = df[df['state_name'].isin(states)]
    y = df['Value']
    X = df.drop(columns=drop_cols + ['Value'], errors='ignore')
    df_exp = pd.concat([X, y.rename("target")], axis=1)

    # Setup de PyCaret para tuning
    setup(
        data=df_exp,
        target='target',
        fold=5,
        experiment_name=experiment + "_rf_tuning",
        log_experiment=config['log_experiment'],
        log_data=True,
        log_plots=True,
        remove_outliers=config['outliers'],
        normalize=config['normalize'],
        remove_multicollinearity=config['multicol'],
        multicollinearity_threshold=0.9,
        pca=config['pca'],
        pca_components=0.95,
        session_id=123
    )

    # Cargar el mejor modelo previamente guardado (Random Forest)
    logger.info("📦 Cargando modelo Random Forest para tuning...")
    rf_model = load_model(model_path)

    # Tuning de hiperparámetros
    with mlflow.start_run(run_name=f"{experiment}_rf_tuning", nested=True):
        logger.info("🔧 Ejecutando tune_model() sobre RandomForestRegressor...")
        tuned_rf = tune_model(rf_model, optimize='RMSE', n_iter=25, verbose=True)

    # Obtener y guardar resultados
    # Obtener resultados del tuning
    tuned_leaderboard = pull()
    tuned_leaderboard.to_csv(base_dir / "rf_tuned_leaderboard.csv")

    # Guardar modelo ajustado
    tuned_model_path = best_models_dir / "pycaret_rf_tuned_model"
    save_model(tuned_rf, tuned_model_path)

    # Guardar resumen de métricas si existe la fila 'Mean'
    summary_path = base_dir / "summary_rf_tuned_model.csv"
    if 'Mean' in tuned_leaderboard.index:
        best_row = tuned_leaderboard.loc['Mean'][['MAE', 'MSE', 'RMSE', 'R2', 'MAPE']]
        best_row.to_frame().T.to_csv(summary_path, index=False)
        logger.info("✅ Guardado resumen del modelo ajustado.")
    else:
        logger.warning("❗ No se encontró la fila 'Mean' en tuned_leaderboard. No se guardó el resumen.")

    # Guardar hiperparámetros ajustados
    params_path = base_dir / "rf_tuned_model_params.txt"
    with open(params_path, "w") as f:
        f.write(f"# Hiperparámetros del modelo ajustado ({type(tuned_rf).__name__})\n")
        for k, v in tuned_rf.get_params().items():
            f.write(f"{k}: {v}\n")
    logger.info(f"📄 Hiperparámetros guardados en {params_path}")

    logger.info("✅ Tuning finalizado exitosamente para Random Forest.")

if __name__ == "__main__":
    main()
