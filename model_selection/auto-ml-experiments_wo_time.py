import os
from pathlib import Path
import logging
import pandas as pd
from dotenv import load_dotenv

import mlflow
import mlflow.sklearn

from pycaret.regression import (
    setup,
    compare_models,
    pull,
    finalize_model,
    save_model,
)

# ────────────────────────────────────────────────────────────────────────────────
# Configuración de logging
# ────────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    """Carga el dataset desde una ruta dada."""
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    return pd.read_csv(path)


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa características (X) y variable objetivo (y)."""
    drop_cols = [
        'Unnamed: 0', 'state_alpha', 'county_code', 'stationTriplet',
        'stationId', 'name', 'lat_centroid', 'lon_centroid', 'unit_desc', 'year', 'month'
    ]
    X = df.drop(columns=drop_cols + ['Value'], errors='ignore')
    y = df['Value']
    return X, y


def run_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
    experiment_name: str
) -> tuple:
    """
    Ejecuta un experimento de PyCaret con la configuración dada.
    Retorna el modelo final y el leaderboard resultante.
    """
    # Habilitar autolog de scikit-learn en MLflow
    mlflow.sklearn.autolog(
        log_model_signatures=True
    )  # autolog registra hiperparámetros, métricas y artefactos automáticamente :contentReference[oaicite:3]{index=3}

    # Concatenar X e y para pasarlos a setup()
    df_exp = pd.concat([X, y.rename("target")], axis=1)

    # Setup sin silent (eliminado en PyCaret 3.x) y con configuración completa
    setup(
        data=df_exp,
        target="target",
        log_experiment=config['log_experiment'],
        log_data=True,
        log_plots=True,
        experiment_name=experiment_name,
        fold=5,
        remove_outliers=config['outliers'],
        normalize=config['normalize'],
        remove_multicollinearity=config['multicol'],
        multicollinearity_threshold=0.9,
        pca=config['pca'],
        pca_components=0.95
    )  # silent ya no existe en PyCaret 3.x :contentReference[oaicite:4]{index=4}

    # Ejecutar y finalizar modelo dentro de un run de MLflow
    with mlflow.start_run(run_name=experiment_name, nested=True):
        best_model = compare_models(sort='RMSE', verbose=False)
        final_model = finalize_model(best_model)

    leaderboard = pull()  # extrae métricas del último experimento :contentReference[oaicite:5]{index=5}
    return final_model, leaderboard


def save_results(
    model,
    leaderboard: pd.DataFrame,
    base_dir: Path,
    best_models_dir: Path,
    summary_path: Path
):
    """Guarda el modelo, el leaderboard completo y un resumen del mejor run."""
    # 1) Guardar pipeline final
    model_file = best_models_dir / "pycaret_best_model"
    save_model(model, model_file)

    # 2) Leaderboard completo
    leaderboard.to_csv(base_dir / "pycaret_leaderboard.csv", index=False)

    # 3) Resumen del mejor run
    best_row = leaderboard.iloc[0][['Model', 'MAE', 'MSE', 'RMSE', 'R2', 'MAPE']]
    best_row.to_frame().T.to_csv(summary_path, index=False)

def make_experiment_name(base: str, config: dict) -> str:
    # Sólo incluimos en el nombre las opciones activas o las que quieras resaltar
    parts = [f"{k[:3]}={v}" for k, v in config.items()]
    # Ejemplo de claves abreviadas: log, out, nor, rar, mul, pca
    return base + "__" + "__".join(parts)

def main():
    # Cargar variables de entorno
    load_dotenv()
    data_dir = os.getenv("SOURCE_DATA_DIRECTORY")
    if not data_dir:
        logger.error("La variable SOURCE_DATA_DIRECTORY no está definida en .env")
        return

    # 1) Carga de datos
    data_path = Path(data_dir) / "historical_monthly_climate_data_apr_sept_by_scan_stations_and_nasa_final.csv"
    df = load_data(data_path)
    X, y = prepare_data(df)

    # 2) Configuración del experimento
    config = {
        'log_experiment': True,
        'outliers'     : True,
        'normalize'    : True,
        'multicol'     : False,
        'pca'          : False
    }
    experiment = make_experiment_name("auto_ml_wo_time", config)

    # 3) Rutas de salida
    base_dir        = Path("results") / "models_results" / experiment
    best_models_dir = base_dir / "best_models"
    summary_path    = base_dir / "summary_best_models.csv"

    # Crear directorios necesarios
    best_models_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directorios creados en {base_dir}")  # uso de pathlib para creación de carpetas 

    # 4) Ejecución del experimento
    logger.info("🔧 Iniciando experimento AutoML con PyCaret")
    best_model, leaderboard = run_experiment(X, y, config, experiment)

    # 5) Guardado de resultados
    logger.info("💾 Guardando resultados")
    save_results(best_model, leaderboard, base_dir, best_models_dir, summary_path)
    logger.info("✅ Proceso completado exitosamente")


if __name__ == "__main__":
    main()
