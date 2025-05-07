import os
import mlflow
import logging
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, r2_score
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    load_dotenv()
    data_dir = os.getenv("SOURCE_DATA_DIRECTORY")
    if not data_dir:
        logger.error("SOURCE_DATA_DIRECTORY no está definida en el .env")
        return

    config = {
        'log_experiment': True,
        'outliers': False,
        'normalize': True,
        'multicol': False,
        'pca': False
    }

    experiment = "gridsearch_rf__" + "__".join([f"{k[:3]}={v}" for k, v in config.items()])
    print(f"experiment: {experiment}")
    base_dir = Path("results") / "models_results" / experiment
    best_models_dir = base_dir / "best_models"
    best_models_dir.mkdir(parents=True, exist_ok=True)
    model_path = best_models_dir / "sklearn_rf_best_model.pkl"
    
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

    # Hiperparámetros para GridSearch
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    rf = RandomForestRegressor(random_state=123)

    scorer = make_scorer(r2_score)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring=scorer, verbose=1, n_jobs=-1)

    logger.info("🔍 Ejecutando GridSearchCV para RandomForestRegressor...")
    with mlflow.start_run(run_name=f"{experiment}_rf_gridsearch", nested=True):
        grid_search.fit(X, y)

        # Loguear los mejores hiperparámetros
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_score_r2", grid_search.best_score_)

        # Guardar modelo
        joblib.dump(grid_search.best_estimator_, model_path)
        mlflow.sklearn.log_model(grid_search.best_estimator_, "model")

    # Guardar resultados completos
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(base_dir / "rf_gridsearch_leaderboard.csv", index=False)

    # Guardar resumen de métricas
    best_params = grid_search.best_params_
    with open(base_dir / "rf_gridsearch_best_params.txt", "w") as f:
        f.write("# Mejores hiperparámetros encontrados:\n")
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")

    logger.info("✅ GridSearchCV finalizado exitosamente.")

if __name__ == "__main__":
    main()
