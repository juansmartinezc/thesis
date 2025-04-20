import os
import pandas as pd
from preprocessor import build_preprocessor 
from sklearn.neural_network import MLPRegressor
from metric_functions import get_scorers
from fine_tuning import tune_model
import xgboost as xgb
from scipy.stats import uniform, randint
from dotenv import load_dotenv

# =========================
# EXPERIMENTO PRINCIPAL
# =========================
source_data_directory = os.environ.get("SOURCE_DATA_DIRECTORY")
save_dir = 'resultados'

xgb_params = {
        'n_estimators': randint(100, 300),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4)
    }

# === Tuning MLP ===
mlp_params = {
    'hidden_layer_sizes': [(100,), (100, 50), (150, 75)],
    'activation': ['relu', 'tanh'],
    'alpha': uniform(0.0001, 0.01),
    'learning_rate_init': uniform(0.001, 0.1)
}

def main():
    print("📦 Cargando dataset...")
    load_dotenv()
    df = pd.read_csv(f'{source_data_directory}/historical_monthly_climate_data_apr_sept_by_scan_stations_and_nasa_final.csv')
    
    ##Dividir entre datos de entrada y salida.
    X = df.drop(columns='Value')
    y = df['Value']

    # === Crear carpeta para resultados ===
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 Carpeta de resultados creada: {save_dir}")

    # === Preprocesamiento ===
    preprocessor, _, _ = build_preprocessor(df)
    scoring = get_scorers()

    # === Tuning XGBoost ===
    best_xgb, score_xgb, params_xgb = tune_model("XGBoost", xgb.XGBRegressor(objective='reg:squarederror', random_state=42), xgb_params, X, y, preprocessor, scoring, save_dir)

    
    best_mlp, score_mlp, params_mlp = tune_model("MLP", MLPRegressor(max_iter=500, random_state=42), mlp_params, X, y, preprocessor, scoring, save_dir)

    # === Guardar resumen general ===
    print("📝 Guardando resumen general de mejores modelos...")
    summary = pd.DataFrame({
        'Model': ['XGBoost', 'MLP'],
        'Best RMSE': [score_xgb, score_mlp]
    })
    summary.to_csv(os.path.join(save_dir, 'tuning_summary.csv'), index=False)
    print("✅ Proceso completado con éxito.")

if __name__ == '__main__':
    main()