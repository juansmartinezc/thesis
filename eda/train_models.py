import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.svm import SVR
from dotenv import load_dotenv
from fine_tuning import tune_model
from scipy.stats import uniform, randint
from metric_functions import get_scorers
from preprocessor import build_preprocessor 
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from save_results.save_randomseach_results import save_best_models
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


##Posibles parametros de entrenamiento.

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

# === Tuning rf ===
rf_params = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['auto', 'sqrt', 'log2']
}

# === Tuning svr ===
svr_params = {
    'C': uniform(1, 10),
    'epsilon': uniform(0.01, 0.5),
    'kernel': ['rbf', 'linear', 'poly'],
    'gamma': ['scale', 'auto']
}

# =========================
# 1 experimento
# Variables para este experimentos
# Metodo de escalizacion: 
# =========================
source_data_directory = os.environ.get("SOURCE_DATA_DIRECTORY")
results_path = 'results'
best_models_path = f"{results_path}/best_models"
summary_path = os.path.join(best_models_path, 'summary_best_models.csv')
models_results_path = f"{results_path}/models_results"

def split_train_test_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    return X_train, X_test, y_train, y_test

def preprocess_data(numerical_cols, categorical_cols):
    preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
])
    return preprocessor

def main():
    print("📦 Cargando dataset...")
    load_dotenv()
    df = pd.read_csv(f'{source_data_directory}/historical_monthly_climate_data_apr_sept_by_scan_stations_and_nasa_final.csv')

    #########Experimento 1##################
    #### Todas las variables son tomadas####
    print("Entrenamiento del modelo")

    # Obtener variables numericas y categoricas.

    ################# 1. Dividir en X e y#####################
    
    categorical_cols = ['state_name', 'county_name']
    numerical_cols = [col for col in df.columns if col not in categorical_cols + ['Value']]
    X = df.drop(columns='Value')
    y = df['Value']
    
    X_train, X_test, y_train, y_test = split_train_test_data(X, y)

    ################# 2. Dividir en X e Y#####################
    preprocessor = preprocess_data(numerical_cols, categorical_cols)

    # === Crear carpeta para resultados ===
    os.makedirs(results_path, exist_ok=True)
    print(f"📁 Carpeta de resultados creada: {results_path}")

    os.makedirs(best_models_path, exist_ok=True)
    print(f"📁 Carpeta para almacenar los mejores resultados creada: {best_models_path}")

    os.makedirs(models_results_path, exist_ok=True)
    print(f"📁 Carpeta de los resultados de los modelos creada: {models_results_path}")
    # === Preprocesamiento ===
    preprocessor, _, _ = build_preprocessor(df)
    scoring = get_scorers()

    # === Tuning XGBoost ===
    best_xgb, score_xgb, params_xgb = tune_model("XGBoost", xgb.XGBRegressor(objective='reg:squarederror', random_state=42), xgb_params, X, y, preprocessor, scoring, models_results_path)

    best_mlp, score_mlp, params_mlp = tune_model("MLP", MLPRegressor(max_iter=500, random_state=42), mlp_params, X, y, preprocessor, scoring, models_results_path)

    best_rf, score_rf, params_rf = tune_model("RandomForest", RandomForestRegressor(random_state=42), rf_params, X, y, preprocessor, scoring, models_results_path)

    best_svr, score_svr, params_svr = tune_model("SVR", SVR(), svr_params, X, y, preprocessor, scoring, models_results_path)

    # === Guardar resumen general ===
    # === Guardar resumen general ===
    print("📝 Guardando resumen general de mejores modelos...")
    summary = pd.DataFrame({
        'Model': ['XGBoost', 'MLP', 'RandomForest', 'SVR'],
        'Best RMSE': [score_xgb, score_mlp, score_rf, score_svr]
    })

    
    summary.to_csv(summary_path, index=False)

    print(f"✅ Resumen guardado en: {summary_path}")
    save_best_models(best_models_path, best_xgb)
    save_best_models(best_models_path, best_mlp)
    save_best_models(best_models_path, best_rf)
    save_best_models(best_models_path, best_svr)

    '''    
    # === 6. Mostrar resumen en consola ===
    for name, res in results.items():
        print(f"\n📊 Resultados de {name}")
        print(f"R² train:     {np.mean(res['train_r2']):.3f} ± {np.std(res['train_r2']):.3f}")
        print(f"R² val:       {np.mean(res['test_r2']):.3f} ± {np.std(res['test_r2']):.3f}")
        print(f"MAE val:      {np.mean(res['test_mae']):.2f}")
        print(f"RMSE val:     {np.mean(res['test_rmse']):.2f}")
    '''
if __name__ == '__main__':
    main()