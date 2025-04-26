import os
import pandas as pd
import xgboost as xgb
from sklearn.svm import SVR
from dotenv import load_dotenv
from scipy.stats import uniform, randint
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from model_selection.fine_tuning import tune_model
from model_selection.metric_functions import get_scorers
from model_selection.aux_functions import build_preprocessor 


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
    'max_features': [None, 'sqrt', 'log2']
}

# === Tuning svr ===
svr_params = {
    'C': uniform(1, 10),
    'epsilon': uniform(0.01, 0.5),
    'kernel': ['rbf', 'linear', 'poly'],
    'gamma': ['scale', 'auto']
}

# =========================
# Experimento random search
# Variables para este experimentos
# Metodo de escalizacion: Standard scaler
# =========================

# =========================
# Definicion de directorios
# =========================

results_path = 'results'
experiment = 'randomsearch'
models_path = 'models_results'
source_data_directory = os.environ.get("SOURCE_DATA_DIRECTORY")
models_results_path = f"{results_path}/{models_path}/{experiment}"
best_models_path = f"{results_path}/{models_path}/{experiment}/best_models"
summary_path = f"{results_path}/{models_path}/{experiment}/summary_best_models.csv"



# =========================
# Main Script
# =========================

def main():
    print("📦 Cargando dataset...")
    load_dotenv()
    df = pd.read_csv(f'{source_data_directory}/historical_monthly_climate_data_apr_sept_by_scan_stations_and_nasa_final.csv')

    #########Experimento 1##################
    #### Todas las variables son tomadas####
    print("Entrenamiento del modelo")

    # Obtener variables numericas y categoricas.

    ################# 1. Capturar variables categoricas y numericas#####################
    
    categorical_cols = ['state_name', 'county_name']
    numerical_cols = [col for col in df.columns if col not in categorical_cols + ['Value']]
    X = df.drop(columns='Value')
    y = df['Value']
    
    ########################### 2. Preprocesar los datos###############################
    
    preprocessor, _, _ = build_preprocessor(numerical_cols, categorical_cols)
    
    ########################### 3. Creacion de directorio###############################
    
    os.makedirs(results_path, exist_ok=True)
    print(f"📁 Carpeta de resultados creada: {results_path}")

    os.makedirs(best_models_path, exist_ok=True)
    print(f"📁 Carpeta para almacenar los mejores resultados creada: {best_models_path}")

    os.makedirs(models_results_path, exist_ok=True)
    print(f"📁 Carpeta de los resultados de los modelos creada: {models_results_path}")

    os.makedirs(models_results_path, exist_ok=True)
    print(f"📁 Carpeta de los resumenes de los modelos creada: {summary_path}")
    
    ###################### 4. Obtener metricas de rendimiento###############################
    
    scoring = get_scorers()
    
    ############### 5. Obtencion de media para normalizar la salidas #######################
    y_mean = y.mean()

    ################################# 6. Tunning de XGBOOST #######################################
    score_xgb = tune_model("XGBoost", xgb.XGBRegressor(objective='reg:squarederror', random_state=42), xgb_params, X, y, preprocessor, scoring, models_results_path, best_models_path)
    nrmse_xgb = (score_xgb / y_mean) * 100
    print("📝 nrmse_xgb: ", nrmse_xgb)
    
    ################################## 7. Tunning de MLP ##########################################
    score_mlp = tune_model("MLP", MLPRegressor(max_iter=500, random_state=42), mlp_params, X, y, preprocessor, scoring, models_results_path, best_models_path)
    nrmse_mlp = (score_mlp / y_mean) * 100
    print("📝 nrmse_mlp: ", nrmse_mlp)
    
    ########################### 8. Tunning de Random forest ########################################
    score_rf = tune_model("RandomForest", RandomForestRegressor(random_state=42), rf_params, X, y, preprocessor, scoring, models_results_path, best_models_path)
    nrmse_rf = (score_rf / y_mean) * 100
    print("📝 nrmse_rf: ", nrmse_rf)
    
    ############################## 9. Tunning de SVR################################################
    score_svr = tune_model("SVR", SVR(), svr_params, X, y, preprocessor, scoring, models_results_path, best_models_path)
    nrmse_svr = (score_svr / y_mean) * 100
    print("📝 nrmse_svr: ", nrmse_svr)
    
    ############################## 10. Guardar el resumen################################################
    print("📝 Guardando resumen general de mejores modelos...")
    summary = pd.DataFrame({
        'Model': ['XGBoost', 'MLP', 'RandomForest', 'SVR'],
        'Best RMSE': [score_xgb, score_mlp, score_rf, score_svr],
        'NRMSE (%)': [nrmse_xgb, nrmse_mlp, nrmse_rf, nrmse_svr]
    })

    summary.to_csv(summary_path, index=False)

    print(f"✅ Resumen guardado en: {summary_path}")
     
############################## Ejecucion principal ################################################
if __name__ == '__main__':
    main()