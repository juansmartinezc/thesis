import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from scipy.stats import uniform, randint
from dotenv import load_dotenv

# =========================
# MÉTRICAS PERSONALIZADAS
# =========================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def get_scorers():
    return {
        'r2': 'r2',
        'mae': make_scorer(mean_absolute_error),
        'neg_root_mean_squared_error': make_scorer(rmse, greater_is_better=False)
    }

# =========================
# PREPROCESADOR
# =========================
def build_preprocessor(df, target='Value'):
    print("⏳ Preparando preprocesador...")
    categorical_cols = ['state_name', 'county_name']
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.difference([target]).tolist()
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
    ])
    return preprocessor, categorical_cols, numerical_cols

# =========================
# FUNCIÓN DE TUNING
# =========================
def tune_model(name, model, param_dist, X, y, preprocessor, scoring, save_dir):
    print(f"\n🚀 Iniciando tuning para {name}...")
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    search = RandomizedSearchCV(
        pipe,
        param_distributions={'regressor__' + k: v for k, v in param_dist.items()},
        scoring=scoring,
        refit='neg_root_mean_squared_error',
        cv=3,
        n_iter=20,
        verbose=1,
        random_state=42,
        return_train_score=True
    )

    search.fit(X, y)
    results = pd.DataFrame(search.cv_results_)

    # Guardar gráfico
    print(f"📈 Guardando gráfico de errores para {name}...")
    plt.figure(figsize=(8, 5))
    plt.plot(-results['mean_train_neg_root_mean_squared_error'], label='Train RMSE', marker='o')
    plt.plot(-results['mean_test_neg_root_mean_squared_error'], label='Validation RMSE', marker='o')
    plt.title(f'RMSE por configuración - {name}')
    plt.xlabel('Iteración')
    plt.ylabel('RMSE (bu/acre)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{name.lower()}_rmse_plot.png'))
    plt.close()

    # Guardar resultados
    print(f"💾 Guardando resultados de tuning para {name}...")
    results.to_csv(os.path.join(save_dir, f'results_{name.lower()}.csv'), index=False)
    with open(os.path.join(save_dir, f'best_params_{name.lower()}.json'), 'w') as f:
        json.dump(search.best_params_, f, indent=4)

    return search.best_estimator_, -search.best_score_, search.best_params_

# =========================
# EXPERIMENTO PRINCIPAL
# =========================
def main():
    print("📦 Cargando dataset...")
    load_dotenv()
    source_data_directory = os.environ.get("SOURCE_DATA_DIRECTORY")
    df = pd.read_csv(f'{source_data_directory}/historical_monthly_climate_data_apr_sept_by_scan_stations_and_nasa_final.csv')
    X = df.drop(columns='Value')
    y = df['Value']

    # === Crear carpeta para resultados ===
    save_dir = 'resultados'
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 Carpeta de resultados creada: {save_dir}")

    # === Preprocesamiento ===
    preprocessor, _, _ = build_preprocessor(df)
    scoring = get_scorers()

    # === Tuning XGBoost ===
    xgb_params = {
        'n_estimators': randint(100, 300),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4)
    }
    best_xgb, score_xgb, params_xgb = tune_model("XGBoost", xgb.XGBRegressor(objective='reg:squarederror', random_state=42), xgb_params, X, y, preprocessor, scoring, save_dir)

    # === Tuning MLP ===
    mlp_params = {
        'hidden_layer_sizes': [(100,), (100, 50), (150, 75)],
        'activation': ['relu', 'tanh'],
        'alpha': uniform(0.0001, 0.01),
        'learning_rate_init': uniform(0.001, 0.1)
    }
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