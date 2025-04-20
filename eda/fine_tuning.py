import os
import json
import pandas as pd
import matplotlib as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_validate
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