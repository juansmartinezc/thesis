import numpy as np
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
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

