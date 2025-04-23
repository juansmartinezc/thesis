import os
import joblib

def save_best_models(best_models_path, name, best_model):
    dir_path = os.path.join(best_models_path, f'{name}')
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f'{name.lower()}_model.pkl')
    joblib.dump(best_model, file_path)