import os
import joblib

def save_best_models(best_models_dir, best_model):
    joblib.dump(f"{best_model}", os.path.join(f"{best_models_dir}", f'{best_model}.pkl'))