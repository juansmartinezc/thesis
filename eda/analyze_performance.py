import matplotlib as plt

def plot_errors(cv_result, name):
    # === 5. Graficar errores ===
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cv_result['train_rmse'], label='Train RMSE', marker='o')
    ax.plot(cv_result['test_rmse'], label='Validation RMSE', marker='o')
    ax.set_title(f'RMSE por fold - {name}')
    ax.set_xlabel('Fold')
    ax.set_ylabel('RMSE (bu/acre)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"{name}_RMSE_plot.png")  # Guardar imagen
    plt.close()


