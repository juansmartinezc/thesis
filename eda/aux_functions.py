import os
import seaborn as sns
import matplotlib as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import cross_validate, train_test_split
###############Analisis de correlación############################



def correlation_analysis(df, report_directory = 'reports'):
    # Correlación numérica
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Matriz de correlación")
    plt.tight_layout()
    plt.savefig(os.path.join(report_directory, 'correlation_matrix.png'))
    plt.show()

def plot_monthly_crop_yield(df, results_directory = 'results'):
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Value'], bins=50, kde=True)
    plt.title("Distribución del rendimiento mensual de maíz ('Value')")
    plt.xlabel("Rendimiento (Value)")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{results_directory}/yield_histogram.png')

def plot_yearly_crop_yield(df_grouped, results_directory = 'results'):
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_grouped, x='year', y='Value', marker='o')
    plt.title("Evolución del rendimiento promedio de maíz por año")
    plt.xlabel("Año")
    plt.ylabel("Rendimiento promedio (Value)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{results_directory}/yield_evolution.png')

def plot_crop_yield_by_status(df, results_directory = 'results'):
    # Visualizar rendimiento promedio por estado
    # Calcular el rendimiento promedio por estado
    state_avg = df.groupby('state_name')['Value'].mean().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=state_avg, x='Value', y='state_name', palette='viridis')
    plt.title("Rendimiento promedio de maíz por estado (2000–2020)")
    plt.xlabel("Rendimiento promedio (Value)")
    plt.ylabel("Estado")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{results_directory}/yield_evolution_by_state.png')

def plot_crop_yield_by_status_top_20(df, results_directory):
    # Visualizar los 20 condados con mayor rendimiento promedio
    # Para condados, seleccionamos los 20 con mayor promedio
    county_avg = df.groupby(['state_name', 'county_name'])['Value'].mean().sort_values(ascending=False).head(20).reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=county_avg, x='Value', y='county_name', hue='state_name', dodge=False)
    plt.title("Top 20 condados con mayor rendimiento promedio de maíz")
    plt.xlabel("Rendimiento promedio (Value)")
    plt.ylabel("Condado")
    plt.legend(title="Estado", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{results_directory}/counties_yield_top_20.png')

def feature_importance_analysis(df):
    df_features = df.copy()
    encoder = OrdinalEncoder()
    df_features[['state_name', 'county_name']] = encoder.fit_transform(df_features[['state_name', 'county_name']])

    # Variables predictoras y objetivo
    X = df_features.drop(columns=['Value'])
    y = df_features['Value']

    # División simple en train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo base
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Importancia de características
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_features = importances.head(25).reset_index()
    top_features.columns = ['Feature', 'Importance']
    print(top_features)
    # Visualización
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_features, x='Importance', y='Feature', palette='Blues_d')
    plt.title("Top 15 características más importantes para predecir el rendimiento")
    plt.xlabel("Importancia")
    plt.ylabel("Variable")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{results_directory}/feature_importance.png')