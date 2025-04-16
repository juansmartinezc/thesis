import os
from dotenv import load_dotenv
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataprep.eda import create_report
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb


# === 1. Definiciones ===
def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))

scoring = {
    'r2': 'r2',
    'mae': make_scorer(mean_absolute_error),
    'rmse': make_scorer(rmse)
}
load_dotenv()
source_data_directory = os.environ.get("SOURCE_DATA_DIRECTORY")
file_path = f'{source_data_directory}/historical_monthly_climate_data_apr_sept_by_scan_stations_and_nasa_final.csv'
df = pd.read_csv(file_path)
report_directory = 'reports'
columns_to_drop = ['Unnamed: 0', 'state_alpha', 'county_code', 'stationTriplet',
       'stationId', 'name', 'lat_centroid', 'lon_centroid', 'latitude',
       'longitude', 'unit_desc']
results_directory = 'results'
df = df.drop(columns=columns_to_drop)
print(df.columns)

os.makedirs(report_directory, exist_ok=True)

def correlation_analysis(df):
    # Correlación numérica
    plt.figure(figsize=(12, 10))
    #sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Matriz de correlación")
    plt.tight_layout()
    plt.savefig(os.path.join(report_directory, 'correlation_matrix.png'))
    plt.show()
'''
def random_forest(df):
    # Separar X y y
    target = 'GY (Kg/H)'  # <- Cambiar esto por tu variable objetivo
    X = df.drop(columns=[target])
    y = df[target]

    # Opcional: seleccionar solo numéricas para un primer análisis simple
    X_numeric = X.select_dtypes(include=['float64', 'int64'])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

    # Modelo de Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Importancia de las características
    importances = pd.Series(model.feature_importances_, index=X_numeric.columns)
    importances = importances.sort_values(ascending=False)

    # Graficar importancias
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=importances.index)
    plt.title("Importancia de características - Random Forest")
    plt.tight_layout()
    plt.savefig(os.path.join(report_directory, 'feature_importance_rf.png'))
    plt.show()

def multicoliniality_analysis():
    # Usamos solo variables numéricas
    X_numeric = df.select_dtypes(include=['float64', 'int64']).drop(columns=['GY (Kg/H)'])  # Ajusta tu variable target

    # Añadimos constante para el análisis
    X_vif = add_constant(X_numeric)

    # Calculamos el VIF
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X_vif.columns
    vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

    # Mostramos
    print(vif_data.sort_values('VIF', ascending=False))
correlation_analysis(df)
    '''


######Preparar el archivo#############
#report = create_report(df)
#report.save(f'{report_directory}/{report}')
df.to_csv(f'{report_directory}/input_data.csv', index=False)

###################Analisis numerico##############
numeric_summary = df.describe()
plt.rcParams['figure.figsize'] = (10, 6)
correlation_matrix = df.corr(numeric_only=True)

########Analisis de rendimiento
# Visualización de la distribución de la variable objetivo (rendimiento)
plt.figure(figsize=(10, 5))
sns.histplot(df['Value'], bins=50, kde=True)
plt.title("Distribución del rendimiento mensual de maíz ('Value')")
plt.xlabel("Rendimiento (Value)")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{results_directory}/yield_histogram.png')

# Evolución del rendimiento a lo largo del tiempo
df_grouped = df.groupby('year')['Value'].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(data=df_grouped, x='year', y='Value', marker='o')
plt.title("Evolución del rendimiento promedio de maíz por año")
plt.xlabel("Año")
plt.ylabel("Rendimiento promedio (Value)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{results_directory}/yield_evolution.png')

# Calcular el rendimiento promedio por estado
state_avg = df.groupby('state_name')['Value'].mean().sort_values(ascending=False).reset_index()

# Visualizar rendimiento promedio por estado
plt.figure(figsize=(12, 6))
sns.barplot(data=state_avg, x='Value', y='state_name', palette='viridis')
plt.title("Rendimiento promedio de maíz por estado (2000–2020)")
plt.xlabel("Rendimiento promedio (Value)")
plt.ylabel("Estado")
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{results_directory}/yield_evolution_by_state.png')

# Para condados, seleccionamos los 20 con mayor promedio
county_avg = df.groupby(['state_name', 'county_name'])['Value'].mean().sort_values(ascending=False).head(20).reset_index()

# Visualizar los 20 condados con mayor rendimiento promedio
plt.figure(figsize=(12, 6))
sns.barplot(data=county_avg, x='Value', y='county_name', hue='state_name', dodge=False)
plt.title("Top 20 condados con mayor rendimiento promedio de maíz")
plt.xlabel("Rendimiento promedio (Value)")
plt.ylabel("Condado")
plt.legend(title="Estado", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{results_directory}/counties_yield_top_20.png')

#########Analisis de feature importance##################

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
##############
#Posibles variables a conservar##########
'''
year

county_name

state_name

ocd

RH2M

silt

SMS_-8

clay

phh2o

TMAX
'''
#########Analisis de random forest##################

'''
sand – redundante con silt y clay

TMIN – muy correlacionada con TMAX

TAVG – redundante con TMAX y TMIN

cec – poca importancia y baja correlación

month – casi sin aporte explicativo

WS10M – poca importancia, sin correlación fuerte

PRCP – baja importancia y correlación marginal

'''


'''
1 er analisis: todas las variables:
Variables temporales: year, month

Variables categóricas: state_name, county_name

Variables edáficas: ocd, cec, phh2o, clay, silt, sand

Variables climáticas: TMAX, TMIN, TAVG, PRCP, RH2M, WS10M, SMS_-8
'''

# Separar variables categóricas y numéricas
categorical_cols = ['state_name', 'county_name']
numerical_cols = [col for col in df.columns if col not in categorical_cols + ['Value']]
print("Entrenamiento del modelo")
# Dividir en X e y
# === 2. Cargar y preparar datos ===
# Asegúrate de tener cargado el DataFrame df (por ejemplo, df = pd.read_csv(...))
categorical_cols = ['state_name', 'county_name']
numerical_cols = [col for col in df.columns if col not in categorical_cols + ['Value']]
X = df.drop(columns='Value')
y = df['Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Preprocesamiento
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
])

# === 3. Definir modelos ===
models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, objective='reg:squarederror', random_state=42),
    "MLP": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    "SVR": SVR(C=10, epsilon=0.2, kernel='rbf')
}

# === 4. Evaluación con validación cruzada ===
results = {}
for name, model in models.items():
    print(f"Entrenando y evaluando modelo: {name}")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    cv_result = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    results[name] = cv_result
    
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

# === 6. Mostrar resumen en consola ===
for name, res in results.items():
    print(f"\n📊 Resultados de {name}")
    print(f"R² train:     {np.mean(res['train_r2']):.3f} ± {np.std(res['train_r2']):.3f}")
    print(f"R² val:       {np.mean(res['test_r2']):.3f} ± {np.std(res['test_r2']):.3f}")
    print(f"MAE val:      {np.mean(res['test_mae']):.2f}")
    print(f"RMSE val:     {np.mean(res['test_rmse']):.2f}")