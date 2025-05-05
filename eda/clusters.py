import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
# Cargar datos
df = pd.read_csv("source_data/historical_monthly_climate_data_apr_sept_by_scan_stations_and_nasa_final.csv")

# Vista general del DataFrame
print(df.head())
print(df.columns)

# Eliminar columnas no deseadas
columns_to_drop = ['Unnamed: 0', 'stationId', 'lat_centroid', 'lon_centroid', 'latitude',
                    'longitude','state_name','county_name', 'state_alpha', 'county_code', 'month', 'year', 'unit_desc']  # ajusta según lo que veas en tu archivo
df_cluster = df.drop(columns=columns_to_drop)

# Eliminar columnas no numéricas si quedan
df_cluster = df_cluster.select_dtypes(include=['number'])

print(df_cluster.columns)
# Escalar las variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)



inertia = []
range_n_clusters = range(2, 10)

# Guardar los valores de silueta
silhouette_scores = []
'''
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"Para n_clusters = {n_clusters}, el coeficiente de silueta promedio es: {silhouette_avg:.4f}")
'''
# Graficar el resultado
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.xlabel("Número de clusters")
plt.ylabel("Coeficiente de silueta promedio")
plt.title("Selección del número óptimo de clusters (Silueta)")
plt.grid(True)
plt.show()

# Aplicar KMeans con 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)
df_cluster['cluster'] = kmeans.fit_predict(X_scaled)

# Agrega la columna de cluster al dataframe original si es necesario
df['cluster'] = df_cluster['cluster']

sns.boxplot(x='cluster', y='yield_variable_name', data=df)  # reemplaza yield_variable_name por la de rendimiento
plt.title("Distribución de rendimiento por cluster")
plt.show()