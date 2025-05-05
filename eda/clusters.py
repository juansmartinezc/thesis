import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from utils.states_codes import state_alpha_to_fips 
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

# Graficar el resultado
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.xlabel("Número de clusters")
plt.ylabel("Coeficiente de silueta promedio")
plt.title("Selección del número óptimo de clusters (Silueta)")
plt.grid(True)
plt.show()
'''
# Aplicar KMeans con 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)
df_cluster['cluster'] = kmeans.fit_predict(X_scaled)

# Agrega la columna de cluster al dataframe original si es necesario
df['cluster'] = df_cluster['cluster']

df['state_fips'] = df['state_alpha'].map(state_alpha_to_fips)
'''
sns.boxplot(x='cluster', y='Value', data=df)  # reemplaza yield_variable_name por la de rendimiento
plt.title("Distribución de rendimiento por cluster")
plt.show()
'''
# 1. Cargar shapefile
shapefile_path = "shape_files/state/cb_2018_us_state_20m.shp"
gdf = gpd.read_file(shapefile_path)
print(gdf.columns)
print(gdf.head())


# 2. Revisar el nombre de las columnas para identificar el campo de estado
print(gdf.columns)

# 4. Agrupar por estado si es necesario (por ejemplo, promedio por cluster en cada estado)
df_state_clusters = df.groupby('state_fips')['cluster'].agg(lambda x: x.value_counts().idxmax()).reset_index()

# 5. Unir el GeoDataFrame con tus datos
gdf = gdf.merge(df_state_clusters, left_on='STATEFP', right_on='state_fips', how='left')  # usa 'STATEFP', 'NAME', 'STUSPS', etc.
print(gdf.columns)

cluster_states = gdf.groupby('cluster')['NAME'].apply(list)
for cluster_id, states in cluster_states.items():
    print(f"Cluster {int(cluster_id)}:")
    for state in states:
        print(f" - {state}")
    print()
# 6. Plotear
fig, ax = plt.subplots(figsize=(12, 8))
gdf.plot(column='cluster', ax=ax, legend=True, cmap='tab10', edgecolor='black')
plt.title('Visualización Geográfica de Clusters por Estado')
plt.axis('off')
plt.show()